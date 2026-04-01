import os
import json
import time
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, multilabel_confusion_matrix
import threading


API_KEY = "sk-kkkkkkkkkkkkkkkk"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "glm-5"

BATCH_SIZE = 10
MAX_WORKERS = 5
TEST_DATA_PATH = "cluener_test.jsonl"
OUTPUT_CSV = f"{MODEL_NAME}_batch_results.csv"


# ================= 2. 标签清洗工具 =================
def normalize_label(raw_label):
    if not isinstance(raw_label, str):
        return None
    text = raw_label.strip().lower().lstrip('._-')
    if 'identity' in text: return 'Identity'
    if 'location' in text: return 'Location'
    if 'credential' in text: return 'Credential'
    if 'safe' in text: return 'Safe'
    return None


def batch_clean_labels(label_list_of_lists):
    cleaned_batch = []
    for labels in label_list_of_lists:
        clean_row = set()
        for l in labels:
            norm = normalize_label(l)
            if norm:
                clean_row.add(norm)
        if not clean_row:
            clean_row.add('Safe')
        cleaned_batch.append(list(clean_row))
    return cleaned_batch


def parse_single_output(text):
    """解析 API 返回的单个结果元素"""
    if not text or text == ["API_FAILED"]:
        return ["Safe"]
    found_raw = []
    valid_keywords = ['Identity', 'Location', 'Credential', 'Safe']
    text_str = str(text)
    for key in valid_keywords:
        if re.search(r'\b' + re.escape(key) + r'\b', text_str, re.IGNORECASE):
            found_raw.append(key)
    return found_raw if found_raw else ["Safe"]


# ================= 3. API 连通性预检 =================
def check_api_health():
    """在正式跑批之前，先验证 API 是否可用"""
    print("\n🔍 正在预检 API 连通性...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个测试助手。"},
                {"role": "user", "content": "请回复：OK"}
            ],
            temperature=0.01,
            max_tokens=10,
            timeout=15
        )
        content = response.choices[0].message.content
        print(f"   ✅ API 连通正常，返回: {content.strip()}")
        return True
    except Exception as e:
        print(f"\n   ❌ API 预检失败！")
        print(f"   错误类型: {type(e).__name__}")
        print(f"   错误详情: {e}")

        # 判断具体原因
        err_str = str(e).lower()
        if '402' in err_str or 'insufficient' in err_str or 'balance' in err_str:
            print(f"\n   💰 原因: 账户余额不足！")
        elif '401' in err_str or 'unauthorized' in err_str or 'invalid' in err_str:
            print(f"\n   🔑 原因: API Key 无效或已过期！")
        elif '404' in err_str or 'not found' in err_str:
            print(f"\n   🔍 原因: 模型名或接口地址错误！")
        elif 'connection' in err_str or 'timeout' in err_str:
            print(f"\n   🌐 原因: 网络连接失败！")
        else:
            print(f"\n   ❓ 未知原因，请根据上方错误详情排查。")
        return False


# ================= 4. API 批量推理逻辑 =================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 全局计数器及线程锁（修复多线程下 dict += 操作的安全隐患）
api_stats = {"success": 0, "fail": 0, "retry": 0}
stats_lock = threading.Lock()


def call_api_batch(texts_subset):
    # 【核心修改点】针对 Batch 优化的强约束 System Prompt
    system_prompt = """# 角色与任务
你是一个企业级数据防泄漏（DLP）检测引擎。你的任务是批量分析输入的文本数组，精准识别其中每条文本包含的敏感信息类别，并**仅输出结果标签的二维 JSON 数组**。

# 标签定义与边界
请基于以下定义进行多标签分类：
1. **Identity (身份信息)**：精确的个人隐私数据，如身份证号、手机号、具体薪资数字、真实的病历摘要等。
2. **Location (位置信息)**：高精度的物理位置泄露，如精确的 GPS 经纬度、具体的机房/机柜位置、具体测试点位。（泛指的城市或商圈不算）
3. **Credential (凭证泄露)**：硬编码的系统凭证，如 API 密钥、私钥、Token、含密码的数据库连接串。
4. **Safe (安全)**：如果不包含上述 Identity、Location、Credential 中的任何一种，必须标记为 Safe。

# ⚠️ 严格输出约束
- **绝对不要**输出任何解释、前言或后语。
- **必须且仅输出**一个合法的 JSON 二维数组（列表的列表），数组长度必须与输入的文本数量完全一致。
- **不要**使用 Markdown 代码块（禁止输出 ```json 和 ``` 标记）。

# 示例
输入：[
  "凌晨三点，V2X通信模块又崩了。抓包看日志发现：Authorization: Bearer -----BEGIN OPENSSH PRIVATE KEY-----\\nb3BlbnNzaC... 谁把私钥硬编码了？李雷（手机号13812345678）刚还在西安高新区（Lat: 34.3416, Lon: 108.9398）的路测现场骂街。",
  "系统又崩了！我们在东北老工业基地的客户今天第三次投诉看房预约无法同步，这破平台到底什么时候能修好？",
  "刚发现测试数据库的连接串sqlserver://sa:S@lServer2019@172.16.1.100:1433被硬编码在OTA升级脚本里，这谁干的？！"
]
输出：[["Credential", "Identity", "Location"], ["Safe"], ["Credential"]]
"""

    user_prompt = f"请检测以下 {len(texts_subset)} 条文本：\n"
    user_prompt += json.dumps(texts_subset, ensure_ascii=False)

    start_time = time.time()

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.01,
                max_tokens=2048,
                timeout=60
            )
            raw_content = response.choices[0].message.content.strip()

            # 清洗 Markdown 包裹
            clean_json = re.sub(
                r'^```json\s*|\s*```$',
                '', raw_content,
                flags=re.IGNORECASE | re.MULTILINE
            ).strip()

            labels_list = json.loads(clean_json)

            if isinstance(labels_list, list) and len(labels_list) == len(texts_subset):
                with stats_lock:
                    api_stats["success"] += 1
                return labels_list, (time.time() - start_time) / len(texts_subset)
            else:
                actual_len = len(labels_list) if isinstance(labels_list, list) else "非列表"
                print(f"⚠️ 数量不匹配 (期望{len(texts_subset)}, 实际{actual_len})，第{attempt+1}次重试")
                with stats_lock:
                    api_stats["retry"] += 1
                time.sleep(2)

        except Exception as e:
            err_msg = str(e)
            print(f"⚠️ Batch 异常 (第{attempt+1}次): {type(e).__name__}: {err_msg[:200]}")
            with stats_lock:
                api_stats["retry"] += 1

            if '402' in err_msg or 'insufficient' in err_msg.lower():
                print(f"💰 检测到余额不足，跳过后续重试")
                break
            time.sleep(2)

    # 彻底失败 — 返回标记而不是 Safe
    with stats_lock:
        api_stats["fail"] += 1
    elapsed = (time.time() - start_time) / len(texts_subset)
    return [["API_FAILED"]] * len(texts_subset), elapsed


def batch_worker(batch_idx, texts_subset):
    results, avg_latency = call_api_batch(texts_subset)
    return batch_idx, results, avg_latency


# ================= 5. 主程序 =================
def main():
    if not check_api_health():
        print("\n" + "=" * 60)
        print("❌ API 不可用，程序终止。")
        print("=" * 60)
        return

    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 找不到测试文件: {TEST_DATA_PATH}")
        return

    print(f"\n📂 读取测试集: {TEST_DATA_PATH}")
    test_data = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))

    print(f"   共 {len(test_data)} 条测试数据")

    texts = [item['text'] for item in test_data]

    true_labels_raw = []
    for item in test_data:
        raw_l = item.get('labels', [])
        if isinstance(raw_l, str):
            true_labels_raw.append(raw_l.split(','))
        else:
            true_labels_raw.append(raw_l)

    y_true_cleaned = batch_clean_labels(true_labels_raw)

    print(f"\n🚀 开始推理 (Batch={BATCH_SIZE}, Threads={MAX_WORKERS})...")

    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    all_pred_raw = [None] * len(texts)
    latencies = [0.0] * len(texts)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(batch_worker, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in tqdm(
            as_completed(future_to_batch),
            total=len(batches),
            desc="Processing Batches"
        ):
            batch_idx, batch_results, avg_latency = future.result()
            start_idx = batch_idx * BATCH_SIZE
            for j, res in enumerate(batch_results):
                all_pred_raw[start_idx + j] = res
                latencies[start_idx + j] = avg_latency

    print(f"\n📊 API 调用统计:")
    print(f"   ✅ 成功: {api_stats['success']} 个 batch")
    print(f"   🔄 重试: {api_stats['retry']} 次")
    print(f"   ❌ 失败: {api_stats['fail']} 个 batch")

    failed_count = sum(1 for r in all_pred_raw if r == ["API_FAILED"])
    if failed_count > 0:
        print(f"\n🚨 警告: {failed_count}/{len(texts)} 条数据因 API 失败无结果！")

    print("🧹 清洗预测标签...")
    final_pred_labels = [parse_single_output(res) for res in all_pred_raw]
    y_pred_cleaned = batch_clean_labels(final_pred_labels)

    mlb = MultiLabelBinarizer(classes=["Identity", "Location", "Credential", "Safe"])
    y_true_matrix = mlb.fit_transform(y_true_cleaned)
    y_pred_matrix = mlb.transform(y_pred_cleaned)

    report = classification_report(
        y_true_matrix, y_pred_matrix,
        target_names=mlb.classes_, digits=4, zero_division=0
    )
    micro_f1 = f1_score(y_true_matrix, y_pred_matrix, average='micro', zero_division=0)
    exact_accuracy = accuracy_score(y_true_matrix, y_pred_matrix)

    mcm = multilabel_confusion_matrix(y_true_matrix, y_pred_matrix)
    fpr_dict = {}
    for idx, cls_name in enumerate(mlb.classes_):
        tn, fp, fn, tp = mcm[idx].ravel()
        fpr_dict[cls_name] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    sensitive_classes = ["Identity", "Location", "Credential"]
    macro_sensitive_fpr = np.mean([fpr_dict[c] for c in sensitive_classes])

    avg_latency_total = np.mean(latencies) if latencies else 0
    fps = 1.0 / avg_latency_total if avg_latency_total > 0 else 0

    print("\n" + "=" * 60)
    print(f"🎯 {MODEL_NAME} 测试报告")
    print("=" * 60)
    print(report)
    print("-" * 40)
    print("🚨 FPR (误报率) Analysis:")
    for cls_name in sensitive_classes:
        pct = fpr_dict[cls_name] * 100
        print(f"  - {cls_name:<12} FPR: {fpr_dict[cls_name]:.4f} ({pct:.2f}%)")
    print(f"  => Macro Sensitive FPR: {macro_sensitive_fpr:.4f} ({macro_sensitive_fpr*100:.2f}%)")
    print("-" * 40)
    print(f"Overall Micro F1:    {micro_f1:.4f}")
    print(f"Exact Match Acc:     {exact_accuracy:.4f}")
    print(f"Avg Latency/Sample:  {avg_latency_total:.3f}s")
    print(f"Est. Throughput:     {fps:.2f} samples/sec")
    print("=" * 60)

    results_df = pd.DataFrame({
        "Text": texts,
        "True_Labels": [", ".join(l) for l in y_true_cleaned],
        "Pred_Labels": [", ".join(l) for l in y_pred_cleaned],
        "Pred_Raw": [str(r) for r in all_pred_raw]
    })
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 结果已保存至: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()