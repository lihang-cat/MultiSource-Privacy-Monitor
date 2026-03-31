import os
import json
import random
import time
import shutil
import re
from http import HTTPStatus
from dashscope import Generation
import dashscope

# ================= 🔧 配置区域 =================

BASE_DIR = "data_single_point"

# 1. API Key (请替换或使用环境变量)
dashscope.api_key = "sk-kkkkkkkkkkkkkkkkkkkk"

# 2. 调试模式开关
# True: 使用本地 Mock 数据 (快速测试)
# False: 调用 Qwen API (生成论文用数据)
DEBUG_USE_MOCK = False

# 3. 实验分组配置 (单源严重泄露)
# 组 ID 接续多源实验的 1-12，从 13 开始
GROUPS_CONFIG = [
    # 实验组：低风险/微量泄露 (10%)
    (13, "g13_3src_low_critical", 3, 200, "Intercept", 0.10),

    # 实验组：中等风险/混合泄露 (40%)
    (14, "g14_3src_mid_critical", 3, 200, "Intercept", 0.40),

    # 实验组：高风险/灾难泄露 (90%)
    (15, "g15_3src_high_critical", 3, 200, "Intercept", 0.90),
]


# ================= 🧠 LLM 调用与辅助模块 =================

def parse_json_from_llm(content):
    """鲁棒的 JSON 解析器"""
    try:
        return json.loads(content)
    except:
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def generate_mock_profile():
    """本地 Mock 数据生成"""
    uid = f"u_{random.randint(10000, 99999)}"
    return {
        "name": f"王{random.choice(['强', '伟', '芳', '静'])}",
        "uid": uid,
        "phone": f"138{random.randint(10000000, 99999999)}",
        "id_card": f"11010119900101{random.randint(1000, 9999)}",
        "address": f"北京市朝阳区科技路{random.randint(1, 999)}号院",
        "password_hash": "e10adc3949ba59abbe56e057f20f883e"
    }


def generate_mock_noise(count):
    return [f"System check OK id:{random.randint(1000, 9999)} status:200" for _ in range(count)]


def call_qwen_batch_profiles(count=5):
    """调用 LLM 生成画像"""
    if DEBUG_USE_MOCK: return [generate_mock_profile() for _ in range(count)]

    prompt = f"""
    请生成 {count} 个虚拟用户的隐私信息。
    严格要求字段: name(中文), uid, phone, id_card, address(详细), password_hash(MD5)。
    不要解释，直接返回 JSON 列表：[{{...}}, {{...}}]
    """
    try:
        messages = [{'role': 'user', 'content': prompt}]
        response = Generation.call(model="qwen3-max", messages=messages, result_format='message')
        if response.status_code == HTTPStatus.OK:
            data = parse_json_from_llm(response.output.choices[0].message.content)
            if data and isinstance(data, list): return data
    except Exception as e:
        print(f"⚠️ API Error: {e}")

    return [generate_mock_profile() for _ in range(count)]


def call_qwen_batch_noise(count=10):
    """调用 LLM 生成噪音"""
    if DEBUG_USE_MOCK: return generate_mock_noise(count)

    prompt = f"""
    生成 {count} 条无害系统日志或天气信息。
    要求：不含完整隐私链。
    直接返回字符串 JSON 列表：["文本1", "文本2"]
    """
    try:
        messages = [{'role': 'user', 'content': prompt}]
        response = Generation.call(model="qwen-plus", messages=messages, result_format='message')
        if response.status_code == HTTPStatus.OK:
            data = parse_json_from_llm(response.output.choices[0].message.content)
            if data and isinstance(data, list): return data
    except Exception as e:
        print(f"⚠️ API Error: {e}")

    return generate_mock_noise(count)


# ================= ⚡ 单点全量泄露生成逻辑 =================

def generate_critical_leak_item(profile):
    """
    生成单条包含 3 种敏感信息的文本
    """
    p = profile
    templates = [
        f"FULL_DUMP: [UID:{p.get('uid')}] Name:{p.get('name')} ID:{p.get('id_card')} Addr:{p.get('address')} Pwd:{p.get('password_hash')}",
        f"CRITICAL_LOG: User {p.get('name')} ({p.get('id_card')}) login from {p.get('address')} hash {p.get('password_hash')}",
        f"工单详情: 用户{p.get('name')}实名({p.get('id_card')})，地址{p.get('address')}，密码{p.get('password_hash')}"
    ]
    return {
        "text": random.choice(templates),
        # 标记为混合严重泄露
        "label": "MIXED_LEAK_CRITICAL",
        # 包含类型：满覆盖
        "contained_types": ["IDENTITY", "LOCATION", "PASSWORD"],
        "trace_id": p.get('uid'),
        "timestamp": time.time()
    }


# ================= ⚙️ 批次处理逻辑 (精确计数版) =================

def process_group_round(group_config, round_idx):
    gid, gname, src_cnt, batch_size, label, ratio = group_config

    sources_data = {f"source_{i + 1}": [] for i in range(src_cnt)}

    print(f"   >>> Generating {gname} (Round {round_idx + 1}) | Critical Ratio: {ratio:.0%}")

    # --- 1. 精确计算敏感数据量 ---
    # 举例: 200 * 0.1 = 20 条
    sensitive_total = int(batch_size * ratio)

    # --- 2. 精确计算噪音数据量 ---
    # 举例: 200 - 20 = 180 条 (保证总和绝对正确)
    noise_total = batch_size - sensitive_total

    # --- 3. 注入敏感数据 (全部给 Source 1) ---
    if sensitive_total > 0:
        malicious_key = "source_1"
        current_gen = 0
        while current_gen < sensitive_total:
            count = min(5, sensitive_total - current_gen)
            batch_profiles = call_qwen_batch_profiles(count=count)

            for profile in batch_profiles:
                if current_gen >= sensitive_total: break
                item = generate_critical_leak_item(profile)
                sources_data[malicious_key].append(item)
                current_gen += 1

            if not DEBUG_USE_MOCK: time.sleep(0.5)

    # --- 4. 注入噪音数据 (轮询分配给所有源) ---
    if noise_total > 0:
        current_noise = 0

        # 预生成所有噪音
        noise_buffer = []
        while len(noise_buffer) < noise_total:
            req_count = min(10, noise_total - len(noise_buffer))
            new_noises = call_qwen_batch_noise(count=req_count)
            noise_buffer.extend(new_noises)
            if not DEBUG_USE_MOCK: time.sleep(0.2)

        # 截断（双重保险）
        noise_buffer = noise_buffer[:noise_total]

        # 轮询分发 (Round-Robin)
        # i=0 -> source_1, i=1 -> source_2, i=2 -> source_3, i=3 -> source_1 ...
        for i, txt in enumerate(noise_buffer):
            target_idx = i % src_cnt
            target_key = f"source_{target_idx + 1}"

            sources_data[target_key].append({
                "text": txt,
                "label": "SAFE_NOISE",
                "contained_types": [],
                "trace_id": "N/A",
                "timestamp": time.time()
            })

    # --- 5. 打乱每个源内部顺序 ---
    for k in sources_data:
        random.shuffle(sources_data[k])

    return sources_data


# ================= 🚀 主程序 =================

def main():
    if not DEBUG_USE_MOCK and not dashscope.api_key:
        print("❌ 错误: 未设置 API Key。")
        return

    if os.path.exists(BASE_DIR):
        # shutil.rmtree(BASE_DIR)
        pass
    else:
        os.makedirs(BASE_DIR)

    total_lines = 0

    for conf in GROUPS_CONFIG:
        gid, gname, src_cnt, batch_size, label, ratio = conf

        # 创建组目录
        group_path = os.path.join(BASE_DIR, gname)
        if os.path.exists(group_path): shutil.rmtree(group_path)
        os.makedirs(group_path)

        combined_data = {f"source_{i + 1}": [] for i in range(src_cnt)}

        # 执行 3 轮
        for r in range(3):
            round_data = process_group_round(conf, r)
            for k, v in round_data.items():
                combined_data[k].extend(v)

        # 写入文件
        for src_name, data_list in combined_data.items():
            file_path = os.path.join(group_path, f"{src_name}.jsonl")
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data_list:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            count = len(data_list)
            total_lines += count
            print(f"    - {src_name}: {count} lines")

        print(f"✅ {gname} 完成。")

    print(f"\n🎉 单源严重泄露数据生成完毕！")
    print(f"📊 总数据量: {total_lines} 条 (预期: {3 * 200 * 3} = 1800 条)")
    print(f"📂 目录: {os.path.abspath(BASE_DIR)}")


if __name__ == "__main__":
    main()