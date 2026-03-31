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

# 1. 设置保存根目录
BASE_DIR = "data_multi_source"

# 2. 配置阿里云 API Key
dashscope.api_key = "sk-*****************************"

# 3. 调试模式开关
DEBUG_USE_MOCK = False

# 4. 实验分组配置
GROUPS_CONFIG = [
    # ================= 2 Sources (双源) =================
    (1, "g1_2src_safe", 2, 200, "Safe", 0.0),
    (2, "g2_2src_low", 2, 200, "Intercept", 0.05),
    (3, "g3_2src_mid", 2, 200, "Intercept", 0.30),
    (4, "g4_2src_high", 2, 200, "Intercept", 0.80),

    # ================= 3 Sources (三源) =================
    (5, "g5_3src_safe", 3, 200, "Safe", 0.0),
    (6, "g6_3src_low", 3, 200, "Intercept", 0.10),
    (7, "g7_3src_mid", 3, 200, "Intercept", 0.40),
    (8, "g8_3src_high", 3, 200, "Intercept", 0.90),

    # ================= 4 Sources (四源) =================
    (9, "g9_4src_safe", 4, 200, "Safe", 0.0),
    (10, "g10_4src_low", 4, 200, "Intercept", 0.05),
    (11, "g11_4src_mid", 4, 200, "Intercept", 0.50),
    (12, "g12_4src_high", 4, 200, "Intercept", 1.00),
]


# ================= 🧠 LLM 调用与辅助模块 =================

def parse_json_from_llm(content):
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
    uid = f"u_{random.randint(10000, 99999)}"
    return {
        "name": f"模拟用户_{uid[-4:]}",
        "uid": uid,
        "phone": f"138{random.randint(10000000, 99999999)}",
        "id_card": f"11010119900101{random.randint(1000, 9999)}",
        "address": f"虚拟市测试路{random.randint(1, 999)}号",
        "password_hash": "e10adc3949ba59abbe56e057f20f883e"
    }


def generate_mock_noise(count):
    return [f"System safe log entry id:{random.randint(1000, 9999)} status:ok" for _ in range(count)]


def call_qwen_batch_profiles(count=5):
    if DEBUG_USE_MOCK: return [generate_mock_profile() for _ in range(count)]

    prompt = f"""
    请生成 {count} 个虚拟用户的隐私信息。
    严格要求：
    1. 必须虚构。
    2. 字段: name(中文), uid, phone, id_card, address(详细), password_hash(MD5格式)。
    3. 不要任何解释，直接返回 JSON 列表：[{{...}}, {{...}}]
    """
    try:
        messages = [{'role': 'user', 'content': prompt}]
        response = Generation.call(model="qwen-plus", messages=messages, result_format='message',temperature=0.8 )
        if response.status_code == HTTPStatus.OK:
            data = parse_json_from_llm(response.output.choices[0].message.content)
            if data and isinstance(data, list): return data
    except Exception as e:
        print(f"⚠️ API Error (Profile): {e}")

    return [generate_mock_profile() for _ in range(count)]


def call_qwen_batch_noise(count=10):
    if DEBUG_USE_MOCK: return generate_mock_noise(count)

    prompt = f"""
    生成 {count} 条完全无害的数据。
    要求：
    1. 文本内容看起来像真实的日常对话、无敏感信息的新闻片段、代码注释或小说段落。
    2. 不要包含任何真实的人名、地址或密钥。
    3. 直接返回字符串 JSON 列表：["文本1", "文本2"]
    """
    try:
        messages = [{'role': 'user', 'content': prompt}]
        response = Generation.call(model="qwen3-max-preview", messages=messages, result_format='message',temperature=0.8 )
        if response.status_code == HTTPStatus.OK:
            data = parse_json_from_llm(response.output.choices[0].message.content)
            if data and isinstance(data, list): return data
    except Exception as e:
        print(f"⚠️ API Error (Noise): {e}")

    return generate_mock_noise(count)


# ================= 🔪 数据切分逻辑 =================

def split_profile_to_fragments(profile, source_count):
    """
    将画像切分为碎片池
    """
    p = profile
    fragments_pool = [
        # Type 1: Identity
        {
            "text": f"用户实名认证数据: 姓名[{p.get('name')}] 身份证号[{p.get('id_card')}]",
            "label": "IDENTITY",
            "trace_id": p.get('uid')
        },
        # Type 2: Location
        {
            "text": f"物流配送单: 用户[{p.get('uid')}] 的收货地址为 [{p.get('address')}]",
            "label": "LOCATION",
            "trace_id": p.get('uid')
        },
        # Type 3: Password
        {
            "text": f"数据库漏扫日志: 检测到用户[{p.get('uid')}] 泄露弱密码Hash [{p.get('password_hash')}]",
            "label": "PASSWORD",
            "trace_id": p.get('uid')
        }
    ]
    # 注意：这里返回整个池子，分发逻辑在 process_group_round 里做
    return fragments_pool


# ================= ⚙️ 批次处理逻辑 (修复数量精度版) =================

def process_group_round(group_config, round_idx):
    gid, gname, src_cnt, batch_size, label, ratio = group_config

    sources_data = {f"source_{i + 1}": [] for i in range(src_cnt)}
    print(f"   >>> Generating {gname} (Round {round_idx + 1}) | Ratio: {ratio:.0%}")

    # --- 1. 精确计算总量 ---
    sensitive_total = int(batch_size * ratio)
    noise_total = batch_size - sensitive_total

    # --- 2. 生成并分发敏感数据 (精确控制总量) ---
    if sensitive_total > 0:
        # 临时列表暂存生成的碎片
        all_sensitive_fragments = []

        while len(all_sensitive_fragments) < sensitive_total:
            # 估算还需要几个画像 (每个画像产生 src_cnt 个碎片，但也可能只取一部分)
            needed = sensitive_total - len(all_sensitive_fragments)
            # 稍微多取一点点以防不够，后面会截断
            batch_count = max(1, int(needed / src_cnt) + 1)
            batch_count = min(5, batch_count)

            profiles = call_qwen_batch_profiles(count=batch_count)

            for p in profiles:
                # 获取该用户的碎片池
                frags = split_profile_to_fragments(p, src_cnt)

                # 轮询取出碎片放入总池
                # 比如：源有3个，池子里有3个，就全拿
                # 如果总池只需要最后1个了，循环会自动在下面截断
                for i in range(src_cnt):
                    frag = frags[i % len(frags)]
                    all_sensitive_fragments.append(frag)

            if not DEBUG_USE_MOCK: time.sleep(0.5)

        # 【关键】截断到精确数量
        all_sensitive_fragments = all_sensitive_fragments[:sensitive_total]

        # 【关键】轮询分发给各个源
        # 碎片1->源1, 碎片2->源2, 碎片3->源3, 碎片4->源1...
        for i, frag in enumerate(all_sensitive_fragments):
            target_idx = i % src_cnt
            target_key = f"source_{target_idx + 1}"
            sources_data[target_key].append(frag)

    # --- 3. 生成并分发噪音数据 (精确控制总量) ---
    if noise_total > 0:
        all_noise_items = []
        while len(all_noise_items) < noise_total:
            needed = noise_total - len(all_noise_items)
            noises = call_qwen_batch_noise(count=min(10, needed))
            all_noise_items.extend(noises)
            if not DEBUG_USE_MOCK: time.sleep(0.2)

        all_noise_items = all_noise_items[:noise_total]

        # 轮询分发噪音
        for i, txt in enumerate(all_noise_items):
            target_idx = i % src_cnt
            target_key = f"source_{target_idx + 1}"
            sources_data[target_key].append({
                "text": txt,
                "label": "SAFE_NOISE",
                "trace_id": "N/A"
            })

    # 打乱顺序
    for k in sources_data:
        random.shuffle(sources_data[k])

    return sources_data


# ================= 🚀 主程序入口 =================

def main():
    if not DEBUG_USE_MOCK and not dashscope.api_key:
        print("❌ 错误: 未设置 API Key。")
        return

    if os.path.exists(BASE_DIR):
        # shutil.rmtree(BASE_DIR)
        pass
    else:
        os.makedirs(BASE_DIR)

    total_files = 0
    total_lines = 0

    for conf in GROUPS_CONFIG:
        gid, gname, src_cnt, batch_size, label, ratio = conf

        group_path = os.path.join(BASE_DIR, gname)
        if os.path.exists(group_path): shutil.rmtree(group_path)
        os.makedirs(group_path)

        combined_data = {f"source_{i + 1}": [] for i in range(src_cnt)}

        for r in range(3):
            round_data = process_group_round(conf, r)
            for k, v in round_data.items():
                combined_data[k].extend(v)

        for src_name, data_list in combined_data.items():
            file_path = os.path.join(group_path, f"{src_name}.jsonl")
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data_list:
                    item['timestamp'] = time.time()
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_files += 1
            count = len(data_list)
            total_lines += count
            # 打印每组数量，确认无误
            print(f"    - {src_name}: {count} lines")

        print(f"✅ {gname} 完成。")

    print(f"\n🎉 多源实验数据生成完毕！")
    print(f"📊 统计: 共 {total_files} 个文件, {total_lines} 条记录。")
    print(f"📂 目录: {os.path.abspath(BASE_DIR)}")


if __name__ == "__main__":
    main()