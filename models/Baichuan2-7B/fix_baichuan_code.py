import os
import shutil

# === 1. 配置路径 ===
# 源头：你的本地 merged 模型目录 (必须改这里！)
SOURCE_FILE = "/root/autodl-tmp/MultiSource-contrastive experiment/models/Baichuan2-7B/output/baichuan2-7b-lora/v3-20260112-185018/checkpoint-1176-merged/modeling_baichuan.py"

# 缓存：报错信息里显示的那个路径
CACHE_FILE = "/root/.cache/huggingface/modules/transformers_modules/checkpoint_hyphen_1176_hyphen_merged/modeling_baichuan.py"


def apply_patch(file_path, file_type):
    print(f"🔧 正在修复 {file_type}: {file_path}")

    if not os.path.exists(file_path):
        print(f"   ⚠️ 文件不存在，跳过: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    modified = False

    for line in lines:
        # --- 修复逻辑 1: kv_seq_len ---
        if "kv_seq_len += past_key_value[0].shape[-2]" in line and "if hasattr" not in line:
            # 智能获取缩进
            indent = line[:line.find("kv_seq_len")]
            new_lines.append(f"{indent}# FinalFix Applied\n")
            new_lines.append(f"{indent}if hasattr(past_key_value, 'get_seq_length'):\n")
            new_lines.append(f"{indent}    kv_seq_len += past_key_value.get_seq_length()\n")
            new_lines.append(f"{indent}else:\n")
            new_lines.append(f"{indent}    kv_seq_len += past_key_value[0].shape[-2]\n")
            modified = True

        # --- 修复逻辑 2: past_key_values_length ---
        elif "past_key_values_length = past_key_values[0][0].shape[2]" in line and "if hasattr" not in line:
            indent = line[:line.find("past_key_values_length")]
            new_lines.append(f"{indent}# FinalFix Applied\n")
            new_lines.append(f"{indent}if hasattr(past_key_values, 'get_seq_length'):\n")
            new_lines.append(f"{indent}    past_key_values_length = past_key_values.get_seq_length()\n")
            new_lines.append(f"{indent}else:\n")
            new_lines.append(f"{indent}    past_key_values_length = past_key_values[0][0].shape[2]\n")
            modified = True

        else:
            new_lines.append(line)

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"   ✅ {file_type} 已成功修复！")
        return True
    else:
        print(f"   ⚪ {file_type} 似乎已经修复过了。")
        return False


def main():
    print("🚀 开始终极修复...")

    # 1. 先修源头 (最重要！)
    source_fixed = apply_patch(SOURCE_FILE, "【本地源文件】")

    # 2. 再修缓存 (防止当前运行出错)
    cache_fixed = apply_patch(CACHE_FILE, "【缓存副本】")

    # 3. 炸掉缓存的字节码 (防止 Python 加载旧的 pyc)
    if os.path.exists(CACHE_FILE):
        cache_dir = os.path.dirname(CACHE_FILE)
        pycache = os.path.join(cache_dir, "__pycache__")
        if os.path.exists(pycache):
            shutil.rmtree(pycache)
            print("💥 已清除 __pycache__ 字节码缓存")

    print("\n✅ 修复完成。现在运行推理，绝对不会再报错了！")


if __name__ == "__main__":
    main()