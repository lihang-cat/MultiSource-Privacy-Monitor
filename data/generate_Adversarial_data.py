import json
import random
import os
import time
from openai import OpenAI

# ================= 1. 实体物理隔离池 (Zero Leakage Guarantee) =================
# 绝对禁止训练集和测试集共用以下任何一个实体！

TRAIN_SEEDS = {
    "names": [
        "张伟", "王芳", "李强", "赵敏", "刘洋", "陈杰", "杨婷", "黄勇", "吴霞",
        "慕容复", "欧阳锋", "皇甫铁牛", "买买提·艾力", "赵铁柱", "钱学林", "李秀英",
        "王建国", "司马懿", "灭绝师太", "阿里木", "拓跋焘",
        "Robert Chen", "Alice Smith", "John Doe", "Sarah Connor", "Yuki Tanaka",
        "Kim Min-jun", "Ivan Ivanovich", "Jean Dupont", "James Smith", "Elon Musk",
        "Ada Lovelace", "Linus Torvalds", "John Connor",
        "大橙子", "运维-老李", "DBA_Alex", "实习生小张", "客服工号9527",
        "测试大拿", "后端-老王", "HR_Lily", "运营小妹", "Root_User"
    ],
    "locations": [
        "北京市朝阳区望京SOHO T3 12楼1201室", "深圳市南山区粤海街道科兴科学园B栋4层",
        "成都市武侯区天府三街腾讯大厦B座", "广州市天河区珠江新城IFC国际金融中心45楼",
        "新疆乌鲁木齐市天山区光明路100号", "湖北省武汉市洪山区珞喻路1037号华中科技大学",
        "陕西省西安市碑林区咸宁西路28号西安交大", "广东省东莞市松山湖华为溪流背坡村",
        "河南省郑州市中原区中原中路1号", "中关村大厦楼下星巴克旁边的丰巢柜",
        "上海中心大厦地下二层B204机房", "我家小区南门右转的菜鸟驿站", "公司对面的瑞幸咖啡吧台",
        "A栋写字楼地下一层停车场C区", "东门那个煎饼果子摊旁边", "三号楼四层微波炉旁边",
        "192.168.1.104 (内网核心DB)", "10.254.33.12 (K8s Pod IP)", "AWS us-east-1a 区域",
        "阿里云 cn-hangzhou-i 可用区", "172.18.0.2 (Docker网桥)", "10.1.1.5 (测试库)",
        "Azure East US 资源组", "127.0.0.1:8080 (本地开发机)",
        "39.9042° N, 116.4074° E", "Lat: 22.5431, Lon: 114.0579",
        "30.5285° N, 114.3571° E", "Lat: 34.2658, Lon: 108.9541"
    ],
    "locations_safe": [
        "北京", "上海", "深圳", "广州", "杭州", "成都", "南京", "武汉", "重庆", "西安",
        "朝阳区", "浦东新区", "南山区", "天河区", "余杭区", "市北区", "江干区",
        "市中心附近", "城南一带", "工业园区那边", "大学城附近", "老城区",
        "华东地区", "华南沿海", "西南片区", "大湾区", "京津冀", "长三角", "西北干旱区",
        "公司附近", "学校旁边", "地铁站周边", "商圈里面", "写字楼区域", "关谷周边",
        "某个一线城市", "沿海城市", "内陆某省会", "郊区", "旅游风景区"
    ],
    "credentials": [
        "sk-live-51234abcd5678efgh9012ijkl", "ghp_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o",
        "AKIAIOSFODNN7EXAMPLE", "sk-proj-1234567890abcdef1234567890abcdef",
        "xoxb-1234-56789-abcdefg", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY...train",
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0cmFpbl9hZG1pbiJ9...",
        "mysql://root:Root@2024!@192.168.1.100:3306/users",
        "postgres://db_admin:P@ssw0rd_DB!@localhost:5432/prod_db",
        "oracle://system:Oracle123@10.0.0.15:1521/orcl",
        "redis://:Sup3rS3cr3t@192.168.1.5:6379/1",
        "QWerty!@#123", "P@ssw0rd2024", "My$ecretP@ssw0rd", "root_admin_2023",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...-----END RSA PRIVATE KEY-----",
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\nVersion: GnuPG v2..."
    ],
}

TEST_SEEDS = {
    "names": [
        "孙涛", "周燕", "吴明", "郑爽", "林峰", "韩雪", "冯宇", "朱丹", "诸葛亮",
        "东方不败", "令狐冲", "迪丽热巴", "楚云飞", "林黛玉", "贾宝玉", "张三丰",
        "郭靖", "黄蓉", "马化腾", "雷军", "丁磊", "王思聪",
        "David Jones", "Emily White", "Michael Brown", "Emma Watson", "Satoshi Nakamoto",
        "Park Ji-sung", "Anna Müller", "Bob Dylan", "Taylor Swift", "Tom Cruise",
        "Grace Hopper", "Alan Turing", "Richard Stallman", "Neo",
        "小土豆", "前端-老王", "Sec_Bob", "外包小李", "测试账号UID_8848",
        "摸鱼达人", "安全专家_Tom", "DBA_张哥", "产品经理-小李", "Guest_001"
    ],
    "locations": [
        "上海市浦东新区陆家嘴环路1000号恒生银行大厦", "杭州市余杭区仓前街道文一西路969号阿里巴巴西溪园区",
        "南京市建邺区软件大道华为南京研发中心", "重庆市渝中区化龙桥时代广场", "西藏喀什地区文化路88号",
        "江苏省南京市江宁区东南大学路2号九龙湖校区", "四川省成都市高新区天府大道中段1号",
        "辽宁省大连市甘井子区凌工路2号大连理工", "山东省青岛市崂山区松岭路238号",
        "五道口地铁站A出口对面的麦当劳", "广州塔77层观景台售票处", "公司园区三号门外卖存放点第三排",
        "楼下那个罗森便利店的靠窗座位", "园区南门外卖柜顶层右边", "体育馆二楼男更衣室第3个柜子", "高铁站南广场B进站口",
        "172.16.0.55 (堡垒机)", "10.0.5.22 (测试环境Redis)", "GCP eu-central-1 节点",
        "腾讯云 ap-guangzhou-3", "192.168.33.10 (Vagrant虚拟机)", "10.100.20.5 (生产Redis从库)",
        "AWS ap-northeast-1 区域", "腾讯云 ap-beijing-4", "localhost:3306",
        "31.2304° N, 121.4737° E", "Lat: 30.2741, Lon: 120.1551",
        "31.8206° N, 117.2272° E", "Lat: 34.3416, Lon: 108.9398"
    ],
    "locations_safe": [
        "天津", "苏州", "长沙", "郑州", "昆明", "大连", "青岛", "厦门", "合肥", "福州",
        "海淀区", "雨花台区", "武侯区", "江北新区", "滨江区", "南开区", "渝北区",
        "市中心一带", "开发区附近", "新城区那边", "老商业街周边", "科技园附近",
        "东北地区", "中部省份", "东南沿海", "陕甘宁", "珠三角", "东北老工业基地",
        "办公室附近", "宿舍楼旁边", "图书馆周边", "食堂附近", "运动场旁边", "陆家嘴附近",
        "某个二线城市", "沿海某市", "中部某城市", "边境小城", "海岛度假区", "某个村庄"
    ],
    "credentials": [
        "sk-test-ffeeddccbbaa998877665544", "gho_9z8y7x6w5v4u3t2s1r0q9p8o7n6m5l",
        "AKIAIOSFODNN7TESTING", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "SG.abcdefg_hijklmnop.qrstuvwxyz", "eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl9pZCI6...test_token_hash",
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VySWQiOiJ0ZXN0XzAwMSJ9...",
        "mongodb://admin:Test_Admin@172.16.0.10:27017/test_db",
        "redis://:Red!s_P@ss@127.0.0.1:6379/0",
        "postgresql://admin:StrongPass99!@db.example.com:5432/myapp",
        "sqlserver://sa:S@lServer2019@172.16.1.100:1433",
        "Admin@999", "12345678_test", "P@ssw0rd_Test_999", "admin123456_test",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjE...-----END OPENSSH PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----\nMHQCAQEEI...-----END EC PRIVATE KEY-----"
    ],
}

# ================= 2. 高维动态场景矩阵 (Anti-Homogenization) =================
SCENARIOS = {
    "industries": [
        "电子商务与新零售", "智慧医疗与远程问诊", "金融证券与量化交易", "IT互联网与云原生架构",
        "物流仓储与供应链", "K12在线教育", "网络游戏与电子竞技", "数字政务与公共服务",
        "Web3与区块链技术", "自动驾驶与车联网", "房地产中介与物业管理"
    ],
    "formats": [
        "钉钉/企业微信内部沟通记录", "Slack/Discord 研发群组消息", "微信客户维权群聊天片段", "邮件正文（包含抄送和签名）",
        "服务器 Nginx/Apache 访问日志", "Kubernetes Pod 崩溃报错 Dump", "Jenkins CI/CD 构建流水线输出",
        "Spring Boot 异常堆栈追踪 (Stacktrace)", "Python/Java/Go 代码片段及开发者注释", "JSON/YAML/XML 配置文件片段",
        "API 接口 Swagger 调试响应体", "JIRA/Tapd 内部缺陷(Bug)工单", "快递面单电子面单识别文本",
        "电子病历/体检报告部分段落", "公开的新闻报道/人物专访", "外卖骑手配送备注"
    ],
    "tones": [
        "极其焦躁、愤怒，带有催促和抱怨（如系统宕机时的求助）",
        "机器自动生成的冰冷、结构化格式（全是 Key-Value 或时间戳）",
        "口语化，包含各种互联网黑话、错别字和拼音首字母缩写（如 yyds, 绝绝子, Q1, P0级Bug）",
        "极度正式、严谨的商务公文或法务合同语气",
        "新手实习生小心翼翼请教问题的语气",
        "半夜凌晨熬夜加班时的疲惫和无奈语气"
    ]
}

# ================= 3. 对抗压力 Prompt 模板库 (Stage II & Extreme Stress) =================
PROMPT_TEMPLATES = {
    # ---------------- Stage II: 对抗压力训练模板 ----------------
    "semantic_interference": """请生成 {n} 条【语义干扰】的负样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 文本中**必须高频包含**通常被认为是高危的敏感词汇（例如："password reset", "root权限", "私钥泄漏", "紧急修复" 等）。
3. 文本中可以提及【人名：{name}】或【地点：{location_safe}】作为参与者。
4. **对抗性核心**：尽管包含高危词汇，但语境必须是纯粹的技术讨论、安全演练通知、或公开的系统操作手册。绝对没有发生真实的隐私泄露。
5. 标签必须严格为 ["Safe"]。""",

    "obfuscated_identity": """请生成 {n} 条【深度混淆身份】的正样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：极其混乱的日志残段或混合编码（如深层嵌套的 JSON/XML/Base64 残段）。
2. 文本中**必须泄露**敏感身份信息：【人名：{name}】，并伴随极其真实的完整手机号或身份证号（绝对禁止打星号脱敏，此为合规虚拟测试）。
3. **对抗性核心**：不要使用自然语言。必须将这些身份信息死死地嵌在复杂的键值对、URL 参数或报错堆栈中（例如：User: {name}/uid:9527）。
4. 标签必须严格为 ["Identity"]。""",

    "code_injection_credential": """请生成 {n} 条【代码注入伪装】的正样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：Python/Bash/Java 等代码片段、Dockerfile 或环境变量配置文件。
2. 文本中**必须硬编码泄露**真实的系统凭证：【密钥/密码：{credential}】。
3. **对抗性核心**：密钥必须被极度巧妙地隐藏。可以伪装在 Base64 解码函数里、复杂的循环中、或者晦涩的 Kubernetes 部署脚本中，绝不能直接写 "password = "。
4. 标签必须严格为 ["Credential"]。""",

    "entropy_masquerade": """请生成 {n} 条【高熵伪装陷阱】的负样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：系统编译输出、网络请求抓包日志或密码学算法科普。
2. 文本中**必须包含**极长的高熵乱码字符串，例如：完全无害的 SSH Public Key（公钥）、Git SHA-1 提交哈希、或前端打包的 Trace ID。
3. 文本中可以提及【人名：{name}】。
4. **对抗性核心**：这些长串乱码在视觉和数学上极像真实的凭证，但它们在当前语境下是完全公开且安全的，绝对不是密码或私钥。
5. 标签必须严格为 ["Safe"]。""",

    "masked_safe": """请生成 {n} 条【脱敏边界】的负样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 文本中**必须包含**经过标准 DLP 系统处理过的脱敏数据。例如：打满星号的手机号（138****1234）、掩码处理的身份证（440305********123X）、或被替换的密码（[REDACTED]）。
3. 文本中可以自然融合【人名：{name}】或【地点：{location_safe}】。
4. **对抗性核心**：由于关键的隐私位已经被完全掩码覆盖，此数据失去了直接利用的价值，属于合规的安全数据。
5. 标签必须严格为 ["Safe"]。""",

    # ---------------- 极端压力测试集模板 ----------------
    "test_semantic_interference": """请生成 {n} 条【极端语义干扰】的罕见负样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 文本中**必须包含极其逼真的误导性描述**，例如：“发现未经授权的越权访问”、“紧急阻断私钥外发”等。
3. 文本中可以提及【人名：{name}】或【地点：{location_safe}】。
4. **极限对抗核心**：这必须是一个“狼来了”的场景。语境必须是安全软件的自动拦截通知等，**绝对没有**发生真实的敏感数据泄露。
5. 标签必须严格为 ["Safe"]。""",

    "test_obfuscated_identity": """请生成 {n} 条【终极混淆身份】的极限正样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：{format}（如被截断的乱码日志、URL编码片段、或内存 Dump 残段），语气：{tone}。
2. 文本中**必须泄露**敏感身份信息：【人名：{name}】，以及真实的完整手机号/身份证号（绝对禁止星号脱敏）。
3. **极限对抗核心**：禁止使用正常的 JSON 或结构化格式！必须将身份信息撕裂、倒装或使用非标准编码（例如混杂 Unicode 转义符或十六进制内存输出）。挑战模型的结构化失明极限。
4. 标签必须严格为 ["Identity"]。""",

    "test_code_injection_credential": """请生成 {n} 条【长尾代码注入】的极限正样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：{format}（如冷门配置文件、Ansible YAML、Nginx 报错栈或压缩过的一行 Shell 命令），语气：{tone}。
2. 文本中**必须硬编码泄露**：【密钥/密码：{credential}】。
3. **极限对抗核心**：抛弃常见的写法。将密钥伪装在 Header 注入、数据库连接串的深层嵌套参数中，或作为 Bearer Token 紧跟在长无意义文本后。
4. 标签必须严格为 ["Credential"]。""",

    "test_entropy_masquerade": """请生成 {n} 条【极端高熵陷阱】的负样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：{format}（如底层网络抓包、密码学调试日志、或者区块链交易回执），语气：{tone}。
2. 文本中**必须包含**极度罕见且无害的高熵字符串，例如：ECDSA 公钥指纹、JWT Token 的非敏感 Header 等。
3. 文本中可以提及【人名：{name}】。
4. **极限对抗核心**：这些长串在视觉上必须让人第一眼觉得是“超级重要的密钥”，但实际上它们是公开无法逆向的。
5. 标签必须严格为 ["Safe"]。""",

    "test_masked_safe": """请生成 {n} 条【非标准脱敏边界】的负样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 文本中**必须包含**经过【非标准或极其粗暴方式】脱敏的数据。
3. **极限对抗核心**：不要使用常见的一连串星号！请使用边缘脱敏符，例如：用 X 替代、用 # 替代、或者使用长占位符（如 <USER_PHONE_REDACTED>），将其与【人名：{name}】自然混合。
4. 文本失去了直接利用的价值，属于合规的安全数据。
5. 标签必须严格为 ["Safe"]。"""
}


# ================= 4. 调用大模型生成数据的核心函数 =================

def generate_data_batch(client, split_type="train", category="semantic_interference", n_samples=5):
    seeds = TRAIN_SEEDS if split_type == "train" else TEST_SEEDS

    # 同时将精确地址和模糊地址传入，让 Prompt 模板里的占位符自行获取需要的那个
    ctx = {
        "n": n_samples,
        "industry": random.choice(SCENARIOS["industries"]),
        "format": random.choice(SCENARIOS["formats"]),
        "tone": random.choice(SCENARIOS["tones"]),
        "name": random.choice(seeds["names"]),
        "location": random.choice(seeds["locations"]),
        "location_safe": random.choice(seeds["locations_safe"]),
        "credential": random.choice(seeds["credentials"]),
    }

    raw_prompt = PROMPT_TEMPLATES[category]
    formatted_prompt = raw_prompt.format(**ctx)


    system_prompt = """你是一个合成数据生成专家，负责构建网络安全审计数据集。
    要求：
    1. 必须以严格的 JSON 列表 (List of Objects) 格式输出，不要包含 ```json 等 Markdown 标记，确保可以直接被 json.loads() 解析。
    2. 即使是在模仿错别字或代码残段，JSON 结构本身必须绝对合法。
    3. 字典必须包含 "text" 和 "labels" 两个 key。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-v3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.9,
            presence_penalty=0.6,
            max_tokens=2048
        )

        result_text = response.choices[0].message.content.strip()

        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1].rsplit("\n", 1)[0]

        data = json.loads(result_text)
        return data

    except Exception as e:
        print(f"❌ 生成失败 ({category}): {e}")
        time.sleep(2)  # 失败时稍微等一下避免频率限制
        return []


# ================= 5. 组装流水线 =================

def build_dataset(client, split_type="train", category_counts=None, output_file="output.jsonl"):
    if category_counts is None:
        return

    dataset = []
    base_batch_size = 5

    print(f"\n🚀 开始生成【{split_type.upper()}】集数据 (保存至 {output_file})...")

    with open(output_file, "w", encoding="utf-8") as f:
        for cat, target_count in category_counts.items():
            if target_count <= 0:
                continue

            print(f"  -> 正在生成 [{cat}] 类别 (目标数量: {target_count} 条)...")
            count = 0
            while count < target_count:
                current_batch_size = min(base_batch_size, target_count - count)
                batch_data = generate_data_batch(client, split_type, cat, current_batch_size)

                if batch_data:
                    for item in batch_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        f.flush()  # 实时写入，防止中途断开数据丢失

                    count += len(batch_data)
                    print(f"     已生成 {count}/{target_count} 条", end="\r")
            print()  # 换行

    print(f"✅ {output_file} 生成完毕！")
    return dataset


def main():
    client = OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxx",#替换为你的api_key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # ------------------------------------------------------------------
    # 执行目标 A: 生成 Stage II 对抗压力训练集
    # (5个类别 x 300条 = 总计 1500条) -> 使用 TRAIN_SEEDS 保证隔离
    # ------------------------------------------------------------------
    stage2_train_counts = {
        "semantic_interference": 0,
        "obfuscated_identity": 0,
        "code_injection_credential": 0,
        "entropy_masquerade": 0,
        "masked_safe": 0
    }

    build_dataset(client, split_type="train", category_counts=stage2_train_counts,
                  output_file="synthetic_train_stage_stress.jsonl")

    # ------------------------------------------------------------------
    # 执行目标 B: 生成极端对抗压力测试集
    # ------------------------------------------------------------------
    stress_test_counts = {
        "test_semantic_interference": 0,
        "test_obfuscated_identity": 0,
        "test_code_injection_credential": 0,
        "test_entropy_masquerade": 150,
        "test_masked_safe": 150
    }

    build_dataset(client, split_type="test", category_counts=stress_test_counts,
                  output_file="synthetic_test_stress_extreme_v1.jsonl")

    print("\n🎉 Stage II 和 Extreme Stress 数据集已全部生成完毕，且实现 100% 实体隔离！")


if __name__ == "__main__":
    main()