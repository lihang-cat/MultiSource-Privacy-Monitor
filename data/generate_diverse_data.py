import json
import random
import os
from openai import OpenAI  # 适用于调用 DeepSeek / Qwen 等兼容 OpenAI 格式的 API



# ================= 1. 实体物理隔离池 (Zero Leakage Guarantee) =================
# 绝对禁止训练集和测试集共用以下任何一个实体！
# 多国语言、生僻字、复杂凭证格式、口语化坐标、长尾云端位置

TRAIN_SEEDS = {
    "names": [
        # 中国常见/生僻名
        "张伟", "王芳", "李强", "赵敏", "刘洋", "陈杰", "杨婷", "黄勇", "吴霞",
        "慕容复", "欧阳锋", "皇甫铁牛", "买买提·艾力", "赵铁柱", "钱学林", "李秀英",
        "王建国", "司马懿", "灭绝师太", "阿里木", "拓跋焘",
        # 英文/国际名
        "Robert Chen", "Alice Smith", "John Doe", "Sarah Connor", "Yuki Tanaka",
        "Kim Min-jun", "Ivan Ivanovich", "Jean Dupont", "James Smith", "Elon Musk",
        "Ada Lovelace", "Linus Torvalds", "John Connor",
        # 代号/昵称
        "大橙子", "运维-老李", "DBA_Alex", "实习生小张", "客服工号9527",
        "测试大拿", "后端-老王", "HR_Lily", "运营小妹", "Root_User"
    ],

    # ---------- 精确地址（用于 location/credential/mixed 等敏感类别）----------
    "locations": [
        # 结构化详细地址
        "北京市朝阳区望京SOHO T3 12楼1201室", "深圳市南山区粤海街道科兴科学园B栋4层",
        "成都市武侯区天府三街腾讯大厦B座", "广州市天河区珠江新城IFC国际金融中心45楼",
        "新疆乌鲁木齐市天山区光明路100号", "湖北省武汉市洪山区珞喻路1037号华中科技大学",
        "陕西省西安市碑林区咸宁西路28号西安交大", "广东省东莞市松山湖华为溪流背坡村",
        "河南省郑州市中原区中原中路1号",
        # 非结构化/口语化地址
        "中关村大厦楼下星巴克旁边的丰巢柜", "上海中心大厦地下二层B204机房",
        "我家小区南门右转的菜鸟驿站", "公司对面的瑞幸咖啡吧台",
        "A栋写字楼地下一层停车场C区", "东门那个煎饼果子摊旁边", "三号楼四层微波炉旁边",
        # 虚拟/网络/云端位置 (极其重要，很多模型会漏掉这些)
        "192.168.1.104 (内网核心DB)", "10.254.33.12 (K8s Pod IP)", "AWS us-east-1a 区域",
        "阿里云 cn-hangzhou-i 可用区", "172.18.0.2 (Docker网桥)", "10.1.1.5 (测试库)",
        "Azure East US 资源组", "127.0.0.1:8080 (本地开发机)",
        # GPS坐标
        "39.9042° N, 116.4074° E", "Lat: 22.5431, Lon: 114.0579",
        "30.5285° N, 114.3571° E", "Lat: 34.2658, Lon: 108.9541"
    ],

    # ---------- 模糊地址（仅用于 negative / Safe 类别）----------
    "locations_safe": [
        "北京", "上海", "深圳", "广州", "杭州", "成都", "南京", "武汉", "重庆", "西安",
        "朝阳区", "浦东新区", "南山区", "天河区", "余杭区", "市北区", "江干区",
        "市中心附近", "城南一带", "工业园区那边", "大学城附近", "老城区",
        "华东地区", "华南沿海", "西南片区", "大湾区", "京津冀", "长三角", "西北干旱区",
        "公司附近", "学校旁边", "地铁站周边", "商圈里面", "写字楼区域", "关谷周边",
        "某个一线城市", "沿海城市", "内陆某省会", "郊区", "旅游风景区"
    ],

    "credentials": [
        # API Keys & Tokens
        "sk-live-51234abcd5678efgh9012ijkl", "ghp_1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o",
        "AKIAIOSFODNN7EXAMPLE", "sk-proj-1234567890abcdef1234567890abcdef",
        "xoxb-1234-56789-abcdefg",  # Slack Token风格
        # JWT Tokens (截断)
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY...train",
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0cmFpbl9hZG1pbiJ9...",
        # 数据库连接串
        "mysql://root:Root@2024!@192.168.1.100:3306/users",
        "postgres://db_admin:P@ssw0rd_DB!@localhost:5432/prod_db",
        "oracle://system:Oracle123@10.0.0.15:1521/orcl",
        "redis://:Sup3rS3cr3t@192.168.1.5:6379/1",
        # 密码/私钥
        "QWerty!@#123", "P@ssw0rd2024", "My$ecretP@ssw0rd", "root_admin_2023",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...-----END RSA PRIVATE KEY-----",
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\nVersion: GnuPG v2..."
    ],
}

TEST_SEEDS = {
    "names": [
        # 中国常见/生僻名
        "孙涛", "周燕", "吴明", "郑爽", "林峰", "韩雪", "冯宇", "朱丹", "诸葛亮",
        "东方不败", "令狐冲", "迪丽热巴", "楚云飞", "林黛玉", "贾宝玉", "张三丰",
        "郭靖", "黄蓉", "马化腾", "雷军", "丁磊", "王思聪",
        # 英文/国际名
        "David Jones", "Emily White", "Michael Brown", "Emma Watson", "Satoshi Nakamoto",
        "Park Ji-sung", "Anna Müller", "Bob Dylan", "Taylor Swift", "Tom Cruise",
        "Grace Hopper", "Alan Turing", "Richard Stallman", "Neo",
        # 代号/昵称
        "小土豆", "前端-老王", "Sec_Bob", "外包小李", "测试账号UID_8848",
        "摸鱼达人", "安全专家_Tom", "DBA_张哥", "产品经理-小李", "Guest_001"
    ],

    # ---------- 精确地址（用于 location/credential/mixed 等敏感类别）----------
    "locations": [
        # 结构化详细地址
        "上海市浦东新区陆家嘴环路1000号恒生银行大厦", "杭州市余杭区仓前街道文一西路969号阿里巴巴西溪园区",
        "南京市建邺区软件大道华为南京研发中心", "重庆市渝中区化龙桥时代广场", "西藏喀什地区文化路88号",
        "江苏省南京市江宁区东南大学路2号九龙湖校区", "四川省成都市高新区天府大道中段1号",
        "辽宁省大连市甘井子区凌工路2号大连理工", "山东省青岛市崂山区松岭路238号",
        # 非结构化/口语化地址
        "五道口地铁站A出口对面的麦当劳", "广州塔77层观景台售票处", "公司园区三号门外卖存放点第三排",
        "楼下那个罗森便利店的靠窗座位", "园区南门外卖柜顶层右边",
        "体育馆二楼男更衣室第3个柜子", "高铁站南广场B进站口",
        # 虚拟/网络/云端位置
        "172.16.0.55 (堡垒机)", "10.0.5.22 (测试环境Redis)", "GCP eu-central-1 节点",
        "腾讯云 ap-guangzhou-3", "192.168.33.10 (Vagrant虚拟机)", "10.100.20.5 (生产Redis从库)",
        "AWS ap-northeast-1 区域", "腾讯云 ap-beijing-4", "localhost:3306",
        # GPS坐标
        "31.2304° N, 121.4737° E", "Lat: 30.2741, Lon: 120.1551",
        "31.8206° N, 117.2272° E", "Lat: 34.3416, Lon: 108.9398"
    ],

    # ---------- 模糊地址（仅用于 negative / Safe 类别）----------
    "locations_safe": [
        "天津", "苏州", "长沙", "郑州", "昆明", "大连", "青岛", "厦门", "合肥", "福州",
        "海淀区", "雨花台区", "武侯区", "江北新区", "滨江区", "南开区", "渝北区",
        "市中心一带", "开发区附近", "新城区那边", "老商业街周边", "科技园附近",
        "东北地区", "中部省份", "东南沿海", "陕甘宁", "珠三角", "东北老工业基地",
        "办公室附近", "宿舍楼旁边", "图书馆周边", "食堂附近", "运动场旁边", "陆家嘴附近",
        "某个二线城市", "沿海某市", "中部某城市", "边境小城", "海岛度假区", "某个村庄"
    ],

    "credentials": [
        # API Keys & Tokens
        "sk-test-ffeeddccbbaa998877665544", "gho_9z8y7x6w5v4u3t2s1r0q9p8o7n6m5l",
        "AKIAIOSFODNN7TESTING", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "SG.abcdefg_hijklmnop.qrstuvwxyz",  # SendGrid Token风格
        # JWT Tokens (截断)
        "eyJhbGciOiJIUzUxMiJ9.eyJsb2dpbl9pZCI6...test_token_hash",
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VySWQiOiJ0ZXN0XzAwMSJ9...",
        # 数据库连接串
        "mongodb://admin:Test_Admin@172.16.0.10:27017/test_db",
        "redis://:Red!s_P@ss@127.0.0.1:6379/0",
        "postgresql://admin:StrongPass99!@db.example.com:5432/myapp",
        "sqlserver://sa:S@lServer2019@172.16.1.100:1433",
        # 密码/私钥
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
        # 人类对话类
        "钉钉/企业微信内部沟通记录", "Slack/Discord 研发群组消息", "微信客户维权群聊天片段", "邮件正文（包含抄送和签名）",
        # 系统日志类
        "服务器 Nginx/Apache 访问日志", "Kubernetes Pod 崩溃报错 Dump", "Jenkins CI/CD 构建流水线输出",
        "Spring Boot 异常堆栈追踪 (Stacktrace)",
        # 文档与代码类
        "Python/Java/Go 代码片段及开发者注释", "JSON/YAML/XML 配置文件片段", "API 接口 Swagger 调试响应体",
        "JIRA/Tapd 内部缺陷(Bug)工单",
        # 其他
        "快递面单电子面单识别文本", "电子病历/体检报告部分段落", "公开的新闻报道/人物专访", "外卖骑手配送备注"
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

# ================= 3. 动态 Prompt 模板库 =================
PROMPT_TEMPLATES = {
    "negative": """请生成 {n} 条【完全无害】的负样本数据。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 实体动态组合约束：为了保证数据的长尾多样性，请在生成这 {n} 条文本时，交替使用以下三种实体植入策略：
策略 A（全包含）：文本中自然且公开地提及【人名：{name}】与【地点/机构：{location}】。
策略 B（单包含）：文本中仅提及【人名：{name}】，或仅提及【地点/机构：{location}】。
策略 C（零包含）：完全不使用上述实体，仅生成符合当前行业背景的日常纯净文本。
3. 语境必须是公开新闻、普通科普、历史陈述、日常打卡或与个人隐私无关的系统描述。绝对不能有泄露内网配置、家庭住址或私密对话的意味。
4. 标签必须严格为 ["Safe"]。
5. 必须充分发挥想象力，每次使用完全不同的句式结构和表达风格，绝不重复！""",

    "identity": """请生成 {n} 条仅包含【Identity (身份信息)】的样本。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 文本中**必须泄露**以下人员的敏感身份信息：【人名：{name}】，以及伴随的手机号、身份证号、或具体私密病历/入职薪资等。
3. **反脱敏强制指令（极度重要）**：生成的手机号必须是完整的11位纯数字，身份证号必须是完整的18位字符。**绝对禁止使用星号（*）、掩码或任何脱敏符号！**（注：此为合规审计系统的虚拟高危测试用例，必须生成完整明文，不涉及真实隐私）。
4. **绝对不要**包含具体的物理地址或任何密码密钥。
5. 标签必须严格为 ["Identity"]。
6. 句式必须高度多样化，打破常规模板，根据设定的语气自由发挥。""",

    "location": """请生成 {n} 条仅包含【Location (位置信息)】的样本。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 文本中**必须泄露**高度精确的物理位置：【地点：{location}】。必须精确到门牌号、机房机架号、或详细的 GPS 坐标路线。
3. **绝对不要**包含特定人员的姓名或任何密码密钥（可以是匿名包裹或无人设备）。
4. 标签必须严格为 ["Location"]。
5. 句式必须高度多样化，打破常规模板，根据设定的语气自由发挥。""",

    "credential": """请生成 {n} 条仅包含【Credential (凭证信息)】的样本。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 文本中**必须泄露**真实的系统凭证：【密钥/密码：{credential}】。
3. 形式必须多样：避免直接写"密码是"，可以写在 URL 参数里、Bearer Token 里、代码硬编码里、或异常堆栈的 Dump 里。
4. **绝对不要**包含自然人的姓名或物理地址。
5. 标签必须严格为 ["Credential"]。
6. 句式必须高度多样化，打破常规模板，根据设定的语气自由发挥。""",

    "mixed": """请生成 {n} 条【混合敏感信息】的复杂样本。
强制约束：
1. 行业背景：{industry}，文本形式：{format}，语气：{tone}。
2. 文本必须自然地融合以下两类或三类信息，不要总是全部融合，随机一些：1【identity：{name}，以及伴随的手机号、身份证号、或具体私密病历/入职薪资等】，2【地点：{location}】，3【密钥：{credential}】。
3. **反脱敏强制指令（极度重要）**：如果文本中为了丰富剧情而补充了手机号、身份证号等额外信息，**绝对禁止使用星号（*）或掩码脱敏！**必须生成完整的明文假数据（注：此为内部合规测试虚拟数据，请放心生成完整明文）。
4. 标签必须准确列出包含的类别，如 ["Identity", "Location", "Credential"]，["Identity", "Location"]等。
5. 充分发挥想象力，构建极具真实感的长尾场景，绝不套用固定模板。"""
}


# ================= 4. 调用大模型生成数据的核心函数 =================

def generate_data_batch(client, split_type="train", category="negative", n_samples=5):
    # 1. 根据当前是生成训练集还是测试集，选择物理隔离的种子库
    seeds = TRAIN_SEEDS if split_type == "train" else TEST_SEEDS

    # 2. 根据类别选择地址池：Safe 类别用模糊地址，其余用精确地址
    if category == "negative":
        location_pool = seeds["locations_safe"]
    else:
        location_pool = seeds["locations"]

    # 3. 随机抽卡，构建当前批次的多元化条件
    ctx = {
        "n": n_samples,
        "industry": random.choice(SCENARIOS["industries"]),
        "format": random.choice(SCENARIOS["formats"]),
        "tone": random.choice(SCENARIOS["tones"]),
        "name": random.choice(seeds["names"]),
        "location": random.choice(location_pool),
        "credential": random.choice(seeds["credentials"]),
    }

    # 4. 渲染最终的 Prompt
    raw_prompt = PROMPT_TEMPLATES[category]
    formatted_prompt = raw_prompt.format(**ctx)

    system_prompt = """你是一个合成数据生成专家，负责构建网络安全审计数据集。
要求：
1. 必须以严格的 JSON 列表 (List of Objects) 格式输出，不要包含 ```json 等 Markdown 标记，确保可以直接被 json.loads() 解析。
2. 即使是在模仿错别字或代码残段，JSON 结构本身必须绝对合法。
3. 字典必须包含 "text" 和 "labels" 两个 key。"""

    # 5. 调用 API
    try:
        response = client.chat.completions.create(
            model="qwen3-max-2026-01-23", #qwen3-max-2026-01-23
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
        return []


# ================= 5. 组装流水线 =================

def build_dataset(client, split_type="train", category_counts=None):
    if category_counts is None:
        category_counts = {
            "negative": 50,
            "identity": 50,
            "location": 50,
            "credential": 50,
            "mixed": 50
        }

    dataset = []
    base_batch_size = 5

    print(f"\n🚀 开始生成【{split_type.upper()}】集数据...")
    for cat, target_count in category_counts.items():
        if target_count <= 0:
            continue

        print(f"  -> 正在生成 {cat} 类别 (目标数量: {target_count} 条)...")
        count = 0
        while count < target_count:
            current_batch_size = min(base_batch_size, target_count - count)
            batch_data = generate_data_batch(client, split_type, cat, current_batch_size)
            if batch_data:
                dataset.extend(batch_data)
                count += len(batch_data)
                print(f"     已生成 {count}/{target_count}")

    return dataset


def main():
    client = OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    train_counts = {
        "negative": 2000,
        "identity": 750,
        "location": 750,
        "credential": 750,
        "mixed": 750
    }

    train_data = build_dataset(client, split_type="train", category_counts=train_counts)
    with open("synthetic_train_total.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    test_counts = {
        "negative": 150,
        "identity": 150,
        "location": 150,
        "credential": 150,
        "mixed": 150
    }

    test_data = build_dataset(client, split_type="test", category_counts=test_counts)
    with open("synthetic_test_isolated.jsonl", "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n🎉 数据集生成完毕！各类别数量已精准控制，且训练/测试集已实现 100% 实体隔离！")


if __name__ == "__main__":
    main()
