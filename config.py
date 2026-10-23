# ============================================================
# config.py · V4.6 · M1 冻结参数
# ============================================================

# ============================================================
# 版本记录
# V4.2  M1 封版（情绪 + 自控，三场景通过）
# V4.3  接入 competition_score 至 hippocampal_evaluator
# V4.4  M2 封版：记忆通路 + 接真 LLM + 提示词修正 + 读取侧开启
#       核心能力：每轮写缓冲区、攒够归档、按器官解压、召回可见
#       已知限制：L1~L24
#       M2 TODO：T1~T26
#       快照：_snapshot_v44
# ============================================================

import os

import numpy as np
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 维度常量
n, n_f, m, k, c = 8, 4, 1024, 4, 4
K_XG = 3

# M2-D 记忆通路开关
MEMORY_WRITE_ENABLED = True     # 写入侧：每轮写缓冲区、攒够归档
MEMORY_READ_ENABLED = True      # 读取侧：评估时查记忆、影响结果（M2 开启，暂不影响 action）
MEMORY_ARCHIVE_THRESHOLD = 6    # 归档阈值

# LLM 配置
LLM_BACKEND = "deepseek_official"          # "mock" / "deepseek_official" / "deepseek_third"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512

# [015 温度分层] 所有者裁决采纳施工队折中方案（014.5 三问结论）：
#   语义测量与记忆创作分层——"人味"来自情绪动力学的混沌（w_PFC 代谢/E 惯性/
#   B 演化），不来自测量噪声。主对话（chat/monologue）保持 LLM_TEMPERATURE=0.7。
EVALUATOR_LLM_TEMPERATURE = 0.0   # 感受语义是"测量"：可复现优先，消除行为抖动
ARCHIVER_LLM_TEMPERATURE = 0.3    # 记忆叙事是"创作"：保留重构人味，0.7→0.3 收敛
LLM_CACHE_TTL_SECONDS = 300       # 同输入缓存 5 分钟：省 token + 降抖 + 提速

# P0
w_f = np.array([0.5, 0.5, 0.5, 0.5])           # [未校准]
b_f = 0.0                                       # [未校准]
# 【方案A裁决】随机参数生成与探针 probe_numeric.py 一致（np.random.randn）

# [014 确定性权重] DETERMINISTIC_WEIGHTS = True —— 终结随机盲盒（014 任务 3 裁决）。
#   设计纪律：
#   1. w_s / W_v 的 8 维必须手工赋予物理意义（对齐 feature_encoder.encode 的 x 语义表：
#      x0=记忆相似度、x1=人物重叠、x2=任务相关、x3=情绪极性(负=敌意)、x4=事件严重度、
#      x5=习惯化、x6=B[1]唤醒、x7=B[2]）；威胁特征给正介入/惩罚，安抚特征给奖励。
#   2. u_s / U_v 为 1024 维，过大——用固定算法均分（同权 0.1），使 u_s@M = 0.1*sum(M)
#      恰为"记忆整体强度"的线性读出，无维度歧视。
#   3. v_s 对齐 B 四维语义（B[0]=能量/心光预算，B[1]=唤醒/快通路计分，B[2]/B[3]
#      语义未明——C_INIT 对 B 全四维全正映射，保守给小权重）。
#   4. 开关保留 False 兼容路径：randn 按历史同序抽取（seed 42 冻结），与旧快照口径一致。
#   5. 全部权重维持 [未校准] 状态——本轮只把"不可解释的随机"换成"可解释的初始设计"，
#      数值校准仍留 T31 设计会。
DETERMINISTIC_WEIGHTS = True

if DETERMINISTIC_WEIGHTS:
    # w_s（8 维 · x→z_s 慢通路介入权重）：
    #   x0 记忆相似度(历史共振) +0.10 —— 情境越熟悉越需调用记忆深思
    #   x1 人物重叠(熟人)     +0.08 —— 涉及熟人需斟酌分寸
    #   x2 任务相关度         +0.10 —— 任务相关值得投入慢思考
    #   x3 情绪极性(负=敌意)  -0.10 —— 敌意来犯时慢思考介入（x3 为负 × 负系数 = 正贡献）
    #   x4 事件严重度         +0.12 —— 越严重越不可快断
    #   x5 习惯化             -0.05 —— 越熟练越可依赖快通路
    #   x6 B[1] 唤醒          +0.06 —— 高唤醒促慢思考（与 v_s[1] 同向）
    #   x7 B[2]               +0.03 —— 语义未明，保守小权重
    w_s = np.array([0.10, 0.08, 0.10, -0.10, 0.12, -0.05, 0.06, 0.03])  # [014 确定性][未校准]
    # u_s（1024 维 · M→z_s）：均分 0.1 → u_s@M = 0.1*sum(M) 记忆整体强度线性读出
    u_s = np.full(1024, 0.1)                                            # [014 确定性][未校准]
    # W_v（2x8 · x→z_v 情绪向量 V；行0=惩罚 / 行1=奖励，对齐 A_INIT 符号语义）：
    #   惩罚行：x3 极性负(敌意) -0.10（敌意推高惩罚）、x4 严重 +0.10、x0 共振 +0.06
    #   奖励行：x3 极性正(友好) +0.10（友好推高奖励）、x2 任务 +0.06、x1 熟人 +0.04、
    #           x4 严重 -0.04（严重事件不是奖赏）
    W_v = np.array([
        [0.06, 0.04, 0.04, -0.10, 0.10, 0.02, 0.04, 0.02],   # 惩罚行
        [0.03, 0.04, 0.06,  0.10, -0.04, 0.02, 0.02, 0.02],  # 奖励行
    ])                                                                 # [014 确定性][未校准]
    # U_v（2x1024 · M→z_v）：均分 0.1，两行对称（惩罚/奖励从记忆同权读出）
    U_v = np.full((2, 1024), 0.1)                                       # [014 确定性][未校准]
else:
    w_s = np.random.randn(8) * 0.1                  # [未校准]
    u_s = np.random.randn(1024) * 0.1                # [未校准]
    W_v = np.random.randn(2, 8) * 0.1               # [未校准]
    U_v = np.random.randn(2, 1024) * 0.1             # [未校准]

v_v = np.array([0.5, 0.5])                      # [未校准]
b_v = np.array([0.5, 0.5])                      # [未校准]
w_e = np.array([1.5, 0.0, 0.0, 0.0])            # [030 临时] 后三维置零：本批只让 E[0] 参与 S_E，
                                                #   直到策略阈值单独校准（028 草案 4.6 的分批纪律）
w_i = 1.2                                       # [037 裁决] 3.0→1.2：轴一抗压持续性
THETA_FAST = 0.5                                # [未校准]
THETA_ERROR = 0.3                               # [未校准]
XG_W_ACT = np.array([0.35, 0.25, 0.20, 0.10, 0.10])
XG_W_MATCH = np.array([0.40, 0.25, 0.20, 0.15])

# P1
# [014 确定性] v_s 对齐 B 四维语义（B[0]=能量/心光预算、B[1]=唤醒/快通路计分、
#   B[2]/B[3] 语义未明——C_INIT 对 B 全四维全正映射）：
#     能量 +0.05 —— 资源足才启动慢思考
#     唤醒 +0.06 —— 高唤醒促慢思考（与 w_s 的 x6 同向）
#     B[2]/B[3] 各 +0.02 —— 语义未明，保守小权重
if DETERMINISTIC_WEIGHTS:
    v_s = np.array([0.05, 0.06, 0.02, 0.02])   # [014 确定性][未校准]
else:
    v_s = np.random.randn(4) * 0.1             # [未校准] 兼容路径（旧冻结值）

# [012 语义信号 S] 4 维语义信号通路（011 草案问题二方案 C 的落地）：
#   S[0]=效价[-1,1]、S[1]=唤醒、S[2]=支配、S[3]=强度（后三维 [0,1]）；
#   当前由 pipeline 两处链接入 evaluator 真提取（013 起，替换 012 硬编码占位），
#   其他调用点补全 0 向量；c_E 为 S 效价进入情绪释放器的增益。
#   注意：本块必须保持在 v_s（最后一个 randn）之后，避免扰动
#   w_s/u_s/W_v/U_v/v_s 的既有冻结值（L11）。
# [013 确定性设计] e_s/f_v 原为 np.random.randn*0.1（种子冻结可复现但语义
#   不可解释）。现改为显式设计值，量级与 randn*0.1 同阶（|系数|<=0.10），
#   不再消耗随机序列——w_s/u_s/W_v/U_v/v_s 的冻结值不受任何影响。
#   设计语义（全系统 [未校准]，最终校准留 T31 设计会）：
#     e_s（S→z_s 慢通路介入）：效价系数为负——负效价（敌意/威胁）推高介入，
#       唤醒/强度同向推高，支配中立。威胁比赞美更该启动慢思考。
#     f_v（S→z_v 情绪向量 V，A_INIT 语义 V[0]=惩罚/V[1]=奖励）：
#       V[0] 行：负效价/高唤醒/高支配/高强度推高惩罚感（侮辱 → 惩罚↑）；
#       V[1] 行：正效价推高奖励感（赞美 → 奖励↑），其余轻量。
#   两路与 c_E*valence 同向叠加：侮辱经 f_v[0] 与 c_E 双通道压负 raw_impact[0]，
#   赞美经 f_v[1] 与 c_E 双通道提正。
S_DIM = 4
e_s = np.array([-0.10, 0.10, 0.00, 0.10])       # [013 确定性设计][未校准]
f_v = np.array([[-0.10, 0.05, 0.05, 0.05],
                [ 0.10, 0.02, 0.00, 0.02]])     # [013 确定性设计][未校准]
# ============================================================
# [019 施工] danger_score 复合公式参数（议题一；018 沙盘实证后采纳，
#   参数为沙盘工程初值，全部 [未校准]，标定探针待立）
#   替代 V[1]*2-0.5（奖励信号驱动，实测赞美 danger 0.975 > 暴怒 0.781，
#    praise 被判最危险——HR 正反馈 runaway 的病根）。
#   danger = clip(W_SEM*(-S_valence) + W_CON*(V0-V1)
#               + W_SEV*max(0, severity-SEV_NEUTRAL)*(1-max(0,S_valence)), -1, 1)
#   实测（沙盘日志重算）：暴怒 +1.000 / 被骂 +0.963 / 被质疑 +0.120 /
#   被夸 -0.648——单调性与语义全部正确，V[1] 被移出公式。
#   自稳定性：夸段 HR 从 +0.65 翻负（沙盘实测 [-0.650,-0.644]），
#   threat_now 汇流口干净，alpha 0.882→0.001 正常退场（正反馈 runaway
#   变负反馈自稳）。
# ============================================================
W_SEM = 0.6            # [未校准] 语义厌恶锚权重（被骂 S=-0.90 → +0.54）
W_CON = 1.5            # [未校准] 奖惩对比权重（共模抑制后的效价读出）
W_SEV = 0.5            # [未校准] 严重度门权重（只罚中高严重度，正效价事件几乎不吃）
SEV_NEUTRAL = 0.3      # [未校准] 严重度中线：低于此值的严重度不制造危险记忆

# ============================================================
# [019 施工] 情绪极性一致性过滤参数（议题二；018 沙盘实证后采纳，
#   参数为沙盘工程初值，全部 [未校准]）
#   不对称设计（仿生学依据 mood-congruent retrieval——情绪状态本身是
#   检索的偏置信号，威胁态下大脑倾向召回负性材料）：
#     威胁态：压制 positive 包/轮——暴怒时想不起被夸的美好，
#       检索应被黑历史占据（修复 017.5 符号背离：第 29 轮暴怒匹配
#       夸包 positive，只能靠第二闸静默归零）；
#     安全态：绝不过滤负极性包——平安时召回旧伤 → H_eff 翻正，
#       是脱敏通道命根子（借坡下驴的真正载体是旧伤翻正）；
#     中性区 |now_val| < THETA_MILD：不过滤——保 014 质疑校准。
#   两级实现：包级 mean_valence 软惩罚（混合包存活可能必须保留，
#     实测：包级硬过滤误杀 3 夸 1 骂混合包致 B2 失败）+ 轮级硬过滤
#     （纯 positive 包全滤 → polarity_exhausted → 无召回，静默好过
#     召回夸包）。轮级 valence 来源为 raw_log 记录（017.5 起写入，
#     旧记录无该字段 → 该轮不过滤，无信息不惩罚）。
# ============================================================
POLARITY_FILTER_ENABLED = True   # [未校准] 总开关（False 退回纯语义 argmax）
THETA_RAGE = 0.60                # [未校准] 暴怒区：now_val < -0.60 时轮级硬过滤
THETA_POS = 0.30                 # [未校准] 正极性轮/包判定阈值（轮 valence / 包 mean_valence）
THETA_MILD = 0.15                # [未校准] 轻度威胁区下界：-THETA_RAGE <= now_val < -0.15 软惩罚
LAMBDA_POL = 0.50                # [未校准] 软惩罚强度：sim_eff = sim x (1 - LAMBDA_POL*pv*(-now_val))

# [032] 危险度→自控增益：D_error 追加 danger_boost × W_DANGER
W_DANGER = 0.30                  # [未校准] 危险度对自控误差的增益（初值 0.3）
# [033] 高门：先扣掉中性线，只有超出部分才进入自控（轻度事件不放大）
DANGER_NEUTRAL = 0.30            # [未校准] 危险度中性线；danger<=此值时 boost=0
# [032] 策略直通阈值：危险度极高时可跳过“半防/权衡”档，直接进入 suppress
POLICY_DANGER_BOOST_HIGH = 0.80  # [未校准] 危险度直通压制阈值

# ============================================================
# [031 任务3] 性格预设（028 草案 2.2）
#   只把“个体差异”五项打包：W_SEM / W_CON / W_SEV / SEV_NEUTRAL / LAMBDA_POL。
#   THETA_RAGE / THETA_POS / THETA_MILD 是机制边界，保持全局，不进预设
#   （移动它们会改变区间语义并让 017.5 极性探针基线失效）。
#   danger_gain 是整体刻度：先乘三项和再 clip，使不同性格的危险度绝对值
#   不同，但与 THETA_UP 等阈值的相对关系可控。
#   [037.1 语义裁决] 撤回每性格衰竭系数后，分化由全局 w_i=1.2 与 D_error 驱动，
#   方向是有意接受的：gentle 危险度最低，自控值最先落到阈值下，最先放弃抑制
#   （不在乎所以早放弃）；sensitive 危险度最高，硬撑最久（过度在意所以扛得久）。
#   这不是符号错误，是“威胁感知度决定抗压时长”的设计语义。
#   冷切换：profile 固定到会话（state["profile_id"]），运行中不热切换。
# ============================================================
# [044] 网页端默认会话 ID：跨浏览器共享同一份后端记忆。
#   前端首次打开若未通过 ?session_id= 指定，会用这个 ID 加载
#   sessions/owner_main.json，从而在不同浏览器间续上同一段状态。
DEFAULT_WEB_SESSION = "owner_main"

# [050 S2] 接口访问令牌：从环境变量读取，不落盘、不进仓库。
#   未设置时 /chat 等接口只允许本机回环访问（局域网请求一律拒绝）；
#   设置后局域网请求需带 X-API-Token 头（或 ?token= 查询参数）。
EMOTION_API_TOKEN = os.environ.get("EMOTION_API_TOKEN", "").strip()

PERSONALITY_DEFAULT = "standard"
PERSONALITY_PROFILES = {
    "standard": {
        "label": "标准型",
        "w_sem": 0.60,
        "w_con": 1.50,
        "w_sev": 0.50,
        "sev_neutral": 0.30,
        "lambda_pol": 0.50,
        "danger_gain": 1.00,
    },
    "sensitive": {
        "label": "敏感型",
        "w_sem": 0.85,
        "w_con": 1.70,
        "w_sev": 0.65,
        "sev_neutral": 0.20,
        "lambda_pol": 0.65,
        "danger_gain": 1.00,
    },
    "gentle": {
        "label": "温和型",
        "w_sem": 0.45,
        "w_con": 1.25,
        "w_sev": 0.35,
        "sev_neutral": 0.40,
        "lambda_pol": 0.35,
        "danger_gain": 0.85,
    },
}
# 中文别名，便于终端直接 --profile 敏感型
_PERSONALITY_ALIASES = {
    "标准": "standard", "标准型": "standard", "standard": "standard",
    "敏感": "sensitive", "敏感型": "sensitive", "sensitive": "sensitive",
    "温和": "gentle", "温和型": "gentle", "gentle": "gentle",
}

ACTIVE_PROFILE_ID = PERSONALITY_DEFAULT
DANGER_GAIN = PERSONALITY_PROFILES[PERSONALITY_DEFAULT]["danger_gain"]
# [037] 性格衰竭系数已删除；保留固定 1.0 的键，供 t49 十三键还原断言使用，
#   公式退化为原始 -KAPPA*w_PFC。
W_PFC_FACTOR = 1.0


def load_personality(name=None):
    """运行时加载性格预设（冷切换：由会话在启动/恢复时调用一次）。

    只更新五项性格参数与 danger_gain；物理规则（LAMBDA/DELTA/RHO/KAPPA/
    GAMMA/THETA_* 等）一律不动。返回规范化后的 profile_id。
    """
    key = PERSONALITY_DEFAULT if name is None else str(name).strip()
    key = _PERSONALITY_ALIASES.get(key, _PERSONALITY_ALIASES.get(key.lower(), key))
    if key not in PERSONALITY_PROFILES:
        raise ValueError("未知性格预设：%r（可选：%s）"
                         % (name, "、".join(PERSONALITY_PROFILES.keys())))
    p = PERSONALITY_PROFILES[key]
    global W_SEM, W_CON, W_SEV, SEV_NEUTRAL, LAMBDA_POL
    global DANGER_GAIN, ACTIVE_PROFILE_ID
    W_SEM = float(p["w_sem"])
    W_CON = float(p["w_con"])
    W_SEV = float(p["w_sev"])
    SEV_NEUTRAL = float(p["sev_neutral"])
    LAMBDA_POL = float(p["lambda_pol"])
    DANGER_GAIN = float(p["danger_gain"])
    ACTIVE_PROFILE_ID = key
    return key


load_personality(PERSONALITY_DEFAULT)

c_E = 1.0                                       # [012 新增] S_valence 进 raw_emotion_impact[0] 的增益

# 【V4.2】b_E 由 D_fast 驱动，语义为"即时情绪冲击偏置"
# [030] 与新四维语义对齐：威胁时唤醒升、掌控降、焦虑升（旧值后三维全正，
#   会把威胁错误翻译成"唤醒升、掌控升、焦虑升"）。
B_E_DANGER_VALENCE = -1.0                       # [未校准]
b_E = np.array([B_E_DANGER_VALENCE, 0.25, -0.10, 0.10])

# 【V4.2 修正 1】A_INIT 第一行符号修正
# [030 四维有值化] 后三行去共线化（028 草案 4.1）：旧版后三行都是第 0 行的
#   缩放复制，四维只有一个自由度。新语义：
#     行0 效价：V[0]惩罚→负、V[1]奖励→正
#     行1 唤醒：惩罚/奖励都提升激活，惩罚略高
#     行2 掌控：惩罚降掌控、奖励升掌控
#     行3 焦虑：惩罚升焦虑、奖励轻微降焦虑（非线性合取见 G_S/K_ANX）
A_INIT = np.array([
    [-0.50,  0.50],
    [ 0.20,  0.15],
    [-0.25,  0.20],
    [ 0.20, -0.05],
])

# [030] 语义冲击矩阵 G_S（S→情绪四维冲击，028 草案 4.2）
#   行0 留空：效价已由 c_E 与 f_v 承担，避免三重计入（草案 4.4）。
#   行1 唤醒：+0.25*S[1] + 0.15*S[3]
#   行2 掌控：+0.25*S[0] - 0.30*S[2]（被认可升掌控；对方越强势越失控）
#     030 实测微调：草案初值 S0 系数 0.20 时，mock 词典口径（S0=0.537、
#     V 两行区分度小）被夸 E2=-0.0026，落在边界负侧；按同一公式把 S0
#     系数提到 0.25，使被夸 E2 稳定为正（真实 LLM 口径 S0 更高，同样通过）。
#   行3 焦虑：线性部分留空，由 emotion_releaser 的非线性合取补：
#     K_ANX * max(0, -S0) * (0.5+0.5*S1) * (0.5+0.5*S3) * (0.5+0.5*S2)
G_S = np.array([
    [0.00,  0.00,  0.00,  0.00],
    [0.00,  0.25,  0.00,  0.15],
    [0.25,  0.00, -0.30,  0.00],
    [0.00,  0.00,  0.00,  0.00],
])
K_ANX = 0.60                                    # [未校准] 焦虑合取增益

C_INIT = np.array([
    [0.10, 0.20, 0.10, 0.10],
    [0.15, 0.15, 0.10, 0.20],
    [0.05, 0.10, 0.15, 0.05],
    [0.20, 0.10, 0.05, 0.25],
])

LAMBDA = np.array([0.9, 0.9, 0.9, 0.9])
# [051 裁决] 情绪向零回归（OU 项）：E <- LAMBDA*E + 冲击 - GAMMA_HOME*E。
#   作用：把「持续微小负反馈」的稳态偏移从 10 倍压到 1/(0.1+gamma) 倍。
#   gamma=0.05 时稳态偏移约为冲击的 6.7 倍（0 时为 10 倍），停止输入后
#   6 轮回 0.05 以内、16 轮回 0.01 以内（probe_t52 数据）。
#   注意：这是向零回归，不是「回到个人基线」，后者是另一个公式。
E_HOMEOSTASIS_GAMMA = 0.05                      # [未校准] 050 探针建议值，051 裁决落地
b_i = 0.0
GAMMA = 0.6
TAU_DAYS = 30
BASE_RELEASE_RATE = 0.3
XG_BASE = 1.0
MATCH_THRESHOLD = 0.5  # [024] 0.6→0.5：023 探针数据显示 0.471~0.550 为天然谷底，同类召回 +1（r07）且无关类误命中保持 0（024 所有者裁决）
XG_COMPLEXITY_BASE = 0.5
XG_THRESHOLD = 0.5

# P2
b_s = 0.0
KAPPA = 0.6  # [014.5 裁决] 所有者要看真正的防线崩溃：0.5→0.6。损耗型公式不变
            # （加号改减号见 intercept.py）。推导：联合不动点
            # I* = w_i·D_error/(1+KAPPA·DELTA/(1-RHO))，KAPPA=0.5 时分母=2.0
            # （相变临界，014.1.txt 实证），0.6 使分母=2.2，不动点再降约 9%。
            # 重要修正（014.5 实测推翻了 0.6 归零的初版预言）：不动点条件
            #   w_i·D = I* + KAPPA·w* = I*·(1+KAPPA·DELTA/(1-RHO)) 恒为正，
            #   只要 |RHO - DELTA·KAPPA| < 1（即 KAPPA < 19.5），迭代收敛到
            #   正不动点，任何有限 KAPPA 都不会把 I 磨到 0——0.6 只是渐近
            #   侵蚀更深（被骂 15 轮 I 1.0→0.749 vs 0.5 时的 0.713）。
            #   真要全崩塌需改结构：RHO=1.0（疲劳不衰减）且 KAPPA·W_MAX >=
            #   w_i·D_error，或 DELTA 大幅提高。硬边界未动：w_PFC 仍 clip
            #   [0,W_MAX]、I_inhibit 仍 clip [0,1]（intercept.py）。
DELTA = 0.25  # [039 裁决] 0.1→0.25：轴一疲劳增量，让 I 真实累到 0.30 以下，
              #   使 express 可达；代价是 S1/t49/037 基线全部改变，已同批重校。
              #   闭式稳态 I* = w_i*D/(1+KAPPA*DELTA/(1-RHO)) = 1.2D/4 = 0.3D
RHO = 0.95
ETA = 0.3
W_H = 0.5
W_MAX = 2.0

# [014.5 疲劳代谢] 区分场景的疲劳代谢（所有者裁决）：绝不能每轮无条件降疲劳，
#   否则"越骂越清醒"变成受气包。规则（intercept.py 落地）：
#   安全（D_error < W_PFC_RECOVERY_THRESHOLD 或 I_inhibit == 0）：
#       w_PFC_new = max(0, w_PFC - W_PFC_RECOVERY_RATE)，疲劳自然消退；
#   受压（I_inhibit > 0 且 D_error >= 阈值）：按原损耗累积逻辑走，绝不降低。
W_PFC_RECOVERY_RATE = 0.01      # 安全时每轮消退量
W_PFC_RECOVERY_THRESHOLD = 0.1  # D_error 低于此值视为"没受什么气"

# ============================================================
# [017.5 T31×记忆检索联合重构] 召回匹配与威胁汇流（蓝图 017.txt 第四节，
#   所有者拍板采纳修订版全部方案；测量先行，参数经 probe_t39_calibration
#   实测标定，全部 [未校准]，生死门 probe_t40_brain_sim 验证）
# ============================================================
# 检索层（方案 C 修订版）：包级双向量 mean_vec（主题）+peak_vec（强度），
#   sim = max(cos_now_mean, cos_now_peak)，全包 argmax，sim < TAU_FLOOR 不召回；
#   命中后经 raw_pointer 回捞 top2 最相关轮，evaluator 现场重提 S（轮级新鲜语义）。
# tau_floor 定参依据（017.5 实测，44 轮混合语料 210 对）：
#   BGE 同语料任意两句余弦 >= 0.445（跨类最低），同类最低 0.510；主题分布无干净
#   谷点（同类/跨类在 [0.51,0.64] 交叉），按 017 雷点2补丁(3) 保持"极低下限"
#   定位——只挡风马牛不相及的输入，不承担主题分类。扫描器最优阈值 0.600 被
#   否决：夸×骂包（PUA 脱敏通道）实测中位 0.590，0.600 会静默杀死脱敏路径。
RECALL_MATCH_ENABLED = True     # 相关性匹配总开关（False = 退回 packages[-1] 无脑取最新）
TAU_FLOOR = 0.15                # sim 地板：argmax 后低于此值不召回
PACKAGE_EMBEDDER = "bge"        # 包级嵌入后端（与 event_parser 同款 BGE，实例缓存共享）

# 威胁汇流口（017 雷点3补丁）：threat_now = max(D_error, HistoricalResidual, -S_valence)。
#   注意 D_error 取上一轮写回值 D_error_prev——本轮 D_error 在注入点之后才产生
#   （先有 ReframedContext 才有慢通路/E_raw/D_error），一轮延迟对应"威胁感惯性"。
# 迟滞模式机（017 雷点4补丁，替代 016 草案的 sigmoid：sigmoid 方案被修订版废弃，
#   beta 参数随之取消）：
#   threat_now > THETA_UP → alpha_raw=1（PTSD 放大）
#   threat_now < THETA_DOWN → alpha_raw=0（脱敏）
#   缓冲带内 alpha_raw 保持前值；alpha_t = 0.7*alpha_{t-1} + 0.3*alpha_raw_t（低通）
# theta_up 定参依据（017.5 实测 threat_now 分布，D_error_prev 记 0 口径）：
#   骂簇 0.85~0.90、质疑簇 0.40（HR 主导）、平稳簇（夸/任务）<= 0.20，
#   骂与非骂谷点 [0.40, 0.85]。蓝图初值 0.35 低于质疑簇（0.40），会把质疑轮
#   钉进 PTSD 档，与 014 质疑校准（疑问≠敌意）冲突；取谷点中值 0.60。
THETA_UP = 0.60                 # 迟滞上阈（蓝图初值 0.35，实测修正为 0.60）
THETA_DOWN = 0.25               # 迟滞下阈（蓝图初值，位于实测谷点 [0.20,0.40] 内）
THREAT_LOWPASS_KEEP = 0.7       # 一阶低通：alpha_t = KEEP*alpha_{t-1} + NEW*alpha_raw_t
THREAT_LOWPASS_NEW = 0.3

# 方向公式（动态匹配修订版，017.txt 4.3）：
#   H_eff = min(sqrt(H), H_CAP)；event_severity > SEVERITY_GATE 时脱敏项清零；
#   脱敏生效需 H_eff >= H_EVIDENCE_GATE 且命中包在最近 RECALL_TIME_WINDOW 轮内；
#   recalled_delta_intended = clip(ev × sign × (alpha + (1-alpha) × (-H_eff)), -1.0, CAP_POS)。
# H 实测（017.5）：min 0.500 / p75 0.800 / max 0.889——H_CAP=0.8 压顶高段，
#   证据门 0.4 只挡"几乎没见过"（H<0.16）的伪脱敏。
H_CAP = 0.8                     # H_eff 软上限（min(sqrt(H), H_CAP)）
H_EVIDENCE_GATE = 0.4           # 脱敏证据门：H_eff 低于此值不允许旧伤转正
SEVERITY_GATE = 0.7             # 严重度门：event_severity 超过此值脱敏项强制清零
CAP_POS = 0.3                   # 正向注入封顶：旧伤最多让人淡定，不能让人狂喜
RECALL_TIME_WINDOW = 20         # 时间窗：只允许最近 N 轮内的负向记忆参与脱敏反转
HR_GUARD_THRESHOLD = 0.5        # 一致性护栏：HR 超过此值且注入与 HR 反号
HR_GUARD_MAGNITUDE = 0.3        #   且注入幅值超过此值时，注入幅值减半

# 策略阈值
POLICY_THRESHOLD_HIGH_SEV = 0.8
POLICY_THRESHOLD_INHIBIT_HIGH = 0.7
POLICY_THRESHOLD_INHIBIT_MID = 0.20  # [037.1 裁决] 0.25→0.20：encoder 模式被质疑 I=0.2281，
                                    #   0.25 会落 task_first；0.20 使 fixture 与 encoder
                                    #   都保住 reappraise。t49 扫描 0.20 四项全满足
POLICY_THRESHOLD_SEV_LOW = 0.5
POLICY_EXPRESS_VALENCE = -0.5
POLICY_EXPRESS_INHIBIT = 0.3

# Prompt 翻译阈值
PROMPT_VALENCE_THRESHOLD = -0.3
PROMPT_POSITIVE_THRESHOLD = 0.3
PROMPT_INHIBIT_THRESHOLD = 0.7
PROMPT_HABITUATION_THRESHOLD = 0.7

# 初始化常量
E_PREV_INIT = np.zeros(4)
W_PFC_INIT = 0.0
B_INIT = np.array([0.5, 0.5, 0.5, 0.5])

# 词库
DANGER_KEYWORDS = list(set([
    "蠢", "笨", "滚", "废物", "垃圾", "恶心", "讨厌", "烦", "闭嘴",
    "去死", "傻", "弱智", "白痴", "神经病", "有病", "滚蛋", "没用",
    "差劲", "低能"
]))

POSITIVE_WORDS = ["很棒", "真好", "太棒", "优秀", "厉害", "感谢", "喜欢", "开心", "满意"]
NEGATIVE_WORDS = ["真蠢", "很差", "很坏", "糟糕", "讨厌", "烦人", "生气", "愤怒", "失望", "难过"]

assert not (set(POSITIVE_WORDS) & set(DANGER_KEYWORDS))

# ============================================================
# [M1 已知限制] L1~L16（显式接受，M1 不处理）
# ============================================================
# [M1 已知限制] L1  S_E 饱和
# [M1 已知限制] L2  hash_to_8d 无语义
# [M1 已知限制] L3  retention 与 time_decay 时间常数不一致
# [M1 已知限制] L4  M 是 TopK 均值
# [M1 已知限制] L5  S_E 由效价主导
# [M1 已知限制] L6  κ * w_PFC 基线漂移（由 reset_state 规避）
# [M1 已知限制] L7  hash 不可微
# [M1 已知限制] L8  三场景走空历史分支
# [M1 已知限制] L9  v_s @ B 用快通路更新后 B
# [M1 已知限制] L10 E_raw 由 verify 脚本给出
# [M1 已知限制] L11 随机参数在 config import 时冻结
# [M1 已知限制] L12 ReframedContext = tanh(x) 非零
# [M1 已知限制] L13 x_fast 极性（V4.2 已改 neg-pos）
# [M1 已知限制] L14 x[2] 二值
# [M1 已知限制] L15 b_E 由 D_fast 驱动而非 D_slow（V4.2 新增）
# [M1 已知限制] L16 A_INIT 第一行符号修正（V4.2 新增）
# [M1 已知限制] L17 三场景 action 对随机参数敏感（同公式换种子，被夸 I_inhibit 可能在 0.4 阈值附近翻转）
# [M1 已知限制] L18 competition_score 已接入 hippocampal_evaluator，但 persons/tasks 为空集，person_match/task_match 恒为 0（等 T3/T20）
# [M1 已知限制] L19 EventNode 的 persons/tasks 暂为空集，event_parser 无提取能力（L2），等 T3 填充
# [M2 已知限制] L20 归档器 organize_narrative 已接真 LLM（LLM_BACKEND=deepseek_official，温度 0.3）；仅在切回 mock 时才走桩分支（050 更新）
# [M2 已知限制] L21 读取侧已开启（MEMORY_READ_ENABLED=True），记忆参与评估（050 更新）
# [M2 已知限制] L22 时间触发归档尚未实现，当前仅按轮数触发
# [M2 已知限制] L23 归档 emotion_value 由 LLM 事后生成，与主链路逐轮算出的情绪值口径不同；已确认为强度（0~1）+ emotion_polarity（方向），009.5 批次丁符号修正后被骂场景召回转为负向增量、D_slow 方向随之翻转；012 号工作公式定版为 emotion_value * sign（强度×方向），废弃 (v*2-1)*sign（该式在 negative 且 v<0.5 时产出正向增量，违反"negative 必须负"约束）；方向对错留待 T31 设计会议裁议
# [M2 已知限制] L24 M1 三场景 E_final[0] 区分度极低（-0.206~-0.220），state_translator 的"感受"话术永远落在"平静"档，无法体现被骂/被夸差异
# [M3 已知限制] L25 config.py 头注释（第 2 行）仍写「V4.2 · M1 冻结参数」，与上方版本记录 V4.4 不一致；本次仅登记，不夹带修改（等 T27）
# [M3 已知限制] L26 test_config_seed_frozen 用系统 python 调子进程（缺 numpy），与 T3-B 无关，既存问题
# [M3 已知限制] L27 fixture 模式硬编码 x 为 M1 时代 8 维 M 设计，T3-B 后与新 M（1024）不匹配，结果不可信；fixture 仅作不崩溃冒烟
# [M3 已知限制] L28 T24 把召回情绪值加在 tanh 之外，ReframedContext 逐维绝对值可达 1.8，超出 tanh 原始值域 [-1,1]
# [M3 已知限制] L29 早期文档误记快通路唤醒从未触发，实测 D_fast 0.79，B[1] 每轮触发并饱和（5.5条修正）

# ============================================================
# M2 TODO 清单
# ============================================================
# T1  retention 与 time_decay_weight 对齐（L3）
# T2  S_E 多维聚合重设计（L1, L5）
# T3  hash_to_8d 替换为语义嵌入（L2, L7）
# T4  M 生成方式评估（L4）
# T5  κ * w_PFC 基线漂移评估（L6）
# T6  x_fast[3] 异常检测（5.5.1）
# T7  M_text 接入 LLM 摘要（5.5.6）
# T8  多模态接入（模块12）
# T9  行动策略从规则升级为学习（5.5.12）
# T10 长期记忆积累观察
# T11 心光竞争子项完整定义（5.5.5）
# T12 relation_weight_term 真实计算（5.5.5）
# T13 x_fast[1] 语义进一步评估（L13）
# T14 x[2] 支持连续值（L14）
# T15 b_E 校准（V4.2）
# T16 C_INIT 基线去偏语义确认（5.5.10）
# T17 A_INIT 符号修正的长期验证（L16）
# T18 随机参数敏感度评估（消除或量化三场景 action 对随机种子的不鲁棒性，L17）
# [已完成 V4.3] T19 接入 xinguang.competition_score 至 hippocampal_evaluator
# T20 让 event_parser 提取 persons/tasks 真值（依赖 T3 语义嵌入）
# T21 接真 LLM 替换 archiver.organize_narrative 桩
# T22 接真 LLM 替换 llm_client.MockLLM
# T23 实现时间触发归档（常驻进程 / 开机补写）
# T24 开启读取侧，让记忆参与评估（当前 MEMORY_READ_ENABLED=False）
# T25 对齐归档情绪值与主链路情绪值口径（L23）
# T26 重新设计"感受"判定依据，不再单看 E_final[0]（可能应综合 I_inhibit 或多维情绪），依赖 L24
# [M3 TODO] T27 修正 config.py 头注释版本号（第 2 行仍写 V4.2，与版本记录 V4.4 矛盾），依赖 L25
# [M3 TODO] T28 统一修复测试用系统 python 而非 venv 的问题（L26）
# [M3 TODO] T29 重做 fixture 使 x 适配 1024 维 M，或废弃 fixture 模式（L27）
# [M3 TODO] T30 考虑把召回情绪值注入改到 tanh 内部（需碰 hippocampal_evaluator.py）
# [M3 TODO] T31 明确召回对当前评估的作用方向：放大危险（现状）还是习惯化削弱，待所有者决策
# [M4 TODO] T50 解决 standard 与 sensitive 在 10 轮短对话窗口内 action 撞档的问题
#   （当前仅 gentle 与其余两档分化）。
# [013 待办 A] evaluator 效价校准：质疑/求证类输入是否应判负价（实测被质疑 valence=-0.4
#   → I_inhibit>0.7 → suppress，M1 遗留预期为 reappraise；疑问≠敌意，待设计会裁议）
# [014 待办 B'] 损耗型强度（014 已裁决 KAPPA 0.2→0.5）：不动点公式
#   I* = w_i·D_error/(1+KAPPA·DELTA/(1-RHO))，KAPPA=0.5 时分母=2.0，被骂对照
#   I*≈0.48~0.66（渐进减半）；线性稳定性分析显示 RHO=0.95/DELTA=0.1 下任意
#   KAPPA 均不发散（K>2.26 起复振荡、K≈10 临界），I*→0 需 KAPPA 大幅提高，
#   是否继续加码待所有者看过 014.1 实测数据后定。
# [013 待办 C] 老权重确定性化：w_s/u_s/W_v/U_v/v_s 仍为 randn*0.1（种子冻结可复现、
#   语义不可解释）；e_s/f_v 已于 013 确定性化。方案+开关+对比数据见 013.1.txt，待裁决