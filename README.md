# emotion_agent · Vesper

> 给大语言模型外挂一套「情绪 + 自控 + 记忆」的控制器。
> 不是让 AI "演得像人"，而是在模型外面造一个真实的人：会累、会记仇、也会被时间治愈。

当前版本：**v6.34**（055 批：对话历史、习惯化重定义、跨天时间感；056 批：关系状态；058 批：地基修复——常驻进程跨天衰减、broken 施密特触发器、亲密度独白、熟人效应防污染验证；059 批：B19 修复——熟人放大的 D_slow 接入复合危险度第四项、事件级软阈值 margin 加固）

---

## 它能做什么（三个可直接观察的现象）

| 场景 | 普通聊天软件 | 这个系统 |
|---|---|---|
| 你连骂它几句 | 礼貌道歉，情绪为零 | 疲劳值 `w_PFC` 一路爬升、理智防线 `I_inhibit` 绷到 1.0，回复口气从解释变成压抑的反驳 |
| 骂完忽然夸它 | 立刻开心道谢 | 缓不过来：情绪有惯性（每轮只朝目标移动 10%），要好几轮才回暖 |
| 隔很久再来骂它 | 完全没记忆 | 会"想起"类似的事；且**放松时被旧伤治愈，暴怒时被旧伤放大** |

三个现象都不需要改 prompt，是内部状态自己在走。

---

## 快速开始（Windows）

```bat
:: 1. 准备环境
python -m venv venv
venv\Scripts\pip install -r requirements.txt

:: 2. 配置密钥（二选一）
::    a) 复制模板填写：copy secrets_local.py.template secrets_local.py
::    b) 启动后在网页侧栏「API 设置」里填（推荐，立即生效）

:: 3. 启动（后端 + 网页同源，自动开浏览器）
启动网页版.bat
```

浏览器打开 `http://127.0.0.1:8000/`，左侧栏「API 设置」→ 填 Base URL / API Key → 获取模型列表 → 选模型 → 保存并生效。

局域网/手机访问（050 起）：后端已监听 `0.0.0.0`，但默认只允许本机调用接口。
需要手机访问时，先在启动前设置令牌，再用带令牌的链接打开：

```bat
set EMOTION_API_TOKEN=你自定义的一串字符
启动网页版.bat
```

手机浏览器打开 `http://<本机局域网IP>:8000/?token=你自定义的一串字符`，
令牌会存进手机浏览器本地，之后直接打开首页即可。未设置令牌时，局域网的接口请求
会被明确拒绝（403），前端会显示"后端已断开，请检查服务"，不再静默回退演示。

其他入口：

| 文件 | 用途 |
|---|---|
| `启动网页版.bat` | FastAPI 后端 + 网页界面（同源，推荐） |
| `启动网页版-Streamlit版.bat` | Streamlit 版界面（备选，需 `pip install streamlit`） |
| `chat_v6.bat` | 纯命令行对话 |

---

## 架构

采集输入 → 18 步链路 → 生成回复 → 写回记忆：

```
01 event_parser            文本 → EventNode（BGE 1024 维）
02 feature_encoder         事件 → x(8) / x_fast(4)
03 xinguang                心力预算 B_xg
04 hippocampal_evaluator   记忆影响判断：HR / H / M / ReframedContext
05 _recall_match           检索记忆包 + 轮级回捞（相关性闸判原始相似度，
                           排序判语义×新鲜度；含极性一致性过滤）
06 fast_path               本能层：D_fast，上调唤醒
07 slow_path               权衡层：D_slow / V（含疲劳、记忆、语义信号）
                           记忆项用 M 减会话均值基后的残差，不再是一个常数
08 compute_danger_score    复合危险度（语义 + 奖惩对比 + 严重度门）
09 emotion_releaser        情绪冲击（习惯化越高释放越少）
10 emotion                 情绪加点（惯性 LAMBDA=0.9）
11 run_recall_intercept    记忆注入（PTSD 放大 / 脱敏治愈）+ 威胁模式迟滞
12 run_intercept           自控层：压制、疲劳累积
13 policy                  策略五档：ask_clarify/suppress/reappraise/express/task_first
14 state_translator        数字 → 中文语气指引（禁止报数）
15 prompt_builder          独白 + 当前输入 + 最近 6 轮历史 → 多轮 messages
16 llm_client              生成回复（messages 数组；DeepSeek / 任意 OpenAI 兼容端点）
17 回写识海                事件带危险度进入长期记忆图
18 归档                    缓冲区满 6 条时归档前 4 条、留 2 条（每 4 轮一包）；
                           同主题包按余弦 0.85 去重合并；叙事失败也照样成包
```

核心机制：

- **情绪惯性**：`E = 0.85 × E_prev + 冲击`（0.9 惯性 − 0.05 OU 回归项），骂完再夸也要十几轮才缓过来
- **自控疲劳**：`I = 1.2 × D_error − 0.6 × w_PFC`，越累越压不住；安全时才缓慢恢复
- **记忆双向**：`factor = alpha + (1−alpha) × (−H_eff)`，同一个式子，暴怒时放大、放松时治愈
- **对话不金鱼**：最近 6 轮问答（token 预算 1000）作为多轮 messages 一起发给模型，
  问"我叫什么"能答出上一轮说的名字；独白与行动指令仍在最后一条 user 里
- **习惯化只认同类**：`H = Σ(w_time × w_sim)/(Σ + 1)`，`w_sim` 只在当前事件与历史
  余弦 > 0.7 时才贡献。聊代码和聊心情不会互相累积"麻木"（旧式是纯计数器，
  第 3 轮就顶到 `H_eff` 封顶）
- **跨天会消退**：两条结算路径共用同一套四条通道——加载会话按 `saved_at`，
  常驻进程轮初按 `last_active`（超过 3600 秒未活动才结算）。情绪（tau=1 天）、
  疲劳（2 天）、习惯化（3 天）各自回落。隔 3 天 ≈ 连聊 18.5 轮的情绪消退量；
  055 之前"气三天"和"没气过"在系统里是同一件事；058 修复了常驻进程
  "开着不关但三天没说话"不消退的问题（B16）
- **关系有账本**：`st["relationships"]` 按稳定实体 ID（"我"→user_001、"老板"→boss_01）
  累积 trust / intimacy。攻击按危险度扣信任（`-= 0.02×danger`），安抚缓慢修复
  （`+0.01`）。破裂是施密特触发器：跌破 0.2 判破裂、涨回 0.3 才解除，
  0.2–0.3 之间保持原状（不会在阈值附近反复横跳）。亲密度两档独白：
  > 0.7 "认识很久"、< 0.2 "还很防备"。熟人效应：信任 > 0.7 放大慢通路思考
  （熟人面前想得多，059 起经 `W_DANGER_SLOW×D_slow` 第四项进入复合危险度，
  传导至 I_inhibit / E_final / w_PFC——熟人面前的负面事件更压情绪、更耗神），
  信任 < 0.3 加快习惯化累积（麻了）。实测骂 10 轮与夸 10 轮，
  trust 分化 0.274；同一串攻击，前 12 轮与"无关系感知"逐位一致，
  之后系统性分道扬镳
- **记忆不因时间消失**：相关性闸（`TAU_FLOOR=0.45`）只判原始语义相似度，
  时间衰减 `exp(−age/20)` 只参与排序。老记忆只会被排到后面，不会被判为"不相关"
- **语义分辨率**：1024 维记忆先减去会话均值基（去掉 BGE 各向异性的共模方向），
  再用零均值方向读出。不同话题的读出值标准差 0.034（旧口径 0.006），
  话题间/话题内信噪比 3.81（旧 0.95）
- **破防机制**：疲劳持续累积、`I_inhibit` 跌破 0.30 且 `E_final[0] < −0.50` 时触发
  `express`（爆发）；每次爆发后进入 10 轮冷却，冷却期内降级为 suppress / task_first，
  之后再次累积、再次爆发（周期性，见 039 批）
- **归档不丢**：叙事生成失败时仍产出"空叙事 + 完整数值"的包，
  原文与数值都不会只在 raw_log 里躺着

---

## 目录结构

```
emotion_agent/
  server_api.py            FastAPI：/chat /state /reset /config /api/models
  session_store.py         会话持久化（JSON + base64 float32 向量）
  pipeline.py              18 步主链路（_run_chain 唯一实现，run_scenario/run_turn 为薄入口）
  config.py                全部参数（大量标注 [未校准]）
  intercept.py             自控层 + 召回注入 + 威胁模式机
  semantic_projection.py   M 减会话均值基（恢复 1024 维语义分辨率）
  hippocampal_evaluator.py 事件级海马体；习惯化 = 同类事件重复度（055）
  relationship.py          关系状态：trust/intimacy 账本 + 熟人效应（056）
  prompt_builder.py        独白 + 历史 → 多轮 messages（055）
  archiver.py              记忆归档（叙事 + 双向量 + 去重合并 + 失败不丢包）
  decompressor.py          轮级回捞
  llm_client.py            LLM 通道（messages 数组 / 重试 / 降级 / 运行时注入 / Token 统计）
  web/index.html           零依赖前端（约 2200 行；内在状态面板含"当前关系"栏）
  scripts/                 命令行入口 + 探针（probe_t24 ~ t58）+ 记忆植入工具
  tests/                   pytest（90 passed / 1 skipped）
  T0xx_封版_hash基准.txt    历代封版 hash 基准（工程纪律产物）
```

密钥相关：`secrets_local.py` / `secrets_override.py` **不在版本库中**（见 `.gitignore`），模板见 `secrets_local.py.template`。

本地测试语料：`data/` 目录存放真实聊天记录的脱敏语料（如 `data/test_corpus.txt`，供 `probe_t58_event_threshold.py` 的深会话诊断使用）。**该语料仅限本地，禁止提交与上传**（已在 `.gitignore` 排除，也不进快照与 hash 基准）；开源克隆中该文件默认缺失，探针会自动跳过对应部分，其余功能不受影响。

---

## 工程纪律（这个项目的可复现性来自这里）

1. **快照链**：每次封版整份复制归档（`_snapshot_v6.x`），排除日志、会话与密钥
2. **hash 基准**：`T0xx_封版_hash基准.txt` 为全部 `.py` 的 sha256 前 16 位，封版必须快照与基准逐位一致
3. **探针体系**：每个行为特性都有独立探针（`scripts/probe_*.py`），改代码先跑探针
4. **黄金轨迹**：`tests/test_golden_trajectory.py` 固定 20 轮输入，逐轮记录
   `w_PFC / I_inhibit / E[4] / action / danger / H`，容差 1e-6 比对。
   重建基准：`GOLDEN_UPDATE=1 pytest tests/test_golden_trajectory.py`
   （注意：基准里冻结了时钟、S 表与 mock 归档后端，只反映数值链路）
5. **报告链**：每次施工产出编号报告，记录实测数据、改动面与遗留项
6. **密钥铁律**：密钥文件不进快照、不进基准、不进日志、不出现在任何输出中

跑测试（必须限定 tests 目录，否则会扫到 `_snapshot_*` 里的同名测试）：

```bat
set HF_HOME=d:/aiqg/hf_cache
set HF_HUB_OFFLINE=1
d:\aiqg\venv\Scripts\python.exe -m pytest tests -q
```

---

## 已知限制（诚实清单）

- 参数大量标注 `[未校准]`：方向正确，速度与力度是工程初值
- 两套检索阈值：事件级锚点 0.5（`MATCH_THRESHOLD`）/ 包级 0.45（`TAU_FLOOR`），
  包级已按实测定参（相关输入最低 0.647、无关输入最高 0.447）；事件级 058 已做
  三语料诊断（真实存档 7/7 查询轮触发、黄金轨迹 17/19 轮触发），"通道死"的
  预设被推翻，病灶在 margin 脆弱与深会话排序稀释；059 落地软阈值——
  余弦 [0.45, 0.60) 为软化带，候选按 `w=clip((s-0.45)/0.15, 0, 1)` 降权后
  进 HistoricalResidual 均值与 M 加权（0.5 处 w=1/3，≥0.60 全权），
  边缘记忆温和参与竞争，阈值微移不再翻转"有无"
- action 已闭环至 prompt 输出（suppress 带 30 字护栏），见 023 批
- 没有遗忘机制：长期记忆图只增不删，检索是 O(n)；写入侧有去重合并（余弦 0.85）
- 四维情绪均已有值且参与释放层，见 030 批；但只有 E[0] 进入自控与独白（`w_e` 后三维为 0）
- 无 HTTPS、无用户体系；050 起有令牌（`EMOTION_API_TOKEN`）与回环限制，未设令牌时局域网接口一律拒绝，页面本身仍可访问
- 单进程单 worker + 全局锁，多会话是排队执行，不是并发实现
- 前端"演示·连骂八轮"是预置剧本，不走真实链路；后端断连时不再自动回退到这套剧本，而是显式报错并禁用输入
- 会话存档每轮整份重写（`events` 每条含 1024 维向量），长会话写放大是已知性能债
- 语义读出用固定种子的随机方向（非主成分）：可区分话题，但不可解释为某个语义轴
- 熟人效应的 D_slow 增益 059 起接入数值链（B19 修复）：经 `compute_danger_score`
  第四项 `W_DANGER_SLOW×D_slow` 传导至危险度、I_inhibit、E_final 与 w_PFC
  （反事实验证：权重置 0 即回到零污染态）；`W_DANGER_SLOW=0.20` 为 [未校准]
  工程初值，标定探针待立
- 习惯化阈值 0.7 让脱敏通道的触发条件变严：只有"同类事件重复"才可能被治愈，
  不同话题的旧伤只会放大、不会翻正（语义上更正确，但覆盖面比旧式窄）

---

## 依赖

Python 3.10+ / numpy / fastapi / uvicorn / requests / sentence-transformers（BGE）/ streamlit（可选界面）
详见 `requirements.txt`。

模型文件首次运行会从 HuggingFace 下载 `BAAI/bge-large-zh-v1.5`（约 1.3 GB）。

---

## 说明

个人研究项目，不是产品。所有"情绪"都是可观测、可复现的数值状态，不是拟人话术。
仓库不含任何 API Key、对话日志或会话记忆；使用者需自备模型服务。
