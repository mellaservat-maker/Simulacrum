# emotion_agent · Vesper

> 给大语言模型外挂一套「情绪 + 自控 + 记忆」的控制器。
> 不是让 AI "演得像人"，而是在模型外面造一个真实的人：会累、会记仇、也会被时间治愈。

当前版本：**v6.9**

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
05 _recall_match           检索记忆包 + 轮级回捞（含极性一致性过滤）
06 fast_path               本能层：D_fast，上调唤醒
07 slow_path               权衡层：D_slow / V（含疲劳、记忆、语义信号）
08 compute_danger_score    复合危险度（语义 + 奖惩对比 + 严重度门）
09 emotion_releaser        情绪冲击（习惯化越高释放越少）
10 emotion                 情绪加点（惯性 LAMBDA=0.9）
11 run_recall_intercept    记忆注入（PTSD 放大 / 脱敏治愈）+ 威胁模式迟滞
12 run_intercept           自控层：压制、疲劳累积
13 policy                  策略五档：ask_clarify/suppress/reappraise/express/task_first
14 state_translator        数字 → 中文语气指引（禁止报数）
15 prompt_builder          拼提示词
16 llm_client              生成回复（DeepSeek / 可运行时切换任意 OpenAI 兼容端点）
17 回写识海                事件带危险度进入长期记忆图
18 归档                    每 6 轮左右压成记忆包（六段叙事 + 双向量）
```

核心机制三件套：

- **情绪惯性**：`E = 0.9 × E_prev + 冲击`，骂完再夸也要十几轮才缓过来
- **自控疲劳**：`I = 3.0 × D_error − 0.6 × w_PFC`，越累越压不住；安全时才缓慢恢复
- **记忆双向**：`factor = alpha + (1−alpha) × (−H_eff)`，同一个式子，暴怒时放大、放松时治愈

---

## 目录结构

```
emotion_agent/
  server_api.py            FastAPI：/chat /state /reset /config /api/models
  session_store.py         会话持久化（JSON + base64 float32 向量）
  pipeline.py              18 步主链路 run_turn
  config.py                全部参数（大量标注 [未校准]）
  intercept.py             自控层 + 召回注入 + 威胁模式机
  archiver.py              记忆归档（六段叙事 + 均值/峰值向量）
  decompressor.py          轮级回捞
  llm_client.py            LLM 通道（重试 / 降级 / 运行时注入 / Token 统计）
  web/index.html           零依赖前端（1542 行）
  scripts/                 命令行入口 + 探针（probe_t24 ~ t43）
  tests/                   pytest
  T0xx_封版_hash基准.txt    历代封版 hash 基准（工程纪律产物）
```

密钥相关：`secrets_local.py` / `secrets_override.py` **不在版本库中**（见 `.gitignore`），模板见 `secrets_local.py.template`。

---

## 工程纪律（这个项目的可复现性来自这里）

1. **快照链**：每次封版 tar 归档（`_snapshot_v6.x`），排除日志、会话与密钥
2. **hash 基准**：`T0xx_封版_hash基准.txt` 为全部 `.py` 的 sha256 前 16 位，封版必须快照与基准逐位一致
3. **探针体系**：每个行为特性都有独立探针（`scripts/probe_*.py`），改代码先跑探针
4. **报告链**：每次施工产出编号报告，记录实测数据、改动面与遗留项
5. **密钥铁律**：密钥文件不进快照、不进基准、不进日志、不出现在任何输出中

---

## 已知限制（诚实清单）

- 参数大量标注 `[未校准]`：方向正确，速度与力度是工程初值
- 两套检索阈值不一致（事件级 0.6 / 包级 0.15），记忆对判断的影响时常为 0
- `policy` 算出的 action 目前只进日志，尚未闭环到 prompt 或输出约束
- 没有遗忘机制：长期记忆图只增不删，检索是 O(n)
- 情绪 4 维中只有第 0 维（效价）有明确语义
- 无鉴权，仅绑定 `127.0.0.1`，不能直接暴露到局域网
- 前端"演示·连骂八轮"是预置剧本，不走真实链路

---

## 依赖

Python 3.10+ / numpy / fastapi / uvicorn / requests / sentence-transformers（BGE）/ streamlit（可选界面）
详见 `requirements.txt`。

模型文件首次运行会从 HuggingFace 下载 `BAAI/bge-large-zh-v1.5`（约 1.3 GB）。

---

## 说明

个人研究项目，不是产品。所有"情绪"都是可观测、可复现的数值状态，不是拟人话术。
仓库不含任何 API Key、对话日志或会话记忆；使用者需自备模型服务。
