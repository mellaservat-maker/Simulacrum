#!/usr/bin/env bash
# gen_hash_t029.sh · 生成 T029 封版 hash 基准（v6.12）
# 本次施工内容（029 号工作，地基改造，行为逐位等价）：
#   任务1 检索网关第一阶段：新增 memory_gateway.py（两路并行、结果层合并，
#     内部仍调用 hippocampal_evaluator.evaluate 与 pipeline._recall_match），
#     pipeline 两条路径（run_scenario / run_turn）统一改为 memory_recall。
#   任务2 配置读取改造：根模块 config 硬导入改为 import config + 运行时属性读取，
#     hippocampal_evaluator.py 按指令保持零改动（内部逻辑冻结）。
#   任务3 写入侧原料：memory_buffer 记录补 danger_score；MemoryPackage 增
#     n_rounds / mean_danger / max_danger；session_store 持久化兼容旧包。
#   新增脚本：scripts/probe_t46_runtime_config.py（运行时 config 生效探针）。
# 口径与 T024 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T024_封版_hash基准.txt（v6.11）。
set -e
cd /d/aiqg/emotion_agent

OUT="T029_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T029 封版 hash 基准（v6.12，检索网关第一阶段 + config 运行时读取 + 写入侧原料）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T024 封版基准同口径，排除 _snapshot_* 备份目录）"
  echo "排除项：secrets_local.py、secrets_override.py、__pycache__、.pytest_cache、logs、_snapshot_*"
  echo "排序：按文件相对路径字母序"
  echo "文件总数：${TOTAL}"
  echo
  find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" \
    | sed 's|^\./||' | sort \
    | xargs sha256sum \
    | awk '{ p=$2; sub(/^\*/,"",p); h=toupper(substr($1,1,16)); printf "%s  %s\n", h, p }'
} > "$OUT"

echo "已生成 $OUT（$TOTAL 个 .py）"
echo "--- 与 T024 封版基准的差异清单（v6.11 至 v6.12 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T024_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
