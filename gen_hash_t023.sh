#!/usr/bin/env bash
# gen_hash_t023.sh · 生成 T023 封版 hash 基准（v6.10）
# 本次施工内容：
#   任务1 Action 闭环：state_translator/prompt_builder 收 action 参数 +
#     四种强指令；pipeline 两处主链路传 action + suppress 暴走护栏（>30 字截断加……）
#   任务2/3 统计先行：probe_t44_calibration 实测事件级/包级匹配与符号校验，
#     数据不支持降阈值（前提"空召回率极高"不成立），MATCH_THRESHOLD 维持 0.6
# 口径与 T022 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T023_改动前_hash基准.txt（与 T022 封版零漂移的施工起点）。
set -e
cd /d/aiqg/emotion_agent

OUT="T023_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T023 封版 hash 基准（v6.10，Action 闭环 + 阈值校准：维持 0.6）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T022 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T023 改动前基准的差异清单（本次改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T023_改动前_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
