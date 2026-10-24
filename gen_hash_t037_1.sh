#!/usr/bin/env bash
# gen_hash_t037_1.sh · 生成 T037.1 封版 hash 基准（v6.18）
# 本次施工内容（037.1 裁决）：MID 0.25→0.20 + 方向反转语义入注释
#   1. config.py：POLICY_THRESHOLD_INHIBIT_MID 0.25→0.20
#      （encoder 模式被质疑 I=0.2281，0.20 使 fixture 与 encoder 均保 reappraise）；
#      性格块补写“方向反转”的设计语义注释（gentle 不在乎所以早放弃，
#      sensitive 过度在意所以硬撑更久）。
#   2. scripts/probe_t49_axis_compare.py：MID 选择改为按当前 config 值报告。
#   S3/S4 旧冻结数按 037.1 丙项正式作废，本批不重出。
# 口径与 T037 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T037_封版_hash基准.txt（v6.17）。
set -e
cd /d/aiqg/emotion_agent

OUT="T037.1_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T037.1 封版 hash 基准（v6.18，MID 0.20）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T037 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T037 封版基准的差异清单（v6.17 至 v6.18 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T037_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
