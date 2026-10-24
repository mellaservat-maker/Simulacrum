#!/usr/bin/env bash
# gen_hash_t019.sh · 生成 T019 改动前 hash 基准（019 号工作，sha256sum 生成，禁手抄）
# 口径与 T017.5 封版基准一致：全部 .py；排除 secrets_local.py / __pycache__ /
#   .pytest_cache / logs；另排除 _snapshot_* 快照目录（备份副本非源文件）。
set -e
cd /d/aiqg/emotion_agent

OUT="T019_改动前_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T019 改动前 hash 基准（019 号工作起点，v6.0 封版态）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T017.5 封版基准同口径，排除 _snapshot_* 备份目录）"
  echo "排除项：secrets_local.py、__pycache__、.pytest_cache、logs、_snapshot_*"
  echo "排序：按文件相对路径字母序"
  echo "文件总数：${TOTAL}"
  echo
  find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" \
    | sed 's|^\./||' | sort \
    | xargs sha256sum \
    | awk '{ p=$2; sub(/^\*/,"",p); h=toupper(substr($1,1,16)); printf "%s  %s\n", h, p }'
} > "$OUT"

echo "已生成 $OUT（$TOTAL 个 .py）"
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -c "secrets_local.py$" "$OUT" || true
