#!/usr/bin/env bash
# gen_hash_t015.sh · 生成 T015 封版 hash 基准（sha256sum 脚本生成，禁手抄）
# 口径与 T014.5 基准一致：全部 .py；排除 secrets_local.py / __pycache__ / .pytest_cache / logs
set -e
cd /d/aiqg/emotion_agent

OUT="T015_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T015 封版 hash 基准"
  echo
  echo "生成日期：2026-10-21"
  echo "工作目录：d://aiqg//emotion_agent（015 封版：温度分层 evaluator=0.0/archiver=0.3 + 同输入 TTL 缓存）"
  echo "收录范围：全部 .py 文件（与 T014.5 基准同口径）"
  echo "排除项：secrets_local.py、__pycache__、.pytest_cache、logs"
  echo "排序：按文件相对路径字母序"
  echo "文件总数：${TOTAL}"
  echo
  find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -name "secrets_local.py" -not -name "secrets_override.py" \
    | sed 's|^\./||' | sort \
    | xargs sha256sum \
    | awk '{ p=$2; sub(/^\*/,"",p); h=toupper(substr($1,1,16)); printf "%s  %s\n", h, p }'
} > "$OUT"

echo "已生成 $OUT（$TOTAL 个 .py）"
echo "--- 与 T014.5 基准交叉 diff（变更面核对）---"
diff <(grep -E "^[0-9A-F]{16}  " T014.5_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -c "secrets_local.py$" "$OUT" || true
