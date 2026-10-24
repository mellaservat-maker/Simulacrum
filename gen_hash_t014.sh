#!/usr/bin/env bash
# gen_hash_t014.sh · 生成 T014 封版 hash 基准（sha256sum 脚本生成，禁手抄）
# 口径与 T013 Phase4 后基准一致：全部 .py；排除 secrets_local.py / __pycache__ / .pytest_cache / logs
# 格式：16 位大写 hash + 两空格 + 相对路径；按路径字母序
set -e
cd /d/aiqg/emotion_agent

OUT="T014_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T014 封版 hash 基准"
  echo
  echo "生成日期：2026-10-21"
  echo "工作目录：d://aiqg//emotion_agent（014 成人礼：KAPPA=0.5 + evaluator 疑问句式校准 + DETERMINISTIC_WEIGHTS=True）"
  echo "收录范围：全部 .py 文件（与 T013 基准同口径）"
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
echo "--- 与 T013 Phase4 后基准交叉 diff（变更面核对）---"
diff <(grep -E "^[0-9A-F]{16}  " T013_Phase4后_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -c "secrets_local.py$" "$OUT" || true
