#!/usr/bin/env bash
# gen_hash_t013.sh · 生成 T013 Phase4 后 hash 基准（sha256sum 脚本生成，禁手抄）
# 口径与 T012 改动后基准一致：全部 .py；排除 secrets_local.py / __pycache__ / .pytest_cache / logs
# 格式：16 位大写 hash + 两空格 + 相对路径；按路径字母序
set -e
cd /d/aiqg/emotion_agent

OUT="T013_Phase4后_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T013 Phase4 后 hash 基准"
  echo
  echo "生成日期：2026-10-21"
  echo "工作目录：d://aiqg//emotion_agent（013 深夜自治 Phase4：pipeline 接入 evaluator 真 S + e_s/f_v 确定性设计）"
  echo "收录范围：全部 .py 文件（与 T012 改动后基准同口径）"
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
echo "--- 与 T013 Phase1 后基准交叉 diff ---"
diff <(grep -E "^[0-9A-F]{16}  " T013_Phase1后_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -c "secrets_local.py$" "$OUT" || true
