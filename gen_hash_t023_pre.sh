#!/usr/bin/env bash
# gen_hash_t023_pre.sh · 生成 T023 改动前 hash 基准（sha256sum 脚本生成，禁手抄）
# 口径与 T022 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py（密钥文件一律不入基准）/ __pycache__ /
#   .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T022_封版_hash基准.txt（v6.9，86 py）。主树已与 GitHub
#   origin/main 同步（e39c3e8），预期零漂移。
set -e
cd /d/aiqg/emotion_agent

OUT="T023_改动前_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T023 改动前 hash 基准（Action 闭环 + 匹配阈值校准施工起点）"
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
echo "--- 与 T022 封版基准的差异清单（须为空=零漂移）---"
diff <(grep -E "^[0-9A-F]{16}  " T022_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
