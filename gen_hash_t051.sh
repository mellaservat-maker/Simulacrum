#!/usr/bin/env bash
# gen_hash_t051.sh · 生成 T051 封版 hash 基准（v6.27）
# 本次施工内容（051 号工作）：OU 情绪回归项落地
#   config.py：新增 E_HOMEOSTASIS_GAMMA = 0.05 [未校准]（050 探针建议值）
#   emotion.py：E_raw 增加 - gamma * E_prev，等价于有效保留系数 0.9-gamma
#   scripts/probe_t52_ou_homeostasis.py：追加「真实链路复核」段，
#     用 emotion.run_emotion 与解析式逐轮比对（051 复核要求）
#   既有行为边界（policy / pipeline / intercept / decompressor）零改动。
# 口径与 T050 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 预期差异 3 条：config.py、emotion.py、probe_t52。
set -e
cd /d/aiqg/emotion_agent

OUT="T051_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T051 封版 hash 基准（v6.27，OU 情绪回归项落地）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T050 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T050 封版基准的差异清单（v6.26 至 v6.27，预期 3 条）---"
diff <(grep -E "^[0-9A-F]{16}  " T050_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- 机制四文件必须零差异 ---"
for f in pipeline.py policy.py intercept.py decompressor.py; do
  grep -E "  ${f}$" T050_封版_hash基准.txt > /tmp/t050_$$.txt || true
  grep -E "  ${f}$" "$OUT" > /tmp/t051_$$.txt || true
  diff /tmp/t050_$$.txt /tmp/t051_$$.txt && echo "OK   $f 未变"
done
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
