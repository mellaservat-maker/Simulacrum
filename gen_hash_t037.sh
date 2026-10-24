#!/usr/bin/env bash
# gen_hash_t037.sh · 生成 T037 封版 hash 基准（v6.17）
# 本次施工内容（037 号工作）：轴一落地 + MID 阈值重校 + 直通闸修法乙
#   1. config.py：w_i 3.0→1.2；删除每性格 w_pfc_factor；
#      W_PFC_FACTOR 固定 1.0（保留键供 t49 十三键还原断言）；
#      POLICY_THRESHOLD_INHIBIT_MID 0.4→0.25。
#   2. intercept.py：疲劳项恢复原始公式 -KAPPA*w_PFC（乘子恒 1.0）。
#   3. policy.py：删除不可达的 danger_boost>=0.80 分支，改为 danger_now>=0.95。
#   4. pipeline.py：策略层改传 danger_now，不再传 danger_boost。
#   5. scripts/probe_t48_personality.py：随 037 撤回性格系数同步修口径。
#   6. scripts/probe_t49_axis_compare.py（新增）：S0-S4 + MID 扫描。
# 口径与 T033 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T033_封版_hash基准.txt（v6.16）。
set -e
cd /d/aiqg/emotion_agent

OUT="T037_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T037 封版 hash 基准（v6.17，轴一 w_i=1.2 + MID 0.25）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T033 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T033 封版基准的差异清单（v6.16 至 v6.17 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T033_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
