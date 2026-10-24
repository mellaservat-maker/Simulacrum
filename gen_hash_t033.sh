#!/usr/bin/env bash
# gen_hash_t033.sh · 生成 T033 封版 hash 基准（v6.16）
# 本次施工内容（033 号工作）：性格影响自控 + danger 高门
#   1. config.py：新增 DANGER_NEUTRAL=0.30（[未校准]）；
#      三套性格新增 w_pfc_factor（sensitive 1.5 / standard 1.0 / gentle 0.5），
#      新增 ACTIVE W_PFC_FACTOR，load_personality 同步切换。
#   2. intercept.py：I_inhibit 的疲劳侵蚀项改为
#      KAPPA * w_PFC * W_PFC_FACTOR（只在自控入口加性格，不碰 policy）。
#   3. pipeline.py：danger_boost 改为 max(0, danger_now - DANGER_NEUTRAL)，
#      两条路径同步；轻度事件（被质疑）不再放大。
#   4. scripts/probe_t48_personality.py：打印 boost 列、自控耐力台账，
#      新增 W_PFC_FACTOR 固定工况单元断言，如实记录 10 轮曲线。
#   5. tests/test_limits.py：D_error 边界注释更新为两代公式并集口径。
#   policy.py 本批未改动（保持 032 判定逻辑原样）。
# 口径与 T032 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T032_封版_hash基准.txt（v6.15）。
set -e
cd /d/aiqg/emotion_agent

OUT="T033_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T033 封版 hash 基准（v6.16，性格衰竭系数 + danger 高门）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T032 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T032 封版基准的差异清单（v6.15 至 v6.16 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T032_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
