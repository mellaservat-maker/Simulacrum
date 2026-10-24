#!/usr/bin/env bash
# gen_hash_t032.sh · 生成 T032 封版 hash 基准（v6.15）
# 本次施工内容（032 号工作）：打通危险度与自控链路
#   1. config.py：新增 W_DANGER=0.30、POLICY_DANGER_BOOST_HIGH=0.80（均 [未校准]）。
#   2. intercept.py：run_intercept 新增 danger_boost（默认 0.0）；
#      D_error = max(0, S_E-THETA_ERROR) + danger_boost * W_DANGER。
#   3. policy.py：policy_minimal 新增 danger_boost；>= POLICY_DANGER_BOOST_HIGH
#      时跳过半防/权衡档直接 suppress。
#   4. pipeline.py：run_scenario / run_turn 两条路径先算 danger_now，
#      作为 danger_boost 送入拦截层与策略层；event.danger_score 复用 danger_now。
#   5. scripts/probe_t48_personality.py：打印 S_E/D_error，增加 W_DANGER 扫描
#      （0.0/0.3/1.0）与探针日志隔离（不污染 logs/ 正式口径）。
#   6. tests/test_limits.py：D_error 边界随公式更新为
#      [-W_DANGER, (1-THETA_ERROR)+W_DANGER]，属预期基线变更。
# 口径与 T031 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T031_封版_hash基准.txt（v6.14）。
set -e
cd /d/aiqg/emotion_agent

OUT="T032_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T032 封版 hash 基准（v6.15，危险度接入自控）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T031 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T031 封版基准的差异清单（v6.14 至 v6.15 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T031_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
