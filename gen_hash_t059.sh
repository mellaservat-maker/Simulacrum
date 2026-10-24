#!/usr/bin/env bash
# gen_hash_t059.sh · 生成 T059 封版 hash 基准（v6.34）
# 本次施工内容（059 号工作）：
#   任务1 B19 修复：compute_danger_score 新增第四项 W_DANGER_SLOW * D_slow
#     （门控：仅熟人放大 _d_slow_gain != 1.0 时接入，默认参数 0.0）；
#     config.py 新增 W_DANGER_SLOW = 0.20 [未校准]；pipeline.py 调用点传入
#     门控后的 D_slow。黄金轨迹 trust 区间内熟人效应从不激活 → 零偏移。
#   任务2 事件级软阈值：hippocampal_evaluator.py 余弦 [0.45, 0.60) 软化带，
#     danger_score 与 M 权重乘衰减系数 (s-0.45)/0.15；config.py 新增
#     MATCH_SOFT_MARGIN_LO/HI [未校准]；黄金基准重建。
# 口径与 T058a 一致：全部 .py；排除 secrets_local.py / secrets_override.py /
#   __pycache__ / .pytest_cache / logs / _snapshot_* / data。
# 预期差异：config.py、pipeline.py、hippocampal_evaluator.py、
#   scripts/probe_t58_gain_leak.py、scripts/probe_t58_event_threshold.py、
#   tests/test_golden_trajectory.py（+ 新增脚本/文件如有）。
set -e
cd /d/aiqg/emotion_agent

OUT="T059_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T059 封版 hash 基准（v6.34：B19 修复 D_slow 接入危险度第四项 + 事件级软阈值）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T058a 同口径，排除 _snapshot_* 备份目录）"
  echo "排除项：secrets_local.py、secrets_override.py、__pycache__、.pytest_cache、logs、_snapshot_*、data/"
  echo "排序：按文件相对路径字母序"
  echo "文件总数：${TOTAL}"
  echo
  find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" \
    | sed 's|^\./||' | sort \
    | xargs sha256sum \
    | awk '{ p=$2; sub(/^\*/,"",p); h=toupper(substr($1,1,16)); printf "%s  %s\n", h, p }'
} > "$OUT"

echo "已生成 $OUT（$TOTAL 个 .py）"
echo "--- 与 T058a 的差异清单（预期：config/pipeline/hippocampal_evaluator/两个探针/golden 测试）---"
diff <(grep -E "^[0-9A-F]{16}  " T058a_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
