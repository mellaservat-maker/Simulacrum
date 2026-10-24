#!/usr/bin/env bash
# gen_hash_t031.sh · 生成 T031 封版 hash 基准（v6.14）
# 本次施工内容（031 号工作）：028 草案任务 3（性格预设，冷切换）
#   1. config.py：新增 PERSONALITY_PROFILES（standard/sensitive/gentle）、
#      中文别名、ACTIVE_PROFILE_ID、DANGER_GAIN 与 load_personality()；
#      只更新五项性格参数与 danger_gain，物理规则不动。
#   2. pipeline.py：compute_danger_score 先乘 DANGER_GAIN 再 clip；
#      run_turn 入口按 state["profile_id"] 冷加载性格。
#   3. state.py：reset_state 增加 profile_id（默认 standard）。
#   4. session_store.py：profile_id 随会话持久化；旧会话缺字段 → standard。
#   5. scripts/chat_cli_v6.py：新增 --profile（中英文别名），冷切换并保存。
#   6. 新增 scripts/probe_t48_personality.py（三性格 10 轮危险度轨迹与持久化）。
#   7. 文档：022.2 使用指南追加 031 增补（性格切换用法）。
# 口径与 T030 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T030_封版_hash基准.txt（v6.13）。
set -e
cd /d/aiqg/emotion_agent

OUT="T031_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T031 封版 hash 基准（v6.14，性格预设 + 冷切换）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T030 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T030 封版基准的差异清单（v6.13 至 v6.14 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T030_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
