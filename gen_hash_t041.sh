#!/usr/bin/env bash
# gen_hash_t041.sh · 生成 T041 封版 hash 基准（v6.20）
# 本次施工内容（041 号工作）：前端适配补齐，不改后端判定逻辑
#   1. server_api.py：/chat 接收 profile_id；响应新增 e[0..3]、action、
#      express_cooldown、danger_now、profile_id/label、thresholds；
#      /state 同步返回性格、四维、冷却与阈值。
#   2. app.py（Streamlit）：侧边栏性格选择（冷切换）；四维情绪四条展示；
#      破防 express 告警与冷却剩余轮数；danger_now 与 MID/HIGH 阈值展示。
#   3. web/index.html：顶部性格下拉；四维 e1/e2/e3、danger/MID、action、
#      破防冷却行；express 红色脉冲；normalizeResponse 与 mock 同步扩展。
#   4. pipeline.py：日志新增观测字段 danger_now（不参与计算）。
# 口径与 T039 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T039_封版_hash基准.txt（v6.19）。
set -e
cd /d/aiqg/emotion_agent

OUT="T041_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T041 封版 hash 基准（v6.20，前端适配补齐）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T039 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T039 封版基准的差异清单（v6.19 至 v6.20 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T039_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
