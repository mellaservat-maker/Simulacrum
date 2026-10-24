#!/usr/bin/env bash
# gen_hash_t043.sh · 生成 T043 封版 hash 基准（v6.21）
# 本次施工内容（043 号工作）：初始状态字段补齐，不改任何判定逻辑
#   1. server_api.py：_state_payload 在无轮次（log is None）时也返回
#      w_PFC / I_inhibit(0.0) / e / e0 / action(task_first)。
#   2. web/index.html：删除写死的演示值（S 初始值、CALM 常量、HTML 静态占位
#      0.18/0.92/+0.16 全部归零，e 数组归零）；syncServerState 改为无条件
#      以服务端 w_PFC / I_inhibit / e0 / e / action / danger_now 覆盖本地；
#      loadChat 切回历史对话时同样以 /state 为准刷新。
# 口径与 T041 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T041_封版_hash基准.txt（v6.20）。
set -e
cd /d/aiqg/emotion_agent

OUT="T043_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T043 封版 hash 基准（v6.21，初始状态字段补齐）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T041 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T041 封版基准的差异清单（v6.20 至 v6.21 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T041_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
