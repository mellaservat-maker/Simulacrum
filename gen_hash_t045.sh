#!/usr/bin/env bash
# gen_hash_t045.sh · 生成 T045 封版 hash 基准（v6.23）
# 本次施工内容（045 号工作）：聊天记录后端持久化 + 局域网/移动端
#   1. session_store.py：dump_state/load_state 增加 messages 列表
#      （只存 role/content/ts），旧存档无此字段按空列表降级。
#   2. server_api.py：成功轮次把 user+assistant 两条消息追加进 st["messages"]
#      并随会话落盘；新增 GET /api/history?session_id= 返回完整记录；
#      启动注释改为 0.0.0.0。
#   3. web/index.html：bootstrapSession 与下拉框切换时拉取 /api/history 渲染；
#      空状态抽成 showEmptyState 复用；后端无记录但本地有对话时保持本地显示。
#   4. 启动脚本：监听地址 127.0.0.1 改为 0.0.0.0（桌面版与项目版）。
# 口径与 T044 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T044_封版_hash基准.txt（v6.22）。
set -e
cd /d/aiqg/emotion_agent

OUT="T045_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T045 封版 hash 基准（v6.23，聊天记录后端持久化）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T044 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T044 封版基准的差异清单（v6.22 至 v6.23 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T044_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
