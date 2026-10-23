#!/usr/bin/env bash
# gen_hash_t046.sh · 生成 T046 封版 hash 基准（v6.24）
# 本次施工内容（046 号工作）：移动端“后端会话”为空的修复
#   1. web/index.html：CONFIG.endpoint 改为跟随页面来源——http(s) 打开时用
#      location.origin + '/chat'，file:// 打开才回落 127.0.0.1:8000/chat。
#      此前写死 127.0.0.1，手机通过局域网 IP 打开页面后所有接口都指向手机
#      自己，/api/sessions、/api/history、/state、/chat 全部失败。
#   2. web/index.html：bootstrapSession 在拉取接口前先写入默认选项，
#      接口失败时下拉框也不会是空白；option 显式给深色底浅色字，
#      避免移动端原生选择器白底白字。
#   3. web/index.html：新增 probeBackend()，启动时真实探测 /health，
#      连接状态不再无条件显示“已连接后端”。
# 口径与 T045 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 注意：本次改动全部在 web/index.html（非 .py），故基准应与 T045 逐位一致。
# 交叉 diff 对象：T045_封版_hash基准.txt（v6.23）。
set -e
cd /d/aiqg/emotion_agent

OUT="T046_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T046 封版 hash 基准（v6.24，移动端接口同源修复）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T045 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T045 封版基准的差异清单（v6.23 至 v6.24 的 .py 改动面，预期为空）---"
diff <(grep -E "^[0-9A-F]{16}  " T045_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
