#!/usr/bin/env bash
# gen_hash_t024.sh · 生成 T024 封版 hash 基准（v6.11）
# 本次施工内容（024 号工作）：
#   任务1.2 阈值裁决：config.py MATCH_THRESHOLD 0.6 -> 0.5
#     （023 探针数据显示 0.471~0.550 天然谷底：同类召回 +1，无关类误命中保持 0/7）
#   任务1.3 代码已提交本地（288ab61，含 023 Action 闭环/护栏/023.5 强制禁代理/
#     023.6 数据入档对应的三支探针），推送待网络恢复
#   任务2 终端持久化：chat_cli_v6.py 接入 session_store（启动恢复 + 每轮落盘 +
#     退出再存），新增 probe_t45_cli_persistence.py 全绿
#   任务3 启动脚本就绪：4 个 bat 加 HF_HUB_OFFLINE=1（023.5 代理残留同根病，
#     BGE 改纯本地加载，启动不再依赖网络）
# 口径与 T023 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T023_封版_hash基准.txt（v6.10）。
set -e
cd /d/aiqg/emotion_agent

OUT="T024_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T024 封版 hash 基准（v6.11，阈值 0.5 + 终端持久化闭环）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T023 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T023 封版基准的差异清单（023.5 之后至 v6.11 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T023_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
