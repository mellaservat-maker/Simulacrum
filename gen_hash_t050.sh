#!/usr/bin/env bash
# gen_hash_t050.sh · 生成 T050 封版 hash 基准（v6.26）
# 本次施工内容（050 号工作）：第一批工程与安全修复 + OU 回归探针
#   S2  server_api.py：新增 _auth_guard 中间件（回环放行 / 局域网令牌）
#       config.py：新增 EMOTION_API_TOKEN（环境变量，不落盘）
#   S1  web/index.html：请求失败显式报错并锁输入，不再静默回退演示
#   A1  server_api.py / session_store.py：未计算字段返回 null，
#       新增 last_metrics 随会话落盘，GET /state 刷新后不再归零
#   A3  session_store.py / archiver.py / scripts/implant_memory.py：
#       _dump_messages → dump_messages；archiver 新增 append_package / replace_packages
#   A5  README.md / config.py：版本号、行数、监听地址、归档器状态同步
#   新增 scripts/probe_t52_ou_homeostasis.py（OU 回归曲线，只出数据）
#   新增 tests/test_state_payload.py（4 条，A1 语义回归）
# 口径与 T047 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 预期差异 7 条：4 个既有文件改动（archiver/config/server_api/session_store）、
#   1 个植入工具改公开接口、2 个新增文件（probe_t52 / test_state_payload）。
#   pipeline.py / policy.py / intercept.py 必须逐位不变。
set -e
cd /d/aiqg/emotion_agent

OUT="T050_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T050 封版 hash 基准（v6.26，第一批工程与安全修复 + OU 回归探针）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T047 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T047 封版基准的差异清单（v6.25 至 v6.26，预期 7 条）---"
diff <(grep -E "^[0-9A-F]{16}  " T047_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- 机制三文件必须零差异 ---"
for f in pipeline.py policy.py intercept.py; do
  grep -E "  ${f}$" T047_封版_hash基准.txt > /tmp/t047_$$.txt || true
  grep -E "  ${f}$" "$OUT" > /tmp/t050_$$.txt || true
  diff /tmp/t047_$$.txt /tmp/t050_$$.txt && echo "OK   $f 未变"
done
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
