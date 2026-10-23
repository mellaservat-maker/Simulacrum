#!/usr/bin/env bash
# gen_hash_t047.sh · 生成 T047 封版 hash 基准（v6.25）
# 本次施工内容（047 号工作）：上帝之手记忆植入工具
#   新增 scripts/implant_memory.py：把一段编造故事以合法 MemoryPackage +
#     EventNode 写入指定会话存档（默认 sessions/owner_main.json）。
#   新增 tests/test_implant_memory.py：落盘安全 + 备份 + dry-run +
#     memory_gateway 包级召回可见性四条用例。
#   既有 .py 零改动（本文件只做新增，不改任何现有模块逻辑）。
# 口径与 T046 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 预期差异：仅新增两条（scripts/implant_memory.py、tests/test_implant_memory.py）。
# 交叉 diff 对象：T046_封版_hash基准.txt（v6.24）。
set -e
cd /d/aiqg/emotion_agent

OUT="T047_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T047 封版 hash 基准（v6.25，上帝之手记忆植入工具）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T046 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T046 封版基准的差异清单（v6.24 至 v6.25，预期仅两条新增）---"
diff <(grep -E "^[0-9A-F]{16}  " T046_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
