#!/usr/bin/env bash
# gen_hash_t017.5.sh · 生成 T017.5 封版 hash 基准（v6.0，sha256sum 脚本生成，禁手抄）
# 口径与 T015 封版基准一致：全部 .py；排除 secrets_local.py / __pycache__ /
#   .pytest_cache / logs；另排除 _snapshot_* 快照目录（备份副本非源文件，
#   否则与 T015 的 71 个源文件口径不可比——历史快照链在环境还原后丢失，
#   T015 基准生成时无快照目录存在）。
# 交叉 diff 对象：T017.5_改动前_hash基准.txt（本次工作的改动前状态）。
set -e
cd /d/aiqg/emotion_agent

OUT="T017.5_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T017.5 封版 hash 基准（v6.0）"
  echo
  echo "生成日期：2026-10-21"
  echo "工作目录：d://aiqg//emotion_agent（T31×记忆检索联合重构封版）"
  echo "收录范围：全部 .py 源文件（与 T015 封版基准同口径，排除 _snapshot_* 备份目录）"
  echo "排除项：secrets_local.py、__pycache__、.pytest_cache、logs、_snapshot_*"
  echo "排序：按文件相对路径字母序"
  echo "文件总数：${TOTAL}"
  echo
  find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" \
    | sed 's|^\./||' | sort \
    | xargs sha256sum \
    | awk '{ p=$2; sub(/^\*/,"",p); h=toupper(substr($1,1,16)); printf "%s  %s\n", h, p }'
} > "$OUT"

echo "已生成 $OUT（$TOTAL 个 .py）"
echo "--- 与 T017.5 改动前基准的差异清单（本次工作真实改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T017.5_改动前_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -c "secrets_local.py$" "$OUT" || true
