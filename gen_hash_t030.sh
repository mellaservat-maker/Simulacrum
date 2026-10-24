#!/usr/bin/env bash
# gen_hash_t030.sh · 生成 T030 封版 hash 基准（v6.13）
# 本次施工内容（030 号工作）：028 草案任务 2 第一批（四维情绪有值化）
#   1. config.py：A_INIT 后三行去共线化；新增 G_S 语义冲击矩阵与 K_ANX；
#      b_E 改为 [-1.0, 0.25, -0.10, 0.10]；w_e 后三维临时置 0（只让 E0 进 S_E）。
#   2. emotion_releaser.py：接收完整 S，G_S@S 与焦虑非线性合取并入
#      released_impact，共享释放率；S=None 时保持旧行为。
#   3. intercept.py：本批无代码改动（w_e 值变化经 config 运行时读取自动生效）。
#   4. emotion.py：本批无代码改动（G_S 经 released_impact 进入，LAMBDA 不动）。
#   5. pipeline.py：两处 run_releaser 调用传入完整 S。
#   6. 新增 scripts/probe_t47_four_dim.py（三场景四维符号验收）。
# 口径与 T029 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T029_封版_hash基准.txt（v6.12）。
set -e
cd /d/aiqg/emotion_agent

OUT="T030_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T030 封版 hash 基准（v6.13，四维情绪有值化第一批）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T029 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T029 封版基准的差异清单（v6.12 至 v6.13 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T029_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
