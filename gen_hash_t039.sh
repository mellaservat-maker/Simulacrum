#!/usr/bin/env bash
# gen_hash_t039.sh · 生成 T039 封版 hash 基准（v6.19）
# 本次施工内容（039 号工作）：DELTA 0.25 + 破防冷却机制
#   1. config.py：DELTA 0.10→0.25（闭式稳态 I*=1.2D/4=0.3D，使 express 可达）。
#   2. state.py：新增跨轮键 express_cooldown=0，并加入缺键断言。
#   3. session_store.py：express_cooldown 随会话持久化，旧会话回落 0。
#   4. policy.py：policy_minimal 新增 express_cooldown；
#      冷却期内 express 降级为 suppress（高严重度）或 task_first（低严重度）。
#   5. pipeline.py：两条路径轮初读取并递减冷却；express 触发后置 10。
#   6. scripts/probe_t49_axis_compare.py：S0 临时还原 DELTA=0.10；
#      S1 断言改为“曲线不完全相同”，MID 选择只认三场景验收。
#   7. scripts/probe_t51_breakdown_cycle.py（新增）：100 轮破防周期验证。
#   MID 保持 0.20：被质疑单轮 I=0.2687，与 DELTA 无关；0.20 同时保住 encoder。
# 口径与 T037.1 封版基准一致：全部 .py；排除 secrets_local.py /
#   secrets_override.py / __pycache__ / .pytest_cache / logs / _snapshot_*。
# 交叉 diff 对象：T037.1_封版_hash基准.txt（v6.18）。
set -e
cd /d/aiqg/emotion_agent

OUT="T039_封版_hash基准.txt"
TOTAL=$(find . -name "*.py" -not -path "*/__pycache__/*" -not -path "./.pytest_cache/*" -not -path "*/logs/*" -not -path "./_snapshot_*" -not -name "secrets_local.py" -not -name "secrets_override.py" | wc -l)

{
  echo "T039 封版 hash 基准（v6.19，DELTA 0.25 + express 冷却）"
  echo
  echo "生成日期：$(date +%Y-%m-%d)"
  echo "工作目录：d://aiqg//emotion_agent"
  echo "收录范围：全部 .py 源文件（与 T037.1 封版基准同口径，排除 _snapshot_* 备份目录）"
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
echo "--- 与 T037.1 封版基准的差异清单（v6.18 至 v6.19 的改动面）---"
diff <(grep -E "^[0-9A-F]{16}  " T037.1_封版_hash基准.txt) <(grep -E "^[0-9A-F]{16}  " "$OUT") || true
echo "--- secrets 条目行精确命中数（须为 0）---"
grep -cE "secrets_local\.py$|secrets_override\.py$" "$OUT" || true
