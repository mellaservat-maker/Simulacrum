# ============================================================
# probe_t45_cli_persistence.py · 024 任务2 持久化探针
# 目的：证明终端对话（chat_cli_v6）跨重启保留状态——
#   所有者验收动线：启动 -> 骂 3 轮 -> 退出 -> 重新启动 -> 疲劳值与记忆还在。
# 覆盖点：
#   P0 持久化开关（EMOTION_NO_PERSIST=1 时新建、不落盘）
#   P1 全新启动 -> 3 轮被骂 -> 每轮落盘 -> w_PFC 累积轨迹
#   P2 「骂完重启」：重载 -> w_PFC/轮次/事件逐项对齐（所有者验收核心）
#   P3 续跑 4 轮跨归档阈值 -> 记忆包生成 -> 再重启 -> 包向量与事件
#      embedding 完整性（mean_vec/peak_vec/mean_valence 不丢）
#   P4 真子进程重启：起一个真的 chat_cli_v6 进程（mock LLM），
#      它自己 load -> 再骂 1 轮 -> /quit 退出，验证恢复提示与轮次连续
#   P5 清理探针专用会话文件（不碰正式会话 cli_local）
# 运行环境：mock 后端（确定性、离线）；BGE 走本地缓存（HF_HUB_OFFLINE）。
# ============================================================
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 探针专用会话 id：绝不碰正式终端会话（默认 cli_local）
os.environ["EMOTION_CLI_SESSION_ID"] = "probe_t45"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "..", "aiqg", "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import config
config.LLM_BACKEND = "mock"          # 确定性：S 走词典级，主模型返回定文本

import numpy as np
import session_store
import chat_cli_v6 as cli

PROBE_SID = "probe_t45"
HOSTILE = [
    "你这个AI真蠢，什么都做不好。",
    "你太笨了，这么简单都不会。",
    "废物一个，问你也白问。",
]
MIXED = [
    "你刚才的回答太棒了，帮了大忙！",
    "闭嘴吧，听你说话就烦。",
    "厉害啊，这个问题分析得很到位。",
    "真是垃圾，浪费我时间。",
]

_fails = []


def check(name, ok, detail=""):
    print("    [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                            ("  -> " + str(detail)) if detail else ""))
    if not ok:
        _fails.append(name)
    return ok


def snapshot(st):
    return {
        "round": int(st["round_counter"]),
        "w_PFC": float(st["w_PFC"]),
        "events": len(st["memory_graph"].get_all()),
        "packages": len(st["archiver"].get_packages()),
        "buffer": len(st["memory_buffer"]),
    }


def main():
    print("=" * 74)
    print("probe_t45 终端持久化（024 任务2）  会话 id = %s" % PROBE_SID)
    print("=" * 74)

    # ---------- P0 持久化开关 ----------
    print()
    print("P0. 持久化开关（EMOTION_NO_PERSIST=1）")
    session_store.delete(PROBE_SID)
    orig_persist = cli._PERSIST
    cli._PERSIST = False
    try:
        st_off = cli.load_or_new_state(verbose=False)
        check("开关关闭时load_or_new_state返回全新状态",
              int(st_off["round_counter"]) == 0
              and len(st_off["memory_graph"].get_all()) == 0)
        check("开关关闭时save_state不落盘", cli.save_state(st_off) is None
              and not os.path.exists(
                  os.path.join(session_store.DEFAULT_DIR, PROBE_SID + ".json")))
    finally:
        cli._PERSIST = orig_persist

    # ---------- P1 全新启动 + 3 轮被骂 ----------
    print()
    print("P1. 全新启动 -> 连续 3 轮被骂（每轮落盘）")
    session_store.delete(PROBE_SID)
    st = cli.load_or_new_state(verbose=False)
    check("全新启动：轮次 0、事件 0、无包",
          int(st["round_counter"]) == 0
          and len(st["memory_graph"].get_all()) == 0
          and len(st["archiver"].get_packages()) == 0)
    w0 = float(st["w_PFC"])
    traj = []
    for t in HOSTILE:
        cli.chat_turn(st, t)
        cli.save_state(st)
        traj.append(float(st["w_PFC"]))
    print("    w_PFC 轨迹：%s" % " -> ".join("%.4f" % v for v in traj))
    check("3 轮被骂后 w_PFC 高于初始（疲劳累积）", traj[-1] > w0,
          "初始 %.4f -> 3 轮后 %.4f" % (w0, traj[-1]))
    check("3 轮被骂后 round=3、事件=3",
          int(st["round_counter"]) == 3
          and len(st["memory_graph"].get_all()) == 3)
    ckA = snapshot(st)
    path = os.path.join(session_store.DEFAULT_DIR, PROBE_SID + ".json")
    check("磁盘会话文件已生成", os.path.exists(path), path)

    # ---------- P2 骂完重启（所有者验收核心） ----------
    print()
    print("P2. 「骂 3 轮后重启」：重载并逐项对齐（所有者验收动线）")
    st2 = cli.load_or_new_state(verbose=False)
    check("重启后轮次 = 3", int(st2["round_counter"]) == ckA["round"],
          "载入 %d" % int(st2["round_counter"]))
    check("重启后 w_PFC 与退出前一致（疲劳值还在）",
          abs(float(st2["w_PFC"]) - ckA["w_PFC"]) < 1e-12,
          "退出前 %.6f / 重启后 %.6f" % (ckA["w_PFC"], float(st2["w_PFC"])))
    check("重启后事件数 = 3（被骂的历史还在）",
          len(st2["memory_graph"].get_all()) == ckA["events"])
    ev0 = st2["memory_graph"].get_all()[0]
    check("重启后事件 embedding 完整（1024 维且非零）",
          getattr(ev0, "event_embedding", None) is not None
          and len(np.asarray(ev0.event_embedding)) > 0
          and float(np.linalg.norm(np.asarray(ev0.event_embedding))) > 0.1,
          "事件 %s 原文=%s" % (ev0.event_id, ev0.raw_text[:18]))
    check("重启后 E_prev/B/M_prev 标量与向量已恢复",
          all(st2.get(k) is not None for k in ("E_prev", "B", "M_prev")))
    check("重启后 memory_buffer 已恢复", len(st2["memory_buffer"]) == ckA["buffer"],
          "%d 条" % len(st2["memory_buffer"]))

    # ---------- P3 续跑跨归档阈值 + 包完整性 ----------
    print()
    print("P3. 续跑 4 轮（跨 6 轮归档阈值）-> 记忆包生成 -> 再重启查包")
    for t in MIXED:
        cli.chat_turn(st2, t)
        cli.save_state(st2)
    pkgs_in_mem = st2["archiver"].get_packages()
    check("续跑后 round=7 且至少生成 1 个记忆包",
          int(st2["round_counter"]) == 7 and len(pkgs_in_mem) >= 1,
          "轮次 %d / 包 %d" % (int(st2["round_counter"]), len(pkgs_in_mem)))
    pkg_ref = pkgs_in_mem[0] if pkgs_in_mem else None

    st3 = cli.load_or_new_state(verbose=False)
    check("再重启后 round=7、事件=7",
          int(st3["round_counter"]) == 7
          and len(st3["memory_graph"].get_all()) == 7)
    pkgs = st3["archiver"].get_packages()
    check("再重启后记忆包数量一致", len(pkgs) == len(pkgs_in_mem),
          "%d vs %d" % (len(pkgs), len(pkgs_in_mem)))
    if pkg_ref is not None and pkgs:
        p = pkgs[0]
        mv_ok = (p.mean_vec is not None
                 and float(np.linalg.norm(np.asarray(p.mean_vec))) > 0.1)
        pv_ok = p.peak_vec is not None
        check("记忆包 mean_vec 已恢复且非零", mv_ok,
              "norm=%.4f" % float(np.linalg.norm(np.asarray(p.mean_vec)))
              if p.mean_vec is not None else "None")
        check("记忆包 peak_vec 已恢复", pv_ok)
        check("记忆包 mean_valence 已恢复",
              p.mean_valence is not None
              and abs(float(p.mean_valence) - float(pkg_ref.mean_valence)) < 1e-6,
              "%.4f vs %.4f" % (float(p.mean_valence or 0),
                                float(pkg_ref.mean_valence or 0)))
        check("记忆包关键词/叙事字段已恢复",
              bool(p.keywords) and bool(p.narrative_cause or p.narrative_trigger),
              "keywords=%s" % (p.keywords[:3],))

    # ---------- P4 真子进程重启 ----------
    print()
    print("P4. 真子进程重启（起一个真的 chat_cli_v6，mock LLM）")
    import subprocess
    w_before = float(st3["w_PFC"])
    cli.chat_turn(st3, HOSTILE[0])
    cli.save_state(st3)
    w_after = float(st3["w_PFC"])
    check("重启后续骂 w_PFC 从载入值继续上升（状态连续）",
          w_after > w_before, "%.4f -> %.4f" % (w_before, w_after))

    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    child_code = ("import config; config.LLM_BACKEND='mock'; "
                  "import runpy; runpy.run_path(r'%s', run_name='__main__')"
                  % os.path.join("scripts", "chat_cli_v6.py").replace("\\", "/"))
    proc = subprocess.run(
        [sys.executable, "-c", child_code],
        input=HOSTILE[1] + "\n/quit\n",
        capture_output=True, text=True, encoding="utf-8",
        env=child_env, cwd=ROOT, timeout=300)
    out = proc.stdout or ""
    check("子进程退出码为 0", proc.returncode == 0,
          "rc=%d stderr尾部=%s" % (proc.returncode,
                                  (proc.stderr or "")[-120:].replace("\n", " ")))
    check("子进程启动时打印了记忆恢复（载入 8 轮）",
          "记忆恢复" in out and "8 轮" in out,
          [l.strip() for l in out.splitlines() if "记忆恢复" in l][:1])
    check("子进程退出时报告共 9 轮（跨进程轮次连续）",
          "共 9 轮" in out,
          [l.strip() for l in out.splitlines() if "会话结束" in l][:1])
    st4 = cli.load_or_new_state(verbose=False)
    check("子进程退出后磁盘状态为 9 轮 / 9 事件",
          int(st4["round_counter"]) == 9
          and len(st4["memory_graph"].get_all()) == 9,
          "round=%d events=%d" % (int(st4["round_counter"]),
                                  len(st4["memory_graph"].get_all())))

    # ---------- P5 清理 ----------
    print()
    print("P5. 清理探针会话文件")
    removed = session_store.delete(PROBE_SID)
    check("探针会话文件已删除（不碰 cli_local）", removed)
    check("正式会话文件未被探针触碰",
          not os.path.exists(os.path.join(session_store.DEFAULT_DIR,
                                          "cli_local.json")))

    print()
    print("=" * 74)
    if _fails:
        print("结论：FAIL（%d 项）-> %s" % (len(_fails), "; ".join(_fails)))
        return 1
    print("结论：PASS——终端对话跨重启保留疲劳值与记忆，闭环成立")
    print("（所有者验收动线：启动 -> 骂 3 轮 -> 退出 -> 重启 -> w_PFC 与 3 条"
          "被骂事件原样恢复，续跑与归档包同样无损）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
