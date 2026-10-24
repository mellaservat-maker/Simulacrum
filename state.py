# ============================================================
# state.py · V4.2 · M1 回合状态初始化
# ============================================================

import numpy as np
import random
import time

import config
from memory_graph import LayeredMemoryGraph
from short_term_buffer import ShortTermBuffer
from raw_log import RawLog
from archiver import Archiver
from decompressor import Decompressor


def reset_state(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    memory_buffer = ShortTermBuffer(default_keep=2)
    raw_log = RawLog()
    archiver = Archiver(memory_buffer, raw_log, threshold=config.MEMORY_ARCHIVE_THRESHOLD)

    decompressor = Decompressor(raw_log)

    st = {
        "E_prev": np.array(config.E_PREV_INIT),
        "w_PFC": config.W_PFC_INIT,
        "B": np.array(config.B_INIT),
        "M_prev": np.zeros(1024),
        "HabituationSignal_prev": 0.0,
        "memory_graph": LayeredMemoryGraph(),
        "memory_buffer": memory_buffer,
        "raw_log": raw_log,
        "archiver": archiver,
        "decompressor": decompressor,
        "round_counter": 0,
        "recalled_memory": None,
        "round": 0,
        # [017.5 T31] 召回拦截层跨轮状态：
        #   alpha_prev   迟滞模式机状态（PTSD↔脱敏，低通平滑后的值）
        #   D_error_prev 上一轮威胁误差——本轮 D_error 在注入点之后才产生，
        #                threat_now 汇流口取上一轮值（一轮延迟=威胁感惯性）
        "alpha_prev": 0.0,
        "D_error_prev": 0.0,
        # [031] 性格预设（冷切换）：会话创建时固定，运行中不热切换；
        #   旧会话缺此字段时按 standard 读取（session_store.load_state）。
        "profile_id": getattr(config, "ACTIVE_PROFILE_ID", "standard"),
        # [039] 破防冷却：express 触发后置 10，轮初递减；
        #   冷却期内即使 I<0.30 且 E0<-0.50 也不走 express。
        "express_cooldown": 0,
        # [055] 距上次会话保存的真实天数（跨天时间感）。
        #   新建会话为 0；session_store.load 时按 saved_at 重算。
        "session_gap_days": 0.0,
        # [058 任务1] 常驻进程空档基准：上次活动时刻（墙钟，秒）。
        #   pipeline 每轮结束推进；轮初 now - last_active 超过
        #   config.RESIDENT_GAP_THRESHOLD_SEC 时按 load 同款四条通道
        #   结算一次跨天衰减（修 B16：LRU 命中不走 load 的假死）。
        #   不落盘——重启路径由 saved_at 兜底（load 时重置为加载时刻），
        #   落盘会与 saved_at 构成双重结算。
        "last_active": time.time(),
        # [056 任务1] 会话级关系账本：{实体ID: {trust, intimacy, interactions}}。
        #   M1 只内置 user_001（当前用户，动力学主对象）与 ai_self（AI 自己，
        #   只计数）；他人实体（boss_01 等）在被提及时按需创建，只计数。
        #   trust ∈ [0,1]：0.5 起点，<0.2 判定关系破裂；
        #   intimacy ∈ [0,1]：随互动次数单调累积（只升不降，M1 规格）；
        #   interactions：累计互动轮数（含 ai_self 与被提及的他人）。
        #   跨轮累积，随会话落盘（session_store.dump/load）。
        "relationships": {
            "user_001": {"trust": 0.5, "intimacy": 0.5,
                         "interactions": 0, "broken": False},
            "ai_self": {"trust": 1.0, "intimacy": 0.5,
                        "interactions": 0, "broken": False},
        },
    }

    # T33：跨轮回写依赖的键，缺一即报错（防再次断裂）
    # [017.5] 追加 alpha_prev / D_error_prev（召回拦截层迟滞机与威胁汇流口）
    # [039] 追加 express_cooldown（破防冷却计数）
    # [056] 追加 relationships（关系账本：pipeline 每轮就地更新 dict 内容，
    #   引用不换，但键缺失说明 reset 被绕过或字段被误删，同样要炸）
    # [058] 追加 last_active（常驻空档基准，pipeline 每轮推进）
    for _k in ("E_prev", "w_PFC", "B", "M_prev", "HabituationSignal_prev",
               "alpha_prev", "D_error_prev", "express_cooldown",
               "relationships", "last_active"):
        assert _k in st, "reset_state 缺少跨轮状态键: %s" % _k

    return st