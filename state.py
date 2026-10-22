# ============================================================
# state.py · V4.2 · M1 回合状态初始化
# ============================================================

import numpy as np
import random

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
    }

    # T33：跨轮回写依赖的键，缺一即报错（防再次断裂）
    # [017.5] 追加 alpha_prev / D_error_prev（召回拦截层迟滞机与威胁汇流口）
    # [039] 追加 express_cooldown（破防冷却计数）
    for _k in ("E_prev", "w_PFC", "B", "M_prev", "HabituationSignal_prev",
               "alpha_prev", "D_error_prev", "express_cooldown"):
        assert _k in st, "reset_state 缺少跨轮状态键: %s" % _k

    return st