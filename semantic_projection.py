# ============================================================
# semantic_projection.py · 054 号工作 · 任务 2
#
# 职责：把海马体输出的 M（1024 维语义均值）从"原始嵌入空间"搬到
#   "去掉共模方向后的残差空间"，供慢通路做语义读出。
#
# 为什么需要这一步（实测依据见 scripts/probe_t54_semantic_res.py）：
#   BGE 向量高度各向异性——任意一句中文的嵌入，其 Σv 都稳定在 -0.5~-0.9
#   这个区间（本机实测 30 条语料 mean -0.684、18 条真实事件 mean -0.745）。
#   均分权重 u_s = 0.1·ones(1024) 恰好只投影到"全 1 方向"，
#   于是 u_s@M ≈ 0.1·ΣM ≈ -0.075，对任何话题都几乎一样（std 0.006），
#   1024 维语义进慢通路只剩一个偏置。
#   减去会话均值基后，共模方向被扣掉，残差向量才携带话题差异。
#
# 只减均值基还不够：均分权重仍然只朝全 1 方向看，实测 std 仅 ×1.5、
#   信噪比反而从 0.95 掉到 0.89。必须同时把权重换成零均值方向
#   （见 config.u_s / U_v 的 [054] 定义），两者配套才有效。
#   配套后实测 std ×5.6、信噪比 0.95 -> 3.81。
# ============================================================

import numpy as np

import config


def mean_basis(memory_graph):
    """会话内全部事件嵌入的均值基；取不到时返回 None（调用方不惩罚）。"""
    if memory_graph is None:
        return None
    getter = getattr(memory_graph, "mean_vector", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


def center(vec, basis):
    """M_centered = normalize(M - basis)。

    降级（不崩、不静默降语义）：vec 或 basis 为空/维度不符/残差过小
    （M 与均值基几乎重合，说明本轮的召回就是"最典型的那一簇"），
    一律原样返回 vec —— 宁可退回旧口径，也不放大噪声。
    """
    if vec is None:
        return vec
    v = np.asarray(vec, dtype=float)
    if v.size == 0:
        return vec
    if not getattr(config, "SEMANTIC_CENTERING", False):
        return vec
    if basis is None:
        return vec
    b = np.asarray(basis, dtype=float)
    if b.shape != v.shape:
        return vec
    r = v - b
    n = float(np.linalg.norm(r))
    # 残差过小（<2% 原向量长度）：说明没有可用的方向信息，不放大噪声
    if n < 1e-8 or n < 0.02 * max(float(np.linalg.norm(v)), 1e-8):
        return vec
    return r / n


def project_M(M, memory_graph):
    """pipeline 用的便捷入口：取均值基 + 居中 + 归一化。"""
    return center(M, mean_basis(memory_graph))
