# ============================================================
# probe_llm_switch.py · LLM 切换探针
# 验证 get_llm() 工厂在各 backend 下的行为（mock 模式下运行）
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from llm_client import get_llm, MockLLM


def check(label, condition, detail=""):
    if condition:
        print(f"  {label} ✓")
        return True
    print(f"  {label} ✗  {detail}")
    return False


def main():
    all_ok = True
    original_backend = config.LLM_BACKEND

    print("=== 检查 1：默认 mock 后端 ===")
    print(f"  LLM_BACKEND = {config.LLM_BACKEND}")
    all_ok &= check("默认值为 'mock'", config.LLM_BACKEND == "mock",
                    f"实际 = {config.LLM_BACKEND}")

    llm = get_llm()
    all_ok &= check("get_llm() 返回 MockLLM 实例", isinstance(llm, MockLLM),
                    f"实际类型 = {type(llm).__name__}")

    result = llm.generate("测试")
    all_ok &= check("generate 返回字符串", isinstance(result, str),
                    f"实际类型 = {type(result).__name__}")
    all_ok &= check("返回包含 '[MOCK]'", "[MOCK]" in result,
                    f"实际 = {result[:50]}")
    print(f"  generate('测试') = {result}")

    print()
    print("=== 检查 2：切到 deepseek_official（密钥为空，应降级 mock） ===")
    config.LLM_BACKEND = "deepseek_official"
    llm2 = get_llm()
    all_ok &= check("密钥为空时降级为 MockLLM", isinstance(llm2, MockLLM),
                    f"实际类型 = {type(llm2).__name__}")
    print(f"  返回类型 = {type(llm2).__name__}")

    print()
    print("=== 检查 3：切到 deepseek_third（密钥为空，应降级 mock） ===")
    config.LLM_BACKEND = "deepseek_third"
    llm3 = get_llm()
    all_ok &= check("密钥为空时降级为 MockLLM", isinstance(llm3, MockLLM),
                    f"实际类型 = {type(llm3).__name__}")
    print(f"  返回类型 = {type(llm3).__name__}")

    print()
    print("=== 检查 4：切到未知值，应抛 ValueError ===")
    config.LLM_BACKEND = "xxx_unknown"
    try:
        get_llm()
        all_ok &= check("未知 backend 抛 ValueError", False, "未抛异常")
    except ValueError as e:
        all_ok &= check("未知 backend 抛 ValueError", True)
        print(f"  异常信息: {e}")

    # 恢复
    config.LLM_BACKEND = original_backend
    print()
    all_ok &= check("LLM_BACKEND 已恢复原值",
                    config.LLM_BACKEND == original_backend,
                    f"当前 = {config.LLM_BACKEND}")

    print()
    if all_ok:
        print("探针完成，全部断言通过")
        return 0
    else:
        print("TODO_REVIEW: 断言失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
