# ============================================================
# run_real_llm_check.py · 手动验证真 API
# 供所有者手动运行，不参与自动测试，不进 pytest
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("步骤1：尝试导入 secrets_local")
import secrets_local
print("导入成功")

print(f"官方 key 长度 = {len(secrets_local.DEEPSEEK_OFFICIAL_API_KEY)}")
print(f"base_url = {secrets_local.DEEPSEEK_OFFICIAL_BASE_URL}")

print("步骤2：构造 DeepSeekLLM")
from llm_client import DeepSeekLLM

llm = DeepSeekLLM(
    api_key=secrets_local.DEEPSEEK_OFFICIAL_API_KEY,
    base_url=secrets_local.DEEPSEEK_OFFICIAL_BASE_URL,
    model=secrets_local.DEEPSEEK_MODEL,
)

print("步骤3：发送测试 prompt")
resp = llm.generate("你好，请用一句话介绍你自己。")

print(f"步骤4：返回结果如下：{resp[:100]}")
