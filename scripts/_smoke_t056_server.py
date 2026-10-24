# -*- coding: utf-8 -*-
"""056 server_api 载荷冒烟：/state 带 relationships 字段。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import server_api
from state import reset_state
import relationship

st = reset_state()
p = server_api._state_payload(st)
print("初始 payload relationships:", p["relationships"])
relationship.update_relationship(st, ["user_001"], 0.9, -0.9)
p2 = server_api._state_payload(st)
print("骂一轮后 user_001:", p2["relationships"]["user_001"])
assert p2["relationships"]["user_001"]["trust"] < 0.5
assert p["relationships"]["user_001"]["trust"] == 0.5
print("SERVER_API SMOKE OK")
