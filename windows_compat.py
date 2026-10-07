"""
Make mlx-lm importable on Windows.

mlx-lm imports the Unix-only `resource` module just to raise the open-file
limit (mlx_lm/utils.py). Windows has no such limit to raise, so a stand-in
with a no-op setrlimit is enough. Import this module before mlx_lm.
"""

import sys
import types

if sys.platform == "win32" and "resource" not in sys.modules:
    resource = types.ModuleType("resource")
    resource.RLIMIT_NOFILE = 7
    resource.getrlimit = lambda limit: (2048, 4096)
    resource.setrlimit = lambda limit, values: None
    sys.modules["resource"] = resource
