import os
import sys

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
import lit.util

config.name = 'MINI_INFER_TEST'
config.suffixes = ['.mlir']
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(('%PATH%', config.environment['PATH']))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])
llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

config.excludes = [
    "CMakeLists.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

tool_dirs = [
    config.mlir_binary_dir,
    config.mlir_tutorial_tool_dir
]
tools = [
    ToolSubst("mini-opt16", config.mlir_tutorial_ns_opt, unresolved="ignore"),
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
