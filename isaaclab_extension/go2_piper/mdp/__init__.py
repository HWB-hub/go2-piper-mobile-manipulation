# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""本任务自己的 MDP 函数子模块（奖励、终止等）。

跟官方行为保持一致：用 lazy_export() 惰性导出，具体暴露哪些名字由 mdp/__init__.pyi 声明。
"""

from isaaclab.utils.module import lazy_export

lazy_export()
