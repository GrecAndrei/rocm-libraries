# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(stinkytofu PRIVATE
    CallGraph.cpp
)

add_subdirectory(asm)
add_subdirectory(controlflow)
add_subdirectory(logical)
