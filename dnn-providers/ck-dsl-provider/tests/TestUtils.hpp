// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <string>

// Two preconditions the GPU-gated unit suite shares with the
// integration test: a HIP-visible device must exist, and that
// device's gcnArchName must contain "gfx950" (the DSL emits
// gfx950-only HSACO today; any other arch fails hipModuleLoadData
// with hipErrorNoBinaryForGpu, which is otherwise confusing).
//
// The helpers are macros rather than free functions so the embedded
// GTEST_SKIP() / ASSERT_EQ short-circuit out of the calling test's
// body -- a free function's return would only exit the helper itself.

#define CK_DSL_PROVIDER_SKIP_IF_NO_GPU(testName)                                           \
    do {                                                                                   \
        int _ckdsl_device_count = 0;                                                       \
        hipError_t _ckdsl_hip_err = hipGetDeviceCount(&_ckdsl_device_count);               \
        if (_ckdsl_hip_err != hipSuccess || _ckdsl_device_count == 0) {                    \
            GTEST_SKIP() << (testName)                                                     \
                         << ": no HIP-visible device (deviceCount=" << _ckdsl_device_count \
                         << ", hipError=" << static_cast<int>(_ckdsl_hip_err) << ")";      \
        }                                                                                  \
        ASSERT_EQ(hipSetDevice(0), hipSuccess);                                            \
    } while (0)

// Sibling of CK_DSL_PROVIDER_SKIP_IF_NO_GPU that also skips when the
// present device's gcnArchName does not contain the requested bare gfx
// token (e.g. "gfx950" or "gfx942"). Use in tests that load a
// DSL-produced HSACO via hipModuleLoadData: the DSL emits arch-specific
// ISA, so a mismatched device fails hipModuleLoadData with
// hipErrorNoBinaryForGpu, which is otherwise confusing.
#define CK_DSL_PROVIDER_SKIP_IF_NOT_ARCH(arch, testName)                          \
    do {                                                                          \
        CK_DSL_PROVIDER_SKIP_IF_NO_GPU(testName);                                 \
        hipDeviceProp_t _ckdsl_props{};                                           \
        ASSERT_EQ(hipGetDeviceProperties(&_ckdsl_props, 0), hipSuccess);          \
        std::string _ckdsl_arch_name = _ckdsl_props.gcnArchName;                  \
        if (_ckdsl_arch_name.find((arch)) == std::string::npos) {                 \
            GTEST_SKIP() << (testName) << ": requires " << (arch)                 \
                         << " (DSL emits arch-specific HSACO); device 0 reports " \
                         << "gcnArchName='" << _ckdsl_arch_name << "'";           \
        }                                                                         \
    } while (0)

// Thin wrapper preserving the original gfx950-specific call sites
// (Oracle / BuildTime fixtures) -- delegates to the parameterized
// macro above so there is a single arch-check implementation.
#define CK_DSL_PROVIDER_SKIP_IF_NOT_GFX950(testName) \
    CK_DSL_PROVIDER_SKIP_IF_NOT_ARCH("gfx950", testName)

// Skip unless the device is one of the arches the DSL SDPA-fwd path
// emits HSACO for (gfx950 or gfx942). The provider routes per detected
// arch, so the fwd correctness/perf fixtures run on either; only a
// device on neither arch is skipped.
#define CK_DSL_PROVIDER_SKIP_IF_SDPA_FWD_ARCH_UNSUPPORTED(testName)                               \
    do {                                                                                          \
        CK_DSL_PROVIDER_SKIP_IF_NO_GPU(testName);                                                 \
        hipDeviceProp_t _ckdsl_props{};                                                           \
        ASSERT_EQ(hipGetDeviceProperties(&_ckdsl_props, 0), hipSuccess);                          \
        std::string _ckdsl_arch_name = _ckdsl_props.gcnArchName;                                  \
        if (_ckdsl_arch_name.find("gfx950") == std::string::npos &&                               \
            _ckdsl_arch_name.find("gfx942") == std::string::npos) {                               \
            GTEST_SKIP() << (testName) << ": requires gfx950 or gfx942 (DSL emits arch-specific " \
                         << "HSACO); device 0 reports gcnArchName='" << _ckdsl_arch_name << "'";  \
        }                                                                                         \
    } while (0)
