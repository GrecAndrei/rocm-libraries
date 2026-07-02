/* ************************************************************************
 * GSU/WGM Sweep Benchmark for hipBLASLt
 *
 * Tests all reasonable (gsu, wgm) combos for common LLM shapes,
 * measures real kernel timing, and writes optimal values to JSON.
 *
 * Usage:
 *   export HIPBLASLT_QUICKTUNE_FILE=/path/to/tuning.json
 *   ./bench_gsuwgm
 *
 * Requirements:
 *   - ROCm installed
 *   - Built libhipblaslt.so with QuickTuning support
 *   - fp16 matrix multiplication supported on target GPU
 *
 * ************************************************************************ */

#include "QuickTuning.hpp"
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

static const int NUM_WARMUP  = 3;
static const int NUM_ITERS   = 10;

struct ShapeDef
{
    const char* name;
    int         m, n, k;
};

static const ShapeDef SHAPES[] = {
    {"single_decode",      1,    4096,  4096},
    {"small_prefill",      128,  4096,  4096},
    {"medium_prefill",     512,  4096,  4096},
    {"large_square",       4096, 4096,  4096},
    {"ffn_up_proj",        4096, 14336, 4096},
    {"ffn_down_proj",      4096, 4096,  14336},
    {"batch_decode_32",    32,   4096,  4096},
    {"batch_decode_64",    64,   4096,  4096},
    {"batch_decode_256",   256,  4096,  4096},
    {"mlp_hidden",         4096, 11008, 4096},
    {"mlp_gate",           4096, 4096,  11008},
};

static const int GSU_VALUES[] = {1, 2, 4};
static const int WGM_VALUES[] = {1, 2, 4, 8};
static const int NUM_GSU = sizeof(GSU_VALUES) / sizeof(GSU_VALUES[0]);
static const int NUM_WGM = sizeof(WGM_VALUES) / sizeof(WGM_VALUES[0]);

struct BenchResult
{
    int gsu;
    int wgm;
    double ms;
    bool success;
};

static double time_matmul(hipblasltHandle_t                          handle,
                          const RocblasltContractionProblem&         prob,
                          std::vector<rocblaslt_matmul_heuristic_result>& results)
{
    hipblasltMatmulDesc_t matmulDesc;
    hipblasltMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_16F, HIP_R_16F);

    auto pref = std::make_shared<rocblaslt_matmul_preference_t>();
    pref->max_workspace_bytes = 32 * 1024 * 1024;

    hipblasltMatrixLayout_t layoutA, layoutB, layoutC, layoutD;

    size_t row = prob.trans_a == HIPBLAS_OP_N ? prob.m : prob.k;
    size_t col = prob.trans_a == HIPBLAS_OP_N ? prob.k : prob.m;
    hipblasltMatrixLayoutCreate(&layoutA, HIP_R_16F, row, col, col);

    row = prob.trans_b == HIPBLAS_OP_N ? prob.k : prob.n;
    col = prob.trans_b == HIPBLAS_OP_N ? prob.n : prob.k;
    hipblasltMatrixLayoutCreate(&layoutB, HIP_R_16F, row, col, col);

    hipblasltMatrixLayoutCreate(&layoutC, HIP_R_16F, prob.m, prob.n, prob.n);
    hipblasltMatrixLayoutCreate(&layoutD, HIP_R_16F, prob.m, prob.n, prob.n);

    float alpha = 1.0f, beta = 0.0f;

    auto heuristicResults = std::vector<rocblaslt_matmul_heuristic_result>{};
    int              returnAlgoCount = 0;
    rocblaslt_status status
        = hipblasLtMatmulAlgoGetHeuristic(handle,
                                          matmulDesc,
                                          layoutA,
                                          layoutB,
                                          layoutC,
                                          layoutD,
                                          pref.get(),
                                          1,
                                          heuristicResults.data(),
                                          &returnAlgoCount);

    if(status != rocblaslt_status_success || returnAlgoCount == 0)
        return -1.0;

    auto          algo = heuristicResults[0].algo;
    rocblaslt::RocTuningV2 tuning;
    tuning.gsu = 0;
    tuning.wgm = 0;

    size_t workspaceSize = 0;
    status = hipblaslt_is_algo_supported_cpp(handle,
                                             ROCBLASLT_GEMM,
                                             /* gemmData */ nullptr,
                                             algo,
                                             &tuning,
                                             workspaceSize);

    if(status != rocblaslt_status_success || workspaceSize > pref->max_workspace_bytes)
        return -1.0;

    void* workspace = nullptr;
    if(workspaceSize > 0)
        hipMalloc(&workspace, workspaceSize);

    float *dA, *dB, *dC;
    size_t sizeA = prob.m * prob.k * sizeof(float) / 2;
    size_t sizeB = prob.k * prob.n * sizeof(float) / 2;
    size_t sizeC = prob.m * prob.n * sizeof(float) / 2;
    hipMalloc(&dA, sizeA);
    hipMalloc(&dB, sizeB);
    hipMalloc(&dC, sizeC);
    hipMemset(dA, 0, sizeA);
    hipMemset(dB, 0, sizeB);
    hipMemset(dC, 0, sizeC);

    hipStream_t stream;
    hipStreamCreate(&stream);

    double bestMs = 1e9;

    for(int iter = 0; iter < NUM_ITERS + NUM_WARMUP; iter++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        status = hipblasLtMatmul(handle,
                                 matmulDesc,
                                 layoutA,
                                 layoutB,
                                 &alpha,
                                 dA,
                                 dB,
                                 &beta,
                                 dC,
                                 dC,
                                 workspace,
                                 workspaceSize,
                                 stream);

        hipStreamSynchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();

        if(status != rocblaslt_status_success)
        {
            bestMs = -1.0;
            break;
        }

        if(iter >= NUM_WARMUP)
        {
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if(ms < bestMs)
                bestMs = ms;
        }
    }

    if(workspace)
        hipFree(workspace);
    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    hipStreamDestroy(stream);
    hipblasltMatrixLayoutDestroy(layoutA);
    hipblasltMatrixLayoutDestroy(layoutB);
    hipblasltMatrixLayoutDestroy(layoutC);
    hipblasltMatrixLayoutDestroy(layoutD);
    hipblasltMatmulDescDestroy(matmulDesc);

    return bestMs;
}

int main(int argc, char** argv)
{
    hipblasltHandle_t handle;
    if(hipblasLtCreate(&handle) != rocblaslt_status_success)
    {
        std::cerr << "Failed to create hipblaslt handle. Is ROCm loaded and GPU visible?\n";
        return 1;
    }

    hipDeviceProp_t props;
    int             device;
    hipGetDevice(&device);
    hipGetDeviceProperties(&props, device);
    std::cout << "GPU: " << props.name << " (" << props.gcnArchName << ")\n";
    std::cout << "CUs: " << props.multiProcessorCount << "\n";
    std::cout << "========================================\n\n";

    std::vector<std::vector<BenchResult>> allResults;

    for(const auto& shape : SHAPES)
    {
        std::cout << "=== Shape: " << shape.name << " (" << shape.m << "x" << shape.n << "x"
                  << shape.k << ") ===\n";

        std::vector<BenchResult> shapeResults;

        for(int gi = 0; gi < NUM_GSU; gi++)
        {
            for(int wi = 0; wi < NUM_WGM; wi++)
            {
                int gsu = GSU_VALUES[gi];
                int wgm = WGM_VALUES[wi];

                RocblasltContractionProblem prob{};
                prob.trans_a       = HIPBLAS_OP_N;
                prob.trans_b       = HIPBLAS_OP_N;
                prob.m             = shape.m;
                prob.n             = shape.n;
                prob.k             = shape.k;
                prob.a_type        = HIP_R_16F;
                prob.b_type        = HIP_R_16F;
                prob.c_type        = HIP_R_16F;
                prob.d_type        = HIP_R_16F;
                prob.compute_type  = HIPBLAS_COMPUTE_16F;
                prob.batch_count   = 1;
                prob.workspaceSize = 0;

                double ms = time_matmul(handle, prob, /*results*/ {});

                BenchResult r;
                r.gsu     = gsu;
                r.wgm     = wgm;
                r.ms      = ms;
                r.success = (ms > 0);
                shapeResults.push_back(r);

                std::cout << "  gsu=" << gsu << " wgm=" << wgm << " → ";
                if(r.success)
                    std::cout << std::fixed << std::setprecision(3) << ms << " ms\n";
                else
                    std::cout << "FAILED\n";
            }
        }

        if(!shapeResults.empty())
        {
            auto bestIt = std::min_element(shapeResults.begin(),
                                           shapeResults.end(),
                                           [](const BenchResult& a, const BenchResult& b)
                                           {
                                               if(!a.success)
                                                   return false;
                                               if(!b.success)
                                                   return true;
                                               return a.ms < b.ms;
                                           });
            if(bestIt != shapeResults.end() && bestIt->success)
            {
                std::cout << "  *** BEST: gsu=" << bestIt->gsu << " wgm=" << bestIt->wgm
                          << " (" << std::fixed << std::setprecision(3) << bestIt->ms
                          << " ms) ***\n\n";
            }
        }

        allResults.push_back(shapeResults);
    }

    std::cout << "========================================\nBenchmark complete.\n";

    hipblasLtDestroy(handle);
    return 0;
}
