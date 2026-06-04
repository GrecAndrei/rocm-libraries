// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration tests for the compiled-plan autotune path:
// create_execution_plans() -> build_plans(ALL) -> autotune() -> execute().
// Also tests the manual benchmark loop using plan-indexed access APIs.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

class IntegrationAutotuneCompiledPlanPath : public hipdnn_tests::IntegrationTestFixture
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testAutotunePluginPath()};
    }

    struct ConvGraphBundle
    {
        ConvGraphBundle(const std::vector<int64_t>& xDims,
                        const std::vector<int64_t>& wDims,
                        const std::vector<int64_t>& yDims)
            : xTensor(Tensor<float>(xDims))
            , wTensor(Tensor<float>(wDims))
            , yTensor(Tensor<float>(yDims))
        {
        }

        std::shared_ptr<Graph> graph;
        std::shared_ptr<TensorAttributes> xAttr;
        std::shared_ptr<TensorAttributes> wAttr;
        std::shared_ptr<TensorAttributes> yAttr;
        Tensor<float> xTensor;
        Tensor<float> wTensor;
        Tensor<float> yTensor;
        std::unordered_map<int64_t, void*> variantPack;

        // Populate the variant pack using the current tensor UIDs.
        // Must be called after build_operation_graph() assigns UIDs.
        void buildVariantPack()
        {
            variantPack.clear();
            variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
            variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
            variantPack[yAttr->get_uid()] = yTensor.memory().deviceData();
        }
    };

    static ConvGraphBundle createConvGraph()
    {
        const std::vector<int64_t> xDims = {1, 4, 4, 4};
        const std::vector<int64_t> wDims = {4, 4, 3, 3};
        const std::vector<int64_t> yDims = {1, 4, 4, 4};

        ConvGraphBundle bundle(xDims, wDims, yDims);

        bundle.xTensor.fillWithValue(1.0f);
        bundle.wTensor.fillWithValue(1.0f);
        bundle.yTensor.fillWithValue(0.0f);

        auto graph = std::make_shared<Graph>();
        graph->set_name("compiled_plan_path_test_conv")
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        auto xAttr = std::make_shared<TensorAttributes>();
        xAttr->set_name("X")
            .set_dim(xDims)
            .set_stride(generateStrides(xDims, TensorLayout::NCHW.strideOrder))
            .set_data_type(DataType::FLOAT);

        auto wAttr = std::make_shared<TensorAttributes>();
        wAttr->set_name("W")
            .set_dim(wDims)
            .set_stride(generateStrides(wDims, TensorLayout::NCHW.strideOrder))
            .set_data_type(DataType::FLOAT);

        ConvFpropAttributes convAttrs;
        convAttrs.set_name("test_conv_fprop");
        convAttrs.set_padding({1, 1});
        convAttrs.set_stride({1, 1});
        convAttrs.set_dilation({1, 1});

        auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttrs);
        yAttr->set_output(true);

        bundle.graph = std::move(graph);
        bundle.xAttr = xAttr;
        bundle.wAttr = wAttr;
        bundle.yAttr = yAttr;

        return bundle;
    }
};

// Test: compiled-plan path autotune end-to-end workflow
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneEndToEnd)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    ASSERT_GE(maxWs, 0);

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    ASSERT_FALSE(results.empty());
    bool anySucceeded = false;
    for(const auto& r : results)
    {
        if(r.succeeded)
        {
            anySucceeded = true;
            break;
        }
    }
    ASSERT_TRUE(anySucceeded) << "No engine succeeded during compiled-plan autotune";

    int64_t ws = 0;
    result = bundle.graph->get_workspace_size(ws);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace execWorkspace(static_cast<size_t>(ws));

    result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// Test: EXHAUSTIVE mode is rejected on the compiled-plan path
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneExhaustiveBlocked)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    EXPECT_NE(result.code, ErrorCode::OK);
}

// Test: manual benchmark loop using plan-indexed access APIs
TEST_F(IntegrationAutotuneCompiledPlanPath, ManualLoopEndToEnd)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GT(planCount, 0);

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    int64_t bestIndex = -1;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            bestIndex = i;
            break;
        }
    }

    ASSERT_GE(bestIndex, 0) << "No plan succeeded during manual loop";

    result = bundle.graph->build_plan_at_index(bestIndex);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t ws = 0;
    result = bundle.graph->get_workspace_size(ws);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace execWorkspace(static_cast<size_t>(ws));

    result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// Test: multi-engine plugin provides multiple plans and autotune benchmarks them
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneMultipleEngines)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // With multi-engine plugin, should have multiple plans
    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2) << "Multi-engine plugin should provide >= 2 plans";

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Multiple engines should produce multiple results
    ASSERT_GE(results.size(), 2u) << "Autotune should benchmark >= 2 engines";

    bool anySucceeded = false;
    for(const auto& r : results)
    {
        if(r.succeeded)
        {
            anySucceeded = true;
            break;
        }
    }
    ASSERT_TRUE(anySucceeded) << "At least one engine should succeed";

    int64_t ws = 0;
    result = bundle.graph->get_workspace_size(ws);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace execWorkspace(static_cast<size_t>(ws));

    result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// Test: build_plans(ALL) compiles multiple engines and they are executable
TEST_F(IntegrationAutotuneCompiledPlanPath, BuildPlansAllCompilesMultipleEngines)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2);

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Verify multiple plans are actually executable
    int successCount = 0;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            ++successCount;
        }
    }
    ASSERT_GE(successCount, 2) << "build_plans(ALL) should compile multiple executable plans";
}

// Test: build_plans(ALL) compiles more plans than HEURISTICS_CHOICE
TEST_F(IntegrationAutotuneCompiledPlanPath, BuildPlansAllVsHeuristicsChoice)
{
    // Graph A: build_plans(HEURISTICS_CHOICE) — only compiles the active plan
    auto bundleA = createConvGraph();
    auto result = bundleA.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = bundleA.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    bundleA.buildVariantPack();
    result = bundleA.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = bundleA.graph->build_plans(BuildPlanPolicy::HEURISTICS_CHOICE);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCountA = bundleA.graph->get_execution_plan_count();
    ASSERT_GE(planCountA, 2) << "Multi-engine plugin should provide >= 2 plans";

    const int64_t maxWsA = bundleA.graph->get_autotune_workspace_size();
    const Workspace workspaceA(static_cast<size_t>(maxWsA));

    int successCountA = 0;
    for(int64_t i = 0; i < planCountA; ++i)
    {
        auto execResult = bundleA.graph->execute_plan_at_index(
            _handle, bundleA.variantPack, workspaceA.get(), i);
        if(execResult.is_good())
        {
            ++successCountA;
        }
    }

    // Graph B: build_plans(ALL) — compiles all plans
    auto bundleB = createConvGraph();
    result = bundleB.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = bundleB.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    bundleB.buildVariantPack();
    result = bundleB.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = bundleB.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWsB = bundleB.graph->get_autotune_workspace_size();
    const Workspace workspaceB(static_cast<size_t>(maxWsB));

    int successCountB = 0;
    for(int64_t i = 0; i < planCountA; ++i)
    {
        auto execResult = bundleB.graph->execute_plan_at_index(
            _handle, bundleB.variantPack, workspaceB.get(), i);
        if(execResult.is_good())
        {
            ++successCountB;
        }
    }

    // ALL should compile more plans than HEURISTICS_CHOICE
    ASSERT_GT(successCountB, successCountA)
        << "build_plans(ALL) should compile more plans than HEURISTICS_CHOICE";
}

// Test: manual benchmark loop iterates over multiple engines
TEST_F(IntegrationAutotuneCompiledPlanPath, ManualLoopMultipleEngines)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2) << "Multi-engine plugin should provide >= 2 plans";

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Iterate all plans, track all successes
    std::vector<int64_t> successIndices;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            successIndices.push_back(i);
        }
    }

    ASSERT_GE(successIndices.size(), 2u) << "At least 2 plans should succeed during manual loop";

    // Select a non-zero index if possible
    const int64_t selectedIndex = successIndices.back();

    result = bundle.graph->build_plan_at_index(selectedIndex);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t ws = 0;
    result = bundle.graph->get_workspace_size(ws);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace execWorkspace(static_cast<size_t>(ws));

    result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// Test: compiled-plan path autotune with engine ID filter selects specific engine
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneWithEngineIdFilter)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify we have multiple plans before filtering
    ASSERT_GE(bundle.graph->get_execution_plan_count(), 2);

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Filter to only engine B
    const int64_t engineBId = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();

    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;
    config.engineIdFilter = {engineBId};

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Only the filtered engine should appear in results
    ASSERT_EQ(results.size(), 1u) << "Filter should select exactly 1 engine";
    ASSERT_EQ(results[0].engineId, engineBId) << "Filtered result should match engine B";

    int64_t ws = 0;
    result = bundle.graph->get_workspace_size(ws);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace execWorkspace(static_cast<size_t>(ws));

    result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// Test: plan-spec path autotune with engine ID filter selects specific engine
TEST_F(IntegrationAutotuneCompiledPlanPath, PlanSpecAutotuneWithEngineIdFilter)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    // Use plan-spec path (add_all_engines) instead of compiled-plan path
    result = bundle.graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Filter to only engine A (the original autotune engine)
    const int64_t engineAId = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();

    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;
    config.engineIdFilter = {engineAId};

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Only engine A's plan specs should be benchmarked
    for(const auto& r : results)
    {
        ASSERT_EQ(r.engineId, engineAId) << "All results should be from engine A";
    }

    int64_t ws = 0;
    result = bundle.graph->get_workspace_size(ws);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace execWorkspace(static_cast<size_t>(ws));

    result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

} // namespace
