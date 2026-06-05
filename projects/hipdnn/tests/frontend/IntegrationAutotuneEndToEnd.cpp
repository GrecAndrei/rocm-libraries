// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for end-to-end autotune -> execute workflow.
// Uses the test_autotune_plugin which supports the autotune knob workflow.

#include <algorithm>
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

class IntegrationAutotuneEndToEnd : public hipdnn_tests::IntegrationTestFixture
{
protected:
    std::vector<std::string> getPluginPaths() const override
    {
        return {hipdnn_tests::plugin_constants::testAutotunePluginPath()};
    }

    // Builds a small conv fprop graph for autotune testing.
    // Returns the graph and populated variant pack with device memory.
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
        graph->set_name("autotune_test_conv")
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

// Test: build graph -> add_all_engines -> autotune (AUTO/SINGLE_SHOT) -> execute -> verify success
TEST_F(IntegrationAutotuneEndToEnd, AutotuneAutoSingleShotThenExecute)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    result = bundle.graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
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

    // Verify at least one engine succeeded
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
    ASSERT_TRUE(anySucceeded) << "No engine succeeded during autotune";

    // Execute with the autotuned plan
    int64_t ws = 0;
    result = bundle.graph->get_workspace_size(ws);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace execWorkspace(static_cast<size_t>(ws));

    result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

// Test: build graph -> get_engine_configs -> filter by workspace -> add_engine_configs
//       -> autotune -> verify workspace constraint -> execute
TEST_F(IntegrationAutotuneEndToEnd, FilteredAutotuneWithWorkspaceConstraint)
{
    auto bundle = createConvGraph();

    auto result = bundle.graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bundle.buildVariantPack();

    // Step 1: Discover available engine configs
    std::vector<EngineConfigInfo> configs;
    result = bundle.graph->get_engine_configs(configs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_FALSE(configs.empty()) << "Expected at least one engine config";

    // Step 2: Filter configs by workspace size (256 MB limit)
    const int64_t workspaceLimit = int64_t{256} * 1024 * 1024;
    std::vector<EngineConfigInfo> filteredConfigs;
    for(const auto& cfg : configs)
    {
        if(cfg.estimatedWorkspaceSize <= workspaceLimit)
        {
            filteredConfigs.push_back(cfg);
        }
    }
    ASSERT_FALSE(filteredConfigs.empty())
        << "No engine configs within workspace limit of " << workspaceLimit << " bytes";

    // Step 3: Add only the filtered engines
    result = bundle.graph->add_engine_configs(filteredConfigs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Step 4: Allocate workspace (capped at the limit)
    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const int64_t allocatedWs = std::min(maxWs, workspaceLimit);
    const Workspace workspace(static_cast<size_t>(allocatedWs));

    // Step 5: Autotune with pre-filtered engines
    AutotuneConfig config;
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Step 6: Verify all results respect the workspace constraint
    for(const auto& r : results)
    {
        EXPECT_LE(r.workspaceSize, workspaceLimit)
            << "Engine " << r.engineName << " (id=" << r.engineId
            << ") reported workspaceSize=" << r.workspaceSize
            << " exceeding workspace limit=" << workspaceLimit;
    }

    // Step 7: Execute with the autotuned plan
    int64_t ws = 0;
    result = bundle.graph->get_workspace_size(ws);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace execWorkspace(static_cast<size_t>(ws));

    result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
}

} // namespace
