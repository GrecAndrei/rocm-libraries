// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for EXHAUSTIVE autotune mode.
// Verifies that EXHAUSTIVE mode primes engine caches via the global.benchmarking
// knob and that AUTO mode does not.

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

class IntegrationAutotuneExhaustive : public hipdnn_tests::IntegrationTestFixture
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
        graph->set_name("autotune_exhaustive_test_conv")
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

// Test: EXHAUSTIVE autotune sets ranExhaustive = true for engines with benchmarking knob
TEST_F(IntegrationAutotuneExhaustive, ExhaustiveModeRunsCachePriming)
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

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;
    config.continueOnPrimingFailure = true;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify at least one engine succeeded
    ASSERT_FALSE(results.empty());
    bool anySucceeded = false;
    bool anyRanExhaustive = false;
    for(const auto& r : results)
    {
        if(r.succeeded)
        {
            anySucceeded = true;
        }
        if(r.ranExhaustive)
        {
            anyRanExhaustive = true;
        }
    }
    ASSERT_TRUE(anySucceeded) << "No engine succeeded during EXHAUSTIVE autotune";
    EXPECT_TRUE(anyRanExhaustive) << "No engine ran exhaustive priming (expected at least one "
                                     "since the test plugin has global.benchmarking knob)";
}

// Test: AUTO mode does not set ranExhaustive on any engine
TEST_F(IntegrationAutotuneExhaustive, AutoModeDoesNotRunCachePriming)
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
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.ranExhaustive)
            << "Engine " << r.engineId << " should not have ran exhaustive in AUTO mode";
    }
}

} // namespace
