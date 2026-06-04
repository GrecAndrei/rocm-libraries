// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for autotune config file persistence round-trip.
// Autotunes to a JSON config file, then verifies the config can be loaded
// via HIPDNN_ENGINE_OVERRIDE_FILE to rebuild and execute the graph.

#include <filesystem>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
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

class IntegrationAutotunePersistence : public hipdnn_tests::IntegrationTestFixture
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
        graph->set_name("autotune_persistence_test_conv")
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

    void TearDown() override
    {
        // Clean up temp config file if it exists
        std::error_code ec;
        std::filesystem::remove(_configFile, ec);

        // Ensure the env var is always cleared
        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_ENGINE_OVERRIDE_FILE");

        IntegrationTestFixture::TearDown();
    }

    std::filesystem::path _configFile = "test_autotune_persistence_config.json";
};

// Test: autotune -> save config -> set override env -> build() -> execute() -> verify success
TEST_F(IntegrationAutotunePersistence, ConfigFileRoundTrip)
{
    // Phase 1: Autotune and write config file
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
        result = bundle.graph->get_max_workspace_size(maxWs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        const Workspace workspace(static_cast<size_t>(maxWs));

        AutotuneConfig config;
        config.mode = TuneMode::AUTO;
        config.strategy = AutotuneStrategy::SINGLE_SHOT;
        config.warmupIterations = 1;

        const AutotuneStorageConfig storageConfig{_configFile, false};

        std::vector<AutotuneResult> results;
        result = bundle.graph->autotune(
            _handle, bundle.variantPack, workspace.get(), config, storageConfig, &results);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        ASSERT_TRUE(std::filesystem::exists(_configFile))
            << "Config file was not created at '" << _configFile << "'";
        ASSERT_FALSE(results.empty());
        ASSERT_TRUE(results[0].succeeded) << "Winning engine did not succeed";
    }

    // Phase 2: Build a new graph using the override config file
    {
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_ENGINE_OVERRIDE_FILE",
                                           _configFile.string().c_str());

        auto bundle = createConvGraph();

        auto result = bundle.graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = bundle.graph->build(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        bundle.buildVariantPack();

        int64_t ws = 0;
        result = bundle.graph->get_workspace_size(ws);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        const Workspace execWorkspace(static_cast<size_t>(ws));

        result = bundle.graph->execute(_handle, bundle.variantPack, execWorkspace.get());
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_ENGINE_OVERRIDE_FILE");
    }
}

} // namespace
