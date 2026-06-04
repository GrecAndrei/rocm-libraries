// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/autotune/AutotuneFileWriter.hpp>
#include <hipdnn_frontend/detail/EngineOverrideConfig.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#endif

using namespace hipdnn_frontend;

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
using namespace hipdnn_frontend::autotune;
using namespace hipdnn_frontend::engine_override;
using namespace hipdnn_data_sdk::utilities;

// ── Test helpers ────────────────────────────────────────────────────────────

namespace
{

/// Create a temporary file path for testing, cleaned up by destructor.
struct TempFile
{
    std::filesystem::path path;

    TempFile()
        : path(std::filesystem::temp_directory_path()
               / ("hipdnn_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".json"))
    {
    }

    ~TempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        // Also remove any .tmp file
        std::filesystem::remove(std::filesystem::path(path.string() + ".tmp"), ec);
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

/// Create a simple AutotuneResult for testing
AutotuneResult makeResult(int64_t engineId,
                          const std::string& engineName,
                          float minTime = 1.0f,
                          bool succeeded = true,
                          int rank = 0)
{
    AutotuneResult r;
    r.engineId = engineId;
    r.engineName = engineName;
    r.minTimeMs = minTime;
    r.avgTimeMs = minTime + 0.5f;
    r.stddevMs = 0.1f;
    r.iterationsRun = 10;
    r.succeeded = succeeded;
    r.modeUsed = TuneMode::AUTO;
    r.converged = true;
    r.workspaceSize = 1024;
    r.rank = rank;
    return r;
}

} // namespace

// ── knobSettingToJson Tests ─────────────────────────────────────────────────

TEST(TestAutotuneFileWriter, KnobSettingToJsonInt)
{
    const KnobSetting setting("TILE_SIZE", int64_t{128});
    auto json = knobSettingToJson(setting);

    EXPECT_EQ(json["knob_id"], "TILE_SIZE");
    EXPECT_EQ(json["type"], "int");
    EXPECT_EQ(json["value"], 128);
}

TEST(TestAutotuneFileWriter, KnobSettingToJsonDouble)
{
    const KnobSetting setting("LEARNING_RATE", 0.001);
    auto json = knobSettingToJson(setting);

    EXPECT_EQ(json["knob_id"], "LEARNING_RATE");
    EXPECT_EQ(json["type"], "double");
    EXPECT_DOUBLE_EQ(json["value"].get<double>(), 0.001);
}

TEST(TestAutotuneFileWriter, KnobSettingToJsonString)
{
    const KnobSetting setting("ALGORITHM", std::string("gemm_v2"));
    auto json = knobSettingToJson(setting);

    EXPECT_EQ(json["knob_id"], "ALGORITHM");
    EXPECT_EQ(json["type"], "string");
    EXPECT_EQ(json["value"], "gemm_v2");
}

// ── buildOverrideEntry Tests ────────────────────────────────────────────────

TEST(TestAutotuneFileWriter, BuildOverrideEntryBasic)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    const std::vector<std::vector<int64_t>> tensorStrides
        = {{150528, 50176, 224, 1}, {147, 49, 7, 1}};

    auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);

    EXPECT_EQ(entry["op"], "conv_fprop");
    EXPECT_EQ(entry["engine_name"], "MIOPEN_ENGINE");
    ASSERT_EQ(entry["tensors"].size(), 2u);
    EXPECT_EQ(entry["tensors"][0]["dim"], std::vector<int64_t>({1, 3, 224, 224}));
    EXPECT_EQ(entry["tensors"][1]["dim"], std::vector<int64_t>({64, 3, 7, 7}));
    EXPECT_EQ(entry["tensors"][0]["stride"], std::vector<int64_t>({150528, 50176, 224, 1}));
    EXPECT_EQ(entry["tensors"][1]["stride"], std::vector<int64_t>({147, 49, 7, 1}));
    EXPECT_FALSE(entry.contains("knobs")); // No knobs → field absent
}

TEST(TestAutotuneFileWriter, BuildOverrideEntryWithKnobs)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    result.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    result.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {{150528, 50176, 224, 1}};

    auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);

    ASSERT_TRUE(entry.contains("knobs"));
    ASSERT_EQ(entry["knobs"].size(), 2u);
    EXPECT_EQ(entry["knobs"][0]["knob_id"], "TILE_SIZE");
    EXPECT_EQ(entry["knobs"][0]["value"], 128);
    EXPECT_EQ(entry["knobs"][1]["knob_id"], "SPLIT_K");
    EXPECT_EQ(entry["knobs"][1]["value"], 2);
}

TEST(TestAutotuneFileWriter, BuildOverrideEntryWithMetadata)
{
    auto result = makeResult(1, "MIOPEN_ENGINE", 1.5f, true, 0);
    result.modeUsed = TuneMode::EXHAUSTIVE;
    result.strategyUsed = AutotuneStrategy::FIXED_AVERAGE;
    result.ranExhaustive = true;
    result.iterationsRun = 20;
    result.deviceName = "gfx942";

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {{150528, 50176, 224, 1}};

    auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);

    ASSERT_TRUE(entry.contains("autotune_metadata"));
    auto& meta = entry["autotune_metadata"];
    EXPECT_FLOAT_EQ(meta["min_time_ms"].get<float>(), 1.5f);
    EXPECT_EQ(meta["iterations_run"], 20);
    EXPECT_EQ(meta["mode"], "exhaustive");
    EXPECT_EQ(meta["strategy"], "fixed_average");
    EXPECT_EQ(meta["rank"], 0);
    EXPECT_EQ(meta["device"], "gfx942");
    EXPECT_TRUE(meta.contains("timestamp"));
    // Timestamp should be ISO 8601 format (basic check for 'T' and 'Z')
    auto ts = meta["timestamp"].get<std::string>();
    EXPECT_NE(ts.find('T'), std::string::npos);
    EXPECT_NE(ts.find('Z'), std::string::npos);
    EXPECT_TRUE(meta["ran_exhaustive"].get<bool>());
}

// ── writeAutotuneResults Tests ──────────────────────────────────────────────

TEST(TestAutotuneFileWriter, WriteToNewFile)
{
    const TempFile tmpFile;

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));
    results.push_back(makeResult(2, "HIPBLASLT_ENGINE", 2.0f, true, 1));

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}, {64, 3, 7, 7}};

    auto err
        = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results, false, tensorDims, {});
    ASSERT_TRUE(err.is_good()) << err.get_message();

    // Verify file exists and is valid JSON
    ASSERT_TRUE(std::filesystem::exists(tmpFile.path));
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    ASSERT_TRUE(json.contains("engine_overrides"));
    EXPECT_EQ(json["engine_overrides"].size(), 2u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "MIOPEN_ENGINE");
    EXPECT_EQ(json["engine_overrides"][1]["engine_name"], "HIPBLASLT_ENGINE");
}

TEST(TestAutotuneFileWriter, WriteSkipsFailedResults)
{
    const TempFile tmpFile;

    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));
    results.push_back(makeResult(2, "HIPBLASLT_ENGINE", 0.0f, false, -1));
    results.push_back(makeResult(3, "FUSILLI_ENGINE", 3.0f, true, 1));

    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};

    auto err
        = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results, false, tensorDims, {});
    ASSERT_TRUE(err.is_good());

    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    // Only succeeded results should be written
    EXPECT_EQ(json["engine_overrides"].size(), 2u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "MIOPEN_ENGINE");
    EXPECT_EQ(json["engine_overrides"][1]["engine_name"], "FUSILLI_ENGINE");
}

TEST(TestAutotuneFileWriter, AppendToExistingFile)
{
    const TempFile tmpFile;

    // Write initial results for conv_fprop
    std::vector<AutotuneResult> results1;
    results1.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims1 = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    auto err1
        = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results1, false, dims1, {});
    ASSERT_TRUE(err1.is_good());

    // Write new results for conv_dgrad (different op)
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 2.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims2 = {{8, 64, 56, 56}};
    auto err2
        = writeAutotuneResults(tmpFile.path.string(), "conv_dgrad", results2, false, dims2, {});
    ASSERT_TRUE(err2.is_good());

    // Both entries should be in the file
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 2u);
    EXPECT_EQ(json["engine_overrides"][0]["op"], "conv_fprop");
    EXPECT_EQ(json["engine_overrides"][1]["op"], "conv_dgrad");
}

TEST(TestAutotuneFileWriter, ReplaceMatchingEntryWithSameKnobs)
{
    const TempFile tmpFile;

    // Write initial result with no knobs
    std::vector<AutotuneResult> results1;
    results1.push_back(makeResult(1, "MIOPEN_ENGINE", 5.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    auto err1
        = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results1, false, dims, {});
    ASSERT_TRUE(err1.is_good());

    // Write updated result for same op + tensors + same (empty) knobs
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 1.0f, true, 0));

    auto err2
        = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results2, false, dims, {});
    ASSERT_TRUE(err2.is_good());

    // Should have replaced the matching entry (same op + same tensors + same knobs)
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "HIPBLASLT_ENGINE");
}

TEST(TestAutotuneFileWriter, PreserveEntriesWithDifferentKnobs)
{
    const TempFile tmpFile;

    // Write initial result with SPLIT_K=2
    std::vector<AutotuneResult> results1;
    auto r1 = makeResult(1, "MIOPEN_ENGINE", 5.0f, true, 0);
    r1.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    results1.push_back(r1);

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    auto err1
        = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results1, false, dims, {});
    ASSERT_TRUE(err1.is_good());

    // Write new result for same op + tensors but DIFFERENT knobs (SPLIT_K=4)
    std::vector<AutotuneResult> results2;
    auto r2 = makeResult(2, "HIPBLASLT_ENGINE", 1.0f, true, 0);
    r2.knobSettings.emplace_back("SPLIT_K", int64_t{4});
    results2.push_back(r2);

    auto err2
        = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results2, false, dims, {});
    ASSERT_TRUE(err2.is_good());

    // Both entries should be preserved (different knob settings)
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 2u);
    EXPECT_EQ(json["engine_overrides"][0]["engine_name"], "MIOPEN_ENGINE");
    EXPECT_EQ(json["engine_overrides"][1]["engine_name"], "HIPBLASLT_ENGINE");
}

TEST(TestAutotuneFileWriter, DeleteAllExistingContent)
{
    const TempFile tmpFile;

    // Write initial results
    std::vector<AutotuneResult> results1;
    results1.push_back(makeResult(1, "MIOPEN_ENGINE", 1.0f, true, 0));
    const std::vector<std::vector<int64_t>> dims1 = {{1, 3, 224, 224}};
    auto err1
        = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results1, false, dims1, {});
    ASSERT_TRUE(err1.is_good());

    // Write new results with deleteAllExisting=true
    std::vector<AutotuneResult> results2;
    results2.push_back(makeResult(2, "HIPBLASLT_ENGINE", 2.0f, true, 0));
    const std::vector<std::vector<int64_t>> dims2 = {{8, 64, 56, 56}};
    auto err2
        = writeAutotuneResults(tmpFile.path.string(), "conv_dgrad", results2, true, dims2, {});
    ASSERT_TRUE(err2.is_good());

    // Only the new results should be in the file
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);

    EXPECT_EQ(json["engine_overrides"].size(), 1u);
    EXPECT_EQ(json["engine_overrides"][0]["op"], "conv_dgrad");
}

// ── Round-trip Tests ────────────────────────────────────────────────────────

TEST(TestAutotuneFileWriter, RoundTripWriteThenLoad)
{
    const TempFile tmpFile;

    // Write results with knobs
    std::vector<AutotuneResult> results;
    auto result = makeResult(MIOPEN_ENGINE_ID, "MIOPEN_ENGINE", 1.0f, true, 0);
    result.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    result.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    results.push_back(result);

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}, {64, 3, 7, 7}};
    auto err = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results, true, dims, {});
    ASSERT_TRUE(err.is_good());

    // Load via EngineOverrideConfig
    auto config = EngineOverrideConfig::load(tmpFile.path.string());
    ASSERT_TRUE(config.has_value());

    // Match against the same tensors
    auto makeTensor = [](const std::vector<int64_t>& d) {
        auto t = std::make_shared<graph::TensorAttributes>();
        t->set_dim(d);
        return t;
    };

    const std::vector<std::shared_ptr<graph::TensorAttributes>> tensors
        = {makeTensor({1, 3, 224, 224}), makeTensor({64, 3, 7, 7})};

    auto matchResult = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(matchResult.has_value());
    EXPECT_EQ(matchResult->engineId, MIOPEN_ENGINE_ID);

    // Verify knobs round-tripped correctly
    ASSERT_EQ(matchResult->knobs.size(), 2u);

    // Knobs may be in any order, so find by name
    bool foundTileSize = false;
    bool foundSplitK = false;
    for(const auto& knob : matchResult->knobs)
    {
        if(knob.knobId() == "TILE_SIZE")
        {
            EXPECT_EQ(std::get<int64_t>(knob.value()), 128);
            foundTileSize = true;
        }
        else if(knob.knobId() == "SPLIT_K")
        {
            EXPECT_EQ(std::get<int64_t>(knob.value()), 2);
            foundSplitK = true;
        }
    }
    EXPECT_TRUE(foundTileSize) << "TILE_SIZE knob not found in round-trip";
    EXPECT_TRUE(foundSplitK) << "SPLIT_K knob not found in round-trip";
}

TEST(TestAutotuneFileWriter, RoundTripNoKnobs)
{
    const TempFile tmpFile;

    // Write results without knobs
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(MIOPEN_ENGINE_ID, "MIOPEN_ENGINE", 1.0f, true, 0));

    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};
    auto err = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results, true, dims, {});
    ASSERT_TRUE(err.is_good());

    // Load and match
    auto config = EngineOverrideConfig::load(tmpFile.path.string());
    ASSERT_TRUE(config.has_value());

    auto makeTensor = [](const std::vector<int64_t>& d) {
        auto t = std::make_shared<graph::TensorAttributes>();
        t->set_dim(d);
        return t;
    };

    auto matchResult = config->matchOperation("conv_fprop", {makeTensor({1, 3, 224, 224})});
    ASSERT_TRUE(matchResult.has_value());
    EXPECT_EQ(matchResult->engineId, MIOPEN_ENGINE_ID);
    EXPECT_TRUE(matchResult->knobs.empty());
}

// ── Strategy/Mode String Tests ──────────────────────────────────────────────

TEST(TestAutotuneFileWriter, StrategyToString)
{
    EXPECT_EQ(strategyToString(AutotuneStrategy::SINGLE_SHOT), "SINGLE_SHOT");
    EXPECT_EQ(strategyToString(AutotuneStrategy::FIXED_AVERAGE), "FIXED_AVERAGE");
    EXPECT_EQ(strategyToString(AutotuneStrategy::RUN_UNTIL_STABLE), "RUN_UNTIL_STABLE");
}

TEST(TestAutotuneFileWriter, StrategyToLowerString)
{
    EXPECT_EQ(strategyToLowerString(AutotuneStrategy::SINGLE_SHOT), "single_shot");
    EXPECT_EQ(strategyToLowerString(AutotuneStrategy::FIXED_AVERAGE), "fixed_average");
    EXPECT_EQ(strategyToLowerString(AutotuneStrategy::RUN_UNTIL_STABLE), "run_until_stable");
}

TEST(TestAutotuneFileWriter, TuneModeToString)
{
    EXPECT_EQ(tuneModeToString(TuneMode::AUTO), "AUTO");
    EXPECT_EQ(tuneModeToString(TuneMode::EXHAUSTIVE), "EXHAUSTIVE");
}

TEST(TestAutotuneFileWriter, TuneModeToLowerString)
{
    EXPECT_EQ(tuneModeToLowerString(TuneMode::AUTO), "auto");
    EXPECT_EQ(tuneModeToLowerString(TuneMode::EXHAUSTIVE), "exhaustive");
}

// ── Error handling Tests ────────────────────────────────────────────────────

TEST(TestAutotuneFileWriter, WriteToInvalidPathFails)
{
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    auto err = writeAutotuneResults("/nonexistent/deep/path/that/does/not/exist/file.json",
                                    "conv_fprop",
                                    results,
                                    true,
                                    dims,
                                    {});

    EXPECT_TRUE(err.is_bad());
}

TEST(TestAutotuneFileWriter, WriteNoSucceededResultsIsOk)
{
    const TempFile tmpFile;

    // All results failed
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE", 0.0f, false, -1));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    auto err = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results, true, dims, {});

    // Should succeed (no error) but write nothing
    EXPECT_TRUE(err.is_good());
}

TEST(TestAutotuneFileWriter, HandleCorruptExistingFile)
{
    const TempFile tmpFile;

    // Write corrupt JSON to the file
    {
        std::ofstream outFile(tmpFile.path);
        outFile << "{ this is not valid json ]}}";
    }

    // Write valid results — should start fresh despite corrupt existing
    std::vector<AutotuneResult> results;
    results.push_back(makeResult(1, "MIOPEN_ENGINE"));
    const std::vector<std::vector<int64_t>> dims = {{1, 3, 224, 224}};

    auto err = writeAutotuneResults(tmpFile.path.string(), "conv_fprop", results, false, dims, {});

    ASSERT_TRUE(err.is_good());

    // Verify valid JSON was written
    std::ifstream file(tmpFile.path);
    auto json = nlohmann::json::parse(file);
    EXPECT_EQ(json["engine_overrides"].size(), 1u);
}

// ── ran_exhaustive / converged metadata tests ───────────────────────────────

TEST(TestAutotuneFileWriter, RanExhaustiveAlwaysWritten)
{
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {};

    // ranExhaustive = false should be written
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.ranExhaustive = false;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        ASSERT_TRUE(meta.contains("ran_exhaustive"));
        EXPECT_FALSE(meta["ran_exhaustive"].get<bool>());
    }

    // ranExhaustive = true should be written
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.ranExhaustive = true;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        ASSERT_TRUE(meta.contains("ran_exhaustive"));
        EXPECT_TRUE(meta["ran_exhaustive"].get<bool>());
    }
}

TEST(TestAutotuneFileWriter, ConvergedOnlyForRunUntilStable)
{
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {};

    // FIXED_AVERAGE: converged should NOT be present
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.strategyUsed = AutotuneStrategy::FIXED_AVERAGE;
        result.converged = true;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        EXPECT_FALSE(meta.contains("converged"));
    }

    // SINGLE_SHOT: converged should NOT be present
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.strategyUsed = AutotuneStrategy::SINGLE_SHOT;
        result.converged = false;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        EXPECT_FALSE(meta.contains("converged"));
    }

    // RUN_UNTIL_STABLE with converged=true: converged should be present
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.strategyUsed = AutotuneStrategy::RUN_UNTIL_STABLE;
        result.converged = true;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        ASSERT_TRUE(meta.contains("converged"));
        EXPECT_TRUE(meta["converged"].get<bool>());
    }

    // RUN_UNTIL_STABLE with converged=false: converged should be present
    {
        auto result = makeResult(1, "MIOPEN_ENGINE");
        result.strategyUsed = AutotuneStrategy::RUN_UNTIL_STABLE;
        result.converged = false;
        auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);
        auto& meta = entry["autotune_metadata"];
        ASSERT_TRUE(meta.contains("converged"));
        EXPECT_FALSE(meta["converged"].get<bool>());
    }
}

TEST(TestAutotuneFileWriter, BuildOverrideEntryWithEmptyStrides)
{
    auto result = makeResult(1, "MIOPEN_ENGINE");
    const std::vector<std::vector<int64_t>> tensorDims = {{1, 3, 224, 224}};
    const std::vector<std::vector<int64_t>> tensorStrides = {}; // No strides

    auto entry = buildOverrideEntry(result, "conv_fprop", tensorDims, tensorStrides);

    ASSERT_EQ(entry["tensors"].size(), 1u);
    EXPECT_EQ(entry["tensors"][0]["dim"], std::vector<int64_t>({1, 3, 224, 224}));
    EXPECT_FALSE(entry["tensors"][0].contains("stride")); // No strides provided
}

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
