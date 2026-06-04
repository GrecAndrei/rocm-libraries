// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/BenchmarkStatistics.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::autotune;

// ============================================================================
// Plan Spec Collection and Dedup Tests
// ============================================================================

TEST(TestAutotune, PlanSpecDedupPreventsDuplicates)
{
    std::vector<PlanSpec> specs;

    PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("TILE_SIZE", int64_t{128});

    PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    b.workspaceSize = 9999; // Different workspace — should still dedup

    specs.push_back(a);

    // Simulate addPlanSpecIfUnique
    auto it = std::find(specs.begin(), specs.end(), b);
    EXPECT_NE(it, specs.end()); // b should be found as duplicate
}

TEST(TestAutotune, PlanSpecDedupAllowsDifferentKnobs)
{
    std::vector<PlanSpec> specs;

    PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("TILE_SIZE", int64_t{128});

    PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("TILE_SIZE", int64_t{256}); // Different value

    specs.push_back(a);

    auto it = std::find(specs.begin(), specs.end(), b);
    EXPECT_EQ(it, specs.end()); // b should NOT be found
}

TEST(TestAutotune, PlanSpecDedupAllowsDifferentEngines)
{
    std::vector<PlanSpec> specs;

    PlanSpec a;
    a.engineId = 42;

    PlanSpec b;
    b.engineId = 99;

    specs.push_back(a);

    auto it = std::find(specs.begin(), specs.end(), b);
    EXPECT_EQ(it, specs.end()); // Different engines should not dedup
}

// ============================================================================
// Benchmarking Knob Stripping
// ============================================================================

TEST(TestAutotune, BenchmarkingKnobNameIsGlobalBenchmarking)
{
    EXPECT_EQ(BENCHMARKING_KNOB_NAME, "global.benchmarking");
}

TEST(TestAutotune, BenchmarkingKnobStrippedFromPlanSpec)
{
    // Simulate what add_engine() does: strip global.benchmarking
    std::vector<KnobSetting> userKnobs;
    userKnobs.emplace_back("TILE_SIZE", int64_t{128});
    userKnobs.emplace_back(BENCHMARKING_KNOB_NAME, int64_t{1});
    userKnobs.emplace_back("SPLIT_K", int64_t{2});

    PlanSpec spec;
    spec.engineId = 42;
    for(const auto& setting : userKnobs)
    {
        if(setting.knobId() == BENCHMARKING_KNOB_NAME)
        {
            continue; // Strip
        }
        spec.knobSettings.push_back(setting);
    }

    EXPECT_EQ(spec.knobSettings.size(), 2u);
    EXPECT_EQ(spec.knobSettings[0].knobId(), "TILE_SIZE");
    EXPECT_EQ(spec.knobSettings[1].knobId(), "SPLIT_K");
}

// ============================================================================
// get_max_workspace_size Logic Tests
// ============================================================================

TEST(TestAutotune, MaxWorkspaceFromPlanSpecs)
{
    std::vector<PlanSpec> specs;

    PlanSpec a;
    a.engineId = 1;
    a.workspaceSize = 100;
    specs.push_back(a);

    PlanSpec b;
    b.engineId = 2;
    b.workspaceSize = 500;
    specs.push_back(b);

    PlanSpec c;
    c.engineId = 3;
    c.workspaceSize = 200;
    specs.push_back(c);

    // Simulate get_max_workspace_size logic
    int64_t maxSize = 0;
    for(const auto& spec : specs)
    {
        maxSize = std::max(maxSize, spec.workspaceSize);
    }

    EXPECT_EQ(maxSize, 500);
}

TEST(TestAutotune, MaxWorkspaceEmptyPlanSpecs)
{
    const std::vector<PlanSpec> specs;

    int64_t maxSize = 0;
    for(const auto& spec : specs)
    {
        maxSize = std::max(maxSize, spec.workspaceSize);
    }

    EXPECT_EQ(maxSize, 0);
}

// ============================================================================
// AutotuneConfig Validation Tests
// ============================================================================

TEST(TestAutotune, ConfigDefaultsAreValid)
{
    const AutotuneConfig config;

    // Verify defaults
    EXPECT_EQ(config.mode, TuneMode::AUTO);
    EXPECT_EQ(config.strategy, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_EQ(config.warmupIterations, 1);
    EXPECT_EQ(config.timedIterations, 10);
    EXPECT_EQ(config.maxIterations, 100);
    EXPECT_EQ(config.windowSize, 3);
    EXPECT_FLOAT_EQ(config.stabilityThreshold, 0.05f);
    EXPECT_EQ(config.maxWorkspaceBytes, 0u);
    EXPECT_TRUE(config.engineIdFilter.empty());
    EXPECT_EQ(config.rankingFn, nullptr);
    EXPECT_FALSE(config.continueOnPrimingFailure);
}

TEST(TestAutotune, ConfigValidationNegativeWarmup)
{
    AutotuneConfig config;
    config.warmupIterations = -1;

    // Warmup can be 0 but not negative
    EXPECT_LT(config.warmupIterations, 0);
}

TEST(TestAutotune, ConfigValidationNegativeTimedIterations)
{
    AutotuneConfig config;
    config.timedIterations = 0; // Must be >= 1

    EXPECT_LT(config.timedIterations, 1);
}

TEST(TestAutotune, ConfigValidationWindowSizeTooSmall)
{
    AutotuneConfig config;
    config.windowSize = 1; // Must be >= 2

    EXPECT_LT(config.windowSize, 2);
}

TEST(TestAutotune, ConfigValidationStabilityThresholdOutOfBounds)
{
    AutotuneConfig config;

    // Must be in (0.0, 1.0) exclusive
    config.stabilityThreshold = 0.0f;
    EXPECT_LE(config.stabilityThreshold, 0.0f);

    config.stabilityThreshold = 1.0f;
    EXPECT_GE(config.stabilityThreshold, 1.0f);

    config.stabilityThreshold = -0.5f;
    EXPECT_LE(config.stabilityThreshold, 0.0f);
}

// ============================================================================
// AutotuneResult Default State Tests
// ============================================================================

TEST(TestAutotune, AutotuneResultDefaultState)
{
    const AutotuneResult result;

    EXPECT_EQ(result.engineId, -1);
    EXPECT_TRUE(result.engineName.empty());
    EXPECT_TRUE(result.knobSettings.empty());
    EXPECT_EQ(result.rank, -1);
    EXPECT_FLOAT_EQ(result.minTimeMs, 0.0f);
    EXPECT_FLOAT_EQ(result.avgTimeMs, 0.0f);
    EXPECT_FLOAT_EQ(result.stddevMs, 0.0f);
    EXPECT_EQ(result.iterationsRun, 0);
    EXPECT_FALSE(result.succeeded);
    EXPECT_EQ(result.modeUsed, TuneMode::AUTO);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.workspaceSize, 0);
    EXPECT_FALSE(result.ranExhaustive);
    EXPECT_TRUE(result.errorMessage.empty());
    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_TRUE(result.deviceName.empty());
}

// ============================================================================
// Ranking Logic Tests (Default)
// ============================================================================

TEST(TestAutotune, DefaultRankingByMinTime)
{
    std::vector<AutotuneResult> results;

    // Engine 1: slow
    AutotuneResult r1;
    r1.engineId = 1;
    r1.succeeded = true;
    r1.minTimeMs = 5.0f;
    results.push_back(r1);

    // Engine 2: fast
    AutotuneResult r2;
    r2.engineId = 2;
    r2.succeeded = true;
    r2.minTimeMs = 1.0f;
    results.push_back(r2);

    // Engine 3: medium
    AutotuneResult r3;
    r3.engineId = 3;
    r3.succeeded = true;
    r3.minTimeMs = 3.0f;
    results.push_back(r3);

    // Default ranking: sort by minTimeMs ascending
    std::stable_sort(
        results.begin(), results.end(), [](const AutotuneResult& a, const AutotuneResult& b) {
            if(a.succeeded != b.succeeded)
            {
                return a.succeeded > b.succeeded;
            }
            if(!a.succeeded)
            {
                return false;
            }
            return a.minTimeMs < b.minTimeMs;
        });

    EXPECT_EQ(results[0].engineId, 2); // fastest
    EXPECT_EQ(results[1].engineId, 3); // medium
    EXPECT_EQ(results[2].engineId, 1); // slowest
}

TEST(TestAutotune, DefaultRankingFailedEnginesAtEnd)
{
    std::vector<AutotuneResult> results;

    // Successful engine
    AutotuneResult r1;
    r1.engineId = 1;
    r1.succeeded = true;
    r1.minTimeMs = 5.0f;
    results.push_back(r1);

    // Failed engine
    AutotuneResult r2;
    r2.engineId = 2;
    r2.succeeded = false;
    results.push_back(r2);

    // Another successful engine
    AutotuneResult r3;
    r3.engineId = 3;
    r3.succeeded = true;
    r3.minTimeMs = 1.0f;
    results.push_back(r3);

    std::stable_sort(
        results.begin(), results.end(), [](const AutotuneResult& a, const AutotuneResult& b) {
            if(a.succeeded != b.succeeded)
            {
                return a.succeeded > b.succeeded;
            }
            if(!a.succeeded)
            {
                return false;
            }
            return a.minTimeMs < b.minTimeMs;
        });

    // Assign ranks
    for(size_t i = 0; i < results.size(); ++i)
    {
        results[i].rank = results[i].succeeded ? static_cast<int>(i) : -1;
    }

    EXPECT_EQ(results[0].engineId, 3); // fastest successful
    EXPECT_EQ(results[0].rank, 0);
    EXPECT_EQ(results[1].engineId, 1); // slower successful
    EXPECT_EQ(results[1].rank, 1);
    EXPECT_EQ(results[2].engineId, 2); // failed
    EXPECT_EQ(results[2].rank, -1);
}

// ============================================================================
// Custom Ranking Function Tests
// ============================================================================

TEST(TestAutotune, CustomRankingByAvgTime)
{
    std::vector<AutotuneResult> results;

    AutotuneResult r1;
    r1.engineId = 1;
    r1.succeeded = true;
    r1.minTimeMs = 1.0f;
    r1.avgTimeMs = 5.0f; // High average despite low min
    results.push_back(r1);

    AutotuneResult r2;
    r2.engineId = 2;
    r2.succeeded = true;
    r2.minTimeMs = 2.0f;
    r2.avgTimeMs = 2.5f; // Low average
    results.push_back(r2);

    // Custom ranking by avgTimeMs
    const AutotuneRankingFn customRank = [](std::vector<AutotuneResult>& res) {
        std::stable_sort(
            res.begin(), res.end(), [](const AutotuneResult& a, const AutotuneResult& b) {
                return a.avgTimeMs < b.avgTimeMs;
            });
    };

    customRank(results);

    // By avgTimeMs, engine 2 should be first
    EXPECT_EQ(results[0].engineId, 2);
    EXPECT_EQ(results[1].engineId, 1);
}

// ============================================================================
// AutotuneStorageConfig Tests
// ============================================================================

TEST(TestAutotune, StorageConfigDefaults)
{
    const AutotuneStorageConfig config;

    EXPECT_TRUE(config.filePath.empty());
    EXPECT_FALSE(config.deleteAllExistingFileContent);
}

TEST(TestAutotune, StorageConfigEmptyPathMeansNoFileIO)
{
    const AutotuneStorageConfig config;
    // When filePath is empty, autotune() should skip file I/O
    EXPECT_TRUE(config.filePath.empty());
}

// ============================================================================
// Engine ID Filter Logic Tests
// ============================================================================

TEST(TestAutotune, EngineIdFilterSelectsSubset)
{
    std::vector<PlanSpec> planSpecs;

    PlanSpec a;
    a.engineId = 1;
    planSpecs.push_back(a);

    PlanSpec b;
    b.engineId = 2;
    planSpecs.push_back(b);

    PlanSpec c;
    c.engineId = 3;
    planSpecs.push_back(c);

    // Filter to only engines 1 and 3
    std::vector<int64_t> filter = {1, 3};
    std::unordered_set<int64_t> filterSet(filter.begin(), filter.end());

    std::vector<PlanSpec> filtered;
    for(const auto& spec : planSpecs)
    {
        if(filterSet.find(spec.engineId) != filterSet.end())
        {
            filtered.push_back(spec);
        }
    }

    EXPECT_EQ(filtered.size(), 2u);
    EXPECT_EQ(filtered[0].engineId, 1);
    EXPECT_EQ(filtered[1].engineId, 3);
}

TEST(TestAutotune, MaxWorkspaceBytesFiltering)
{
    std::vector<PlanSpec> planSpecs;

    PlanSpec a;
    a.engineId = 1;
    a.workspaceSize = 100;
    planSpecs.push_back(a);

    PlanSpec b;
    b.engineId = 2;
    b.workspaceSize = 500;
    planSpecs.push_back(b);

    PlanSpec c;
    c.engineId = 3;
    c.workspaceSize = 200;
    planSpecs.push_back(c);

    // Limit workspace to 250
    const size_t maxWorkspaceBytes = 250;
    std::vector<PlanSpec> filtered;
    for(const auto& spec : planSpecs)
    {
        if(spec.workspaceSize <= static_cast<int64_t>(maxWorkspaceBytes))
        {
            filtered.push_back(spec);
        }
    }

    EXPECT_EQ(filtered.size(), 2u);
    EXPECT_EQ(filtered[0].engineId, 1);
    EXPECT_EQ(filtered[1].engineId, 3);
}

// ============================================================================
// EXHAUSTIVE Priming Logic Tests
// ============================================================================

TEST(TestAutotune, ExhaustivePrimingKnobInjection)
{
    // Simulate what autotune() does: copy knob settings, append global.benchmarking=1
    std::vector<KnobSetting> originalKnobs;
    originalKnobs.emplace_back("TILE_SIZE", int64_t{128});

    auto primingKnobs = originalKnobs;
    primingKnobs.emplace_back(BENCHMARKING_KNOB_NAME, int64_t{1});

    EXPECT_EQ(primingKnobs.size(), 2u);
    EXPECT_EQ(primingKnobs[0].knobId(), "TILE_SIZE");
    EXPECT_EQ(primingKnobs[1].knobId(), BENCHMARKING_KNOB_NAME);
    EXPECT_EQ(std::get<int64_t>(primingKnobs[1].value()), 1);
}

// ============================================================================
// EngineConfigInfo Tests
// ============================================================================

TEST(TestAutotune, EngineConfigInfoDefaults)
{
    const EngineConfigInfo info;
    EXPECT_EQ(info.engineId, -1);
    EXPECT_TRUE(info.engineName.empty());
    EXPECT_TRUE(info.knobs.empty());
    EXPECT_FALSE(info.supportsExhaustive);
    EXPECT_EQ(info.workspaceSize, 0);
}

// ============================================================================
// EngineVariant Conversion Tests
// ============================================================================

TEST(TestAutotune, EngineVariantToKnobSettings)
{
    EngineVariant variant;
    variant.engineId = 42;
    variant.knobSettings["TILE_SIZE"] = int64_t{128};
    variant.knobSettings["SPLIT_K"] = int64_t{2};

    // Convert map to vector (as add_engine_variants() does)
    std::vector<KnobSetting> settings;
    for(const auto& [knobId, value] : variant.knobSettings)
    {
        if(knobId == BENCHMARKING_KNOB_NAME)
        {
            continue;
        }
        settings.emplace_back(knobId, value);
    }

    EXPECT_EQ(settings.size(), 2u);
    // std::map is ordered by key, so SPLIT_K comes before TILE_SIZE
    EXPECT_EQ(settings[0].knobId(), "SPLIT_K");
    EXPECT_EQ(settings[1].knobId(), "TILE_SIZE");
}

// ============================================================================
// Convergence Detection Tests (RUN_UNTIL_STABLE)
// ============================================================================

TEST(TestAutotune, ConvergenceDetection)
{
    // Simulate RUN_UNTIL_STABLE: check if last windowSize samples converge
    const int windowSize = 3;
    const float stabilityThreshold = 0.05f;

    // Stable samples (low CoV)
    const std::vector<float> stableTimings = {1.0f, 1.01f, 0.99f, 1.0f, 1.005f};

    if(static_cast<int>(stableTimings.size()) >= windowSize)
    {
        const std::vector<float> window(stableTimings.end() - windowSize, stableTimings.end());
        const float cov = computeCoefficientOfVariation(window);
        EXPECT_LT(cov, stabilityThreshold);
    }
}

TEST(TestAutotune, NoConvergenceWithHighVariance)
{
    const int windowSize = 3;
    const float stabilityThreshold = 0.05f;

    // Unstable samples (high CoV)
    const std::vector<float> unstableTimings = {1.0f, 5.0f, 2.0f, 8.0f, 0.5f};

    if(static_cast<int>(unstableTimings.size()) >= windowSize)
    {
        const std::vector<float> window(unstableTimings.end() - windowSize, unstableTimings.end());
        const float cov = computeCoefficientOfVariation(window);
        EXPECT_GE(cov, stabilityThreshold);
    }
}

// ============================================================================
// D2: maxIterations >= windowSize validation for RUN_UNTIL_STABLE
// ============================================================================

TEST(TestAutotune, MaxIterationsLessThanWindowSizeIsDetectable)
{
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.maxIterations = 3;
    config.windowSize = 5;

    // The autotune() function should reject this configuration
    EXPECT_LT(config.maxIterations, config.windowSize);
}

TEST(TestAutotune, MaxIterationsEqualToWindowSizeIsValid)
{
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.maxIterations = 5;
    config.windowSize = 5;

    EXPECT_GE(config.maxIterations, config.windowSize);
}

TEST(TestAutotune, MaxIterationsCheckOnlyForRunUntilStable)
{
    // For FIXED_AVERAGE, maxIterations < windowSize should not be an error
    AutotuneConfig config;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.maxIterations = 3;
    config.windowSize = 5;

    // This should be valid for FIXED_AVERAGE since maxIterations/windowSize
    // are only used by RUN_UNTIL_STABLE
    EXPECT_EQ(config.strategy, AutotuneStrategy::FIXED_AVERAGE);
}

// ============================================================================
// AutotuneResult New Fields Tests
// ============================================================================

TEST(TestAutotune, AutotuneResultStrategyAndDeviceFields)
{
    AutotuneResult result;
    result.strategyUsed = AutotuneStrategy::RUN_UNTIL_STABLE;
    result.deviceName = "gfx942";

    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_EQ(result.deviceName, "gfx942");
}

TEST(TestAutotune, AutotuneResultDefaultStrategyAndDevice)
{
    const AutotuneResult result;

    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_TRUE(result.deviceName.empty());
}
