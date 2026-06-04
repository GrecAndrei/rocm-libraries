// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/detail/EngineOverrideConfig.hpp>

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
#include <cstdio>
#include <fstream>
#endif

using namespace hipdnn_frontend::engine_override;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

// ── helpers ─────────────────────────────────────────────────────────────────

static std::shared_ptr<TensorAttributes> makeTensor(const std::vector<int64_t>& dims)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim(dims);
    return t;
}

static std::shared_ptr<TensorAttributes> makeTensorWithStride(const std::vector<int64_t>& dims,
                                                              const std::vector<int64_t>& strides)
{
    auto t = std::make_shared<TensorAttributes>();
    t->set_dim(dims);
    t->set_stride(strides);
    return t;
}

static TensorPattern makePattern(std::vector<int64_t> dims)
{
    TensorPattern p;
    p.dim = std::move(dims);
    return p;
}

static TensorPattern makePatternWithStride(std::vector<int64_t> dims, std::vector<int64_t> strides)
{
    TensorPattern p;
    p.dim = std::move(dims);
    p.stride = std::move(strides);
    return p;
}

// Construct a single-rule config inline (no JSON required).
static EngineOverrideConfig makeConfig(std::vector<OperationRule> rules)
{
    return EngineOverrideConfig(std::move(rules));
}

// ── Test 1: exact dim match, single rule ────────────────────────────────────

TEST(TestEngineOverrideConfig, ExactDimMatchSingleRule)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224}), makePattern({64, 3, 7, 7})};

    auto config = makeConfig({std::move(rule)});

    const std::vector<std::shared_ptr<TensorAttributes>> tensors
        = {makeTensor({1, 3, 224, 224}), makeTensor({64, 3, 7, 7})};

    auto result = config.matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->engineId, MIOPEN_ENGINE_ID);
}

// ── Test 2: first matching rule wins ────────────────────────────────────────

TEST(TestEngineOverrideConfig, FirstMatchingRuleWins)
{
    OperationRule rule1;
    rule1.op = "conv_fprop";
    rule1.engineName = MIOPEN_ENGINE_NAME;
    rule1.tensors = {makePattern({1, 3, 224, 224})};

    OperationRule rule2;
    rule2.op = "conv_fprop";
    rule2.engineName = HIPBLASLT_ENGINE_NAME;
    rule2.tensors = {makePattern({1, 3, 224, 224})};

    auto config = makeConfig({std::move(rule1), std::move(rule2)});

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};

    auto result = config.matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->engineId, MIOPEN_ENGINE_ID); // first rule wins
}

// ── Test 3: no rule matches (wrong dims) ────────────────────────────────────

TEST(TestEngineOverrideConfig, NoRuleMatchesWrongDims)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224})};

    auto config = makeConfig({std::move(rule)});

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {
        makeTensor({1, 3, 112, 112}) // different spatial dims
    };

    auto result = config.matchOperation("conv_fprop", tensors);
    EXPECT_FALSE(result.has_value());
}

// ── Test 4: wildcard (-1) in one dimension ──────────────────────────────────

TEST(TestEngineOverrideConfig, WildcardInOneDimension)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = HIPBLASLT_ENGINE_NAME;
    rule.tensors = {makePattern({-1, 64, 56, 56})}; // batch dim is wildcard

    auto config = makeConfig({std::move(rule)});

    for(const int64_t batch : {1, 4, 8, 32})
    {
        const std::vector<std::shared_ptr<TensorAttributes>> tensors
            = {makeTensor({batch, 64, 56, 56})};
        auto result = config.matchOperation("conv_fprop", tensors);
        ASSERT_TRUE(result.has_value()) << "batch=" << batch << " should match";
        EXPECT_EQ(result->engineId, HIPBLASLT_ENGINE_ID);
    }

    // Non-matching channel dim should still fail
    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({4, 128, 56, 56})};
    EXPECT_FALSE(config.matchOperation("conv_fprop", tensors).has_value());
}

// ── Test 5: all-wildcard rule matches any shape ─────────────────────────────

TEST(TestEngineOverrideConfig, AllWildcardRuleMatchesAnyShape)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = FUSILLI_ENGINE_NAME;
    rule.tensors = {makePattern({-1, -1, -1, -1})};

    auto config = makeConfig({std::move(rule)});

    for(const auto& shape :
        std::vector<std::vector<int64_t>>{{1, 3, 224, 224}, {8, 64, 56, 56}, {32, 256, 14, 14}})
    {
        const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor(shape)};
        auto result = config.matchOperation("conv_fprop", tensors);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->engineId, FUSILLI_ENGINE_ID);
    }
}

// ── Test 6: wrong op name → nullopt ─────────────────────────────────────────

TEST(TestEngineOverrideConfig, WrongOpNameReturnsNullopt)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224})};

    auto config = makeConfig({std::move(rule)});

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};

    EXPECT_FALSE(config.matchOperation("conv_dgrad", tensors).has_value());
    EXPECT_FALSE(config.matchOperation("conv_wgrad", tensors).has_value());
    EXPECT_FALSE(config.matchOperation("matmul", tensors).has_value());
}

// ── Test 7: wrong tensor count in rule → nullopt ────────────────────────────

TEST(TestEngineOverrideConfig, WrongTensorCountReturnsNullopt)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224}), makePattern({64, 3, 7, 7})}; // 2 patterns

    auto config = makeConfig({std::move(rule)});

    // Provide only 1 tensor where 2 are expected
    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    EXPECT_FALSE(config.matchOperation("conv_fprop", tensors).has_value());

    // Provide 3 tensors where 2 are expected
    const std::vector<std::shared_ptr<TensorAttributes>> tensors3
        = {makeTensor({1, 3, 224, 224}), makeTensor({64, 3, 7, 7}), makeTensor({64, 1, 1, 1})};
    EXPECT_FALSE(config.matchOperation("conv_fprop", tensors3).has_value());
}

// ── Tests 11–12: cross-partition ordering (exact vs wildcard) ───────────────
//
// These tests verify that first-match-wins semantics are preserved when an
// exact rule and a wildcard rule sit in different partitions.

// Test 11: wildcard declared before exact — wildcard must win
TEST(TestEngineOverrideConfig, WildcardBeforeExactBothMatch)
{
    OperationRule wildcard;
    wildcard.op = "conv_fprop";
    wildcard.engineName = FUSILLI_ENGINE_NAME;
    wildcard.tensors = {makePattern({-1, 3, 224, 224})}; // order 0, wildcard

    OperationRule exact;
    exact.op = "conv_fprop";
    exact.engineName = HIPBLASLT_ENGINE_NAME;
    exact.tensors = {makePattern({1, 3, 224, 224})}; // order 1, exact

    auto config = makeConfig({std::move(wildcard), std::move(exact)});

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config.matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->engineId, FUSILLI_ENGINE_ID); // wildcard (order 0) beats exact (order 1)
}

// Test 12: exact declared before wildcard — exact must win
TEST(TestEngineOverrideConfig, ExactBeforeWildcardBothMatch)
{
    OperationRule exact;
    exact.op = "conv_fprop";
    exact.engineName = HIPBLASLT_ENGINE_NAME;
    exact.tensors = {makePattern({1, 3, 224, 224})}; // order 0, exact

    OperationRule wildcard;
    wildcard.op = "conv_fprop";
    wildcard.engineName = FUSILLI_ENGINE_NAME;
    wildcard.tensors = {makePattern({-1, 3, 224, 224})}; // order 1, wildcard

    auto config = makeConfig({std::move(exact), std::move(wildcard)});

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config.matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->engineId, HIPBLASLT_ENGINE_ID); // exact (order 0) beats wildcard (order 1)
}

// ── Stride matching tests ────────────────────────────────────────────────────

// Test 13: exact stride match selects the correct engine
TEST(TestEngineOverrideConfig, ExactStrideMatchSelectsEngine)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = MIOPEN_ENGINE_NAME;
    rule.tensors = {makePatternWithStride({1, 3, 224, 224}, {150528, 50176, 224, 1})};

    auto config = makeConfig({std::move(rule)});

    auto matching = makeTensorWithStride({1, 3, 224, 224}, {150528, 50176, 224, 1});
    auto result = config.matchOperation("conv_fprop", {matching});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->engineId, MIOPEN_ENGINE_ID);

    // Different stride must not match
    auto wrongStride = makeTensorWithStride({1, 3, 224, 224},
                                            {1, 224, int64_t{224} * 3, int64_t{224} * 3 * 224});
    EXPECT_FALSE(config.matchOperation("conv_fprop", {wrongStride}).has_value());
}

// Test 14: wildcard stride element (-1) matches any value in that slot
TEST(TestEngineOverrideConfig, WildcardStrideElement)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = HIPBLASLT_ENGINE_NAME;
    // Wildcard on last two stride slots
    rule.tensors = {makePatternWithStride({1, 3, 224, 224}, {150528, 50176, -1, -1})};

    auto config = makeConfig({std::move(rule)});

    // Should match regardless of the last two stride values
    for(const int64_t s2 : {224, 112, 56})
    {
        auto t = makeTensorWithStride({1, 3, 224, 224}, {150528, 50176, s2, 1});
        auto result = config.matchOperation("conv_fprop", {t});
        ASSERT_TRUE(result.has_value()) << "stride[2]=" << s2;
        EXPECT_EQ(result->engineId, HIPBLASLT_ENGINE_ID);
    }

    // First two stride slots must still match
    auto wrongStride = makeTensorWithStride({1, 3, 224, 224}, {999, 50176, 224, 1});
    EXPECT_FALSE(config.matchOperation("conv_fprop", {wrongStride}).has_value());
}

// Test 15: empty stride in pattern matches any tensor stride (no constraint)
TEST(TestEngineOverrideConfig, EmptyStridePatternMatchesAnyStride)
{
    OperationRule rule;
    rule.op = "conv_fprop";
    rule.engineName = FUSILLI_ENGINE_NAME;
    rule.tensors = {makePattern({1, 3, 224, 224})}; // no stride field

    auto config = makeConfig({std::move(rule)});

    // Should match regardless of stride
    for(const auto& strides : std::vector<std::vector<int64_t>>{
            {150528, 50176, 224, 1}, {1, 3, 672, 150528}, {999, 888, 777, 666}})
    {
        auto t = makeTensorWithStride({1, 3, 224, 224}, strides);
        auto result = config.matchOperation("conv_fprop", {t});
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->engineId, FUSILLI_ENGINE_ID);
    }
}

// ── Tests 8–10: JSON-dependent ──────────────────────────────────────────────

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB

// Test 8: load from valid JSON file → parses rules, matches correctly

TEST(TestEngineOverrideConfig, LoadFromValidJsonFile)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "comment": "test rule for ResNet first conv",
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] },
        { "dim": [64, 3, 7, 7] }
      ]
    },
    {
      "comment": "wildcard catch-all",
      "op": "conv_fprop",
      "engine_name": "FUSILLI_ENGINE",
      "tensors": [
        { "dim": [-1, -1, -1, -1] },
        { "dim": [-1, -1, -1, -1] }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    // Exact match hits the first rule
    const std::vector<std::shared_ptr<TensorAttributes>> exact
        = {makeTensor({1, 3, 224, 224}), makeTensor({64, 3, 7, 7})};
    auto r1 = config->matchOperation("conv_fprop", exact);
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->engineId, MIOPEN_ENGINE_ID);

    // Different shape falls through to the wildcard rule
    const std::vector<std::shared_ptr<TensorAttributes>> other
        = {makeTensor({8, 64, 56, 56}), makeTensor({64, 64, 3, 3})};
    auto r2 = config->matchOperation("conv_fprop", other);
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r2->engineId, FUSILLI_ENGINE_ID);
}

// Test 9: load from missing file → nullopt, no crash

TEST(TestEngineOverrideConfig, LoadFromMissingFileReturnsNullopt)
{
    auto config = EngineOverrideConfig::load("/nonexistent/path/hipdnn_no_such_file.json");
    EXPECT_FALSE(config.has_value());
}

// Test 10: HIPDNN_ENGINE_OVERRIDE_FILE unset → loadFromEnv() returns nullptr

TEST(TestEngineOverrideConfig, EnvVarUnsetReturnsNullptr)
{
    // HIPDNN_ENGINE_OVERRIDE_FILE is not set in the unit-test environment.
    // loadFromEnv() caches on first call, so this also verifies the pointer
    // is stable across repeated calls.
    const auto* config = EngineOverrideConfig::loadFromEnv();
    EXPECT_EQ(config, nullptr);
    EXPECT_EQ(EngineOverrideConfig::loadFromEnv(), config); // same cached pointer
}

// Test 16: JSON with stride constraint is parsed and matched correctly
TEST(TestEngineOverrideConfig, JsonWithStrideConstraint)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224], "stride": [150528, 50176, 224, 1] },
        { "dim": [64, 3, 7, 7] }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    auto x = makeTensorWithStride({1, 3, 224, 224}, {150528, 50176, 224, 1});
    auto w = makeTensor({64, 3, 7, 7});

    auto r1 = config->matchOperation("conv_fprop", {x, w});
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(r1->engineId, MIOPEN_ENGINE_ID);

    // Wrong stride must not match
    auto xWrong = makeTensorWithStride({1, 3, 224, 224},
                                       {1, 224, int64_t{224} * 3, int64_t{224} * 3 * 224});
    EXPECT_FALSE(config->matchOperation("conv_fprop", {xWrong, w}).has_value());
}

// ── Knob parsing tests ──────────────────────────────────────────────────────

// Test 17: JSON entry with integer knobs
TEST(TestEngineOverrideConfig, JsonWithIntegerKnobs)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ],
      "knobs": [
        { "name": "TILE_SIZE", "type": "int", "value": 128 },
        { "name": "SPLIT_K", "type": "int", "value": 2 }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->engineId, MIOPEN_ENGINE_ID);

    ASSERT_EQ(result->knobs.size(), 2u);
    EXPECT_EQ(result->knobs[0].knobId(), "TILE_SIZE");
    EXPECT_EQ(std::get<int64_t>(result->knobs[0].value()), 128);
    EXPECT_EQ(result->knobs[1].knobId(), "SPLIT_K");
    EXPECT_EQ(std::get<int64_t>(result->knobs[1].value()), 2);
}

// Test 18: JSON entry with double knobs
TEST(TestEngineOverrideConfig, JsonWithDoubleKnobs)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ],
      "knobs": [
        { "name": "LEARNING_RATE", "type": "double", "value": 0.001 }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());

    ASSERT_EQ(result->knobs.size(), 1u);
    EXPECT_EQ(result->knobs[0].knobId(), "LEARNING_RATE");
    EXPECT_DOUBLE_EQ(std::get<double>(result->knobs[0].value()), 0.001);
}

// Test 19: JSON entry with string knobs
TEST(TestEngineOverrideConfig, JsonWithStringKnobs)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ],
      "knobs": [
        { "name": "ALGORITHM", "type": "string", "value": "gemm_v2" }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());

    ASSERT_EQ(result->knobs.size(), 1u);
    EXPECT_EQ(result->knobs[0].knobId(), "ALGORITHM");
    EXPECT_EQ(std::get<std::string>(result->knobs[0].value()), "gemm_v2");
}

// Test 20: JSON entry without knobs field → knobs vector is empty
TEST(TestEngineOverrideConfig, JsonWithoutKnobsFieldGivesEmptyKnobs)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->knobs.empty());
}

// Test 21: JSON entry with empty knobs array → knobs vector is empty
TEST(TestEngineOverrideConfig, JsonWithEmptyKnobsArray)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ],
      "knobs": []
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->knobs.empty());
}

// Test 22: Wildcard rule with knobs returns correct knobs
TEST(TestEngineOverrideConfig, WildcardRuleWithKnobs)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [-1, -1, -1, -1] }
      ],
      "knobs": [
        { "name": "TILE_SIZE", "type": "int", "value": 256 }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    // Any 4D tensor should match the wildcard
    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({8, 64, 56, 56})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->engineId, MIOPEN_ENGINE_ID);

    ASSERT_EQ(result->knobs.size(), 1u);
    EXPECT_EQ(result->knobs[0].knobId(), "TILE_SIZE");
    EXPECT_EQ(std::get<int64_t>(result->knobs[0].value()), 256);
}

// Test 23: JSON with mixed knob types
TEST(TestEngineOverrideConfig, JsonWithMixedKnobTypes)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ],
      "knobs": [
        { "name": "TILE_SIZE", "type": "int", "value": 128 },
        { "name": "SCALE", "type": "double", "value": 0.5 },
        { "name": "ALGO", "type": "string", "value": "implicit_gemm" }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());

    ASSERT_EQ(result->knobs.size(), 3u);
    EXPECT_EQ(result->knobs[0].knobId(), "TILE_SIZE");
    EXPECT_EQ(std::get<int64_t>(result->knobs[0].value()), 128);
    EXPECT_EQ(result->knobs[1].knobId(), "SCALE");
    EXPECT_DOUBLE_EQ(std::get<double>(result->knobs[1].value()), 0.5);
    EXPECT_EQ(result->knobs[2].knobId(), "ALGO");
    EXPECT_EQ(std::get<std::string>(result->knobs[2].value()), "implicit_gemm");
}

// Test 24: "integer" as type alias for "int"
TEST(TestEngineOverrideConfig, JsonKnobTypeIntegerAlias)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ],
      "knobs": [
        { "name": "TILE_SIZE", "type": "integer", "value": 64 }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());

    ASSERT_EQ(result->knobs.size(), 1u);
    EXPECT_EQ(std::get<int64_t>(result->knobs[0].value()), 64);
}

// Test 25: "float" as type alias for "double"
TEST(TestEngineOverrideConfig, JsonKnobTypeFloatAlias)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ],
      "knobs": [
        { "name": "RATE", "type": "float", "value": 1.5 }
      ]
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());

    ASSERT_EQ(result->knobs.size(), 1u);
    EXPECT_DOUBLE_EQ(std::get<double>(result->knobs[0].value()), 1.5);
}

// Test 26: knobs with autotune_metadata present (backward compat)
TEST(TestEngineOverrideConfig, JsonWithAutotuneMetadata)
{
    constexpr const char* CONTENTS = R"({
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224] }
      ],
      "knobs": [
        { "name": "TILE_SIZE", "type": "int", "value": 128 }
      ],
      "autotune_metadata": {
        "min_time_ms": 1.23,
        "rank": 0
      }
    }
  ]
})";

    auto config = EngineOverrideConfig::loadFromContent(CONTENTS);
    ASSERT_TRUE(config.has_value());

    // autotune_metadata should be ignored by the parser; knobs still work
    const std::vector<std::shared_ptr<TensorAttributes>> tensors = {makeTensor({1, 3, 224, 224})};
    auto result = config->matchOperation("conv_fprop", tensors);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->engineId, MIOPEN_ENGINE_ID);
    ASSERT_EQ(result->knobs.size(), 1u);
    EXPECT_EQ(result->knobs[0].knobId(), "TILE_SIZE");
}

#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB
