/* ************************************************************************
 *
 * QuickTuning Test - Compile and run to verify JSON loading and approximate matching
 *
 * ************************************************************************ */

#include "QuickTuning.hpp"
#include <iostream>
#include <cassert>

using namespace TensileLite;

int main()
{
    std::cout << "=== QuickTuning Test ===" << std::endl;

    // Test 1: Load the gfx1032 tuning JSON
    QuickTuningMap& map = QuickTuningMap::getMap();
    bool loaded = map.loadFromFile("/home/alex/TheRock/rocm-libraries/projects/hipblaslt/library/src/amd_detail/rocblaslt/src/tuning_gfx1032.json");
    std::cout << "Load JSON: " << (loaded ? "PASS" : "FAIL") << std::endl;
    if(!loaded) return 1;

    std::cout << "Entries loaded: " << (!map.empty() ? "PASS" : "FAIL") << std::endl;

    // Test 2: Exact match for 4096x4096x4096 fp16 NN
    {
        ProblemOverride exact(true, true, rocisa::DataType::Half, rocisa::DataType::Half,
                              rocisa::DataType::Half, rocisa::DataType::Half, 4096, 4096, 4096, 1);
        auto result = map.findBestMatch(exact);
        std::cout << "Exact match: " << (result.has_value() ? "PASS" : "FAIL") << std::endl;
        if(result.has_value())
        {
            std::cout << "  GSU=" << result->gsu.value_or(-1) << " WGM=" << result->wgm.value_or(-1) << std::endl;
        }
    }

    // Test 3: Approximate match - 4000x4000x4000 (within 10% of 4096)
    {
        ProblemOverride approx(true, true, rocisa::DataType::Half, rocisa::DataType::Half,
                               rocisa::DataType::Half, rocisa::DataType::Half, 4000, 4000, 4000, 1);
        auto result = map.findBestMatch(approx);
        std::cout << "Approx match (4000): " << (result.has_value() ? "PASS" : "FAIL") << std::endl;
        if(result.has_value())
        {
            std::cout << "  exactMatch=" << result->exactMatch << std::endl;
        }
    }

    // Test 4: Too far - should fail (2000 is > 15% from 4096)
    {
        ProblemOverride faroff(true, true, rocisa::DataType::Half, rocisa::DataType::Half,
                               rocisa::DataType::Half, rocisa::DataType::Half, 2000, 2000, 2000, 1);
        auto result = map.findBestMatch(faroff);
        std::cout << "Far match (2000): " << (!result.has_value() ? "PASS (correctly rejected)" : "FAIL") << std::endl;
    }

    // Test 5: Type mismatch should fail
    {
        ProblemOverride wrongType(true, true, rocisa::DataType::Float, rocisa::DataType::Float,
                                  rocisa::DataType::Float, rocisa::DataType::Float, 4096, 4096, 4096, 1);
        auto result = map.findBestMatch(wrongType);
        std::cout << "Type mismatch: " << (!result.has_value() ? "PASS (correctly rejected)" : "FAIL") << std::endl;
    }

    std::cout << "=== All tests completed ===" << std::endl;
    return 0;
}
