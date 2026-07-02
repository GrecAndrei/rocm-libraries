/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "UserDrivenTuningParser.hpp"
#include <optional>
#include <variant>

namespace TensileLite
{
    /**
     * @brief A tuning entry that can specify either a solution index or parameters.
     *
     * This is more flexible than the current OverrideMap which only stores
     * solution indices. Parameters (GSU, WGM) are preferred because they
     * survive library rebuilds (solution indices change when the library is
     * regenerated, but parameters are stable).
     */
    struct TuningEntry
    {
        // Exact match criteria
        ProblemOverride problem;

        // Either a specific solution index, or parameters to apply
        std::optional<int> solutionIndex;
        std::optional<int> gsu;
        std::optional<int> wgm;

        // Metadata for approximate matching
        bool        exactMatch = true;
        int         tolerancePct = 10;  // Allow +/- 10% size variation
        std::string description;         // For debugging
    };

    /**
     * @brief Enhanced tuning map with approximate search.
     *
     * The current OverrideMap only does exact match via multimap::equal_range.
     * This version tries exact match first, then falls back to nearest neighbor
     * within tolerance.
     */
    class QuickTuningMap
    {
    public:
        static QuickTuningMap& getMap()
        {
            static QuickTuningMap gInstance;
            return gInstance;
        }

        QuickTuningMap() {}
        ~QuickTuningMap() {}
        QuickTuningMap(const QuickTuningMap&) = delete;
        QuickTuningMap& operator=(const QuickTuningMap&) = delete;

        /**
         * @brief Load tuning data from a JSON or CSV file.
         */
        bool loadFromFile(const std::string& path);

        /**
         * @brief Find the best tuning entry for a given problem.
         *
         * Tries exact match first. If no exact match, finds the nearest
         * problem within tolerancePct.
         */
        std::optional<TuningEntry> findBestMatch(const ProblemOverride& prob) const;

        bool empty() const;
        void clear();
        void add(const TuningEntry& entry);

    private:
        std::vector<TuningEntry>      m_entries;
        mutable std::shared_timed_mutex m_mutex;

        bool loadJson(const std::string& path);
        bool loadCsv(const std::string& path);
        double problemDistance(const ProblemOverride& a, const ProblemOverride& b) const;
    };

    inline bool initQuickTuningFromEnv()
    {
        char* env = getenv("HIPBLASLT_QUICKTUNE_FILE");
        if(env && env[0] != '\0')
        {
            return QuickTuningMap::getMap().loadFromFile(env);
        }
        return false;
    }

} // namespace TensileLite
