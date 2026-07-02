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

#include "QuickTuning.hpp"
#include "Debug.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

// nlohmann/json is already available via msgpack in this project
#include <nlohmann/json.hpp>

namespace TensileLite
{
    static rocisa::DataType stringToDataType(const std::string& str)
    {
        if(str == "fp32" || str == "float" || str == "Float")
            return rocisa::DataType::Float;
        if(str == "fp64" || str == "double" || str == "Double")
            return rocisa::DataType::Double;
        if(str == "fp16" || str == "half" || str == "Half")
            return rocisa::DataType::Half;
        if(str == "bf16" || str == "bfloat16" || str == "BFloat16")
            return rocisa::DataType::BFloat16;
        if(str == "int8" || str == "Int8")
            return rocisa::DataType::Int8;
        if(str == "int32" || str == "Int32")
            return rocisa::DataType::Int32;
        if(str == "fp8" || str == "Float8")
            return rocisa::DataType::Float8;
        if(str == "bf8" || str == "BFloat8")
            return rocisa::DataType::BFloat8;
        return rocisa::DataType::None;
    }

    bool QuickTuningMap::loadFromFile(const std::string& path)
    {
        if(path.length() > 5 && path.substr(path.length() - 5) == ".json")
        {
            return loadJson(path);
        }
        else
        {
            return loadCsv(path);
        }
    }

    bool QuickTuningMap::loadJson(const std::string& path)
    {
        std::ifstream file(path);
        if(!file.is_open())
        {
            log_error(__func__, "Could not open tuning file:", path.c_str());
            return false;
        }

        try
        {
            nlohmann::json j;
            file >> j;

            if(!j.contains("entries"))
            {
                log_error(__func__, "JSON tuning file missing entries array");
                return false;
            }

            int globalTolerance = 10;
            if(j.contains("tolerance_pct"))
                globalTolerance = j["tolerance_pct"].get<int>();

            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            for(const auto& entry : j["entries"])
            {
                TuningEntry te;
                te.tolerancePct = globalTolerance;

                if(entry.contains("tolerance_pct"))
                    te.tolerancePct = entry["tolerance_pct"].get<int>();

                const auto& match = entry["match"];

                bool transA = true;
                bool transB = true;
                if(match.contains("transA"))
                    transA = (match["transA"].get<std::string>() != "N");
                if(match.contains("transB"))
                    transB = (match["transB"].get<std::string>() != "N");

                size_t m = match.value("m", 0);
                size_t n = match.value("n", 0);
                size_t k = match.value("k", 0);
                size_t batch = match.value("batch", 1);

                rocisa::DataType aType = rocisa::DataType::Float;
                rocisa::DataType bType = rocisa::DataType::Float;
                rocisa::DataType cType = rocisa::DataType::Float;
                rocisa::DataType compType = rocisa::DataType::Float;

                if(match.contains("a_type"))
                    aType = stringToDataType(match["a_type"].get<std::string>());
                if(match.contains("b_type"))
                    bType = stringToDataType(match["b_type"].get<std::string>());
                if(match.contains("c_type"))
                    cType = stringToDataType(match["c_type"].get<std::string>());
                if(match.contains("compute_type"))
                    compType = stringToDataType(match["compute_type"].get<std::string>());

                te.problem = ProblemOverride(transA, transB, aType, bType, compType, cType, m, n, k, batch);

                if(entry.contains("solution_index"))
                {
                    te.solutionIndex = entry["solution_index"].get<int>();
                }

                if(entry.contains("params"))
                {
                    const auto& params = entry["params"];
                    if(params.contains("gsu"))
                        te.gsu = params["gsu"].get<int>();
                    if(params.contains("wgm"))
                        te.wgm = params["wgm"].get<int>();
                }

                if(entry.contains("description"))
                    te.description = entry["description"].get<std::string>();

                m_entries.push_back(te);
            }

            log_info(__func__, "Loaded", m_entries.size(), "tuning entries from JSON");
            return true;
        }
        catch(const std::exception& e)
        {
            log_error(__func__, "JSON parse error:", e.what());
            return false;
        }
    }

    bool QuickTuningMap::loadCsv(const std::string& path)
    {
        getContractionProblemsFromFile(path);
        OverrideMap& oldMap = OverrideMap::getMap();

        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

        log_info(__func__, "CSV loaded via legacy parser; consider migrating to JSON for approximate matching");
        return true;
    }

    std::optional<TuningEntry> QuickTuningMap::findBestMatch(const ProblemOverride& prob) const
    {
        std::shared_lock<std::shared_timed_mutex> lock(m_mutex);

        if(m_entries.empty())
            return std::nullopt;

        for(const auto& entry : m_entries)
        {
            if(entry.problem == prob)
                return entry;
        }

        const TuningEntry* bestEntry = nullptr;
        double bestDist = std::numeric_limits<double>::max();

        for(const auto& entry : m_entries)
        {
            if(entry.problem.inputTypeA() != prob.inputTypeA()
               || entry.problem.inputTypeB() != prob.inputTypeB()
               || entry.problem.computeType() != prob.computeType()
               || entry.problem.outputType() != prob.outputType())
                continue;

            if(entry.problem.transA() != prob.transA()
               || entry.problem.transB() != prob.transB())
                continue;

            double tolerance = entry.tolerancePct / 100.0;

            bool withinTolerance = true;
            if(prob.m() > 0 && entry.problem.m() > 0)
            {
                double relDiff = std::abs((double)entry.problem.m() - (double)prob.m()) / prob.m();
                if(relDiff > tolerance)
                    withinTolerance = false;
            }
            if(prob.n() > 0 && entry.problem.n() > 0)
            {
                double relDiff = std::abs((double)entry.problem.n() - (double)prob.n()) / prob.n();
                if(relDiff > tolerance)
                    withinTolerance = false;
            }
            if(prob.k() > 0 && entry.problem.k() > 0)
            {
                double relDiff = std::abs((double)entry.problem.k() - (double)prob.k()) / prob.k();
                if(relDiff > tolerance)
                    withinTolerance = false;
            }

            if(!withinTolerance)
                continue;

            double dist = problemDistance(entry.problem, prob);
            if(dist < bestDist)
            {
                bestDist = dist;
                bestEntry = &entry;
            }
        }

        if(bestEntry)
        {
            TuningEntry result = *bestEntry;
            result.exactMatch = false;
            return result;
        }

        return std::nullopt;
    }

    double QuickTuningMap::problemDistance(const ProblemOverride& a, const ProblemOverride& b) const
    {
        double mDist = 0.0;
        double nDist = 0.0;
        double kDist = 0.0;

        if(a.m() > 0 && b.m() > 0)
            mDist = std::log2(std::max((double)a.m(), (double)b.m()) / std::min((double)a.m(), (double)b.m()));
        if(a.n() > 0 && b.n() > 0)
            nDist = std::log2(std::max((double)a.n(), (double)b.n()) / std::min((double)a.n(), (double)b.n()));
        if(a.k() > 0 && b.k() > 0)
            kDist = std::log2(std::max((double)a.k(), (double)b.k()) / std::min((double)a.k(), (double)b.k()));

        return std::sqrt(mDist * mDist + nDist * nDist + kDist * kDist);
    }

    bool QuickTuningMap::empty() const
    {
        std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
        return m_entries.empty();
    }

    void QuickTuningMap::clear()
    {
        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
        m_entries.clear();
    }

    void QuickTuningMap::add(const TuningEntry& entry)
    {
        std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
        m_entries.push_back(entry);
    }

} // namespace TensileLite
