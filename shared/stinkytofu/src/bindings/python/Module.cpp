/* ************************************************************************
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
#include "stinkytofu/bindings/python/Module.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <unordered_map>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/pipeline/Backend.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"

namespace stinkytofu {
namespace {
/**
 * @brief Range of instructions that belongs to a group.
 * @note This is a helper struct to manage instruction groups.
 */
struct InstructionGroupRange {
    IntrusiveListIterator<IRBase> first; /**< The first instruction in the group */
    IntrusiveListIterator<IRBase> last;  /**< The last instruction in the group */

    InstructionGroupRange()
        : first(IntrusiveListIterator<IRBase>()), last(IntrusiveListIterator<IRBase>()) {}

    IntrusiveListIterator<IRBase> begin() const {
        return first;
    }

    IntrusiveListIterator<IRBase> end() const {
        return ++IntrusiveListIterator<IRBase>(last.getNodePtr());
    }
};

}  // anonymous namespace

struct StinkyAsmModule::Impl {
    std::string name;
    std::string outputName;  // If non-empty, used for cost file basename (full kernel name)
    std::string outputDir;   // If non-empty, cost file goes to outputDir/<kernel_name>/
    std::array<int, 3> arch;

    // This map maintains the defined group names and the range of instructions for each group.
    std::unordered_map<std::string, InstructionGroupRange> instructionGroups;

    // Multi-Function ownership. functions[0] is always the entry Function and is
    // returned by the no-arg getFunction(). Callees (created via createFunction)
    // are appended after it in insertion order. Function objects are stored via
    // unique_ptr to keep their address stable across vector growth (Function
    // members reference `this` for their BasicBlockList parent pointer).
    std::vector<std::unique_ptr<Function>> functions;

    // Total instruction encoding size in bytes (for .amdhsa_inst_pref_size). -1 if not set.
    int64_t totalInstructionBytes = -1;

    Impl(const std::string& name, const std::array<int, 3>& arch) : name(name), arch(arch) {
        // Entry Function is created with empty name (matches today's shape).
        // Callees added later via StinkyAsmModule::createFunction get a name.
        functions.emplace_back(std::make_unique<Function>());
        functions[0]->createBasicBlock("entry");
    }

    ~Impl() {
        // Function destructors clean up BasicBlocks and their IRLists, and the
        // IRBases owned by those IRLists.
    }
};

StinkyAsmModule::StinkyAsmModule(const std::string& name, const std::array<int, 3>& arch,
                                 const ModuleOptions& moduleOptions)
    : pImpl(std::make_unique<Impl>(name, arch)), moduleOptions(moduleOptions) {
    // SwPrefetchScratchSgpr == -1: do not run SwPrefetchInsertionPass (see Gfx1250Backend).
    this->moduleOptions.EnableSwPrefetchInsertion =
        (this->moduleOptions.SwPrefetchScratchSgpr != -1);
}

StinkyAsmModule::~StinkyAsmModule() = default;

StinkyAsmModule::StinkyAsmModule(StinkyAsmModule&&) noexcept = default;
StinkyAsmModule& StinkyAsmModule::operator=(StinkyAsmModule&&) noexcept = default;

std::string StinkyAsmModule::getName() const {
    return pImpl->name;
}

void StinkyAsmModule::setOutputName(const std::string& name) {
    pImpl->outputName = name;
}

std::string StinkyAsmModule::getOutputName() const {
    return pImpl->outputName;
}

void StinkyAsmModule::setOutputDir(const std::string& dir) {
    pImpl->outputDir = dir;
}

std::string StinkyAsmModule::getOutputDir() const {
    return pImpl->outputDir;
}

std::array<int, 3> StinkyAsmModule::getArch() const {
    return pImpl->arch;
}

Function& StinkyAsmModule::getFunction() {
    return *pImpl->functions[0];
}

const Function& StinkyAsmModule::getFunction() const {
    return *pImpl->functions[0];
}

Function& StinkyAsmModule::createFunction(const std::string& name, bool isCallee) {
    assert(!name.empty() && "createFunction requires a non-empty name");
    assert(getFunction(name) == nullptr &&
           "createFunction: Function name must be unique within this StinkyAsmModule");
    auto& f = pImpl->functions.emplace_back(std::make_unique<Function>(name));
    f->createBasicBlock("entry");
    f->setIsCallee(isCallee);
    return *f;
}

Function* StinkyAsmModule::getFunction(std::string_view name) {
    auto it = std::find_if(
        pImpl->functions.begin(), pImpl->functions.end(),
        [name](const std::unique_ptr<Function>& f) { return f && f->getName() == name; });
    return it == pImpl->functions.end() ? nullptr : it->get();
}

const Function* StinkyAsmModule::getFunction(std::string_view name) const {
    auto it = std::find_if(
        pImpl->functions.begin(), pImpl->functions.end(),
        [name](const std::unique_ptr<Function>& f) { return f && f->getName() == name; });
    return it == pImpl->functions.end() ? nullptr : it->get();
}

std::vector<Function*> StinkyAsmModule::getFunctions() {
    std::vector<Function*> out;
    out.reserve(pImpl->functions.size());
    for (auto& f : pImpl->functions) out.push_back(f.get());
    return out;
}

std::vector<const Function*> StinkyAsmModule::getFunctions() const {
    std::vector<const Function*> out;
    out.reserve(pImpl->functions.size());
    for (const auto& f : pImpl->functions) out.push_back(f.get());
    return out;
}

size_t StinkyAsmModule::numFunctions() const {
    return pImpl->functions.size();
}

std::string StinkyAsmModule::emitAssembly() const {
    // Configure the emitter with default options
    stinkytofu::AsmEmitterOptions options;
    options.emitComments = true;
    options.emitCycleInfo = false;
    options.indent = 0;
    options.emitBlankLines = false;
    options.useSymbolicNames = true;  // Enable symbolic register names

    stinkytofu::StinkyAsmEmitter emitter(options);
    return emitter.emit(getFunction());
}

void StinkyAsmModule::runOptimizationPipeline() {
    Backend backend(*this);
    backend.runOptimization();
}

std::optional<uint64_t> StinkyAsmModule::getMetaDataU64(const std::string& key) const {
    return getFunction().getMetaData(key);
}

void StinkyAsmModule::addGroup(const std::string& name) {
    if (pImpl->instructionGroups.find(name) != pImpl->instructionGroups.end()) {
        return;
    }
    pImpl->instructionGroups.emplace(name, InstructionGroupRange());
}

bool StinkyAsmModule::hasGroup(const std::string& name) const {
    return pImpl->instructionGroups.find(name) != pImpl->instructionGroups.end();
}

std::optional<std::pair<IntrusiveListIterator<IRBase>, IntrusiveListIterator<IRBase>>>
StinkyAsmModule::findGroupRange(const std::string& groupName) const {
    if (!hasGroup(groupName)) {
        return std::nullopt;
    }

    const auto& range = pImpl->instructionGroups.at(groupName);
    // Return nullopt if the group was registered but never populated
    if (range.first == IntrusiveListIterator<IRBase>()) {
        return std::nullopt;
    }

    return std::make_optional(std::make_pair(range.begin(), range.end()));
}

void StinkyAsmModule::updateInstructionGroups(const std::vector<const std::string*>& groups,
                                              size_t instsCountBefore) {
    for (const auto& groupName : groups) {
        if (hasGroup(*groupName)) {
            BasicBlock& bb = *getFunction().getEntryBlock();
            auto newInstructionCount = bb.size() - instsCountBefore;
            auto& groupRange = pImpl->instructionGroups.at(*groupName);
            if (groupRange.first == IntrusiveListIterator<IRBase>()) {
                auto it = bb.rbegin();
                for (size_t i = 1; i < newInstructionCount; ++i) {
                    it++;
                }
                groupRange.first = IntrusiveListIterator<IRBase>(it.getNodePtr());
            }

            groupRange.last = IntrusiveListIterator<IRBase>(bb.rbegin().getNodePtr());
        }
    }
}

void StinkyAsmModule::setGroupRange(const std::string& groupName,
                                    IntrusiveListIterator<IRBase> first,
                                    IntrusiveListIterator<IRBase> last) {
    if (!hasGroup(groupName)) return;
    auto& groupRange = pImpl->instructionGroups.at(groupName);
    groupRange.first = first;
    groupRange.last = last;
}

const StinkyAsmModule::ModuleOptions& StinkyAsmModule::getModuleOptions() const {
    return this->moduleOptions;
}

void StinkyAsmModule::setModuleOptions(const ModuleOptions& moduleOptions) {
    this->moduleOptions = moduleOptions;
}

void StinkyAsmModule::setTotalInstructionBytes(int64_t totalBytes) {
    pImpl->totalInstructionBytes = totalBytes;
}

int64_t StinkyAsmModule::getTotalInstructionBytes() const {
    return pImpl->totalInstructionBytes;
}

}  // namespace stinkytofu
