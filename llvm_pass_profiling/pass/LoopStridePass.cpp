//===-- LoopStridePass.cpp - report memory stride per loop ---------------===//
//
// Out-of-tree analysis pass for the new pass manager. For every natural loop
// it walks the loads/stores and asks ScalarEvolution for the pointer's
// evolution. Accesses whose address is an add-recurrence {base,+,step} in the
// loop have a compile-time-known stride; everything else is either
// loop-invariant or irregular (gather/scatter).
//
// The per-loop verdict mirrors the first question the loop vectoriser asks:
// are the memory accesses affine and contiguous? Running this next to
// -Rpass-missed=loop-vectorize shows *why* a loop was or wasn't vectorised,
// from the IR's point of view rather than the source's.
//
// Usage:
//   clang -O2 -fpass-plugin=libLoopStridePass.{so,dylib} file.c   (auto-runs)
//   opt -load-pass-plugin=... -passes='function(loop-stride)' file.ll
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
// Moved from llvm/Passes/ to llvm/Plugins/ in LLVM 22.
#if __has_include("llvm/Plugins/PassPlugin.h")
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct LoopStridePass : PassInfoMixin<LoopStridePass> {

  static void classifyAccess(Instruction &I, Loop *L, ScalarEvolution &SE,
                             const DataLayout &DL, unsigned &Affine,
                             unsigned &Unit, unsigned &Total) {
    Value *Ptr = nullptr;
    Type *AccTy = nullptr;
    if (auto *LD = dyn_cast<LoadInst>(&I)) {
      Ptr = LD->getPointerOperand();
      AccTy = LD->getType();
    } else if (auto *ST = dyn_cast<StoreInst>(&I)) {
      Ptr = ST->getPointerOperand();
      AccTy = ST->getValueOperand()->getType();
    } else {
      return;
    }

    Total++;
    uint64_t ElemBytes = DL.getTypeStoreSize(AccTy);
    const char *Kind = isa<LoadInst>(I) ? "load " : "store";

    const SCEV *PtrScev = SE.getSCEV(Ptr);
    const auto *AR = dyn_cast<SCEVAddRecExpr>(PtrScev);

    if (!AR || AR->getLoop() != L) {
      if (SE.isLoopInvariant(PtrScev, L))
        errs() << "    " << Kind << " " << ElemBytes
               << " B: loop-invariant address\n";
      else
        errs() << "    " << Kind << " " << ElemBytes
               << " B: irregular (non-affine, gather/scatter)\n";
      return;
    }

    Affine++;
    const SCEV *Step = AR->getStepRecurrence(SE);
    if (const auto *C = dyn_cast<SCEVConstant>(Step)) {
      int64_t StrideBytes = C->getAPInt().getSExtValue();
      if ((uint64_t)std::abs(StrideBytes) == ElemBytes) {
        Unit++;
        errs() << "    " << Kind << " " << ElemBytes << " B: unit stride ("
               << StrideBytes << " B/iter)\n";
      } else {
        errs() << "    " << Kind << " " << ElemBytes << " B: strided ("
               << StrideBytes << " B/iter = " << StrideBytes / (int64_t)ElemBytes
               << " elems)\n";
      }
    } else {
      errs() << "    " << Kind << " " << ElemBytes
             << " B: affine, stride unknown at compile time\n";
    }
  }

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    auto &LI = FAM.getResult<LoopAnalysis>(F);
    auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
    const DataLayout &DL = F.getParent()->getDataLayout();

    SmallVector<Loop *, 8> Loops = LI.getLoopsInPreorder();
    if (Loops.empty())
      return PreservedAnalyses::all();

    errs() << "[loop-stride] function '" << F.getName() << "'\n";
    for (Loop *L : Loops) {
      errs() << "  loop";
      if (DebugLoc DLoc = L->getStartLoc())
        errs() << " at line " << DLoc.getLine();
      errs() << " (depth " << L->getLoopDepth() << ")\n";

      unsigned Affine = 0, Unit = 0, Total = 0;
      for (BasicBlock *BB : L->getBlocks()) {
        // Attribute each access to its innermost loop only.
        if (LI.getLoopFor(BB) != L)
          continue;
        for (Instruction &I : *BB)
          classifyAccess(I, L, SE, DL, Affine, Unit, Total);
      }

      if (Total == 0) {
        errs() << "    (no memory accesses at this depth)\n";
        continue;
      }
      errs() << "    verdict: " << Affine << "/" << Total << " affine, "
             << Unit << "/" << Total << " unit-stride -> ";
      if (Unit == Total)
        errs() << "contiguous, vectorisation-friendly\n";
      else if (Affine == Total)
        errs() << "affine but strided, expect poor cache-line utilisation\n";
      else
        errs() << "contains gather/scatter, hard to vectorise\n";
    }
    return PreservedAnalyses::all();
  }

  // Run even on optnone functions so -O0 IR can be inspected too.
  static bool isRequired() { return true; }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopStridePass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            // `opt -passes='function(loop-stride)'`
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "loop-stride") {
                    FPM.addPass(LoopStridePass());
                    return true;
                  }
                  return false;
                });
            // Auto-run under `clang -O1/-O2/-O3 -fpass-plugin=...`, after the
            // simplification pipeline so loops are rotated and SCEV is usable.
            PB.registerOptimizerEarlyEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel,
                   ThinOrFullLTOPhase) {
                  MPM.addPass(
                      createModuleToFunctionPassAdaptor(LoopStridePass()));
                });
          }};
}
