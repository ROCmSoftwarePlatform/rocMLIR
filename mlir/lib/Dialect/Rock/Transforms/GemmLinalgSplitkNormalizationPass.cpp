//===- GemmLinalgSplitkNormalizationPass.cpp ------------===//
//
// Copyright 2025 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
//
// This pass modifies linalg.generic for split-k fusions. It converts any
// arith.addf/arith.subf gemmOut, other to arith.addf gemmOut,
// other/splitkFactor.
//
//===-----------------------------------------------------===//
#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGEMMLINALGSPLITKNORMALIZATIONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gemm-linalg-splitk-normalization"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockGemmLinalgSplitkNormalizationPass
    : public rock::impl::RockGemmLinalgSplitkNormalizationPassBase<
          RockGemmLinalgSplitkNormalizationPass> {
  void runOnOperation() override;
};
} // end namespace

static LogicalResult divideAddBySplitkFactor(linalg::GenericOp genericOp,
                                             Value gemmResult,
                                             int64_t splitKFactor,
                                             IRRewriter &b) {
  SmallVector<std::tuple<Operation *, int>> adds;
  if (failed(checkValidOutputFusion(genericOp, gemmResult, adds)))
    return failure();

  for (auto [arithOp, gemmOutIndex] : adds) {
    assert(gemmOutIndex == 0 || gemmOutIndex == 1);
    LLVM_DEBUG(llvm::dbgs() << "Op to modify: " << arithOp << "\n");
    b.setInsertionPoint(arithOp);
    Value gemmOut = arithOp->getOperand(gemmOutIndex);
    Value otherValue = (gemmOutIndex == 0) ? arithOp->getOperand(1) : arithOp->getOperand(0);
    auto splitKFactorValue = createConstantFloatOp(
        b, arithOp->getLoc(), otherValue.getType(), otherValue.getType(),
        static_cast<float>(splitKFactor));
    Value otherBySplitk = b.createOrFold<arith::DivFOp>(
        arithOp->getLoc(), otherValue, splitKFactorValue);
    if (isa<arith::AddFOp>(arithOp)) {
      b.replaceOpWithNewOp<arith::AddFOp>(arithOp, gemmOut, otherBySplitk);
    } else if (isa<arith::SubFOp>(arithOp)) {
      if (gemmOutIndex == 0)
        b.replaceOpWithNewOp<arith::SubFOp>(arithOp, gemmOut, otherBySplitk);
      else
        b.replaceOpWithNewOp<arith::SubFOp>(arithOp, otherBySplitk, gemmOut);
    } else {
      return failure();
    }
  }
  return success();
}

static LogicalResult
rewriteLinalgForSplitK(func::FuncOp &func,
                       BufferDependencyAnalysis &bufferDeps) {
  IRRewriter rewriter(func->getContext());
  SmallVector<linalg::GenericOp> genericOps;
  func.walk([&genericOps](linalg::GenericOp genericOp) {
    genericOps.push_back(genericOp);
  });
  const auto &writersTable = bufferDeps.getWritersTable();
  LLVM_DEBUG(llvm::dbgs() << "Found " << genericOps.size()
                          << " linalg::GenericOp\n");

  for (linalg::GenericOp op : genericOps) {
    SmallVector<Value> gemmOut;
    SmallVector<int64_t> splitKFactors;
    for (auto operand : op->getOperands()) {
      auto genericOpInputAlloc = findMemrefAlloc(operand);
      if (succeeded(genericOpInputAlloc)) {
        if (writersTable.contains(genericOpInputAlloc.value())) {
          for (OpOperand *op : writersTable.at(genericOpInputAlloc.value())) {
            if (auto gemm = dyn_cast<GemmOp>(op->getOwner())) {
              const int64_t splitKFactor = gemm.getParams()->getSplitKFactor();
              LLVM_DEBUG(llvm::dbgs()
                         << "splitKFactor=" << splitKFactor << "\n");
              if (splitKFactor > 1) {
                gemmOut.push_back(genericOpInputAlloc.value());
                splitKFactors.push_back(splitKFactor);
              }
            }
          }
        }
      }
    }
    assert(gemmOut.empty() || gemmOut.size() == 1);
    assert(gemmOut.size() == splitKFactors.size());
    if (gemmOut.size() == 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Found linalg::GenericOp that reads GEMM output, let's "
                    "modify it if it has addf and/or subf\n");
      if (failed(divideAddBySplitkFactor(op, gemmOut[0], splitKFactors[0],
                                         rewriter)))
        return failure();
    } else {
      LLVM_DEBUG(llvm::dbgs() << "We didn't find any linalg::GenericOp that needs to be modified\n");
    }
  }
  return success();
}

void RockGemmLinalgSplitkNormalizationPass::runOnOperation() {
  func::FuncOp func = getOperation();
  BufferDependencyAnalysis &bufferDeps =
      getAnalysis<BufferDependencyAnalysis>();

  if (failed(rewriteLinalgForSplitK(func, bufferDeps))) {
    return signalPassFailure();
  }
} // namespace
