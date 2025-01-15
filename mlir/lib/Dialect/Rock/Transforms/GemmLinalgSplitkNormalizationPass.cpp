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
// arith.addf/arith.subf gemmOut, other to arith.addf gemmOut, other/splitkFactor.
//
//===-----------------------------------------------------===//
#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <memory>
#include <sstream>

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
    : public rock::impl::RockGemmLinalgSplitkNormalizationPassBase<RockGemmLinalgSplitkNormalizationPass> {
  void runOnOperation() override;
};
} // end namespace

static LogicalResult divideAddBySplitkFactor(linalg::GenericOp genericOp, Value gemmResult, int64_t splitKFactor, IRRewriter &b) {
  SmallVector<std::tuple<Operation*, int>> adds;
  if(failed(checkValidOutputFusion(genericOp, gemmResult, adds)))
    return failure();

  for(auto [op, index] : adds) {
    assert(index == 0 || index == 1);
    LLVM_DEBUG(llvm::dbgs() << "Op to modify: "<<op<<"\n");
    b.setInsertionPoint(op);
    Value gemmOut = op->getOperand(index);
    Value otherValue = (index == 0) ? op->getOperand(1) : op->getOperand(0);
    auto splitKFactorValue = createConstantFloatOp(b, op->getLoc(), otherValue.getType(), otherValue.getType(), static_cast<float>(splitKFactor), otherValue.getType().isF32() ? APFloat::opOK : APFloat::opInexact);
    Value otherBySplitk = b.createOrFold<arith::DivFOp>(op->getLoc(), otherValue, splitKFactorValue);
    if(isa<arith::AddFOp>(op)) {
      b.replaceOpWithNewOp<arith::AddFOp>(
        op, gemmOut, otherBySplitk);
    } else if(isa<arith::SubFOp>(op)) {
      if(index == 0)
        b.replaceOpWithNewOp<arith::SubFOp>(
          op, gemmOut, otherBySplitk);
      else
        b.replaceOpWithNewOp<arith::SubFOp>(
          op, otherBySplitk, gemmOut);
    } else {
      return failure();
    }
  }
  return success();
}

static LogicalResult rewriteLinalgForSplitK(func::FuncOp &func, BufferDependencyAnalysis &bufferDeps) {
  IRRewriter rewriter(func->getContext());
  SmallVector<linalg::GenericOp> genericOps;
  func.walk([&genericOps](linalg::GenericOp genericOp) {genericOps.push_back(genericOp);});
  const auto &writersTable = bufferDeps.getWritersTable();

  for(linalg::GenericOp op : genericOps) {
    SmallVector<Value> operands;
    SmallVector<int64_t> splitKFactors;
    for(auto operand : op->getOperands()) {
      if(auto alloc = operand.getDefiningOp<memref::AllocOp>()) {
        if (writersTable.contains(alloc)) {
          for (OpOperand *op : writersTable.at(alloc)) {
            if (auto gemm = dyn_cast<GemmOp>(op->getOwner())) {
              const int64_t splitKFactor = gemm.getParams()->getSplitKFactor();
              if(splitKFactor > 1) {
                operands.push_back(operand);
                splitKFactors.push_back(splitKFactor);
              }
            }
          }
        }
      }
    }
    assert(operands.empty() || operands.size() == 1);
    assert(operands.size() == splitKFactors.size());
    if(operands.size() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "Found linalg::GenericOp that reads GEMM output, let's modify it if it has addf and/or subf\n");
      if(failed(divideAddBySplitkFactor(op, operands[0], splitKFactors[0], rewriter)))
        return failure();
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
