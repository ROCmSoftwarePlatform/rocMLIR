//===- MIGraphX.cpp - C Interface for MIGraphX dialect
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Pipeline/Pipeline.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/RocmDeviceName.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TargetSelect.h"
#include <mutex>
#include <vector>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx,
                                      mlir::migraphx::MIGraphXDialect)

MlirTypeID rocmlirMIXRShapedTypeGetTypeId() {
  return wrap(mlir::migraphx::MIXRShapedType::getTypeID());
}

bool rocmlirIsAMIXRShapedType(MlirType type) {
  return llvm::isa<mlir::migraphx::MIXRShapedType>(unwrap(type));
}

MlirType rocmlirMIXRShapedTypeGet(intptr_t rank, const int64_t *shape,
                                  const int64_t *strides,
                                  MlirType elementType) {
  return wrap(mlir::migraphx::MIXRShapedType::get(
      llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      llvm::ArrayRef(strides, static_cast<size_t>(rank)), unwrap(elementType)));
}

MlirType rocmlirMIXRShapedTypeAsTensor(MlirType type) {
  return wrap(
      llvm::cast<mlir::migraphx::MIXRShapedType>(unwrap(type)).asTensor());
}

// Returns the required buffer size if called with null buffer
// and fill information in the passed ptr when provided.
MLIR_CAPI_EXPORTED
void mlirGetKernelInfo(MlirModule module, int *size, void *data) {
  auto mod = unwrap(module);
  int argNum = 0;
  int argIdx = 0;
  llvm::StringRef kernelName;

  // Either of pointers should be provided.
  assert((size != nullptr || data != nullptr) &&
         "Either size or data pointer should be provided");
  std::vector<int> info;
  mod.walk([&](mlir::func::FuncOp f) {
    auto args = f.getArguments();
    for (auto arg : args) {
      argNum++;
      auto sType = mlir::cast<mlir::ShapedType>(arg.getType());
      auto rank = sType.getRank();
      info.push_back(rank);
      for (int i = 0; i < rank; i++)
        info.push_back(sType.getDimSize(i));
      argIdx += rank;
    }
    kernelName = f.getName();
  });
  if (data == nullptr && size != nullptr) {
    *size = (1 + argNum + argIdx) * sizeof(int) + kernelName.size();
  } else if (data != nullptr) {
    int argSize = argNum + argIdx;
    int *argData = (int *)data;
    argData[0] = argNum;
    for (int i = 0; i < argSize; i++)
      argData[i + 1] = info[i];
    char *nameData = (char *)(argData + argSize + 1);
    for (size_t i = 0, e = kernelName.size(); i < e; ++i) {
      nameData[i] = kernelName[i];
    }
  }
}

// Returns block_size and grid_size as uint32_t[2]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs) {
  auto mod = unwrap(module);
  size_t count = 0;
  mod.walk([&](mlir::gpu::BinaryOp binary) {
    mlir::gpu::KernelTableAttr metadata =
        mlir::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]).getKernels();
    for (auto kernel : metadata) {
      auto block = kernel.getAttr<mlir::IntegerAttr>("block_size");
      auto grid = kernel.getAttr<mlir::IntegerAttr>("grid_size");
      if (!block || !grid)
        continue;
      attrs[0] = block.getInt();
      attrs[1] = grid.getInt();
      ++count;
    }
  });
  assert(count == 1 && "invalid number of kernels");
}

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, size_t *size,
                                      char *bin) {
  bool success = false;
  auto mod = unwrap(module);
  if (bin == nullptr && size == nullptr)
    return success;
  mod.walk([&](mlir::gpu::BinaryOp binary) {
    auto object = llvm::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]);
    if (bin != nullptr) { // return binary regardless the presence of *size
      llvm::StringRef hsaco = object.getObject().getValue();
      std::copy(hsaco.begin(), hsaco.end(), bin);
      success = true;
    } else {
      *size = object.getObject().getValue().size();
      success = true;
    }
  });
  return success;
}

// pipelines

MLIR_CAPI_EXPORTED
void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm) {
  auto passMan = unwrap(pm);
  if (failed(applyPassManagerCLOptions(*passMan)))
    llvm::errs() << "Failed to apply command-line options.\n";
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::migraphx::addHighLevelPipeline(*passMan);
  mlir::rock::buildBufferizePipeline(*passMan);
}

MLIR_CAPI_EXPORTED void
mlirMIGraphXAddApplicabilityPipeline(MlirPassManager pm) {
  auto *passMan = unwrap(pm);
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::rock::KernelOptions opts;
  opts.enableApplicability = true;
  // This is the default, but we set it paranoidly.
  opts.tuningFallback = false;
  mlir::rock::buildKernelPipeline(*passMan, opts);
}

MLIR_CAPI_EXPORTED bool mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                                                       const char *arch) {
  auto *passMan = unwrap(pm);
  if (failed(applyPassManagerCLOptions(*passMan)))
    return false;
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::rock::KernelOptions kOpts;
  kOpts.tuningFallback = false;
  mlir::rock::buildKernelPipeline(*passMan, kOpts);
  llvm::StringRef archStr(arch);
  mlir::RocmDeviceName devName;
  if (archStr.empty() || mlir::failed(devName.parse(archStr))) {
    llvm::errs() << "Invalid architecture: " << archStr << "\n";
    return false;
  }
  mlir::rock::BackendOptions opts;
  opts.triple = devName.getTriple().str();
  opts.chip = devName.getChip().str();
  opts.features = devName.getFeaturesForBackend();
  opts.optLevel = 3;
  mlir::rock::buildBackendPipeline(*passMan, opts);

  return true;
}
