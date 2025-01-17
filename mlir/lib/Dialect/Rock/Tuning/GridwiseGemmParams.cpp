#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/MfmaInsnGroup.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/IR/WmmaInsnGroup.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

llvm::raw_ostream &mlir::rock::operator<<(llvm::raw_ostream &os,
                                          GemmDimension dim) {
  switch (dim) {
  case GemmDimension::G:
    return os << "GemmDimmension::G";
  case GemmDimension::K:
    return os << "GemmDimension::K";
  case GemmDimension::MorN:
    return os << "GemmDimension::MorN";
  }
  return os;
}

/// Non-xdlops
// clang-format off
#define NonAccel_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef NonAccel_DEFINITIONS_GEN
// clang-format on

PopulateParamsInfo PopulateParamsInfo::fromOp(RockGemmWrapperInterface op) {
  PopulateParamsInfo info{op.getGemmSize(), op.getArch(),  op.getGemmFeatures(),
                          op.getAType(),    op.getBType(), op.getKernelType()};

  if (auto convOp = dyn_cast<ConvBwdWeightOp>(*op)) {
    auto convDims = ConvolutionDims::fromOp(op);
    info.numCu = convOp.getNumCU();
    info.batchSize = convDims.n;
  }
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  WalkResult wRes = func.walk(
      [&](ReduceOp rOp) -> WalkResult { return WalkResult::interrupt(); });
  info.hasFusedReduction = wRes.wasInterrupted();
  return info;
}

std::optional<GemmSize> mlir::rock::calculatePadding(int64_t kPerBlock,
                                                     int64_t mPerBlock,
                                                     int64_t nPerBlock,
                                                     const GemmSize &gemmSize,
                                                     int64_t kPack) {
  int64_t kExtra = (kPerBlock * kPack) -
                   math_util::mod_1_to_n(gemmSize.k, kPerBlock * kPack);
  int64_t mExtra = mPerBlock - math_util::mod_1_to_n(gemmSize.m, mPerBlock);
  int64_t nExtra = nPerBlock - math_util::mod_1_to_n(gemmSize.n, nPerBlock);
  if (mExtra == 0 && kExtra == 0 && nExtra == 0)
    return std::nullopt;
  return GemmSize(0, mExtra, kExtra, nExtra);
}

GemmSize mlir::rock::calculatePaddedGemmSize(const InitParams &params,
                                             GemmSize gemmSize, int64_t kPack) {
  auto gemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize, kPack);

  if (gemmExtraPad.has_value()) {
    gemmSize.m += gemmExtraPad->m;
    gemmSize.k += gemmExtraPad->k;
    gemmSize.n += gemmExtraPad->n;
  }
  return gemmSize;
}

std::optional<GemmSize> mlir::rock::requiredPadding(Attribute params,
                                                    GemmSize gemmSize) {
  int64_t kPerBlock, mPerBlock, nPerBlock;
  int64_t kPack = 1;
  if (auto generalParams = dyn_cast<GeneralGemmParamsAttr>(params)) {
    kPerBlock = generalParams.getKPerBlock();
    mPerBlock = generalParams.getMPerBlock();
    nPerBlock = generalParams.getNPerBlock();
  } else if (auto accelParams =
                 dyn_cast<RockAccelTuningParamAttrInterface>(params)) {
    kPerBlock = accelParams.getKpackPerBlock();
    mPerBlock = accelParams.getMPerBlock();
    nPerBlock = accelParams.getNPerBlock();
    kPack = accelParams.getKpack();
  } else {
    llvm_unreachable("The tuning paramaters are general or xdlops");
  }
  return calculatePadding(kPerBlock, mPerBlock, nPerBlock, gemmSize, kPack);
}

int64_t mlir::rock::obtainBlockSize(int64_t waveSize, int64_t mPerBlock,
                                    int64_t nPerBlock, int64_t mPerWave,
                                    int64_t nPerWave) {
  return waveSize * (mPerBlock / mPerWave) * (nPerBlock / nPerWave);
}

int64_t mlir::rock::obtainBlockSize(int64_t waveSize,
                                    RockAccelTuningParamAttrInterface params) {
  return obtainBlockSize(waveSize, params.getMPerBlock(), params.getNPerBlock(),
                         params.getMPerWave(), params.getNPerWave());
}

LogicalResult PopulateParams::calculateBlockGemmPerformanceParameters(
    const InitParamsNonAccel &param) {

  FailureOr<GeneralGemmBlockStructure> maybeDerived =
      deriveGeneralGemmBlockStructure(param.blockSize);
  if (failed(maybeDerived))
    return failure();
  GeneralGemmBlockStructure derived = *maybeDerived;

  if (!(param.gemmMPerThread >= 2 && param.gemmMPerThread <= 4))
    return failure();

  if (!(param.gemmNPerThread >= 2 && param.gemmNPerThread <= 4))
    return failure();

  if (!(param.gemmMPerBlock % param.gemmMPerThread == 0 &&
        param.gemmNPerBlock % param.gemmNPerThread == 0))
    return failure();

  int64_t threadGemmMPerCluster = param.gemmMPerThread *
                                  derived.mThreadsPerCuwave *
                                  derived.mCuwavesPerBlock;
  int64_t threadGemmNPerCluster = param.gemmNPerThread *
                                  derived.nThreadsPerCuwave *
                                  derived.nCuwavesPerBlock;

  if ((param.gemmMPerBlock % threadGemmMPerCluster != 0) ||
      (param.gemmNPerBlock % threadGemmNPerCluster != 0)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "M per block or N per block aren't divisible by M/N per cluster\n");
    return failure();
  }

  return success();
}

LogicalResult
PopulateParams::populateDerived(const InitParamsNonAccel &params) {
  LogicalResult res = calculateBlockGemmPerformanceParameters(params);

  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Incoherent blockGemm tuning parameter "
                            << " size.\n");
    return failure();
  }

  return success();
}

Attribute
PopulateParams::getGemmParamsAttr(OpBuilder &b,
                                  const InitParamsNonAccel &params) const {
  return b.getAttr<GeneralGemmParamsAttr>(
      params.blockSize, params.gemmKPerBlock, params.gemmMPerBlock,
      params.gemmNPerBlock,
      /*kPerThread=*/1, params.gemmMPerThread, params.gemmNPerThread,
      /*kpack=*/1, params.splitKFactor);
}

LogicalResult
PopulateParams::paramsProbablyValid(OpBuilder &b,
                                    const PopulateParamsInfo &info,
                                    const InitParamsNonAccel &params) {
  return populateDerived(params);
}

static LogicalResult couldFusedReductionBePerformant(const GemmSize &gemmSize,
                                                     int64_t mPerBlock,
                                                     int64_t nPerBlock) {
  // 16 is practically lowest m in MFMAs/WMMAs
  // that could be performant. If the gemm sizes
  // are not divisible by that, then we definitely
  // need padding. Therefore, it can't use blockwise
  // reductions.

  // Thus, it becomes a competition among
  // atomic_store based reduction kernels.
  // So basically, all configs could be performant relative to each other.
  if (gemmSize.m % 16 != 0) {
    return success();
  }
  if (gemmSize.n % 16 != 0) {
    return success();
  }
  // We can skip knowing that dPerBlock=16
  // is there on the tuning space that should
  // be faster than anyone that use m or n
  // padding.
  if (gemmSize.m % mPerBlock != 0) {
    return failure();
  }
  if (gemmSize.n % nPerBlock != 0) {
    return failure();
  }
  return success();
}

LogicalResult
PopulateParams::couldBePerformant(const PopulateParamsInfo &info,
                                  const InitParamsNonAccel &params) {
  if (info.hasFusedReduction) {
    return couldFusedReductionBePerformant(info.gemmSize, params.gemmMPerBlock,
                                           params.gemmNPerBlock);
  }
  return success();
}

LogicalResult PopulateParams::obtainTuningParameters(
    OpBuilder &b, const PopulateParamsInfo &info, const StringRef perfConfig,
    InitParamsNonAccel &validParams) {

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isValidPerfConfig = validParams.deserialize(perfConfig.str());
    if (isValidPerfConfig) {
      LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams));
      return populateDerived(validParams);
    }
    // Signal the client if perfCofnig is passed in but is invalid
    return failure();
  }

  // Backup path: Use the set of default tuning parameters
  LogicalResult res = failure();
  auto paramSets =
      getTuningParameters(info.kernelType, info.gemmAType, info.gemmBType);
  for (auto &params : orderInitParams(paramSets, info.gemmSize)) {
    res = populateDerived(params);
    if (failed(res)) {
      continue;
    }

    validParams = params;
    break;
  }
  LLVM_DEBUG(llvm::dbgs() << genDebugForParams(validParams) << "\n");

  return res;
}

LogicalResult
PopulateParams::obtainTuningParameters(RockGemmWrapperInterface op,
                                       const StringRef perfConfig,
                                       InitParamsNonAccel &validParams) {
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
  OpBuilder b(op);
  return obtainTuningParameters(b, info, perfConfig, validParams);
}

std::vector<InitParamsNonAccel>
PopulateParams::getTuningParameters(KernelType opType, Type dataTypeA,
                                    Type dataTypeB) const {
  ArrayRef<InitParamsNonAccel> params;
  if (opType == KernelType::Gemm) {
    params = {initParametersGemm, nInitParametersGemm};
  } else {
    params = {initParametersConv, nInitParametersConv};
  }
  return std::vector<InitParamsNonAccel>(params);
}

static int64_t calculatePaddingComplexity(const GemmSize &paddingAmount,
                                          const GemmSize &gemmSize) {
  int64_t nonPaddedComplexity = gemmSize.m * gemmSize.k * gemmSize.n;
  int64_t paddedComplexity = (gemmSize.m + paddingAmount.m) *
                             (gemmSize.k + paddingAmount.k) *
                             (gemmSize.n + paddingAmount.n);
  return paddedComplexity - nonPaddedComplexity;
}

int64_t PopulateParams::calculatePaddingAmount(const InitParamsNonAccel &params,
                                               const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize);
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

// Acceleration common interface implementation
std::unique_ptr<PopulateParamsAccel>
PopulateParamsAccel::select(GemmFeatures features) {
  if (bitEnumContainsAll(features, GemmFeatures::mfma)) {
    return std::make_unique<PopulateParamsXDL>();
  } else if (bitEnumContainsAll(features, GemmFeatures::wmma)) {
    return std::make_unique<PopulateParamsWmma>();
  } else {
    return nullptr;
  }
}

int64_t
PopulateParamsAccel::calculatePaddingAmount(const InitParamsAccel &params,
                                            const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.gemmKPerBlock, params.gemmMPerBlock,
                       params.gemmNPerBlock, gemmSize, params.gemmKPack);
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

LogicalResult
PopulateParamsAccel::paramsProbablyValid(OpBuilder &b,
                                         const PopulateParamsInfo &info,
                                         const InitParamsAccel &params) {
  Attribute params0 = getGemmParamsAttr(b, params);
  RockAccelTuningParamAttrInterface accelParams0;
  if (auto xdlopsParams0 = dyn_cast<XdlopsGemmParamsAttr>(params0)) {
    int64_t mWaves = params.gemmMPerBlock / params.gemmMPerWave;
    if (mWaves > maxWavesPerWG) {
      return failure();
    }
    auto xdlopsDerivedParams0 = XdlopsGemmDerivedParamsAttr::get(xdlopsParams0);
    accelParams0 = xdlopsDerivedParams0;
  } else {
    accelParams0 = cast<RockAccelTuningParamAttrInterface>(params0);
  }
  return isValidBlockwiseGemm(accelParams0, info.gemmAType, info.gemmBType,
                              info.arch, false, false);
}

LogicalResult
PopulateParamsAccel::couldBePerformant(const PopulateParamsInfo &info,
                                       const InitParamsAccel &params) {
  if (info.hasFusedReduction) {
    return couldFusedReductionBePerformant(info.gemmSize, params.gemmMPerBlock,
                                           params.gemmNPerBlock);
  }
  return specificCouldBePerformant(params, info.gemmAType, info.gemmBType);
}

LogicalResult PopulateParamsAccel::obtainTuningParameters(
    OpBuilder &b, const PopulateParamsInfo &info, const StringRef perfConfig,
    InitParamsAccel &validParams) {

  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    bool isValidPerfConfig = validParams.deserialize(perfConfig.str());
    if (isValidPerfConfig) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Got perf config: " << genDebugForParams(validParams));
      return paramsProbablyValid(b, info, validParams);
    }
    // Signal the client if perfCofnig is passed in but is invalid
    return failure();
  }

  LogicalResult res = failure();
  auto paramSets = getTuningParameters(info.kernelType, info.gemmAType,
                                       info.gemmBType, info.arch);

  for (const auto &params : orderInitParams(paramSets, info.gemmSize)) {
    res = paramsProbablyValid(b, info, params);
    if (failed(res)) {
      continue;
    }
    validParams = params;
    break;
  }
  LLVM_DEBUG(llvm::dbgs() << "perf config: " << genDebugForParams(validParams)
                          << "\n");
  return res;
}

LogicalResult
PopulateParamsAccel::obtainTuningParameters(RockGemmWrapperInterface op,
                                            const StringRef perfConfig,
                                            InitParamsAccel &validParams) {
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
  OpBuilder b(op);
  auto res = obtainTuningParameters(b, info, perfConfig, validParams);
  if (failed(res)) {
    LLVM_DEBUG(llvm::dbgs() << "Couldn't pick heuristic values for ");
    LLVM_DEBUG(op->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  return res;
}

/// Xdlops acceleration
// clang-format off
#define XDL_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef XDL_DEFINITIONS_GEN
// clang-format on

LogicalResult PopulateParamsXDL::isValidBlockwiseGemm(
    RockAccelTuningParamAttrInterface param, Type dataTypeA, Type dataTypeB,
    StringRef arch, bool enableBlockSizeUpperLimit,
    bool enableDPerWaveFiltering) {

  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;
  int64_t blockSize = obtainBlockSize(waveSize, param);
  if (blockSize > maxHardwareWorkgroupSize)
    return failure();
  // TBD: support fp16/bf16

  // clang-format off
  std::vector<std::tuple<int, int, int>> validWaveGemmSize =
  {
    std::make_tuple(128, 128, 2),
    std::make_tuple(128, 64, 2),
    std::make_tuple(64, 128, 2),
    std::make_tuple(64, 64, 2),
    std::make_tuple(64, 32, 2),
    std::make_tuple(32, 64, 2),
    std::make_tuple(32, 32, 2),
    std::make_tuple(64, 16, 4),
    std::make_tuple(16, 64, 4),
    std::make_tuple(16, 16, 4),
  };
  // clang-format on

  XdlopsGemmDerivedParamsAttr xdlopsDerivedParams =
      cast<XdlopsGemmDerivedParamsAttr>(param);
  if (xdlopsDerivedParams.getMnPerXdl() > xdlopsDerivedParams.getMPerWave() ||
      xdlopsDerivedParams.getMnPerXdl() > xdlopsDerivedParams.getNPerWave()) {
    LLVM_DEBUG(llvm::dbgs()
               << "mnPerXdl is too large:" << xdlopsDerivedParams << "\n");
    return failure();
  }

  // Add broadcasts for non 8-bit types.
  bool is8BitReduceOnly = dataTypeA.getIntOrFloatBitWidth() == 8;
  if (!is8BitReduceOnly) {
    validWaveGemmSize.emplace_back(8, 64, 1);
    validWaveGemmSize.emplace_back(4, 64, 1);
  }

  // Check for valid repeats and k distributions
  int64_t minDPerWave = std::min(param.getMPerWave(), param.getNPerWave());
  int64_t validKPerWaveFactor = 2;
  if (minDPerWave <= 16) {
    validKPerWaveFactor = 4;
  }
  if (!((param.getMPerBlock() % minDPerWave == 0) &&
        (param.getNPerBlock() % minDPerWave == 0) &&
        ((param.getKpackPerBlock() * param.getKpack()) % validKPerWaveFactor ==
         0))) {
    return failure();
  }

  if (enableDPerWaveFiltering) {
    if (!std::any_of(validWaveGemmSize.cbegin(), validWaveGemmSize.cend(),
                     [param](const auto it) noexcept -> bool {
                       int validMPerWave, validNPerWave, validKPerWave;
                       std::tie(validMPerWave, validNPerWave, validKPerWave) =
                           it;
                       return (param.getMPerWave() == validMPerWave) &&
                              (param.getNPerWave() == validNPerWave) &&
                              (param.getKpackPerBlock() * param.getKpack() %
                                   validKPerWave ==
                               0);
                     })) {
      return failure();
    }
  }

  if (blockSize < waveSize) {
    return failure();
  }

  // fail with blockSize >= 512
  // \todo fix the issue with blockSize >= 512
  if (enableBlockSizeUpperLimit && blockSize > 4 * waveSize) {
    return failure();
  }

  if ((param.getMPerBlock() % param.getMPerWave()) != 0) {
    return failure();
  }

  if ((param.getNPerBlock() % param.getNPerWave()) != 0) {
    return failure();
  }

  // Reject invalid blockSize
  int64_t kPerBlock = param.getKpackPerBlock() * param.getKpack();
  int64_t mPerBlock = param.getMPerBlock();
  int64_t nPerBlock = param.getNPerBlock();
  if (!isValidBlockSize(blockSize, kPerBlock, mPerBlock, nPerBlock)) {
    LLVM_DEBUG(llvm::dbgs() << "tuning: Block size too large.\n");
    return failure();
  }

  // Sledgehammer hotfix because not unrolling sometimes makes the register
  // allocator break. This should be refined quickly.
  if (cast<RockTuningParamAttrInterface>(param).getForceUnroll() == false) {
    return failure();
  }

  // Reject invalid KPACK values.
  int64_t mnPerXdl = std::min(param.getMPerWave(), param.getNPerWave());
  if (auto derivedParam = cast<XdlopsGemmDerivedParamsAttr>(param)) {
    mnPerXdl = derivedParam.getMnPerXdl();
  }
  auto maybeMfmaInsnGroup =
      MfmaInsnGroup::select(dataTypeA, dataTypeB, arch, mnPerXdl);
  if (failed(maybeMfmaInsnGroup)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to select xdlops instruction group.\n");
    return failure();
  }
  MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
  if (!mfmaGroup.isCoherentWithK(param.getKpack(), param.getKpackPerBlock())) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Mfma instruction group selection is not compatible with k.\n");
    return failure();
  }

  return success();
}

std::vector<InitParamsAccel>
PopulateParamsXDL::getTuningParameters(KernelType opType, Type dataTypeA,
                                       Type dataTypeB, StringRef arch) const {
  ArrayRef<InitParamsAccel> params;
  if (opType == KernelType::Gemm) {
    switch (dataTypeA.getIntOrFloatBitWidth()) {
    case 8:
      params = {initParametersI8Gemm, nInitParametersI8Gemm};
      break;
    case 16:
      params = {initParametersFp16Gemm, nInitParametersFp16Gemm};
      break;
    default:
      params = {initParametersGemm, nInitParametersGemm};
    }
  } else {
    switch (dataTypeA.getIntOrFloatBitWidth()) {
    case 8:
      params = {initParametersForward8BitConv, nInitParametersForward8BitConv};
      break;
    case 16:
      params = {initParametersFp16Conv, nInitParametersFp16Conv};
      break;
    default:
      params = {initParametersConv, nInitParametersConv};
    }
  }
  std::vector<InitParamsAccel> res;
  // Only return valid XDLOp params
  std::copy_if(
      params.begin(), params.end(), std::back_inserter(res),
      [&](const InitParamsAccel &param) {
        int64_t mnPerXdl = param.gemmNPerWaveOrMnPerXdl;
        auto maybeMfmaInsnGroup =
            MfmaInsnGroup::select(dataTypeA, dataTypeB, arch, mnPerXdl);
        if (failed(maybeMfmaInsnGroup)) {
          return false;
        }
        MfmaInsnGroup mfmaGroup = *maybeMfmaInsnGroup;
        if (!mfmaGroup.isCoherentWithK(param.gemmKPack, param.gemmKPerBlock)) {
          return false;
        }
        return true;
      });
  return res;
}

LogicalResult
PopulateParamsXDL::specificCouldBePerformant(const InitParamsAccel &params,
                                             Type dataTypeA, Type dataTypeB) {
  // Implement this if needed.
  (void)params;
  (void)dataTypeA;
  (void)dataTypeB;
  return success();
}

Attribute
PopulateParamsXDL::getGemmParamsAttr(OpBuilder &builder,
                                     const InitParamsAccel &validParams) const {
  return builder.getAttr<XdlopsGemmParamsAttr>(
      validParams.gemmKPerBlock, validParams.gemmMPerBlock,
      validParams.gemmNPerBlock, validParams.gemmKPack,
      validParams.gemmMPerWave, validParams.gemmNPerWaveOrMnPerXdl,
      validParams.splitKFactor, validParams.gemmAThreadCopyMoreGemmK);
}

/// Wmma acceleration
// clang-format off
#define Wmma_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Wmma_DEFINITIONS_GEN
// clang-format on

LogicalResult PopulateParamsWmma::isValidBlockwiseGemm(
    RockAccelTuningParamAttrInterface param, Type dataTypeA, Type dataTypeB,
    StringRef arch, bool enableBlockSizeUpperLimit,
    bool enableDPerWaveFiltering) {

  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;
  int64_t blockSize = obtainBlockSize(waveSize, param);
  if (blockSize > maxHardwareWorkgroupSize)
    return failure();

  // clang-format off
  std::vector<std::tuple<int, int, int>> validWaveGemmSize =
  {
    std::make_tuple(128, 128, 2),
    std::make_tuple(128, 64, 2),
    std::make_tuple(64, 128, 2),
    std::make_tuple(64, 64, 2),
    std::make_tuple(64, 32, 2),
    std::make_tuple(32, 64, 2),
    std::make_tuple(32, 32, 2),
    std::make_tuple(32, 16, 2),
    std::make_tuple(16, 32, 2),
    std::make_tuple(32, 32, 2),
    std::make_tuple(64, 16, 2),
    std::make_tuple(16, 64, 2),
    std::make_tuple(16, 16, 2),
  };
  // clang-format on

  // Check for valid repeats and k distributions
  int64_t minDPerWave = std::min(param.getMPerWave(), param.getNPerWave());
  int64_t validKPerWaveFactor = 2;
  if (minDPerWave <= 16) {
    validKPerWaveFactor = 4;
  }
  if (!((param.getMPerBlock() % minDPerWave == 0) &&
        (param.getNPerBlock() % minDPerWave == 0) &&
        (param.getKpackPerBlock() % validKPerWaveFactor == 0))) {
    return failure();
  }

  if (enableDPerWaveFiltering) {
    if (!std::any_of(validWaveGemmSize.cbegin(), validWaveGemmSize.cend(),
                     [param](const auto it) noexcept -> bool {
                       int validMPerWave, validNPerWave, validKPerWave;
                       std::tie(validMPerWave, validNPerWave, validKPerWave) =
                           it;
                       return (param.getMPerWave() == validMPerWave) &&
                              (param.getNPerWave() == validNPerWave) &&
                              (param.getKpackPerBlock() % validKPerWave == 0);
                     }))
      return failure();
  }

  if (blockSize < waveSize)
    return failure();

  // fail with blockSize >= 512
  // \todo fix the issue with blockSize >= 512
  if (enableBlockSizeUpperLimit && blockSize > 4 * waveSize) {
    return failure();
  }

  if ((param.getMPerBlock() % param.getMPerWave()) != 0)
    return failure();

  if ((param.getNPerBlock() % param.getNPerWave()) != 0)
    return failure();

  // Sledgehammer hotfix because not unrolling sometimes makes the register
  // allocator break. This should be refined quickly.
  if (param.getForceUnroll() == false) {
    return failure();
  }

  // Reject invalid KPACK values.
  auto maybeWmmaInsn =
      WmmaInsn::select(dataTypeA, dataTypeB, waveSize, arch,
                       param.getMPerWave(), param.getNPerWave());
  if (failed(maybeWmmaInsn)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to select wmma instruction.\n");
    return failure();
  }
  WmmaInsn wmmaInsn = *maybeWmmaInsn;
  if (!wmmaInsn.isCoherentWithK(param.getKpack(), param.getKpackPerBlock())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Wmma instruction selection is not compatible with k.\n");
    return failure();
  }

  return success();
}

std::vector<InitParamsAccel>
PopulateParamsWmma::getTuningParameters(KernelType opType, Type dataTypeA,
                                        Type dataTypeB, StringRef arch) const {
  ArrayRef<InitParamsAccel> params;
  std::vector<InitParamsAccel> res;
  if (opType == KernelType::Gemm) {
    switch (dataTypeA.getIntOrFloatBitWidth()) {
    case 8:
      params = {initParametersI8Gemm, nInitParametersI8Gemm};
      break;
    case 16:
      params = {initParametersFp16Gemm, nInitParametersFp16Gemm};
      break;
    default:
      return res;
    }
  } else {
    switch (dataTypeA.getIntOrFloatBitWidth()) {
    case 8:
      params = {initParametersForward8BitConv, nInitParametersForward8BitConv};
      break;
    case 16:
      params = {initParametersFp16Conv, nInitParametersFp16Conv};
      break;
    default:
      return res;
    }
  }
  // Only return valid Wmma params
  const int64_t waveSize = mlir::rock::lookupArchInfo(arch).waveSize;
  std::copy_if(
      params.begin(), params.end(), std::back_inserter(res),
      [&](const InitParamsAccel &param) {
        auto maybeWmmaInsn =
            WmmaInsn::select(dataTypeA, dataTypeB, waveSize, arch,
                             param.gemmMPerWave, param.gemmNPerWaveOrMnPerXdl);
        if (failed(maybeWmmaInsn)) {
          return false;
        }
        WmmaInsn wmmaInsn = *maybeWmmaInsn;
        if (!wmmaInsn.isCoherentWithK(param.gemmKPack, param.gemmKPerBlock)) {
          return false;
        }
        return true;
      });
  return res;
}

LogicalResult
PopulateParamsWmma::specificCouldBePerformant(const InitParamsAccel &params,
                                              Type dataTypeA, Type dataTypeB) {
  // Implement this if needed.
  (void)params;
  (void)dataTypeA;
  (void)dataTypeB;
  return success();
}

Attribute PopulateParamsWmma::getGemmParamsAttr(
    OpBuilder &builder, const InitParamsAccel &validParams) const {
  return builder.getAttr<WmmaGemmParamsAttr>(
      validParams.gemmKPerBlock, validParams.gemmMPerBlock,
      validParams.gemmNPerBlock, validParams.gemmKPack,
      validParams.gemmMPerWave, validParams.gemmNPerWaveOrMnPerXdl,
      validParams.splitKFactor, validParams.gemmAThreadCopyMoreGemmK);
}
