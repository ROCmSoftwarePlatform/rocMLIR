//===- TosaToRock.cpp - Lowering Tosa to Rock Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Rock dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToRock/TosaToRock.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include <utility>

#define DEBUG_TYPE "convert-tosa-to-rock"

using namespace mlir;

namespace {

static bool isZeroAttribute(Attribute value) {
  if (auto intValue = dyn_cast<IntegerAttr>(value))
    return intValue.getValue().isZero();
  if (auto fpValue = dyn_cast<FloatAttr>(value))
    return fpValue.getValue().isZero();
  if (auto splatValue = dyn_cast<SplatElementsAttr>(value))
    return isZeroAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = dyn_cast<ElementsAttr>(value))
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto elementsValue = dyn_cast<DenseElementsAttr>(value))
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = dyn_cast<ArrayAttr>(value))
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
}

static bool isConstantZero(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    return isZeroAttribute(cst.getValue());
  if (auto cst = v.getDefiningOp<tosa::ConstOp>())
    return isZeroAttribute(cst->getAttr("value"));
  return false;
}

static bool isNegInfAttribute(Attribute value) {
  if (auto fpValue = dyn_cast<FloatAttr>(value)) {
    auto value = fpValue.getValue();

    std::pair<APFloat, llvm::detail::opStatus> floatRes = rock::createAPFloat(
        fpValue.getType(), -std::numeric_limits<float>::infinity());
    auto expectedValue = floatRes.first;
    auto status = floatRes.second;
    assert(status == APFloat::opOK);

    return value.compare(expectedValue) == llvm::APFloat::cmpEqual;
  }
  if (auto splatValue = dyn_cast<SplatElementsAttr>(value))
    return isNegInfAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = dyn_cast<ElementsAttr>(value))
    return llvm::all_of(elementsValue.getValues<Attribute>(),
                        isNegInfAttribute);
  if (auto elementsValue = dyn_cast<DenseElementsAttr>(value))
    return llvm::all_of(elementsValue.getValues<Attribute>(),
                        isNegInfAttribute);
  if (auto arrayValue = dyn_cast<ArrayAttr>(value))
    return llvm::all_of(arrayValue.getValue(), isNegInfAttribute);

  return false;
}

static bool isConstIsNegInf(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    return isNegInfAttribute(cst.getValue());
  if (auto cst = v.getDefiningOp<tosa::ConstOp>())
    return isNegInfAttribute(cst->getAttr("value"));
  return false;
}

static bool isConstRangeAttribute(Attribute value, int32_t expectedVal) {
  if (auto intValue = dyn_cast<IntegerAttr>(value)) {
    auto value = intValue.getValue();

    std::pair<APInt, bool> intRes =
        rock::createAPInt(intValue.getType(), expectedVal);
    auto expectedValue = intRes.first;
    bool overflow = intRes.second;
    if (overflow)
      return false;

    return expectedValue == value;
  }
  if (auto splatValue = dyn_cast<SplatElementsAttr>(value))
    return false;
  if (auto elementsValue = dyn_cast<ElementsAttr>(value))
    return llvm::all_of(llvm::enumerate(elementsValue.getValues<Attribute>()),
                        [](const auto &indexedAttr) {
                          return isConstRangeAttribute(indexedAttr.value(),
                                                       indexedAttr.index());
                        });
  if (auto elementsValue = dyn_cast<DenseElementsAttr>(value))
    return llvm::all_of(llvm::enumerate(elementsValue.getValues<Attribute>()),
                        [](const auto &indexedAttr) {
                          return isConstRangeAttribute(indexedAttr.value(),
                                                       indexedAttr.index());
                        });
  if (auto arrayValue = dyn_cast<ArrayAttr>(value))
    return llvm::all_of(llvm::enumerate(arrayValue.getValue()),
                        [](const auto &indexedAttr) {
                          return isConstRangeAttribute(indexedAttr.value(),
                                                       indexedAttr.index());
                        });

  return false;
}

bool isConstRange(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    return isConstRangeAttribute(cst.getValue(), 0);
  if (auto cst = v.getDefiningOp<tosa::ConstOp>())
    return isConstRangeAttribute(cst->getAttr("value"), 0);
  return false;
}

// Note:  we want something a bit more general than SmallString<8> for
// the layout string, but it has to allow for inserting a character into
// the string for the caller to see.
static Value expandTensor(ConversionPatternRewriter &rw, Operation *op,
                          Value operand, SmallString<8> &layout,
                          StringRef lowerName, int64_t g, uint32_t idx = 4) {
  auto loc = op->getLoc();
  auto oprType = cast<ShapedType>(operand.getType());
  if (!oprType.hasStaticShape()) {
    (void)rw.notifyMatchFailure(
        op, "tosa to rock conversion expects statically shaped tensors");
    return Value();
  }
  ArrayRef<int64_t> shape = oprType.getShape();

  SmallVector<uint32_t, 8> endDims;
  SmallVector<uint32_t, 8> startDims;
  SmallVector<StringRef, 8> startNames;

  // find the lower dimension that encodes the g dimension
  std::optional<uint32_t> groupFoldedDim = std::nullopt;

  for (uint32_t i = 0, e = shape.size(); i < e; ++i) {
    startNames.push_back(layout.substr(i, 1));
    if (layout[i] == lowerName[0]) {
      groupFoldedDim = i;
    } else {
      startDims.push_back(i);
      endDims.push_back(groupFoldedDim.has_value() ? i + 1 : i);
    }
  }

  if (!groupFoldedDim.has_value()) {
    (void)rw.notifyMatchFailure(op, "tosa conv has an invalid layout");
    return Value();
  }

  uint32_t lowerDim = groupFoldedDim.value();
  // insert 'g' dimension into layout
  rock::BottomUpTMBuilder transform(rw, ArrayRef<StringRef>(startNames), shape,
                                    loc);
  transform.passThrough(endDims, startDims);
  transform.unmerge({"g", lowerName}, {lowerDim, lowerDim + 1}, lowerName,
                    {{g, shape[lowerDim] / g}});
  layout = Twine(layout.substr(0, lowerDim) + "g" +
                 layout.substr(lowerDim, layout.size() - lowerDim))
               .str();

  return rw.create<rock::TransformOp>(loc, operand, transform.get());
}

static std::tuple<StringAttr, std::optional<uint32_t>, rock::GemmFeatures>
getArchAttributes(Operation *op, Type inputType) {
  auto func = op->getParentOfType<func::FuncOp>();
  // auto mod = func->getParentOfType<ModuleOp>();

  // TODO(sjw): get these from options
  StringAttr arch = StringAttr::get(op->getContext(), "");
  FailureOr<StringAttr> maybeArch = rock::getArch(op);
  if (succeeded(maybeArch)) {
    arch = maybeArch.value();
  }
  std::optional<uint32_t> num_cu = std::nullopt;
  FailureOr<int64_t> maybeNumCU = rock::getNumCU(op);
  if (succeeded(maybeNumCU)) {
    num_cu = (uint32_t)maybeNumCU.value();
  }
  std::optional<bool> xdlopsV2 = std::nullopt;

  if (auto attr = op->getAttrOfType<BoolAttr>("xdlopsV2"))
    xdlopsV2 = attr.getValue();
  else if (auto attr = func->getAttrOfType<BoolAttr>("xdlopsV2"))
    xdlopsV2 = attr.getValue();

  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  rock::GemmFeatures features = archInfo.getDefaultFeatures(inputType);
  if (xdlopsV2.has_value())
    features = rock::bitEnumSet(features, rock::GemmFeatures::mfma, *xdlopsV2);

  return {arch, num_cu, features};
}

static FailureOr<rock::ConvOp>
makeRockConv(ConversionPatternRewriter &rw, Operation *op, Value input,
             Value filter, Value output, DenseI64ArrayAttr pad,
             DenseI64ArrayAttr stride, DenseI64ArrayAttr dilation,
             int64_t group) {
  Location loc = op->getLoc();

  SmallString<8> filterLayout("kyxc");
  if (auto attr = op->getAttrOfType<StringAttr>("filter_layout"))
    filterLayout = attr.getValue();
  else if (cast<ShapedType>(filter.getType()).getRank() > 4)
    filterLayout = "k012c";
  SmallString<8> inputLayout("nhwc");
  if (auto attr = op->getAttrOfType<StringAttr>("input_layout"))
    inputLayout = attr.getValue();
  else if (cast<ShapedType>(input.getType()).getRank() > 4)
    inputLayout = "n012c";
  SmallString<8> outputLayout("nhwk");
  if (auto attr = op->getAttrOfType<StringAttr>("output_layout"))
    outputLayout = attr.getValue();
  else if (cast<ShapedType>(output.getType()).getRank() > 4)
    outputLayout = "n012k";

  // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
  // and add 'g into the layout
  auto inputExp = expandTensor(rw, op, input, inputLayout, "c", group);
  auto filterExp = expandTensor(rw, op, filter, filterLayout, "k", group);
  auto outputExp = expandTensor(rw, op, output, outputLayout, "k", group);

  StringAttr arch;
  std::optional<uint32_t> num_cu;
  rock::GemmFeatures features;
  std::tie(arch, num_cu, features) = getArchAttributes(op, input.getType());

  IntegerAttr numCUAttr =
      num_cu.has_value() ? rw.getI32IntegerAttr(num_cu.value()) : nullptr;
  auto cop = rw.create<rock::ConvOp>(
      loc, outputExp.getType(), filterExp, inputExp, outputExp, arch,
      rw.getAttr<rock::GemmFeaturesAttr>(features),
      /*blockSize=*/nullptr, /*gridSize=*/nullptr, rw.getIndexArrayAttr(pad),
      rw.getIndexArrayAttr(stride), rw.getIndexArrayAttr(dilation),
      /*params=*/nullptr, numCUAttr);

  // specify layout attributes
  SmallVector<StringAttr, 5> filterLayoutSpec;
  SmallVector<StringAttr, 5> inputLayoutSpec;
  SmallVector<StringAttr, 5> outputLayoutSpec;
  for (size_t i = 0; i < filterLayout.size(); ++i) {
    filterLayoutSpec.push_back(rw.getStringAttr(filterLayout.substr(i, 1)));
    inputLayoutSpec.push_back(rw.getStringAttr(inputLayout.substr(i, 1) + "i"));
    outputLayoutSpec.push_back(
        rw.getStringAttr(outputLayout.substr(i, 1) + "o"));
  }

  // arch-specific attributes
  // TODO: remove these
  if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
    cop->setAttr("perf_config", attr);

  // convolution config attributes
  cop->setAttr("filter_layout",
               rw.getArrayAttr(ArrayRef<Attribute>(filterLayoutSpec.begin(),
                                                   filterLayoutSpec.end())));
  cop->setAttr("input_layout",
               rw.getArrayAttr(ArrayRef<Attribute>(inputLayoutSpec.begin(),
                                                   inputLayoutSpec.end())));
  cop->setAttr("output_layout",
               rw.getArrayAttr(ArrayRef<Attribute>(outputLayoutSpec.begin(),
                                                   outputLayoutSpec.end())));
  return cop;
}

static bool isTosaReduction(Operation *op) {
  return isa<tosa::ReduceMaxOp, tosa::ReduceSumOp, tosa::ReduceMinOp,
             tosa::ReduceProdOp, tosa::ReduceAllOp, tosa::ReduceAnyOp>(op);
}

static Value traceToRes(Value tensor, DenseMap<Value, Value> &cache,
                        Value expectedTensor) {
  if (cache.contains(tensor))
    return cache.at(tensor);

  Value res = nullptr;
  if (tensor.getDefiningOp()) {
    if (isTosaReduction(tensor.getDefiningOp()) && expectedTensor == tensor) {
      res = tensor;
    } else if (auto view = tensor.getDefiningOp<ViewLikeOpInterface>()) {
      res = traceToRes(view.getViewSource(), cache, expectedTensor);
    } else if (auto collapse =
                   tensor.getDefiningOp<tensor::CollapseShapeOp>()) {
      res = traceToRes(collapse.getSrc(), cache, expectedTensor);
    } else if (auto tosaOp = tensor.getDefiningOp<tosa::TosaOp>()) {
      for (auto operand : tosaOp->getOperands()) {
        if (llvm::isa<TensorType>(operand.getType())) {
          res = traceToRes(operand, cache, expectedTensor);
          if (res)
            break;
        }
      }
    }
  }

  cache.insert({tensor, res});
  return res;
}

static FailureOr<int64_t> traceToRes(Value expectedTensor, func::FuncOp func) {
  llvm::DenseMap<Value, Value> cache;

  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
  assert(returns.size() == 1 && "Number of returns is not one");
  func::ReturnOp returnOp = returns[0];

  for (auto [i, res] : llvm::enumerate(returnOp->getOperands())) {
    Value out = traceToRes(res, cache, expectedTensor);
    if (out == expectedTensor) {
      return i;
    }
  }
  return failure();
}

template <typename OpT>
class ConvConverter final : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    auto operands = adaptor.getOperands();
    auto loc = op->getLoc();
    auto *context = op->getContext();
    auto input = operands[0];
    auto filter = operands[1];
    auto bias = operands[2];
    auto outputType = cast<RankedTensorType>(op.getType());

    Value output =
        rw.create<bufferization::AllocTensorOp>(loc, outputType, ValueRange{});

    int64_t group = 1;
    if (auto attr = op->template getAttrOfType<IntegerAttr>("group"))
      group = attr.getInt(); // Use op.getGroup() when all OpT have it.
    FailureOr<rock::ConvOp> rockConv =
        makeRockConv(rw, op, input, filter, output, op.getPadAttr(),
                     op.getStrideAttr(), op.getDilationAttr(), group);
    if (failed(rockConv))
      return failure();

    Value result = rw.create<rock::TensorUntransformCastOp>(
        loc, outputType, rockConv->getResult(), rockConv->getOutput());
    // test for zero bias, and ignore
    if (!isConstantZero(op.getOperand(2))) {
      // non-zero bias, replace with tosa.add w/ broadcast
      auto biasType = cast<ShapedType>(bias.getType());
      if (!biasType.hasStaticShape())
        return failure();

      int64_t nDims = cast<ShapedType>(input.getType()).getRank();
      SmallVector<int64_t> biasShape;
      for (int i = 0; i < nDims - 1; i++)
        biasShape.push_back(1);
      biasShape.push_back(biasType.getShape()[0]);
      auto newType =
          RankedTensorType::get(biasShape, biasType.getElementType());

      // [[0, 1, 2, 3]]
      ReassociationExprs exprs;
      for (int i = 0; i < nDims; i++)
        exprs.push_back(getAffineDimExpr(i, context));
      SmallVector<ReassociationExprs, 1> reassociations;
      reassociations.push_back(exprs);

      auto biasExpand =
          rw.create<tensor::ExpandShapeOp>(loc, newType, bias, reassociations);

      result = rw.create<tosa::AddOp>(loc, op.getType(),
                                      ValueRange{result, biasExpand});
    }

    rw.replaceOp(op, result);

    return success();
  }
};

static Value insertBroadcast(Value inp, ArrayRef<int64_t> outShape,
                             Location loc, OpBuilder &b) {
  ArrayRef<int64_t> inpShape = cast<ShapedType>(inp.getType()).getShape();
  bool broadcastDone = false;
  rock::BottomUpTMBuilder broadcastDims(b, inpShape, loc);
  for (unsigned int i = 0; i < outShape.size(); i++) {
    if (inpShape[i] == 1 && outShape[i] != 1) {
      broadcastDims.broadcast({i}, {outShape[i]});
      broadcastDone = true;
    } else {
      broadcastDims.passThrough({i}, {i});
    }
  }
  if (!broadcastDone) {
    return inp;
  }
  return b.create<rock::TransformOp>(loc, inp, broadcastDims.get());
}

class MatMulConverter final : public OpConversionPattern<tosa::MatMulOp> {
public:
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

  UnitAttr getTranspose(tosa::MatMulOp op, StringRef name) const {
    if (auto attr = op->getAttrOfType<BoolAttr>(name)) {
      if (attr.getValue())
        return UnitAttr::get(op->getContext());
    }
    return nullptr;
  }

  std::tuple<int64_t, int64_t> getLastDims(UnitAttr transposed,
                                           RankedTensorType type) const {
    ArrayRef<int64_t> shape = type.getShape();
    int64_t rank = type.getRank();
    if (transposed) {
      return {shape[rank - 1], shape[rank - 2]};
    }
    return {shape[rank - 2], shape[rank - 1]};
  }

  void setLastDims(UnitAttr transposed, SmallVectorImpl<int64_t> &shape,
                   std::pair<int64_t, int64_t> lastDims) const {
    size_t rank = shape.size();
    if (transposed) {
      shape[rank - 1] = lastDims.first;
      shape[rank - 2] = lastDims.second;
    } else {
      shape[rank - 2] = lastDims.first;
      shape[rank - 1] = lastDims.second;
    }
  }

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                tosa::MatMulOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    Location loc = op->getLoc();
    auto outputType = cast<RankedTensorType>(op.getType());
    Value output =
        rw.create<bufferization::AllocTensorOp>(loc, outputType, ValueRange{});

    UnitAttr transposeA = getTranspose(op, "transpose_a"),
             transposeB = getTranspose(op, "transpose_b"),
             transposeC = getTranspose(op, "transpose_c");

    StringAttr arch;
    std::optional<uint32_t> num_cu;
    rock::GemmFeatures features;
    std::tie(arch, num_cu, features) =
        getArchAttributes(op, op.getA().getType());

    auto [mDim, nDim] = getLastDims(transposeC, outputType);

    int64_t kDimOfA;
    std::tie(std::ignore, kDimOfA) =
        getLastDims(transposeA, cast<RankedTensorType>(op.getA().getType()));
    int64_t kDimOfB;
    std::tie(kDimOfB, std::ignore) =
        getLastDims(transposeB, cast<RankedTensorType>(op.getB().getType()));
    int kDim = (kDimOfA > kDimOfB) ? kDimOfA : kDimOfB;

    SmallVector<int64_t, 3> aShape = llvm::to_vector<3>(
        cast<RankedTensorType>(op.getA().getType()).getShape());
    setLastDims(transposeA, aShape, {mDim, kDim});
    Value brA = insertBroadcast(adaptor.getA(), aShape, loc, rw);

    SmallVector<int64_t, 3> bShape = llvm::to_vector<3>(
        cast<RankedTensorType>(op.getB().getType()).getShape());
    setLastDims(transposeB, bShape, {kDim, nDim});
    Value brB = insertBroadcast(adaptor.getB(), bShape, loc, rw);

    IntegerAttr numCUAttr =
        num_cu.has_value() ? rw.getI32IntegerAttr(num_cu.value()) : nullptr;
    auto rockGemm = rw.create<rock::GemmOp>(
        loc, outputType, brA, brB, output, transposeA, transposeB, transposeC,
        arch, numCUAttr, rw.getAttr<rock::GemmFeaturesAttr>(features),
        rw.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
        /*blockSize=*/nullptr, /*gridSize=*/nullptr,
        /*params=*/nullptr);

    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      rockGemm->setAttr("perf_config", attr);

    rw.replaceOp(op, rockGemm.getResult());

    return success();
  }
};

static void permuteLayout(Operation *op, const char *attrKey,
                          const char *layoutDefault, ArrayRef<int32_t> permDims,
                          bool isInput = false) {
  StringRef currentLayout(layoutDefault);
  if (auto attr = op->getAttrOfType<StringAttr>(attrKey))
    currentLayout = attr.getValue();
  SmallString<4> layout(currentLayout);
  if (isInput) {
    for (int i = 0, e = permDims.size(); i < e; ++i)
      layout[permDims[i]] = currentLayout[i];
  } else {
    for (int i = 0, e = permDims.size(); i < e; ++i)
      layout[i] = currentLayout[permDims[i]];
  }
  op->setAttr(attrKey, StringAttr::get(op->getContext(), layout));
}

struct TransposeRewritePattern : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern<tosa::TransposeOp>::OpRewritePattern;

  LogicalResult getTransposeDims(Value v, SmallVector<int32_t> &perms) const {
    Operation *cval = v.getDefiningOp();
    if (isa<arith::ConstantOp>(cval) || isa<tosa::ConstOp>(cval)) {
      auto cattr = cast<DenseElementsAttr>(cval->getAttr("value"));
      auto vals = cattr.tryGetValues<int32_t>();
      if (succeeded(vals)) {
        perms.assign((*vals).begin(), (*vals).end());
        return success();
      }
      auto vals64 = cattr.tryGetValues<int64_t>();
      if (succeeded(vals64)) {
        perms.assign((*vals64).begin(), (*vals64).end());
        return success();
      }
    }
    return failure();
  }

  void setTranspose(Operation *op, StringRef name, bool isNonTrivial) const {
    bool currentValue = false;
    if (auto attr = op->getAttrOfType<BoolAttr>(name))
      currentValue = attr.getValue();
    bool newValue = currentValue ^ isNonTrivial;
    op->setAttr(name, BoolAttr::get(op->getContext(), newValue));
  }

  LogicalResult checkInputHasUses(PatternRewriter &rewriter,
                                  tosa::TransposeOp top, Value tInput) const {
    // if the input has uses (apart from this one), we can't do this
    if (!tInput.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          top, "abandoning attempt to fuse transpose "
               "because the operation has other uses");
    }
    return success();
  }

  LogicalResult checkMatMulTransposeValid(tosa::MatMulOp matmulOp,
                                          ArrayRef<int32_t> dims) const {
    // batch dimension is expected to be 3rd from the last.
    if (dims.size() >= 3 && dims[dims.size() - 3] != (int32_t)dims.size() - 3) {
      return matmulOp.emitWarning(
          "Transposing the batch dimension out of place lowers performance");
    }
    return success();
  }

  bool isMatMulNonTrivial(ArrayRef<int32_t> dims) const {
    int32_t lastDim = dims.size() - 1;
    int32_t prevLastDim = dims.size() - 2;
    return (dims[prevLastDim] == lastDim && dims[lastDim] == prevLastDim);
  }

  // This function traverses the uses of tOutput and then modifies
  // the uses to indicate the input are transposed and replaces them
  // with tInput. If there are collapse shapes encountered, the collapse
  // is applied on the tInput.
  LogicalResult mergeTransposeWithGemmLikeOp(PatternRewriter &rewriter,
                                             Value tOutput,
                                             ArrayRef<int32_t> dims,
                                             Value tInput) const {
    for (auto &use : llvm::make_early_inc_range(tOutput.getUses())) {
      if (auto op = dyn_cast<tensor::CollapseShapeOp>(use.getOwner())) {
        SmallVector<ReassociationIndices, 4> reassocIndices =
            op.getReassociationIndices();
        // This is to capture new reassociations above the transpose
        llvm::SmallDenseMap<int32_t, ReassociationIndices> newReassocIdxMap;
        ArrayRef<int64_t> inShape = op.getSrcType().getShape();

        // This loops maps reassociated dims back to pre transposed dims.
        SmallVector<int32_t, 4> newDims;

        llvm::SmallDenseSet<int64_t> preTpUnitDims;
        for (ReassociationIndices indices : reassocIndices) {
          ReassociationIndices newReassocIdx;
          size_t numNonUnitDimsMerged = 0;
          for (size_t i = 0, e = indices.size(); i < e; ++i) {
            if (inShape[indices[i]] == 1) {
              preTpUnitDims.insert(dims[indices[i]]);
            } else {
              numNonUnitDimsMerged += 1;
            }
            newReassocIdx.push_back(dims[indices[i]]);
          }
          if (numNonUnitDimsMerged > 1) {
            // Per MIGraphX bug #2692, this transpsoe/collaspe swap logic
            // will be incorrect in cases like the following
            //   %0 = expand_shape [[0], [1, 2], [3]] %arg0 : tensor<7x6x5xT>
            //   to tensor<7x3x2x5xT> %1 = transpose %0, [0, 2, 1, 3] :
            //   tensor<7x2x3x5xT> %2 = collapse_shape [[0], [1, 2], [2]] %1 :
            //   tensor<7x2x3x5xT> to tensor<7x6x5xT>
            // by way of creating a trivial expand/collapse pair that isn't
            // correct.
            //
            // Therefore, as a sledgehammer fix, don't handle any cases where
            // non-trivial collapses are performed.
            return rewriter.notifyMatchFailure(
                op, "abandoning attempt to interchange transpose and "
                    "non-trivial collapse");
          }
          if (newReassocIdx.size() > 1) {
            llvm::sort(newReassocIdx);
            // Remove unit dims from larger end of reassociation indices
            // but we need at least one for the reassociation
            while (newReassocIdx.size() > 1 &&
                   preTpUnitDims.contains(newReassocIdx.back())) {
              newReassocIdx.pop_back();
            }
            for (size_t i = 1; i < newReassocIdx.size(); i++) {
              if (newReassocIdx[i] - newReassocIdx[i - 1] != 1) {
                return rewriter.notifyMatchFailure(
                    op, "CollapseShape op following transpose collapses "
                        "non-contigous pre-transpose dims.");
              }
            }
          }
          newDims.push_back(newReassocIdx[0]);
          // minIdx is the representative of a group that is
          // being collapsed. For e.g. for a collapse of [3,4,5] is assigned
          // with 3 as the representative. I also note that we only allow
          // collapsing of contigous pre-transpose dims.
          newReassocIdxMap[newReassocIdx[0]] = newReassocIdx;
        }

        // Assign the ordering index of reassociated dims as the dim index
        SmallVector<int32_t, 4> newDimsSorted = newDims;
        llvm::sort(newDimsSorted);
        SmallVector<ReassociationIndices, 4> newReassocIndicesSorted;
        DenseMap<int32_t, int32_t> dimMap;
        // The vector of newDims (may) contain a discontinous
        // a range of representative minIdxs. Here we make
        // it contigous by assigning order idx.
        for (size_t i = 0; i < newDimsSorted.size(); i++) {
          dimMap[newDimsSorted[i]] = i;
          newReassocIndicesSorted.push_back(newReassocIdxMap[newDimsSorted[i]]);
        }
        // HOTFIX: glue trailing unit dimensions onto collapses that need
        // them. This is because a case like
        // %t = transpose %aRaw [0, 1, 3, 2] : tensor<1x1xKxM> ->
        // tensor<1x1xMxK> %a = collapse_shape [[0, 1], [2], [3]]
        //    : tensor<1x1xMxK> -> tensor<1xMxK>
        // will, with the above unit-dimension-removal logic, lead to the
        // invalid reassociation [[0], [2], [3]], causing a crash.
        // See MIGraphX bug #2365.
        // The entire logic here should be reviewed, or at least made less
        // complex if possible, but ... release-critical bug, what can we do?
        for (size_t i = 0, e = newReassocIndicesSorted.size() - 1; i < e; ++i) {
          ReassociationIndices &theseIndices = newReassocIndicesSorted[i];
          const ReassociationIndices &nextIndices =
              newReassocIndicesSorted[i + 1];
          while (theseIndices.back() + 1 < nextIndices[0]) {
            theseIndices.push_back(theseIndices.back() + 1);
          }
        }
        // do the same for the last set of indices too
        // where it does not match upto the rank of the input.
        ReassociationIndices &lastIndices = newReassocIndicesSorted.back();
        while (lastIndices.back() + 1 < (int64_t)inShape.size()) {
          lastIndices.push_back(lastIndices.back() + 1);
        }

        for (size_t i = 0; i < newDims.size(); i++) {
          newDims[i] = dimMap[newDims[i]];
        }

        tensor::CollapseShapeOp newCollapseShapeOp =
            rewriter.create<tensor::CollapseShapeOp>(op.getLoc(), tInput,
                                                     newReassocIndicesSorted);

        if (mergeTransposeWithGemmLikeOp(rewriter, op.getResult(), newDims,
                                         newCollapseShapeOp.getResult())
                .failed()) {
          rewriter.eraseOp(newCollapseShapeOp);
          return failure();
        }
        if (op->use_empty())
          rewriter.eraseOp(op);
      } else if (auto op = dyn_cast<tensor::ExpandShapeOp>(use.getOwner())) {
        return rewriter.notifyMatchFailure(
            op, "We dont support expand shapes yet.");
      } else if (auto convOp = dyn_cast<tosa::Conv2DOp>(use.getOwner())) {
        if (convOp.getInput() == tOutput) {
          permuteLayout(convOp, "input_layout", "nhwc", dims, true);
          convOp.getInputMutable().assign(tInput);
        } else if (convOp.getWeight() == tOutput) {
          permuteLayout(convOp, "filter_layout", "kyxc", dims, true);
          convOp.getWeightMutable().assign(tInput);
        } else {
          return convOp.emitWarning("transpose found leading to a conv2D input "
                                    "other than data or weight");
        }
      } else if (auto matMulOp = dyn_cast<tosa::MatMulOp>(use.getOwner())) {
        if (checkMatMulTransposeValid(matMulOp, dims).failed()) {
          return failure();
        }
        bool mmNonTrivial = isMatMulNonTrivial(dims);
        if (matMulOp.getA() == tOutput) {
          setTranspose(matMulOp, "transpose_a", mmNonTrivial);
          matMulOp.getAMutable().assign(tInput);
        } else if (matMulOp.getB() == tOutput) {
          setTranspose(matMulOp, "transpose_b", mmNonTrivial);
          matMulOp.getBMutable().assign(tInput);
        } else {
          return matMulOp.emitWarning(
              "transpose found leading to a matmul input other than A or B");
        }
      } else {
        return failure();
      }
    }
    return success();
  }

  // Fold transpose ops and convert convolution into changed layout.
  // case #0 : fold TP(NCHW2NHWC)+tosa.conv.NHWC+TP(NHWC2NCHW) back to
  //           rock.conv.NCHW
  // Pattern match start from the output transpose
  LogicalResult matchAndRewrite(tosa::TransposeOp top,
                                PatternRewriter &b) const final {
    SmallVector<int32_t> dims;
    if (failed(getTransposeDims(top.getOperand(1), dims))) {
      return failure();
    }

    Value tInput = top.getOperand(0);
    Value tOutput = top.getResult();

    if (tosa::Conv2DOp convOp = tInput.getDefiningOp<tosa::Conv2DOp>()) {
      if (checkInputHasUses(b, top, tInput).failed()) {
        return failure();
      }
      // tosa.conv2d output is transpose
      permuteLayout(convOp, "output_layout", "nhwk", dims);
      convOp->getResult(0).setType(tOutput.getType());
      top->replaceAllUsesWith(convOp);
    } else if (tosa::MatMulOp matMulOp =
                   tInput.getDefiningOp<tosa::MatMulOp>()) {

      if (checkInputHasUses(b, top, tInput).failed()) {
        return failure();
      }
      if (checkMatMulTransposeValid(matMulOp, dims).failed()) {
        return failure();
      }
      setTranspose(matMulOp, "transpose_c", isMatMulNonTrivial(dims));
      matMulOp->getResult(0).setType(tOutput.getType());
      top->replaceAllUsesWith(matMulOp);
    } else {
      if (mergeTransposeWithGemmLikeOp(b, tOutput, dims, tInput).failed()) {
        return failure();
      }
    }

    if (top.use_empty())
      b.eraseOp(top);
    return success();
  }
};

// In Tosa canonicalize, a transpose of NCHW to NHWC where H==W==1 will
// convert to a reshape because it does not change memory layout. Then in
// TosaToTensor conversion, the reshape is replaced by this pattern:
//     %0 = collapse(filters[KCHW]) -> [KC]
//     %1 = expand(%0[KC]) -> [KHWC]
// If this feeds into a conv2d as filter, we will drop the collapse/expand and
// update the filter_layout attribute.
struct CollapseExpandRewritePattern
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  bool checkExpand(tensor::ExpandShapeOp expOp) const {
    auto srcSh = cast<ShapedType>(expOp.getOperand(0).getType()).getShape();
    auto resSh = cast<ShapedType>(expOp.getResultType()).getShape();
    // [[0, 1, 2], [3]]
    // NC -> NHWC
    if (srcSh.size() == 2 && resSh.size() == 4 && srcSh[0] == resSh[0] &&
        srcSh[1] == resSh[3] && resSh[1] == 1 && resSh[2] == 1) {
      return true;
    }
    return false;
  }

  bool checkCollapse(tensor::CollapseShapeOp colOp) const {
    auto srcSh = cast<ShapedType>(colOp.getOperand().getType()).getShape();
    auto resSh = cast<ShapedType>(colOp.getResultType()).getShape();
    // [[0], [1, 2, 3]]
    // NCHW -> NC
    if (srcSh.size() == 4 && resSh.size() == 2 && srcSh[0] == resSh[0] &&
        srcSh[1] == resSh[1] && srcSh[2] == 1 && srcSh[3] == 1) {
      return true;
    }
    return false;
  }

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expOp,
                                PatternRewriter &b) const final {
    LogicalResult lres = failure();
    Value expInp = expOp.getOperand(0);
    Value expOut = expOp.getResult();

    if (!checkExpand(expOp))
      return failure();

    auto colOp = expInp.getDefiningOp<tensor::CollapseShapeOp>();
    if (colOp && checkCollapse(colOp)) {
      auto colInp = colOp.getOperand();

      for (Operation *usr : expOut.getUsers()) {
        if (auto convOp = dyn_cast<tosa::Conv2DOp>(usr)) {
          if (convOp.getOperand(1) == expOut) {
            // update filter_layout
            SmallVector<int32_t> dims{0, 2, 3, 1};
            permuteLayout(convOp, "filter_layout", "kyxc", dims, true);
            // replace filter input with collapse source
            convOp->replaceUsesOfWith(expOut, colInp);

            lres = success();
          }
        }
      }
    }

    return lres;
  }
};

// Tosa ops can broadcast values along axes, which allows for
// element-wise operations without fully-matching dimensions.  The
// Elementwise trait is strict about matching dimensions, but
// broadcastable ops are also element-wise, and we know that an
// additional set of ops are also element-wise.
static bool isElementwiseOp(Operation *op) {
  return op->hasTrait<OpTrait::Elementwise>() ||
         op->hasTrait<OpTrait::ResultsBroadcastableShape>() ||
         // clang-format off
    isa<tosa::CastOp,
        tosa::ClampOp,
        tosa::ErfOp,
        tosa::SigmoidOp,
        tosa::TanhOp,
        tosa::AbsOp,
        tosa::CeilOp,
        tosa::ClzOp,
        tosa::ExpOp,
        tosa::FloorOp,
        tosa::LogOp,
        tosa::LogicalNotOp,
        tosa::NegateOp,
        tosa::ReciprocalOp,
        tosa::RsqrtOp,
        tosa::SelectOp,
        tosa::EqualOp,
        tosa::GreaterOp,
        tosa::GreaterEqualOp
       >(op);
  // clang-format on
}

struct AttentionRewritePattern : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern<tosa::MatMulOp>::OpRewritePattern;

  Value getValueNonReshapeOp(Value val) const {
    while (val.getDefiningOp() &&
           (val.getDefiningOp<tensor::CollapseShapeOp>() ||
            val.getDefiningOp<tensor::ExpandShapeOp>())) {
      val = val.getDefiningOp()->getOperand(0);
    }
    return val;
  }

  template <typename TosaOp>
  TosaOp getDefiningNonReshapeOp(Value val) const {
    while (val.getDefiningOp<tensor::CollapseShapeOp>() ||
           val.getDefiningOp<tensor::ExpandShapeOp>()) {
      val = val.getDefiningOp()->getOperand(0);
    }
    return val.getDefiningOp<TosaOp>();
  }

  template <typename TosaOp>
  TosaOp getDefiningNonReshapeOpNonCastOp(Value val) const {
    while (val.getDefiningOp<tensor::CollapseShapeOp>() ||
           val.getDefiningOp<tensor::ExpandShapeOp>() ||
           val.getDefiningOp<tosa::CastOp>()) {
      val = val.getDefiningOp()->getOperand(0);
    }
    return val.getDefiningOp<TosaOp>();
  }

  FailureOr<Value> addBroadcast(Value val) const {
    if (auto add = getDefiningNonReshapeOp<tosa::AddOp>(val)) {
      // this is a broadcast add, one of the arguments comes is the actual
      // value, the other is a 0 constant
      Value nonZero;
      if (auto constOp =
              getDefiningNonReshapeOp<tosa::ConstOp>(add.getInput1())) {
        if (isConstantZero(constOp.getResult()))
          nonZero = add.getInput2();
      } else if (auto constOp = getDefiningNonReshapeOp<arith::ConstantOp>(
                     add.getInput1())) {
        if (isConstantZero(constOp.getResult()))
          nonZero = add.getInput2();
      }

      if (auto constOp =
              getDefiningNonReshapeOp<tosa::ConstOp>(add.getInput2())) {
        if (isConstantZero(constOp.getResult()))
          nonZero = add.getInput1();
      } else if (auto constOp = getDefiningNonReshapeOp<arith::ConstantOp>(
                     add.getInput2())) {
        if (isConstantZero(constOp.getResult()))
          nonZero = add.getInput1();
      }
      if (nonZero)
        return nonZero;
    }
    return failure();
  }

  FailureOr<std::pair<Value, Value>> getKVCache(Value softmaxInput) const {
    auto select = getDefiningNonReshapeOp<tosa::SelectOp>(softmaxInput);
    if (select) {
      // Check onTrue is -inf
      auto onTrue = select.getOnTrue();
      bool isConsNegInf = false;
      if (auto constOp = getDefiningNonReshapeOp<arith::ConstantOp>(onTrue))
        isConsNegInf = isConstIsNegInf(constOp.getResult());
      else if (auto constOp = getDefiningNonReshapeOp<tosa::ConstOp>(onTrue))
        isConsNegInf = isConstIsNegInf(constOp.getResult());

      if (!isConsNegInf)
        return failure();

      auto pred = select.getPred();
      if (auto greaterEqual =
              getDefiningNonReshapeOpNonCastOp<tosa::GreaterEqualOp>(pred)) {
        // input1 is a constant with a range from 0 to maxSeqLen
        auto input1 = greaterEqual.getInput1();
        FailureOr<Value> maybeNonZero1 = addBroadcast(input1);
        if (failed(maybeNonZero1))
          return failure();

        // check that maybeNonZero1 is a const with range 0..maxSeqLen
        bool isRange = false;
        if (auto constRange = getDefiningNonReshapeOp<arith::ConstantOp>(
                maybeNonZero1.value()))
          isRange = isConstRange(constRange.getResult());
        if (auto constRange =
                getDefiningNonReshapeOp<tosa::ConstOp>(maybeNonZero1.value()))
          isRange = isConstRange(constRange.getResult());

        if (!isRange)
          return failure();

        // input2 comes from argument: currentSeqLen
        auto input2 = greaterEqual.getInput2();
        FailureOr<Value> maybeNonZero2 = addBroadcast(input2);
        if (failed(maybeNonZero2))
          return failure();

        Value currentSeqLen = getValueNonReshapeOp(maybeNonZero2.value());
        Value result = select.getOnFalse();
        return std::make_pair(result, currentSeqLen);
      }
    }
    return failure();
  }

  FailureOr<std::tuple<Value, bool, Value>>
  maybeSoftmaxNumerator(Value val) const {
    Value currentSeqLen;
    tosa::ExpOp exp = getDefiningNonReshapeOp<tosa::ExpOp>(val);
    if (!exp)
      return failure();

    auto sub = getDefiningNonReshapeOp<tosa::SubOp>(exp.getInput1());
    if (!sub)
      return failure();

    bool hasTosaReduce = false;
    Value result;
    auto rmax = getDefiningNonReshapeOp<tosa::ReduceMaxOp>(sub.getInput2());
    if (rmax) {
      if (rmax.getInput() != sub.getInput1())
        return failure();

      hasTosaReduce = true;
      result = rmax.getInput();
    } else {
      if (sub.getInput1() != sub.getInput2())
        return failure();

      hasTosaReduce = false;
      result = sub.getInput1();
    }
    // Note that non KV-Cache fusions might have tosa.select
    // so, if the checks for kv-cache fail, we just keep going
    auto maybeKVCache = getKVCache(result);
    if (succeeded(maybeKVCache))
      std::tie(result, currentSeqLen) = maybeKVCache.value();

    return std::make_tuple(result, hasTosaReduce, currentSeqLen);
  }

  FailureOr<std::tuple<Value, bool, Value>>
  maybeSoftmaxDenominator(Value val) const {
    FailureOr<std::tuple<Value, bool, Value>> result;
    auto rsum = getDefiningNonReshapeOp<tosa::ReduceSumOp>(val);
    if (rsum) {
      result = maybeSoftmaxNumerator(rsum.getInput());
      if (succeeded(result) && !(std::get<1>(result.value()))) {
        // if we see tosa::Reduce Op in the denominator then we expect to see
        // tosa::Reduce Op in the numerator as well
        return failure();
      }
      return result;
    }
    result = maybeSoftmaxNumerator(val);
    if (succeeded(result) && std::get<1>(result.value())) {
      // if we don't see tosa::Reduce Op in the denominator then we expect to
      // not see any tosa::Reduce Op in the numerator as well
      return failure();
    }
    return result;
  }

  FailureOr<std::tuple<Value, bool, Value>> maybeSoftmax(Value val) const {
    auto mul = getDefiningNonReshapeOp<tosa::MulOp>(val);
    if (!mul) {
      return failure();
    }
    if (auto rec =
            getDefiningNonReshapeOp<tosa::ReciprocalOp>(mul.getInput1())) {
      return maybeSoftmaxDenominator(rec.getInput1());
    } else if (auto rec = getDefiningNonReshapeOp<tosa::ReciprocalOp>(
                   mul.getInput2())) {
      return maybeSoftmaxDenominator(rec.getInput1());
    } else {
      return failure();
    }
  }

  Value normalizeInputTensor(PatternRewriter &rewriter, Location loc,
                             TypedValue<TensorType> inputTensor) const {
    if (!inputTensor) {
      return inputTensor;
    }
    ArrayRef<int64_t> shape = inputTensor.getType().getShape();
    SmallVector<int64_t, 4> reverseInputShape =
        llvm::to_vector<4>(llvm::reverse(shape));
    SmallVector<int64_t, 4> normalizedShape;
    int collapsedBatchLen = 1;
    for (int64_t dimLen : ArrayRef<int64_t>{reverseInputShape}.slice(2)) {
      collapsedBatchLen *= dimLen;
    }
    normalizedShape.push_back(collapsedBatchLen);
    normalizedShape.push_back(reverseInputShape[1]);
    normalizedShape.push_back(reverseInputShape[0]);
    auto normalizedType = RankedTensorType::get(
        normalizedShape, inputTensor.getType().getElementType());
    auto reshapeOp = rewriter.create<tosa::ReshapeOp>(
        loc, normalizedType, inputTensor,
        rewriter.getDenseI64ArrayAttr(normalizedShape));
    return reshapeOp;
  }

  Value addBlockArgument(OpBuilder &b, Value val, Block *block,
                         Location loc) const {
    RankedTensorType valType = cast<RankedTensorType>(val.getType());
    val = block->addArgument(
        MemRefType::get(valType.getShape(), valType.getElementType()), loc);
    val = rock::getAsTensor(b, loc, val);
    return val;
  }

  // This function traverse an upward tree where the root is the softmax
  // input. It traverses the tree until it hit the gemm or last elemwise
  // operation that may or maynot be interleaved with reshape-like ops. Note
  // there is a TODO to explore relaxing reshape-like ops constraints to more
  // of rock.transforms. (See the implementation for the TODO)
  std::tuple<Value, FailureOr<tosa::MatMulOp>> getPreSoftmaxElementwiseRegion(
      Value input, OpBuilder &regionBuilder, Block *block,
      SmallVector<Value> &elementwiseArgs,
      std::optional<Location> loc = std::nullopt, bool doRewrite = false,
      int recDepth = 0) const {
    PatternRewriter::InsertionGuard guard(regionBuilder);
    regionBuilder.setInsertionPointToEnd(block);
    // If the matmul is found, we return this information to the
    // root.
    LLVM_DEBUG(llvm::dbgs()
               << std::string(recDepth, '\t')
               << "getPreSoftmaxElementwiseRegion:input=" << input << "\n");
    if (tosa::MatMulOp matmul = input.getDefiningOp<tosa::MatMulOp>()) {
      Value matmulMemRef;
      if (doRewrite) {
        matmulMemRef =
            addBlockArgument(regionBuilder, input, block, loc.value());
        rock::AttentionOp attnOp =
            cast<rock::AttentionOp>(block->getParentOp());
        attnOp.setFirstGemmIdx(block->getArguments().size() - 1);
      }
      LLVM_DEBUG(llvm::dbgs() << std::string(recDepth, '\t')
                              << "matmul found. terminating recursion.\n");
      return {matmulMemRef, matmul};
    }
    if (tosa::ConstOp constOp = input.getDefiningOp<tosa::ConstOp>()) {
      Value newConstOpRes;
      if (doRewrite) {
        auto *newConstOp = regionBuilder.clone(*constOp);
        newConstOpRes = newConstOp->getResult(0);
      }
      LLVM_DEBUG(llvm::dbgs() << std::string(recDepth, '\t')
                              << "const found. terminating recursion.\n");
      return {newConstOpRes, failure()};
    }
    Operation *op = input.getDefiningOp();
    // Right now, this is a bit restricted that we only allow reshape-like
    // ops between in the elementwise tree that get fused to the fusion point.
    // TODO: however, the latest code gridwise-gemm-to-blockwise should tackle
    // more cases. The absolute restriction is gemm0Output to Linalg block
    // should contain invertible transforms, but that's future work.
    if (!op || (!isElementwiseOp(op) &&
                !isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op))) {
      Value blockArg;
      if (doRewrite) {
        blockArg = addBlockArgument(regionBuilder, input, block, loc.value());
      }
      elementwiseArgs.push_back(input);
      LLVM_DEBUG(llvm::dbgs()
                 << std::string(recDepth, '\t')
                 << "unsupported region op found. terminating recursion.\n");
      return {blockArg, failure()};
    }
    // Following section recursively calls into the left and right
    // sub-tree to grab as much of the elementwise tree rooted on softmax
    // input.
    mlir::IRMapping mapper;
    SmallVector<Value> newOperands;

    FailureOr<mlir::tosa::MatMulOp> maybeMatMul = failure();
    for (auto operand : op->getOperands()) {
      auto [result, maybeSubTreeMatMul] = getPreSoftmaxElementwiseRegion(
          operand, regionBuilder, block, elementwiseArgs, loc, doRewrite,
          recDepth + 1);
      mapper.map(operand, result);
      newOperands.push_back(result);
      if (succeeded(maybeSubTreeMatMul)) {
        maybeMatMul = maybeSubTreeMatMul;
      }
    }

    Value res;
    if (doRewrite) {
      auto *newOp = regionBuilder.clone(*op, mapper);
      res = newOp->getResult(0);
    }
    // We convey to the caller the result
    // of the cloning as well if this subtree
    // contains the first matmul.
    if (succeeded(maybeMatMul)) {
      LLVM_DEBUG(llvm::dbgs() << std::string(recDepth, '\t')
                              << "a subtree have a matmul in it.\n");
      return {res, maybeMatMul};
    }
    LLVM_DEBUG(llvm::dbgs() << std::string(recDepth, '\t')
                            << "none of subtress have a matmul in it.\n");
    return {res, failure()};
  }

  LogicalResult match(tosa::MatMulOp op) const override {
    FailureOr<std::tuple<Value, bool, Value>> softmaxInputResult =
        maybeSoftmax(op.getA());
    if (failed(softmaxInputResult)) {
      return failure();
    }

    Value softmaxInput;
    bool hasReduceOp;
    std::tie(softmaxInput, hasReduceOp, std::ignore) =
        softmaxInputResult.value();
    OpBuilder b{op};
    SmallVector<Value> vec;
    FailureOr<tosa::MatMulOp> maybeFirstMatMul;
    std::tie(std::ignore, maybeFirstMatMul) =
        getPreSoftmaxElementwiseRegion(softmaxInput, b, nullptr, vec);

    if (succeeded(maybeFirstMatMul)) {
      TypedValue<TensorType> matC = maybeFirstMatMul.value().getC();
      ArrayRef<int64_t> shapeC = matC.getType().getShape();
      bool isDotProduct = *(std::prev(shapeC.end(), 1)) == 1;
      isDotProduct &= *(std::prev(shapeC.end(), 2)) == 1;

      LLVM_DEBUG(llvm::dbgs()
                 << "first matmul = " << maybeFirstMatMul.value() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "hasReduceOp = " << hasReduceOp << "\n");
      if (isDotProduct && hasReduceOp) {
        return failure();
      }
      if (!isDotProduct && !hasReduceOp) {
        return failure();
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "first matmul not found\n");
    }

    return maybeFirstMatMul;
  }

  void rewrite(tosa::MatMulOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value softmaxInput, currentSeqLen;
    std::tie(softmaxInput, std::ignore, currentSeqLen) =
        maybeSoftmax(op.getA()).value();
    auto outputType = cast<RankedTensorType>(op.getType());
    Value output = rewriter.create<bufferization::AllocTensorOp>(
        loc, outputType, ValueRange{});
    StringAttr arch;
    std::optional<uint32_t> numCu;
    rock::GemmFeatures features;
    std::tie(arch, numCu, features) = getArchAttributes(op, op.getType());
    SmallVector<Value> elementwiseOtherArgs;

    FailureOr<tosa::MatMulOp> maybeFirstMatMul;
    std::tie(std::ignore, maybeFirstMatMul) = getPreSoftmaxElementwiseRegion(
        softmaxInput, rewriter, nullptr, elementwiseOtherArgs);
    // This is guranteed by the matcher
    tosa::MatMulOp firstMatMulOp = maybeFirstMatMul.value();
    IntegerAttr numCUAttr =
        numCu.has_value() ? rewriter.getI32IntegerAttr(numCu.value()) : nullptr;

    rock::AttentionOp attnOp = rewriter.create<rock::AttentionOp>(
        loc, outputType, firstMatMulOp.getA(), firstMatMulOp.getB(), op.getB(),
        elementwiseOtherArgs, currentSeqLen, output,
        // TODO(implement transpose fusion support here)
        /*qTransposed=*/nullptr,
        /*kTransposed=*/nullptr,
        /*vTransposed=*/nullptr,
        /*oTransposed=*/nullptr, arch,
        rewriter.getAttr<rock::GemmFeaturesAttr>(features), numCUAttr,
        /*params0=*/nullptr, /*params1=*/nullptr,
        /*firstGemmIdx=*/rewriter.getI32IntegerAttr(0));

    Block *preSoftmaxElemwiseBlock = &attnOp.getPreSoftmaxBody().emplaceBlock();
    FailureOr<tosa::MatMulOp> maybeMatMul;
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(preSoftmaxElemwiseBlock);
      Value res;
      std::tie(res, maybeMatMul) = getPreSoftmaxElementwiseRegion(
          softmaxInput, rewriter, preSoftmaxElemwiseBlock, elementwiseOtherArgs,
          loc, true);
      RankedTensorType resTensorType = cast<RankedTensorType>(res.getType());
      MemRefType resMemRefType = MemRefType::get(
          resTensorType.getShape(), resTensorType.getElementType());
      Value resMemref =
          rewriter.create<bufferization::ToMemrefOp>(loc, resMemRefType, res);
      Value outMemref =
          preSoftmaxElemwiseBlock->addArgument(resMemRefType, loc);
      rewriter.create<memref::CopyOp>(loc, resMemref, outMemref);
      rewriter.create<rock::YieldOp>(loc);
    }
    rewriter.replaceOp(op, attnOp.getResult());
  }
};

template <typename TosaReduceOp>
typename std::enable_if_t<
    std::is_same<TosaReduceOp, tosa::ReduceSumOp>::value ||
        std::is_same<TosaReduceOp, tosa::ReduceMaxOp>::value,
    LogicalResult> static matchAndRewriteReductions(TosaReduceOp op,
                                                    rock::ReduceMethod rMethod,
                                                    Attribute outputInitVal,
                                                    ConversionPatternRewriter
                                                        &rw) {
  Location loc = op->getLoc();
  auto outputType = cast<RankedTensorType>(op.getType());
  Value output =
      rw.create<bufferization::AllocTensorOp>(loc, outputType, ValueRange{});
  StringAttr arch;
  std::optional<uint32_t> num_cu;
  rock::GemmFeatures features;
  std::tie(arch, num_cu, features) = getArchAttributes(op, op.getType());

  int32_t blockSize = 256;
  auto elementCount =
      cast<ShapedType>(op.getInput().getType()).getNumElements();
  int32_t gridSize = (elementCount + blockSize - 1) / blockSize;
  if (num_cu.has_value()) {
    gridSize = std::min((int32_t)(20 * num_cu.value()), gridSize);
  }

  auto rockReduce = rw.create<rock::ReduceOp>(
      loc, outputType, op.getInput(), output,
      rw.getAttr<rock::GemmFeaturesAttr>(features),
      rw.getAttr<rock::ReduceMethodAttr>(rMethod),
      rw.getIndexAttr(op.getAxis()), rw.getI32IntegerAttr(blockSize),
      rw.getI32IntegerAttr(gridSize),
      /*useLDS=*/nullptr,
      /*useDPP=*/nullptr);

  func::FuncOp func = op->template getParentOfType<func::FuncOp>();
  FailureOr<int64_t> maybeRes = traceToRes(op.getOutput(), func);
  if (failed(maybeRes))
    return op.emitOpError(
        "can't trace the reduction output to a kernel result");
  int64_t resNumber = maybeRes.value();

  func.setResultAttr(resNumber, rock::PrefillAttr::getMnemonic(),
                     outputInitVal);
  func.setResultAttr(resNumber, "mhal.read_access", rw.getUnitAttr());
  // The original function also need the read access attr for the output.
  if (func->hasAttr("original_func")) {
    if (ModuleOp rootMod =
            func->getParentOfType<ModuleOp>()->getParentOfType<ModuleOp>()) {
      SymbolTable symTable(rootMod);
      SymbolRefAttr originalFuncAttr =
          func->getAttrOfType<SymbolRefAttr>("original_func");
      if (func::FuncOp originalFunc = dyn_cast<func::FuncOp>(
              symTable.lookupSymbolIn(rootMod, originalFuncAttr))) {
        originalFunc.setResultAttr(resNumber, "mhal.read_access",
                                   rw.getUnitAttr());
      }
    }
  }
  rw.replaceOp(op, rockReduce.getResult());
  return success();
}

class ReduceSumConverter final : public OpConversionPattern<tosa::ReduceSumOp> {
public:
  using OpConversionPattern<tosa::ReduceSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::ReduceSumOp op,
                                tosa::ReduceSumOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    Type elementType =
        cast<ShapedType>(op.getInput().getType()).getElementType();
    if (!elementType.isF32() && !elementType.isF16()) {
      return rw.notifyMatchFailure(
          op, "We only support F32 and F16 reductions, yet.");
    }
    Attribute outputInitVal = rw.getFloatAttr(elementType, 0.0000);
    return matchAndRewriteReductions(op, rock::ReduceMethod::Sum, outputInitVal,
                                     rw);
  }
};

class ReduceMaxConverter final : public OpConversionPattern<tosa::ReduceMaxOp> {
public:
  using OpConversionPattern<tosa::ReduceMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::ReduceMaxOp op,
                                tosa::ReduceMaxOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    Type elementType =
        cast<ShapedType>(op.getInput().getType()).getElementType();
    Attribute outputInitVal;
    if (elementType.isF32()) {
      outputInitVal = rw.getFloatAttr(
          elementType, APFloat::getInf(APFloat::IEEEsingle(), true));
    } else {
      return rw.notifyMatchFailure(op, "We only support F32 reductions, yet.");
    }
    return matchAndRewriteReductions(op, rock::ReduceMethod::Max, outputInitVal,
                                     rw);
  }
};

// We identify the pattern dummy add with implicit broadcasting
// and rewrite it to be rock.transform broadcast
class AddSplatZeroRewritePattern final : public OpRewritePattern<tosa::AddOp> {
public:
  using OpRewritePattern<tosa::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::AddOp op,
                                PatternRewriter &rw) const final {
    Location loc = op.getLoc();
    TypedValue<TensorType> inp1 = op.getInput1();
    TypedValue<TensorType> inp2 = op.getInput2();
    TypedValue<TensorType> out = op.getOutput();

    TypedValue<TensorType> bcastInput;
    if (isConstantZero(inp1))
      bcastInput = inp2;
    if (isConstantZero(inp2)) {
      if (bcastInput) {
        return rw.notifyMatchFailure(op, "both inputs are splat zeros");
      }
      bcastInput = inp1;
    }
    if (bcastInput) {
      Value bcast =
          insertBroadcast(bcastInput, out.getType().getShape(), loc, rw);
      rw.replaceOp(op, bcast);
      return success();
    }
    return rw.notifyMatchFailure(op, "none of the inputs are splat zeros");
  }
};

} // namespace

void tosa::populateTosaToRockConversionPatterns(MLIRContext *context,
                                                RewritePatternSet &patterns) {
  patterns.add<ConvConverter<tosa::Conv2DOp>, ConvConverter<tosa::Conv3DOp>,
               MatMulConverter, ReduceSumConverter, ReduceMaxConverter>(
      context);
}

void tosa::populateTosaToRockTensorConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<AttentionRewritePattern, TransposeRewritePattern,
               CollapseExpandRewritePattern, AddSplatZeroRewritePattern>(
      context);
}
