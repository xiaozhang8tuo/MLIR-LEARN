#include "MiniInfer/MiniInferOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

using namespace mlir;
using namespace mlir::mini_infer;

// const op fold: 直接返回常量属性
OpFoldResult ConstOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

// add(x, 0) -> x;  add(0, x) -> x
OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<DenseElementsAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<DenseElementsAttr>(adaptor.getRhs());

  if (rhsAttr && rhsAttr.isSplat()) {
    auto elemType = rhsAttr.getElementType();
    if (isa<FloatType>(elemType) &&
        rhsAttr.getSplatValue<APFloat>().isZero())
      return getLhs();
  }
  if (lhsAttr && lhsAttr.isSplat()) {
    auto elemType = lhsAttr.getElementType();
    if (isa<FloatType>(elemType) &&
        lhsAttr.getSplatValue<APFloat>().isZero())
      return getRhs();
  }
  return {};
}
