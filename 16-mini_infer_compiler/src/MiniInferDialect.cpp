#include "MiniInfer/MiniInferOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "MiniInfer/MiniInferDialect.cpp.inc"

#define GET_OP_CLASSES
#include "MiniInfer/MiniInferOps.cpp.inc"

namespace mlir::mini_infer {

void MiniInferDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MiniInfer/MiniInferOps.cpp.inc"
      >();
}

// fold产出常量时，框架调这个函数把Attribute物化成Op
Operation *MiniInferDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  if (auto elemAttr = dyn_cast<ElementsAttr>(value))
    return builder.create<ConstOp>(loc, type, elemAttr);
  return nullptr;
}

} // namespace mlir::mini_infer
