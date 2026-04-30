// mini-opt: 用于lit测试
#include "MiniInfer/MiniInferOps.h"
#include "MiniInfer/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::mini_infer::MiniInferDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tensor::TensorDialect>();

  mlir::registerTransformsPasses();
  mlir::mini_infer::registerPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Mini Infer Compiler Opt", registry));
}
