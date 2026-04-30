// mini inference compiler demo
// 构建一个简单的推理图: matmul -> add(bias) -> relu
// 然后跑fold+lower pass，展示编译过程

#include "MiniInfer/MiniInferOps.h"
#include "MiniInfer/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main() {
  DialectRegistry registry;
  registry.insert<mini_infer::MiniInferDialect, func::FuncDialect,
                  arith::ArithDialect, linalg::LinalgDialect,
                  tensor::TensorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  // 创建module
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  // func @inference(%input: tensor<4x8xf32>) -> tensor<4x4xf32>
  auto inputType = RankedTensorType::get({4, 8}, builder.getF32Type());
  auto outputType = RankedTensorType::get({4, 4}, builder.getF32Type());
  auto funcType = builder.getFunctionType({inputType}, {outputType});
  auto func = builder.create<func::FuncOp>(loc, "inference", funcType);

  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value input = entry->getArgument(0);

  // weights: 8x4 全1矩阵 (简化)
  auto weightType = RankedTensorType::get({8, 4}, builder.getF32Type());
  auto weightAttr = DenseElementsAttr::get(weightType, 1.0f);
  Value weights = builder.create<mini_infer::ConstOp>(loc, weightType,
                                                       weightAttr);

  // bias: 4x4 全0 (会被fold掉)
  auto biasAttr = DenseElementsAttr::get(outputType, 0.0f);
  Value bias = builder.create<mini_infer::ConstOp>(loc, outputType, biasAttr);

  // matmul(input, weights) -> 4x4
  Value mm = builder.create<mini_infer::MatMulOp>(loc, outputType, input,
                                                   weights);

  // add(mm, bias) -> 被fold优化掉 (因为bias全0)
  Value added = builder.create<mini_infer::AddOp>(loc, outputType, mm, bias);

  // relu(added)
  Value result = builder.create<mini_infer::ReLUOp>(loc, outputType, added);

  builder.create<func::ReturnOp>(loc, result);

  llvm::outs() << "=== 原始 IR ===\n";
  module->dump();

  // 跑pass: canonicalize (触发fold) -> lower to linalg
  PassManager pm(&context);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mini_infer::createLowerToLinalgPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "pipeline failed!\n";
    return 1;
  }

  llvm::outs() << "\n=== Lowering 后 ===\n";
  module->dump();
  return 0;
}
