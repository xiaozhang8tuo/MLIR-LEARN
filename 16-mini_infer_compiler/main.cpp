// mini inference compiler demo
// 构建一个简单的推理图: matmul -> add(bias) -> relu
// 然后跑 fold + lower + bufferize + LLVM lowering，最终 JIT 执行

#include "MiniInfer/MiniInferOps.h"
#include "MiniInfer/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>

using namespace mlir;

int main() {
  // --- 注册所有需要的 Dialect ---
  DialectRegistry registry;
  registry.insert<mini_infer::MiniInferDialect, func::FuncDialect,
                  arith::ArithDialect, linalg::LinalgDialect,
                  tensor::TensorDialect, memref::MemRefDialect,
                  scf::SCFDialect, cf::ControlFlowDialect,
                  bufferization::BufferizationDialect,
                  LLVM::LLVMDialect>();
  // ExecutionEngine 需要 LLVM IR 翻译
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  // 注册 bufferization 接口扩展
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);

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
  // 让 convert-func-to-llvm 生成 C 接口包装函数
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(&context));

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

  // =============================================
  // Phase 1: Optimize + Lower to Linalg
  // =============================================
  PassManager pm(&context);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(mini_infer::createLowerToLinalgPass());

  if (failed(pm.run(module))) {
    llvm::errs() << "Phase 1 (lower to linalg) failed!\n";
    return 1;
  }

  llvm::outs() << "\n=== Lowering 到 Linalg ===\n";
  module->dump();

  // =============================================
  // Phase 2: Bufferize (tensor -> memref)
  // =============================================
  PassManager pm2(&context);
  pm2.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm2.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  pm2.addPass(memref::createExpandStridedMetadataPass());
  pm2.addPass(createCanonicalizerPass());

  if (failed(pm2.run(module))) {
    llvm::errs() << "Phase 2 (bufferization) failed!\n";
    return 1;
  }

  // =============================================
  // Phase 3: Linalg -> Loops -> LLVM
  // =============================================
  PassManager pm3(&context);
  pm3.addPass(createConvertLinalgToLoopsPass());
  pm3.addPass(createLowerAffinePass());
  pm3.addPass(createConvertSCFToCFPass());
  pm3.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm3.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  pm3.addPass(createConvertControlFlowToLLVMPass());
  pm3.addPass(createConvertFuncToLLVMPass());
  pm3.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm3.run(module))) {
    llvm::errs() << "Phase 3 (lower to LLVM) failed!\n";
    return 1;
  }

  llvm::outs() << "\n=== 最终 LLVM IR Dialect ===\n";
  module->dump();

  // =============================================
  // Phase 4: JIT 执行
  // =============================================
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto jitOrErr = ExecutionEngine::create(module);
  if (!jitOrErr) {
    llvm::errs() << "JIT 创建失败: " << jitOrErr.takeError() << "\n";
    return 1;
  }
  auto &jit = *jitOrErr.get();

  // 准备输入: 4x8 全1矩阵
  float inputData[4 * 8];
  std::fill_n(inputData, 4 * 8, 1.0f);
  StridedMemRefType<float, 2> inputRef{};
  inputRef.basePtr = inputData;
  inputRef.data = inputData;
  inputRef.offset = 0;
  inputRef.sizes[0] = 4;
  inputRef.sizes[1] = 8;
  inputRef.strides[0] = 8;
  inputRef.strides[1] = 1;

  // 输出 descriptor (函数内部会分配数据并填充这个descriptor)
  StridedMemRefType<float, 2> outputRef{};
  memset(&outputRef, 0, sizeof(outputRef));

  // _mlir_ciface_inference(result_ptr, input_ptr) - 直接调用 C 接口
  auto sym = jit.lookup("_mlir_ciface_inference");
  if (!sym) {
    llvm::errs() << "找不到 _mlir_ciface_inference: " << sym.takeError() << "\n";
    return 1;
  }
  auto inferFn = reinterpret_cast<void (*)(void *, void *)>(*sym);
  inferFn(&outputRef, &inputRef);

  // 打印结果
  // 期望: 4x4 全8.0 (input全1 × weights全1 → 每个元素=8, relu(8)=8)
  llvm::outs() << "\n=== JIT 执行结果 (期望: 4x4 全 8.0) ===\n";
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      llvm::outs() << outputRef.data[i * 4 + j] << " ";
    llvm::outs() << "\n";
  }

  return 0;
}
