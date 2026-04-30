# MLIR-LEARN

MLIR 学习笔记 + 实践代码。基于 [violetDelia/MLIR-Tutorial](https://github.com/violetDelia/MLIR-Tutorial) 教程，从 CH2 到 CH15 跟着写，CH16 是自己设计的一个 mini 推理编译器。

## 环境

- LLVM 19.1.7（third_party/llvm-project，作为 submodule 拉取，不提交源码）
- CMake + Ninja + g++

## 构建

```bash
git submodule update --init --recursive
cmake . -G Ninja -B build
cd build
ninja CH-16 mini-opt16
```

## CH16: Mini Inference Compiler

自己写的一个简单推理编译器，把之前学的串起来。做了一个 `mini_infer` dialect，定义了 4 个 Op（const / matmul / add / relu），然后实现了：

1. **Fold**：ConstOp 返回属性值参与级联折叠；AddOp 做代数简化 `add(x, 0) → x`
2. **Lowering**：mini_infer → linalg（matmul → linalg.matmul, relu → linalg.generic + maximumf, add → linalg.generic + addf）
3. **Bufferization**：tensor → memref（OneShotBufferize）
4. **Lower to LLVM**：linalg → loops → scf → cf → llvm
5. **JIT 执行**：用 ExecutionEngine 跑出结果

Demo 构建了 `y = ReLU(W*x + b)` 计算图，input[4×8] 全1，weights[8×4] 全1，bias 全0（被 fold 消除），最终 JIT 输出 4×4 全 8.0。

### 文件结构

```
16-mini_infer_compiler/
├── main.cpp                 # C++ 构建推理图 + 跑完整 pipeline + JIT 执行
├── mini-opt.cpp             # mlir-opt 工具（lit 测试用）
├── include/MiniInfer/
│   ├── MiniInferOps.td      # TableGen 定义 dialect + 4个 Op
│   ├── MiniInferOps.h
│   └── Passes.h
├── src/
│   ├── MiniInferDialect.cpp # dialect 注册 + materializeConstant
│   ├── MiniInferFold.cpp    # ConstOp::fold + AddOp::fold
│   └── LowerToLinalg.cpp   # 4 个 ConversionPattern + Pass
└── test/
    ├── fold.mlir            # 验证 add(x,0) 被消除
    └── lower.mlir           # 验证 matmul+relu 降为 linalg
```

### 运行

```bash
cd build/16-mini_infer_compiler
./CH-16    # 打印原始IR → Linalg IR → LLVM IR → JIT执行结果
```

## 教程章节索引

| CH | 主题 | 目录 |
|----|------|------|
| 2 | Dialect 定义 | 2-define_dialect |
| 3 | Type 定义 | 3-define_type |
| 4 | Attribute 定义 | 4-define_attribute |
| 5 | Operation 定义 | 5-define_operation |
| 6 | Interface 定义 | 6-define_interface |
| 7 | IR 结构 | 7-ir_struct |
| 8 | Pass 定义 | 8-define_pass |
| 9 | Rewrite Pattern | 9-rewrite_pattern |
| 10 | mlir-opt + debug | 10-mlir_opt-and-debug |
| 11 | LIT 测试 | 11-lit_for_test |
| 12 | Dialect Conversion | 12-operation_lowing_pass |
| 13 | Pass Manager | 13-pass_manager |
| 14 | Fold / Canonicalization | 14-fold_and_canonicalization |
| 15 | Lowering to LLVM | 15-lowing_to_llvm |
| **16** | **Mini Inference Compiler（自己设计）** | **16-mini_infer_compiler** |
