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

### 运行输出

```
=== 原始 IR ===
module {
  func.func @inference(%arg0: tensor<4x8xf32>) -> tensor<4x4xf32> attributes {llvm.emit_c_interface} {
    %0 = "mini_infer.const"() <{value = dense<1.000000e+00> : tensor<8x4xf32>}> : () -> tensor<8x4xf32>
    %1 = "mini_infer.const"() <{value = dense<0.000000e+00> : tensor<4x4xf32>}> : () -> tensor<4x4xf32>
    %2 = "mini_infer.matmul"(%arg0, %0) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    %3 = "mini_infer.add"(%2, %1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = "mini_infer.relu"(%3) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }
}

=== Lowering 到 Linalg ===
module {
  func.func @inference(%arg0: tensor<4x8xf32>) -> tensor<4x4xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<1.000000e+00> : tensor<8x4xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%arg0, %cst : tensor<4x8xf32>, tensor<8x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = tensor.empty() : tensor<4x4xf32>
    %4 = linalg.generic {indexing_maps = [...], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<4x4xf32>) outs(%3 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_1 = arith.constant 0.000000e+00 : f32
      %5 = arith.maximumf %in, %cst_1 : f32
      linalg.yield %5 : f32
    } -> tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }
}

=== 最终 LLVM IR Dialect ===
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_8x4xf32(dense<1.000000e+00> : tensor<8x4xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<8 x array<4 x f32>>
  llvm.func @inference(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    // 构建输入 memref descriptor
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.constant(8 : index) : i64
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(4 : index) : i64
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %13 = llvm.mlir.constant(8 : index) : i64
    %14 = llvm.mlir.constant(4 : index) : i64
    %15 = llvm.mlir.constant(1 : index) : i64
    %16 = llvm.mlir.constant(32 : index) : i64
    %17 = llvm.mlir.zero : !llvm.ptr
    %18 = llvm.getelementptr %17[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    // 获取全局常量 weights[8x4] 的指针
    %20 = llvm.mlir.addressof @__constant_8x4xf32 : !llvm.ptr
    %21 = llvm.getelementptr %20[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<4 x f32>>
    // 构建 weights 的 memref descriptor
    %22 = llvm.mlir.constant(3735928559 : index) : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.insertvalue %23, %24[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %21, %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.insertvalue %27, %26[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %13, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %14, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %14, %30[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %15, %31[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    // malloc 分配 matmul 输出 buffer (4x4xf32 = 64 bytes, 64-byte aligned)
    %33 = llvm.mlir.constant(4 : index) : i64
    %34 = llvm.mlir.constant(4 : index) : i64
    %35 = llvm.mlir.constant(1 : index) : i64
    %36 = llvm.mlir.constant(16 : index) : i64
    %37 = llvm.mlir.zero : !llvm.ptr
    %38 = llvm.getelementptr %37[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %39 = llvm.ptrtoint %38 : !llvm.ptr to i64
    %40 = llvm.mlir.constant(64 : index) : i64
    %41 = llvm.add %39, %40 : i64
    %42 = llvm.call @malloc(%41) : (i64) -> !llvm.ptr
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.mlir.constant(1 : index) : i64
    %45 = llvm.sub %40, %44 : i64
    %46 = llvm.add %43, %45 : i64
    %47 = llvm.urem %46, %40  : i64
    %48 = llvm.sub %46, %47 : i64
    %49 = llvm.inttoptr %48 : i64 to !llvm.ptr
    // 构建 matmul 输出的 memref descriptor
    %50 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %51 = llvm.insertvalue %42, %50[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.insertvalue %49, %51[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.mlir.constant(0 : index) : i64
    %54 = llvm.insertvalue %53, %52[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %33, %54[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %34, %55[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %34, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %35, %57[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%11 : i64)
  // === linalg.fill 展开: 双重循环将 matmul 输出清零 ===
  ^bb1(%59: i64):  // for i = 0..4
    %60 = llvm.icmp "slt" %59, %10 : i64
    llvm.cond_br %60, ^bb2, ^bb6
  ^bb2:
    llvm.br ^bb3(%11 : i64)
  ^bb3(%61: i64):  // for j = 0..4
    %62 = llvm.icmp "slt" %61, %10 : i64
    llvm.cond_br %62, ^bb4, ^bb5
  ^bb4:  // output[i][j] = 0.0
    %63 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.mlir.constant(4 : index) : i64
    %65 = llvm.mul %59, %64 : i64
    %66 = llvm.add %65, %61 : i64
    %67 = llvm.getelementptr %63[%66] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %12, %67 : f32, !llvm.ptr
    %68 = llvm.add %61, %9 : i64
    llvm.br ^bb3(%68 : i64)
  ^bb5:
    %69 = llvm.add %59, %9 : i64
    llvm.br ^bb1(%69 : i64)
  ^bb6:
    llvm.br ^bb7(%11 : i64)
  // === linalg.matmul 展开: 三重循环 C[i][j] += A[i][k] * B[k][j] ===
  ^bb7(%70: i64):  // for i = 0..4
    %71 = llvm.icmp "slt" %70, %10 : i64
    llvm.cond_br %71, ^bb8, ^bb15
  ^bb8:
    llvm.br ^bb9(%11 : i64)
  ^bb9(%72: i64):  // for j = 0..4
    %73 = llvm.icmp "slt" %72, %10 : i64
    llvm.cond_br %73, ^bb10, ^bb14
  ^bb10:
    llvm.br ^bb11(%11 : i64)
  ^bb11(%74: i64):  // for k = 0..8
    %75 = llvm.icmp "slt" %74, %8 : i64
    llvm.cond_br %75, ^bb12, ^bb13
  ^bb12:  // matmul 核心: C[i][j] += A[i][k] * B[k][j]
    %76 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.extractvalue %7[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.getelementptr %76[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %79 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mul %70, %79 : i64
    %81 = llvm.extractvalue %7[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mul %74, %81 : i64
    %83 = llvm.add %80, %82 : i64
    %84 = llvm.getelementptr %78[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %85 = llvm.load %84 : !llvm.ptr -> f32                   // A[i][k]
    %86 = llvm.extractvalue %32[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.mlir.constant(4 : index) : i64
    %88 = llvm.mul %74, %87 : i64
    %89 = llvm.add %88, %72 : i64
    %90 = llvm.getelementptr %86[%89] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %91 = llvm.load %90 : !llvm.ptr -> f32                   // B[k][j]
    %92 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.mlir.constant(4 : index) : i64
    %94 = llvm.mul %70, %93 : i64
    %95 = llvm.add %94, %72 : i64
    %96 = llvm.getelementptr %92[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %97 = llvm.load %96 : !llvm.ptr -> f32                   // C[i][j] (累加值)
    %98 = llvm.fmul %85, %91  : f32                          // A[i][k] * B[k][j]
    %99 = llvm.fadd %97, %98  : f32                          // C[i][j] += ...
    %100 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.mlir.constant(4 : index) : i64
    %102 = llvm.mul %70, %101 : i64
    %103 = llvm.add %102, %72 : i64
    %104 = llvm.getelementptr %100[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %99, %104 : f32, !llvm.ptr                    // store C[i][j]
    %105 = llvm.add %74, %9 : i64
    llvm.br ^bb11(%105 : i64)
  ^bb13:
    %106 = llvm.add %72, %9 : i64
    llvm.br ^bb9(%106 : i64)
  ^bb14:
    %107 = llvm.add %70, %9 : i64
    llvm.br ^bb7(%107 : i64)
  // === malloc 分配 relu 输出 buffer ===
  ^bb15:
    %108 = llvm.mlir.constant(4 : index) : i64
    %109 = llvm.mlir.constant(4 : index) : i64
    %110 = llvm.mlir.constant(1 : index) : i64
    %111 = llvm.mlir.constant(16 : index) : i64
    %112 = llvm.mlir.zero : !llvm.ptr
    %113 = llvm.getelementptr %112[%111] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %114 = llvm.ptrtoint %113 : !llvm.ptr to i64
    %115 = llvm.mlir.constant(64 : index) : i64
    %116 = llvm.add %114, %115 : i64
    %117 = llvm.call @malloc(%116) : (i64) -> !llvm.ptr
    %118 = llvm.ptrtoint %117 : !llvm.ptr to i64
    %119 = llvm.mlir.constant(1 : index) : i64
    %120 = llvm.sub %115, %119 : i64
    %121 = llvm.add %118, %120 : i64
    %122 = llvm.urem %121, %115  : i64
    %123 = llvm.sub %121, %122 : i64
    %124 = llvm.inttoptr %123 : i64 to !llvm.ptr
    // 构建 relu 输出的 memref descriptor
    %125 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %126 = llvm.insertvalue %117, %125[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.insertvalue %124, %126[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.mlir.constant(0 : index) : i64
    %129 = llvm.insertvalue %128, %127[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %108, %129[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.insertvalue %109, %130[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %132 = llvm.insertvalue %109, %131[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %110, %132[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb16(%11 : i64)
  // === linalg.generic(relu) 展开: 双重循环 relu_out[i][j] = max(matmul_out[i][j], 0) ===
  ^bb16(%134: i64):  // for i = 0..4
    %135 = llvm.icmp "slt" %134, %10 : i64
    llvm.cond_br %135, ^bb17, ^bb21
  ^bb17:
    llvm.br ^bb18(%11 : i64)
  ^bb18(%136: i64):  // for j = 0..4
    %137 = llvm.icmp "slt" %136, %10 : i64
    llvm.cond_br %137, ^bb19, ^bb20
  ^bb19:  // relu 核心: out[i][j] = max(in[i][j], 0.0)
    %138 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %139 = llvm.mlir.constant(4 : index) : i64
    %140 = llvm.mul %134, %139 : i64
    %141 = llvm.add %140, %136 : i64
    %142 = llvm.getelementptr %138[%141] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %143 = llvm.load %142 : !llvm.ptr -> f32                 // load matmul_result[i][j]
    %144 = llvm.intr.maximum(%143, %12)  : (f32, f32) -> f32 // max(x, 0)
    %145 = llvm.extractvalue %133[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.mlir.constant(4 : index) : i64
    %147 = llvm.mul %134, %146 : i64
    %148 = llvm.add %147, %136 : i64
    %149 = llvm.getelementptr %145[%148] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %144, %149 : f32, !llvm.ptr                   // store relu_result[i][j]
    %150 = llvm.add %136, %9 : i64
    llvm.br ^bb18(%150 : i64)
  ^bb20:
    %151 = llvm.add %134, %9 : i64
    llvm.br ^bb16(%151 : i64)
  ^bb21:
    llvm.return %133 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
  // C 接口包装函数: 解包输入 memref descriptor → 调用 inference → 存回输出 descriptor
  llvm.func @_mlir_ciface_inference(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.call @inference(%1, %2, %3, %4, %5, %6, %7) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %8, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
  }
}

=== JIT 执行结果 (期望: 4x4 全 8.0) ===
8.000000e+00 8.000000e+00 8.000000e+00 8.000000e+00
8.000000e+00 8.000000e+00 8.000000e+00 8.000000e+00
8.000000e+00 8.000000e+00 8.000000e+00 8.000000e+00
8.000000e+00 8.000000e+00 8.000000e+00 8.000000e+00
```

注意原始 IR 中的 `mini_infer.add(%2, %1)` 在 Lowering 后消失了——被 canonicalize pass 中的 `AddOp::fold` 优化掉（因为 bias 全零）。
