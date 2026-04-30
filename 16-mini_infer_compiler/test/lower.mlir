// RUN: mini-opt16 %s --lower-mini-infer | FileCheck %s

// matmul + relu 应该被lower成linalg
// CHECK-LABEL: @test_lower
// CHECK: linalg.matmul
// CHECK: arith.maximumf
func.func @test_lower(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %0 = "mini_infer.matmul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  %1 = "mini_infer.relu"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
