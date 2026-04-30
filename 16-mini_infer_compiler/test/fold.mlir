// RUN: mini-opt16 %s --canonicalize | FileCheck %s

// add(x, 0) 应该被fold掉
// CHECK-LABEL: @test_fold_add_zero
// CHECK-NOT: mini_infer.add
// CHECK: return %arg0
func.func @test_fold_add_zero(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %zero = "mini_infer.const"() <{value = dense<0.0> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
  %1 = "mini_infer.add"(%arg0, %zero) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
