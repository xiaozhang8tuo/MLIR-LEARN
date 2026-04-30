add_library(MLIRMiniInfer16 STATIC
  MiniInferDialect.cpp
  MiniInferFold.cpp
  LowerToLinalg.cpp
)
target_link_libraries(MLIRMiniInfer16
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)
add_dependencies(MLIRMiniInfer16 MiniInferOpsIncGen16)
