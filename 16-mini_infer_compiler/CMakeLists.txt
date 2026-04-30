set(ch_num 16)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

set(TUTORIA_TOOL_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(TUTORIA_NS_OPT ${TUTORIA_TOOL_DIR}/mini-opt16)

add_subdirectory(include)
add_subdirectory(src)

# 主demo程序
add_executable(CH-${ch_num} main.cpp)
target_link_libraries(CH-${ch_num}
  MLIRMiniInfer16
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRPass
  MLIRTransforms
  MLIRParser
)

# lit测试用的opt工具
add_executable(mini-opt${ch_num} mini-opt.cpp)
target_link_libraries(mini-opt${ch_num}
  MLIRMiniInfer16
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRPass
  MLIRTransforms
  MLIRParser
  MLIRMlirOptMain
)

add_subdirectory(test)
