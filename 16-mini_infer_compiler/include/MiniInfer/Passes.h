#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::mini_infer {
std::unique_ptr<Pass> createLowerToLinalgPass();
void registerPasses();
} // namespace mlir::mini_infer
