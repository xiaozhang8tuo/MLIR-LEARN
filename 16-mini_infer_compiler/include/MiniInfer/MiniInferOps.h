#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "MiniInfer/MiniInferDialect.h.inc"

#define GET_OP_CLASSES
#include "MiniInfer/MiniInferOps.h.inc"
