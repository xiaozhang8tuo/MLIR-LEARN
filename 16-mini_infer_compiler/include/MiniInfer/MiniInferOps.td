// mini inference compiler - 自定义dialect
// 定义一个简单的推理图dialect，包含matmul/relu/add/const

#ifndef MINI_INFER_OPS_TD
#define MINI_INFER_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MiniInfer_Dialect : Dialect {
  let name = "mini_infer";
  let cppNamespace = "::mlir::mini_infer";
  let hasConstantMaterializer = 1;
}

class MiniInfer_Op<string mnemonic, list<Trait> traits = []> :
    Op<MiniInfer_Dialect, mnemonic, traits>;

def ConstOp : MiniInfer_Op<"const", [Pure, ConstantLike]> {
  let summary = "constant tensor";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyTensor:$result);
  let hasFolder = 1;
}

def MatMulOp : MiniInfer_Op<"matmul"> {
  let summary = "matrix multiply";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
}

def ReLUOp : MiniInfer_Op<"relu"> {
  let summary = "relu activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
}

def AddOp : MiniInfer_Op<"add"> {
  let summary = "element-wise add";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let hasFolder = 1;
}

#endif // MINI_INFER_OPS_TD
