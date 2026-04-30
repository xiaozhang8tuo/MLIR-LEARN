// mini_infer -> linalg lowering pass
#include "MiniInfer/MiniInferOps.h"
#include "MiniInfer/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct ConstOpLowering : public OpConversionPattern<mini_infer::ConstOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mini_infer::ConstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

struct MatMulOpLowering : public OpConversionPattern<mini_infer::MatMulOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mini_infer::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType = resultType.getElementType();

    // 创建零初始化的输出tensor (matmul累加需要)
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
    Value filled = rewriter.create<linalg::FillOp>(loc, zero, init)
                       .getResult(0);

    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{filled});
    rewriter.replaceOp(op, matmul.getResults());
    return success();
  }
};

struct ReLUOpLowering : public OpConversionPattern<mini_infer::ReLUOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mini_infer::ReLUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    auto elemType = resultType.getElementType();
    int64_t rank = resultType.getRank();

    Value init =
        rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
    SmallVector<AffineMap> maps(2,
                                rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank,
                                           utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType}, ValueRange{adaptor.getInput()},
        ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value zero = b.create<arith::ConstantOp>(loc,
                                                    b.getZeroAttr(elemType));
          Value relu = b.create<arith::MaximumFOp>(loc, args[0], zero);
          b.create<linalg::YieldOp>(loc, relu);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct AddOpLowering : public OpConversionPattern<mini_infer::AddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mini_infer::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    Value init =
        rewriter.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
    SmallVector<AffineMap> maps(3,
                                rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank,
                                           utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resultType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{init},
        maps, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

struct LowerMiniInferToLinalgPass
    : public PassWrapper<LowerMiniInferToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMiniInferToLinalgPass)
  StringRef getArgument() const final { return "lower-mini-infer"; }
  StringRef getDescription() const final {
    return "lower mini_infer ops to linalg";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    tensor::TensorDialect>();
  }
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    target.addIllegalDialect<mini_infer::MiniInferDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstOpLowering, MatMulOpLowering, ReLUOpLowering,
                 AddOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mini_infer::createLowerToLinalgPass() {
  return std::make_unique<LowerMiniInferToLinalgPass>();
}

void mlir::mini_infer::registerPasses() {
  PassRegistration<LowerMiniInferToLinalgPass>();
}
