echo running ExpressionEvaluatorIntTest
mkdir -p $CMSSW_BASE/src/ExpressionEvaluatorTests
cp -r $CMSSW_BASE/src/CommonTools/Utils/test/ExpressionEvaluatorTests/EEIntTest $CMSSW_BASE/src/ExpressionEvaluatorTests/EEIntTest
scram b ExpressionEvaluatorTests/EEIntTest
ls -l $CMSSW_BASE/include/$SCRAM_ARCH/ExpressionEvaluatorTests/EEIntTest/src

