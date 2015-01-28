echo running ExpressionEvaluatorUnitTest
mkdir -p $CMSSW_BASE/src/ExpressionEvaluatorTests
cp -r $CMSSW_BASE/src/CommonTools/Utils/test/ExpressionEvaluatorTests/EEUnitTest $CMSSW_BASE/src/ExpressionEvaluatorTests/EEUnitTest
scram b ExpressionEvaluatorTests/EEUnitTest
ls -l $CMSSW_BASE/include/$SCRAM_ARCH/ExpressionEvaluatorTests/EEUnitTest/src
