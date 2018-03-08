# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:phase1_2017_realistic -n 10 --era Run2_2017 --eventcontent RECOSIM,DQM --runUnscheduled -s RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM --datatier GEN-SIM-RECO,DQMIO --geometry DB:Extended --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//00D31729-751E-E811-B1C6-0CC47A4D76C8.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//02B41E7C-741E-E811-B0CC-0CC47A7C34A6.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//2E3AC027-751E-E811-BD0A-0CC47A7452D8.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//3E26988B-741E-E811-8361-0025905A48F0.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//4AE89523-751E-E811-9105-0CC47A4D765A.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//50D5D97E-741E-E811-8E90-0CC47A7C354C.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//520BE32D-751E-E811-A5B3-0025905B8600.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//6C866F80-741E-E811-A63E-0CC47A4C8E34.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//70CC289B-741E-E811-A8D5-0025905B8606.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//78ADB42A-751E-E811-AE4A-0CC47A4C8E56.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//84711E41-751E-E811-9A4D-0025905A60DA.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//9E47F330-751E-E811-B105-003048FFD7AA.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//A291237C-741E-E811-B360-0CC47A7C34C4.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//A4C44D30-751E-E811-A7AA-0025905B855E.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//A6AAAF87-741E-E811-83DC-0025905B8562.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//AEE88480-741E-E811-8DA6-0CC47A4C8E34.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//C079F98B-741E-E811-A383-0025905A60B8.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//D0EDAA28-751E-E811-A6E4-0CC47A4C8E14.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//DC12581D-751E-E811-98DB-0CC47A4D766C.root',
    '/store/relval/CMSSW_10_1_0_pre2/RelValMinBias_13/GEN-SIM-DIGI-RAW/100X_upgrade2018_realistic_v11-v1/20000//E2DE9D28-751E-E811-B7A4-0CC47A4C8EBA.root',
),
    secondaryFileNames = cms.untracked.vstring()
)


#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(True)
)

process.Timing = cms.Service("Timing"
    ,summaryOnly = cms.untracked.bool(True)
)


process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True),
    numberOfThreads = cms.untracked.uint32(8),
    numberOfStreams = cms.untracked.uint32(8),
    wantSummary = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:step3.root'),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('file:step3_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction_trackingOnly)
process.prevalidation_step = cms.Path(process.globalPrevalidationTrackingOnly)
process.load('RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeExtractor_cfi')
process.clusterShapeExtractor.noBPIX1=False
process.clusterShapeExtractor_step = cms.Path(process.clusterShapeExtractor)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.clusterShapeExtractor_step)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)
from FWCore.ParameterSet.Utilities import cleanUnscheduled
process=cleanUnscheduled(process)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
