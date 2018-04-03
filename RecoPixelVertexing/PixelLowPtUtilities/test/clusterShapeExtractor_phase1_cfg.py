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
    input = cms.untracked.int32(20000)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/48D2453E-682D-E811-9F55-0025905A6082.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/4CDEB54A-682D-E811-A594-0025905B85DE.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/5226F8B3-692D-E811-A6A1-0025905B857A.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/56BA1C44-682D-E811-AC3B-0025905B85DA.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/5EFC4945-682D-E811-A2FF-0025905B8598.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/6A52C535-682D-E811-AA8D-0CC47A4D7698.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/6A95D4AF-692D-E811-9D95-0025905B85B8.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/84A9E4A8-692D-E811-AC93-0CC47A745294.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/8C7CB9A3-692D-E811-A78F-0CC47A7C340E.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/96133CEA-682D-E811-AD05-0CC47A4D764A.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/983AD7BE-6B2D-E811-A2E5-0CC47A7C3424.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/A2C419A4-692D-E811-B109-0CC47A4D7690.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/A2ED6045-672D-E811-B6B8-0CC47A78A3EE.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/A8B637F8-682D-E811-B734-003048FFCBB8.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/B02153AF-692D-E811-9A1D-0025905B8598.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/B2B3EFA1-692D-E811-B8DC-0CC47A4D75F6.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/C8780C47-682D-E811-9E60-0025905B85DC.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/D6D576D5-682D-E811-B5D6-0CC47A78A3EE.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/EAD209D7-682D-E811-8E8F-0025905B85FC.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realisticv3_PixelRealScenario_v1-v1/10000/FA0FE9DC-682D-E811-97B7-0CC47A4D7636.root',
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
