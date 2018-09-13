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
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//002CD53A-B538-E811-A45A-0CC47A7C353E.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//083FD50B-B538-E811-8230-0CC47A7C3450.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//0E3CA40C-B538-E811-BF3A-0CC47A4D768E.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//3478DAB1-B338-E811-8489-0CC47A7C354C.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//3CB8FF06-B538-E811-B422-0CC47A7C353E.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//50C35A28-B538-E811-891F-0025905A60EE.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//58462D48-B538-E811-9AE8-0CC47A7C349C.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//98F4CC3C-B538-E811-80B5-0CC47A4D7604.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//9E3C82FC-B538-E811-BBBC-0CC47A4D75EE.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//B4FF1CDA-B438-E811-8D56-0CC47A78A2F6.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//C4969AF3-B238-E811-BCD0-0CC47A7C34C4.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//CE47FE0E-B538-E811-A170-0CC47A4C8F0A.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//DA3298AF-B538-E811-9B7F-0CC47A78A3F4.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//DE9D0407-B538-E811-ADB4-0CC47A7C35A8.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//F0588306-B538-E811-A721-0CC47A4D768E.root',
    '/store/relval/CMSSW_10_1_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_design_v7-v1/10000//F869423D-B538-E811-A4B9-003048FFD7A4.root',
),
    secondaryFileNames = cms.untracked.vstring()
)


## Input source
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//00B4851A-7731-E811-8D30-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//021F63CD-7931-E811-B20D-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//088DE856-7931-E811-AC1B-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//12C1DCA1-7931-E811-9B90-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//1A021820-7731-E811-8BCE-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//1A4980DC-7931-E811-8089-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//48AF84C1-7931-E811-915D-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//48E45DBD-7931-E811-BC8A-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//6ED4EE31-7731-E811-B41F-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//8828A4BB-7931-E811-835D-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//8C238969-7931-E811-ABD7-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//9EBA1919-7731-E811-B2EB-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//BC2AEE3F-7A31-E811-AA6A-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//C29C94BD-7931-E811-8E9C-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//D856E4E8-7931-E811-B845-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//F257604D-7931-E811-91AD-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//F4AD6A47-7731-E811-8FAF-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//F4BA9C2B-7731-E811-B40D-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//FAF4334D-7931-E811-8AE1-0242AC130002.root',
#),
#    secondaryFileNames = cms.untracked.vstring()
#)


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
