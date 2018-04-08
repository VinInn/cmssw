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
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//0456796E-7531-E811-850E-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//0CAD0F9C-7531-E811-9C72-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//18497066-7531-E811-B296-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//485A26AB-7531-E811-8E94-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//4C2F021F-7531-E811-8249-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//5E01CD65-7531-E811-AC4B-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//6010776E-7531-E811-B25D-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//7CF1D2B0-7531-E811-BD31-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//82867023-7531-E811-9BC9-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//9C07C94D-7531-E811-9214-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//B2B0343C-7531-E811-B850-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//B45DB912-7531-E811-A024-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//C6322637-7531-E811-83ED-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//CC1C8F4C-7531-E811-84E1-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//E2A2229A-7531-E811-B975-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//ECF61972-7531-E811-805D-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//F2C06A8D-7531-E811-B6C4-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValNuGun/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//F699A57A-7531-E811-ADCA-0242AC130002.root',
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
