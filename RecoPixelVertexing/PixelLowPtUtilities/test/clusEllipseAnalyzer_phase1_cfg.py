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
    input = cms.untracked.int32(1000)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//00B4851A-7731-E811-8D30-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//021F63CD-7931-E811-B20D-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//088DE856-7931-E811-AC1B-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//12C1DCA1-7931-E811-9B90-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//1A021820-7731-E811-8BCE-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//1A4980DC-7931-E811-8089-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//48AF84C1-7931-E811-915D-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//48E45DBD-7931-E811-BC8A-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//6ED4EE31-7731-E811-B41F-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//8828A4BB-7931-E811-835D-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//8C238969-7931-E811-ABD7-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//9EBA1919-7731-E811-B2EB-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//BC2AEE3F-7A31-E811-AA6A-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//C29C94BD-7931-E811-8E9C-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//D856E4E8-7931-E811-B845-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//F257604D-7931-E811-91AD-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//F4AD6A47-7731-E811-8FAF-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//F4BA9C2B-7731-E811-B40D-0242AC130002.root',
    '/store/relval/CMSSW_10_1_0_pre3/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_101X_upgrade2018_realistic_v3_cc7-v2/10000//FAF4334D-7931-E811-8AE1-0242AC130002.root',
),
    secondaryFileNames = cms.untracked.vstring()
)

#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/0806614B-A92F-E811-A810-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/206F3519-A42F-E811-BC66-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/244C7D92-A72F-E811-B2DA-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/2A51D35D-A82F-E811-ADF3-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/2ACB189F-A72F-E811-8C16-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/3EA75845-A42F-E811-9FFB-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/407BDCE7-A72F-E811-869C-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/58E72D41-A62F-E811-BA80-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/62F31674-A62F-E811-8DD9-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/88BF8E6A-A52F-E811-8D85-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/96A32E5D-A82F-E811-95C1-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/9A5B30E6-A72F-E811-A84A-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/9EA33F90-A72F-E811-8F83-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/9EBDD868-A32F-E811-AB37-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/B061A75A-A52F-E811-A324-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/BCAC126B-AA2F-E811-9E23-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/C4485B08-A82F-E811-9DBB-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/E8E11A7E-A62F-E811-B2F7-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/F81A08B0-A72F-E811-AB73-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/FAAD8110-A72F-E811-B8AB-0242AC130002.root',
#    '/store/relval/CMSSW_10_1_0_pre3/RelValMinBias_13/GEN-SIM-DIGI-RAW/101X_upgrade2018_realistic_v3_cc7-v1/10000/FAC9D661-A82F-E811-BC83-0242AC130002.root',
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

process.clusEllipseAnalyzer = cms.EDAnalyzer("ClusEllipseAnalyzer",
    tracks = cms.InputTag("generalTracks"),
)

process.clusEllipseAnalyzer_step = cms.Path(process.clusEllipseAnalyzer)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.clusEllipseAnalyzer_step)

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
