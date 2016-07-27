# Imports
import FWCore.ParameterSet.Config as cms

# Create a new CMS process
process = cms.Process('Phas2ITPixelRecHitTest')

# Import all the necessary files
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2023tiltedReco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# temporary: use fake conditions for LA 
process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkTilted_cff')

# Number of events (-1 = all)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:/tmp/emiglior/step3.root')
)

# Output
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('file:phase2ITpixel_rechits_validation.root')
)

process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.siPixelRecHits.src = cms.InputTag("phase2ITPixelClusters")
# CPE for other steps
process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')

# PixelCPEGeneric #
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.PixelCPEGenericESProducer.useLAWidthFromDB = cms.bool(False)
process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
process.PixelCPEGenericESProducer.useLAAlignmentOffsets = cms.bool(False)
process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
# Analyzer
process.hitmap_analysis = cms.EDAnalyzer('Phase2ITPixelRecHitsValidation',
   phase2ITPixelRecHitSrc = cms.InputTag("siPixelRecHits"),
)



# Processes to run
process.rechits_step = cms.Path(process.siPixelRecHits)
process.validation_step = cms.Path(process.hitmap_analysis)

process.schedule = cms.Schedule(process.rechits_step, process.validation_step)

