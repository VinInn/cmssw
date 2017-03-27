import FWCore.ParameterSet.Config as cms

process = cms.Process("Refitting")

### Standard Configurations
#process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
#process.load("Configuration.StandardSequences.MagneticField_cff") 
#process.load('Configuration.Geometry.GeometryRecoDB_cff')
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff") 

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
# choose!
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data_GRun', '')
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_GRun', '')


### Track refitter specific stuff
import RecoTracker.TrackProducer.TrackRefitter_cfi
import CommonTools.RecoAlgos.recoTrackRefSelector_cfi
process.mytkselector = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
process.mytkselector.quality = ['highPurity']
process.mytkselector.minHit = 5
process.mytkselector.min3DLayer = 2
process.mytkselector.ptMin = 0.5
process.mytkselector.tip = 20.0
process.mytkselector.minPixelHit=2

import RecoTracker.TrackProducer.TrackRefitterWithHitFilter_cfi
process.myRefittedTracks = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
process.myRefittedTracks.src= 'mytkselector'
process.myRefittedTracks.NavigationSchool = ''
process.myRefittedTracks.Fitter = 'FlexibleKFFittingSmoother'

process.myNewRefittedTracks = RecoTracker.TrackProducer.TrackRefitterWithHitFilter_cfi.TrackRefitterWithHitFilter.clone()
process.myNewRefittedTracks.src= 'mytkselector'
process.myNewRefittedTracks.hitList= 'shhits'
process.myNewRefittedTracks.NavigationSchool = ''
process.myNewRefittedTracks.Fitter = 'FlexibleKFFittingSmoother'


import RecoTracker.FinalTrackSelectors.SharingHitTrackSelector_cfi
process.shhits = RecoTracker.FinalTrackSelectors.SharingHitTrackSelector_cfi.SharingHitTrackSelector.clone()
process.shhits.src='mytkselector'

### and an analyzer
process.trajCout = cms.EDAnalyzer('TrajectoryAnalyzer',
   trajectoryInput=cms.InputTag('myRefittedTracks')
)

process.trackCout = cms.EDAnalyzer('TwoTrackAnalyzer',
  oriSrc=cms.InputTag('mytkselector'),
  refitSrc=cms.InputTag('myNewRefittedTracks'),
  hitList= cms.InputTag('shhits')
)


process.source = cms.Source ("PoolSource",
                             fileNames=cms.untracked.vstring('file:pickevents_1.root',
                            ),
                             skipEvents=cms.untracked.uint32(0)
                             )

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.Path = cms.Path(process.mytkselector+process.shhits+process.myNewRefittedTracks+process.trackCout)
# process.Path = cms.Path(process.mytkselector+process.myRefittedTracks+process.trajCout)

