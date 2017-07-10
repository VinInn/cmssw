import FWCore.ParameterSet.Config as cms

clusterShapeExtractor = cms.EDAnalyzer("PixelClusterShapeExtractor",
    tracks = cms.InputTag("generalTracks"),
    clusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache'),
    pixelSimLinkSrc = cms.InputTag('simSiPixelDigis', 'Pixel'),
    hasSimHits     = cms.bool(True),
    hasRecTracks   = cms.bool(False),
# for the associator
    associateStrip      = cms.bool(False),
    associatePixel      = cms.bool(True),
    associateRecoTracks = cms.bool(False),
    ROUList = cms.vstring(
      'TrackerHitsPixelBarrelLowTof',
      'TrackerHitsPixelBarrelHighTof',
      'TrackerHitsPixelEndcapLowTof',
      'TrackerHitsPixelEndcapHighTof')
)

