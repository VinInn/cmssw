import FWCore.ParameterSet.Config as cms

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
QuadrupletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()



# SEEDS (a la detacthed)
#from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
#PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
#PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
#import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
#QuadrupletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#QuadrupletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'QuadrupletStepSeedLayers'
#QuadrupletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
#QuadrupletStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
#QuadrupletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
#QuadrupletStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
#QuadrupletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.5
#
#QuadrupletStepSeeds.SeedComparitorPSet = cms.PSet(
#        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
#        FilterAtHelixStage = cms.bool(False),
#        FilterPixelHits = cms.bool(True),
#        FilterStripHits = cms.bool(False),
#        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
#        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache')
#    )


# seeding
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
QuadrupletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.6,
    originRadius = 0.02,
    nSigmaZ = 4.0
    )
    )
    )
QuadrupletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'QuadrupletStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
QuadrupletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor



# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
QuadrupletStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2
    )
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
QuadrupletStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
QuadrupletStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('QuadrupletStepTrajectoryFilterBase')),
    #    cms.PSet( refToPSet_ = cms.string('QuadrupletStepTrajectoryFilterShape'))
    ),
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimatorESProducer_cfi
QuadrupletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimatorESProducer_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('QuadrupletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose')),
    pTChargeCutThreshold = cms.double(15.)
)

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
QuadrupletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('QuadrupletStepTrajectoryFilter')),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = cms.string('QuadrupletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
QuadrupletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('QuadrupletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('QuadrupletStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
QuadrupletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'QuadrupletStepTrackCandidates',
    AlgorithmName = cms.string('quadrupletStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )


#vertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
firstStepPrimaryVertices=RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
firstStepPrimaryVertices.TrackLabel = cms.InputTag("QuadrupletStepTracks")
firstStepPrimaryVertices.vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               )
      ]
    )
 

# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *

QuadrupletStepClassifier1 = TrackMVAClassifierPrompt.clone()
QuadrupletStepClassifier1.src = 'QuadrupletStepTracks'
QuadrupletStepClassifier1.GBRForestLabel = 'MVASelectorIter0_13TeV'
QuadrupletStepClassifier1.qualityCuts = [-0.9,-0.8,-0.7]

from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepClassifier1
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStep
QuadrupletStepClassifier2 = detachedTripletStepClassifier1.clone()
QuadrupletStepClassifier2.src = 'QuadrupletStepTracks'
QuadrupletStepClassifier3 = lowPtTripletStep.clone()
QuadrupletStepClassifier3.src = 'QuadrupletStepTracks'



from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
QuadrupletStep = ClassifierMerger.clone()
QuadrupletStep.inputClassifiers=['QuadrupletStepClassifier1','QuadrupletStepClassifier2','QuadrupletStepClassifier3']



# Final sequence
InitialStep = cms.Sequence(QuadrupletStepSeedLayers*
                           QuadrupletStepSeeds*
                           QuadrupletStepTrackCandidates*
                           QuadrupletStepTracks*
                           firstStepPrimaryVertices*
                           QuadrupletStepClassifier1*QuadrupletStepClassifier2*QuadrupletStepClassifier3*
                           QuadrupletStep)
