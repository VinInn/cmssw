import FWCore.ParameterSet.Config as cms

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
quadrupletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()



# SEEDS (a la detacthed)
#from RecoPixelVertexing.PixelTriplets.PixelTripletLargeTipGenerator_cfi import *
#PixelTripletLargeTipGenerator.extraHitRZtolerance = 0.0
#PixelTripletLargeTipGenerator.extraHitRPhitolerance = 0.0
#import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
#quadrupletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#quadrupletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'quadrupletStepSeedLayers'
#quadrupletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(PixelTripletLargeTipGenerator)
#quadrupletStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
#quadrupletStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
#quadrupletStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
#quadrupletStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.5
#
#quadrupletStepSeeds.SeedComparitorPSet = cms.PSet(
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
quadrupletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.6,
    originRadius = 0.02,
    nSigmaZ = 4.0
    )
    )
    )
quadrupletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'quadrupletStepSeedLayers'

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
quadrupletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor



# building
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
quadrupletStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
#    seedExtention=1,
    minimumNumberOfHits = 3,
    minPt = 0.2
    )

import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
quadrupletStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
quadrupletStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('quadrupletStepTrajectoryFilterBase')),
    #    cms.PSet( refToPSet_ = cms.string('quadrupletStepTrajectoryFilterShape'))
    ),
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimatorESProducer_cfi
quadrupletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimatorESProducer_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('quadrupletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose')),
    pTChargeCutThreshold = cms.double(15.)
)

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
quadrupletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('quadrupletStepTrajectoryFilter')),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = cms.string('quadrupletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
quadrupletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('quadrupletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('quadrupletStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
quadrupletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'quadrupletStepTrackCandidates',
    AlgorithmName = cms.string('quadrupletStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )


#vertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
firstStepPrimaryVertices=RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
firstStepPrimaryVertices.TrackLabel = cms.InputTag("quadrupletStepTracks")
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

quadrupletStepClassifier1 = TrackMVAClassifierPrompt.clone()
quadrupletStepClassifier1.src = 'quadrupletStepTracks'
quadrupletStepClassifier1.GBRForestLabel = 'MVASelectorIter0_13TeV'
quadrupletStepClassifier1.qualityCuts = [-0.9,-0.8,-0.7]

from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepClassifier1
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStep
quadrupletStepClassifier2 = detachedTripletStepClassifier1.clone()
quadrupletStepClassifier2.src = 'quadrupletStepTracks'
quadrupletStepClassifier3 = lowPtTripletStep.clone()
quadrupletStepClassifier3.src = 'quadrupletStepTracks'



from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
quadrupletStep = ClassifierMerger.clone()
quadrupletStep.inputClassifiers=['quadrupletStepClassifier1','quadrupletStepClassifier2','quadrupletStepClassifier3']



# Final sequence
QuadrupletStep = cms.Sequence(quadrupletStepSeedLayers*
                           quadrupletStepSeeds*
                           quadrupletStepTrackCandidates*
                           quadrupletStepTracks*
                           firstStepPrimaryVertices*
                           quadrupletStepClassifier1*quadrupletStepClassifier2*quadrupletStepClassifier3*
                           quadrupletStep)
