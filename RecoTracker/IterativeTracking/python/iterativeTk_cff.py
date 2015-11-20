import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.InitialStepPreSplitting_cff import *

from RecoTracker.IterativeTracking.QuadrupletStep_cff import *
from RecoTracker.IterativeTracking.InitialTripletStep_cff import *
# from RecoTracker.IterativeTracking.InitialStep_cff import *

from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *
from RecoTracker.IterativeTracking.JetCoreRegionalStep_cff import *

from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

detachedTripletStepClusters.oldClusterRemovalInfo='initialTripletStepClusters'

iterTracking = cms.Sequence(InitialStepPreSplitting*
                            QuadrupletStep*
                            InitialStep*
                            DetachedTripletStep*
                            LowPtTripletStep*
                            PixelPairStep*
                            MixedTripletStep*
                            PixelLessStep*
                            TobTecStep*
			    JetCoreRegionalStep *	
                            earlyGeneralTracks*
                            muonSeededStep*
                            preDuplicateMergingGeneralTracks*
                            generalTracksSequence*
                            ConvStep*
                            conversionStepTracks
                            )
