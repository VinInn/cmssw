#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkers.h"


DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  ECALAndBREMLinker, 
		  "ECALAndBREMLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  ECALAndECALLinker, 
		  "ECALAndECALLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  ECALAndHCALCaloJetLinker, 
		  "ECALAndHCALCaloJetLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  ECALAndHCALLinker, 
		  "ECALAndHCALLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  GSFAndBREMLinker, 
		  "GSFAndBREMLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  GSFAndECALLinker, 
		  "GSFAndECALLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  GSFAndGSFLinker, 
		  "GSFAndGSFLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  GSFAndHCALLinker, 
		  "GSFAndHCALLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  HCALAndBREMLinker, 
		  "HCALAndBREMLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  HCALAndHOLinker, 
		  "HCALAndHOLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  HFEMAndHFHADLinker, 
		  "HFEMAndHFHADLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  PreshowerAndECALLinker, 
		  "PreshowerAndECALLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  SCAndECALLinker, 
		  "SCAndECALLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndECALLinker, 
		  "TrackAndECALLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndGSFLinker, 
		  "TrackAndGSFLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndHCALLinker, 
		  "TrackAndHCALLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndHOLinker, 
		  "TrackAndHOLinker");

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  TrackAndTrackLinker, 
		  "TrackAndTrackLinker");

