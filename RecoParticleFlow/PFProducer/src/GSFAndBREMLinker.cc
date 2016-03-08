#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkers.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

double GSFAndBREMLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {   
  double dist = -1.0;
  const reco::PFBlockElementGsfTrack * gsfelem(NULL);
  const reco::PFBlockElementBrem * bremelem(NULL);
  if( elem1->type() < elem2->type() ) {
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack *>(elem1);
    bremelem = static_cast<const reco::PFBlockElementBrem *>(elem2);
  } else {
    gsfelem = static_cast<const reco::PFBlockElementGsfTrack *>(elem2);
    bremelem = static_cast<const reco::PFBlockElementBrem *>(elem1);
  }  
  const reco::GsfPFRecTrackRef& gsfref = gsfelem->GsftrackRefPF();
  const reco::GsfPFRecTrackRef& bremref = bremelem->GsftrackRefPF();
  if( gsfref.isNonnull() && bremref.isNonnull() && gsfref == bremref ) {
    dist = 0.001;
  }
  return dist;
}
