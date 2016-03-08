#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkers.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

double TrackAndHOLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {  
  constexpr reco::PFTrajectoryPoint::LayerType HOLayer =
    reco::PFTrajectoryPoint::HOLayer;
  const reco::PFBlockElementTrack   *tkelem(NULL);
  const reco::PFBlockElementCluster *hoelem(NULL);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    hoelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    hoelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFClusterRef& horef = hoelem->clusterRef();
  const reco::PFRecTrackRef& tkref = tkelem->trackRefPF();  
  if( horef.isNull() || tkref.isNull() ) {
    throw cms::Exception("BadClusterRefs") 
      << "PFBlockElementCluster's refs are null!";
  }  
  if ( tkelem->trackRef()->pt() > 3.00001 && 
       tkref->extrapolatedPoint( HOLayer ).isValid() ) {
    dist = LinkByRecHit::testTrackAndClusterByRecHit( *tkref, *horef, 
						      false, _debug );
  } else {
    dist = -1.;
  }
  return dist;
}
