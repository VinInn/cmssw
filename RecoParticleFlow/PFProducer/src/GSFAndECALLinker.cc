#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkers.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

double GSFAndECALLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {  
  constexpr reco::PFTrajectoryPoint::LayerType ECALShowerMax =
    reco::PFTrajectoryPoint::ECALShowerMax;
  const reco::PFBlockElementCluster  *ecalelem(NULL);
  const reco::PFBlockElementGsfTrack *gsfelem(NULL);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    gsfelem  = static_cast<const reco::PFBlockElementGsfTrack*>(elem2);
  } else {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    gsfelem  = static_cast<const reco::PFBlockElementGsfTrack*>(elem1);
  }
  const reco::PFRecTrack& track = gsfelem->GsftrackPF();
  const reco::PFClusterRef& clusterref = ecalelem->clusterRef();
  const reco::PFTrajectoryPoint& tkAtECAL =
    track.extrapolatedPoint( ECALShowerMax );
  if( tkAtECAL.isValid() ) {
    dist = LinkByRecHit::testTrackAndClusterByRecHit( track, *clusterref, 
						      false, _debug );
  }
  if ( _debug ) {
    if ( dist > 0. ) {
      std::cout << " Here a link has been established" 
		<< " between a GSF track an Ecal with dist  " 
		<< dist <<  std::endl;
    } else {
      if( _debug ) std::cout << " No link found " << std::endl;
    }
  }
  
  return dist;
}
