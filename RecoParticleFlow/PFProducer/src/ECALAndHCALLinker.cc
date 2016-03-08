#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkers.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

double ECALAndHCALLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {  
  const reco::PFBlockElementCluster *hcalelem(NULL), *ecalelem(NULL);
  double dist(-1.0);
  if( elem1->type() < elem2->type() ) {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFClusterRef& ecalref = ecalelem->clusterRef();
  const reco::PFClusterRef& hcalref = hcalelem->clusterRef();
  const reco::PFCluster::REPPoint& ecalreppos = ecalref->positionREP();
  if( hcalref.isNull() || ecalref.isNull() ) {
    throw cms::Exception("BadClusterRefs") 
      << "PFBlockElementCluster's refs are null!";
  }  
  dist = ( std::abs(ecalreppos.Eta()) > 2.5 ?
	   LinkByRecHit::computeDist( ecalreppos.Eta(),
				      ecalreppos.Phi(), 
				      hcalref->positionREP().Eta(), 
				      hcalref->positionREP().Phi() )
	   : -1.0 );
  return (dist < 0.2 ? dist : -1.0);
}
