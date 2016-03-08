#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkers.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"

double SCAndECALLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const { 
  double dist = -1.0;  
  const reco::PFBlockElementCluster* ecalelem(NULL);    
  const reco::PFBlockElementSuperCluster* scelem(NULL); 
  if( elem1->type() < elem2->type() ) {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    scelem = static_cast<const reco::PFBlockElementSuperCluster*>(elem2);
  } else {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    scelem = static_cast<const reco::PFBlockElementSuperCluster*>(elem1);
  }
  const reco::PFClusterRef& clus = ecalelem->clusterRef();
  const reco::SuperClusterRef& sclus = scelem->superClusterRef();
  if( sclus.isNull() ) {
    throw cms::Exception("BadRef")
      << "SuperClusterRef is invalid!";
  }
  
  if( _superClusterMatchByRef ) {
    if( sclus == ecalelem->superClusterRef() ) dist = 0.001;
  } else {
    if( ClusterClusterMapping::overlap(*sclus,*clus) ) {
      dist = LinkByRecHit::computeDist( sclus->position().eta(),
					sclus->position().phi(), 
					clus->positionREP().Eta(), 
					clus->positionREP().Phi() );
    }
  }
  return dist;
}
