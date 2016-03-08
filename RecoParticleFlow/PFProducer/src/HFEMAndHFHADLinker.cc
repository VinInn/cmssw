#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkers.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"


DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, 
		  HFEMAndHFHADLinker, 
		  "HFEMAndHFHADLinker");

double HFEMAndHFHADLinker::testLink
  ( const reco::PFBlockElement* elem1,
    const reco::PFBlockElement* elem2) const {  
  const reco::PFBlockElementCluster *hfemelem(NULL), *hfhadelem(NULL);
  if( elem1->type() < elem2->type() ) {
    hfemelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    hfhadelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    hfemelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    hfhadelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFClusterRef& hfemref  = hfemelem->clusterRef();
  const reco::PFClusterRef& hfhadref = hfhadelem->clusterRef();
  if( hfemref.isNull() || hfhadref.isNull() ) {
    throw cms::Exception("BadClusterRefs") 
      << "PFBlockElementCluster's refs are null!";
  }    
  return LinkByRecHit::testHFEMAndHFHADByRecHit( *hfemref, *hfhadref, _debug );
}
