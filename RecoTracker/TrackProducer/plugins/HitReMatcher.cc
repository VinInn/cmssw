#include "HitReMatcher.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"


TransientTrackingRecHit::RecHitContainer
HitReMatcher::operator()(int,trackingRecHit_iterator first, trackingRecHit_iterator last,
			 const TkTransientTrackingRecHitBuilder * builder) {
  
  // WARNING: here we assume that the hits are correcly sorted according to seedDir
  TransientTrackingRecHit::RecHitContainer hits;       
  for (auto i=first; i!=last; i++){
    if (reMatchSplitHits_){
      //re-match hits that belong together
      auto next = i; next++;
      if (next != last && (*i)->isValid()){
	//check whether hit and nexthit are on glued module
	DetId hitId = (**i).geographicalId();
	
	if(hitId.det() == DetId::Tracker) {
	  if(GeomDetEnumerators::isTrackerStrip((**i).det()->subDetector())) {
	    SiStripDetId stripId(hitId);
	    if (stripId.partnerDetId() == (*next)->geographicalId().rawId()){
	      //yes they are parterns in a glued geometry.
	      DetId gluedId = stripId.glued();
	      const SiStripRecHit2D *mono=0;
	      const SiStripRecHit2D *stereo=0;
	      if (stripId.stereo()==0){
		mono=dynamic_cast<const SiStripRecHit2D *>(&**i);
		stereo=dynamic_cast<const SiStripRecHit2D *>(&**next);
	      }
	      else{
		mono=dynamic_cast<const SiStripRecHit2D *>(&**next);
		stereo=dynamic_cast<const SiStripRecHit2D *>(&**i);
	      }
	      if (!mono	|| !stereo){
		edm::LogError("TrackProducerAlgorithm")
		  <<"cannot get a SiStripRecHit2D from the rechit."<<hitId.rawId()
		  <<" "<<gluedId.rawId()
		  <<" "<<stripId.partnerDetId()
		  <<" "<<(*next)->geographicalId().rawId();
	      }
	      LocalPoint pos;//null
	      LocalError err;//null
	      hits.push_back(std::make_shared<SiStripMatchedRecHit2D>(pos,err, *builder->geometry()->idToDet(gluedId), mono,stereo));
	      //the local position and error is dummy but the fitter does not need that anyways
	      i++;
	      continue;//go to next.
	    }//consecutive hits are on parterns module.
	  }}//is a strip module
      }//next is not the end of hits
    }//if rematching option is on.
       
    if ((**i).geographicalId()!=0U)  hits.push_back( (**i).cloneForFit(*builder->geometry()->idToDet( (**i).geographicalId() ) ) );
  }
  
  return hits;
}
