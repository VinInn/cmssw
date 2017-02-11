#ifndef   TrackProducer_HitFilterFromTrack_H
#define   TrackProducer_HitFilterFromTrack_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

class HitFilterFromTrack {
public:
  explicit HitFilterFromTrack() {}
  virtual ~HitFilterFromTrack(){}
  
  virtual
  TransientTrackingRecHit::RecHitContainer
  operator()(int itr,
	     trackingRecHit_iterator first, trackingRecHit_iterator last,
	     const TkTransientTrackingRecHitBuilder * builder)=0;
  
};



#endif
