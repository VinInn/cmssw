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
  operator()(trackingRecHit_iterator first, trackingRecHit_iterator last,
	     const TkTransientTrackingRecHitBuilder * builder) const =0;
  
};



#endif
