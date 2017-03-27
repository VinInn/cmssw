#ifndef   TrackProducer_HitReMatcher_H
#define   TrackProducer_HitReMatcher_H


#include "RecoTracker/TrackProducer/interface/HitFilterFromTrack.h"

class dso_hidden HitReMatcher final : public HitFilterFromTrack {
public:
  explicit HitReMatcher(bool reMatch): reMatchSplitHits_(reMatch) {}
  virtual ~HitReMatcher(){}
  
  TransientTrackingRecHit::RecHitContainer
  operator()(int,trackingRecHit_iterator first, trackingRecHit_iterator last,
	     const TkTransientTrackingRecHitBuilder * builder) override;

private:

  bool reMatchSplitHits_;
  
};



#endif
