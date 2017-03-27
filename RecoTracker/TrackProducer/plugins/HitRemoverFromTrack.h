#ifndef   TrackProducer_HitRemover_H
#define   TrackProducer_HitRemover_H


#include "RecoTracker/TrackProducer/interface/HitFilterFromTrack.h"
#include<vector>
#include<array>

class dso_hidden HitRemoverFromTrack final : public HitFilterFromTrack {
 public:
  using HitList = std::vector<std::array<int,4>>;

public:
  explicit HitRemoverFromTrack() {}
  virtual ~HitRemoverFromTrack(){}

  void setHits(HitList const * h) {  m_hits=h; ntr=0;}

  
  TransientTrackingRecHit::RecHitContainer
    operator()(int itr, trackingRecHit_iterator first, trackingRecHit_iterator last,
	     const TkTransientTrackingRecHitBuilder * builder) override {
    assert(m_hits);

    TransientTrackingRecHit::RecHitContainer rtr;

    auto const & hitList = (*m_hits);
    if (itr!=hitList[ntr][0]) return rtr;

    auto go = [&](auto i) {
      if ((**i).geographicalId()!=0U)  rtr.push_back( (**i).cloneForFit(*builder->geometry()->idToDet( (**i).geographicalId() ) ) );
    };

    if (hitList[ntr][1]==0) go(first);
    if (hitList[ntr][2]==0) go(first+1);    
    for (auto i=first+2; i!=last; i++){
      go(i);
    }

    // remove invalid hits from head
    while (!(**rtr.begin()).isValid()) rtr.erase(rtr.begin());
    
    std::cout << "hits " << itr << ' '  << last-first  << ' ' << rtr.size() <<std::endl;
    ++ntr;
    return rtr;
  }

private:
 HitList const * m_hits=nullptr;
 int ntr=0;
};



#endif
