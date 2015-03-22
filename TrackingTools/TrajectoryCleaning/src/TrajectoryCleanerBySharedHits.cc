#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/RecHitComparatorByPosition.h"

#include "TrackingTools/TrajectoryCleaning/src/OtherHashMaps.h"


//#define DEBUG_PRINT(X) X
#define DEBUG_PRINT(X) 

namespace {

// Define when two rechits are equals
struct EqualsBySharesInput { 
    bool operator()(const TransientTrackingRecHit *h1, const TransientTrackingRecHit *h2) const {
        return (h1 == h2) || ((h1->geographicalId() == h2->geographicalId()) && (h1->hit()->sharesInput(h2->hit(), TrackingRecHit::some)));
    }
};
// Define a hash, i.e. a number that must be equal if hits are equal, and should be different if they're not
struct HashByDetId : std::unary_function<const TransientTrackingRecHit *, std::size_t> {
    std::size_t operator()(const TransientTrackingRecHit *hit) const { 
        boost::hash<uint32_t> hasher; 
        return hasher(hit->geographicalId().rawId());
    }
};

using RecHitMap = cmsutil::SimpleAllocHashMultiMap<const TransientTrackingRecHit*, Trajectory *, HashByDetId, EqualsBySharesInput>;
using TrajMap = cmsutil::UnsortedDumbVectorMap<Trajectory*, int>;

struct Maps {
  Maps() : theRecHitMap(128,256,1024){} // allocate 128 buckets, one row for 256 keys and one row for 512 values
  RecHitMap theRecHitMap;
  TrajMap theTrajMap;
};

thread_local Maps theMaps;
}

using namespace std;

void TrajectoryCleanerBySharedHits::clean( TrajectoryPointerContainer & tc) const
{
  if (tc.size() <= 1) return; // nothing to clean

  auto & theRecHitMap = theMaps.theRecHitMap;

  theRecHitMap.clear(10*tc.size());           // set 10*tc.size() active buckets
                                              // numbers are not optimized

  DEBUG_PRINT(std::cout << "Filling RecHit map" << std::endl);
  for (auto it : tc) {
    DEBUG_PRINT(std::cout << "  Processing trajectory " << it << " (" << it->foundHits() << " valid hits)" << std::endl);
    auto const & pd = it->measurements();
    for (auto const & mea : pd ) {
      auto theRecHit = &(*mea.recHit());
      if (theRecHit->isValid()) {
        DEBUG_PRINT(std::cout << "    Added hit " << theRecHit << " for trajectory " << it << std::endl);
        theRecHitMap.insert(theRecHit, it);
      }
    }
  }
  DEBUG_PRINT(theRecHitMap.dump());

  DEBUG_PRINT(std::cout << "Using RecHit map" << std::endl);
  // for each trajectory fill theTrajMap
  auto & theTrajMap = theMaps.theTrajMap;
  for ( auto tt : tc) {
    if(!tt->isValid()) continue;  
      DEBUG_PRINT(std::cout << "  Processing trajectory " << tt << " (" << tt->foundHits() << " valid hits)" << std::endl);
      theTrajMap.clear();
      auto const & pd =  tt->measurements();
      for (auto const & mea : pd) {
	auto theRecHit  = &(*mea.recHit());
        if (theRecHit->isValid()) {
          DEBUG_PRINT(std::cout << "    Searching for overlaps on hit " << theRecHit << " for trajectory " << *itt << std::endl);
          for (auto ivec = theRecHitMap.values(theRecHit);
                ivec.good(); ++ivec) {
              if (*ivec != tt){
                if ((*ivec)->isValid()){
                    theTrajMap[*ivec]++;
                }
              }
          }
	}
      }
      //end filling theTrajMap

      auto score = [=](auto tp) { return validHitBonus_*tp->foundHits() - missingHitPenalty_*tp->lostHits() - tp->chiSquared();};
    
      // check for duplicated tracks
      if(!theTrajMap.empty() > 0)
	for(auto mapp :  theTrajMap) {
	  if(mapp.second > 0 ){
	    int innerHit = 0;
	    if ( allowSharedFirstHit ) {
	      const TrajectoryMeasurement & innerMeasure1 = ( tt->direction() == alongMomentum ) ? 
		tt->firstMeasurement() : tt->lastMeasurement();
	      const TransientTrackingRecHit* h1 = &(*(innerMeasure1).recHit());
	      const TrajectoryMeasurement & innerMeasure2 = ( mapp.first->direction() == alongMomentum ) ? 
		mapp.first->firstMeasurement() : mapp.first->lastMeasurement();
	      const TransientTrackingRecHit* h2 = &(*(innerMeasure2).recHit());
	      if ( (h1 == h2) || ((h1->geographicalId() == h2->geographicalId()) && 
				  (h1->hit()->sharesInput(h2->hit(), TrackingRecHit::some))) ) {
		innerHit = 1;
	      }
	    }
	    int nhit1 = tt->foundHits();
	    int nhit2 = mapp.first->foundHits();
	    if( ( mapp.second - innerHit) >= ( (std::min(nhit1, nhit2)-innerHit) * theFraction) ){
	      auto badtraj = (score(tt) > score(mapp.first)) ? mapp.first : tt;
	      badtraj->invalidate();  // invalidate this trajectory
              if (badtraj== tt ) break; //  is invalid, no need to loop further
	    }
	  }
	}  // end TrajMap loop
    
    
  }
}
