#ifndef SeedExtentionTrajectoryFilter_H
#define SeedExtentionTrajectoryFilter_H

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

class SeedExtentionTrajectoryFilter final : public TrajectoryFilter {
public:

  explicit SeedExtentionTrajectoryFilter() {} 
  
  explicit SeedExtentionTrajectoryFilter(edm::ParameterSet const & pset, edm::ConsumesCollector&) :
     theExtention( pset.getParameter<int>("seedExtention")) {}

  virtual bool qualityFilter( const Trajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }
  virtual bool qualityFilter( const TempTrajectory& traj) const { return TrajectoryFilter::qualityFilterIfNotContributing; }

  virtual bool toBeContinued( TempTrajectory& traj) const { return TBC<TempTrajectory>(traj);}
  virtual bool toBeContinued( Trajectory& traj) const{ return TBC<Trajectory>(traj);}

  virtual std::string name() const{return "LostHitsFractionTrajectoryFilter";}

protected:

  template<class T> bool TBC(const T& traj) const {
    return (traj.foundHits()>int(traj.seedNHits())+theExtention) | (0==traj.lostHits());
  }

 int theExtention = 0;


};

#endif
