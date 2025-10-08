#ifndef OptimalHelixPlaneCrossing_H
#define OptimalHelixPlaneCrossing_H

#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

#include <memory>

#ifndef CROSSING_TYPE
#define CROSSING_TYPE double;
#endif

#ifdef VI_DEBUG
#include<iostream>
#endif

class OptimalHelixPlaneCrossing {
public:
#ifdef VI_DEBUG
  constexpr static bool videbug=true;
#else
  constexpr static bool videbug=false;
#endif

  using Base = HelixPlaneCrossing;
  using F = CROSSING_TYPE;

  template <typename... Args>
  OptimalHelixPlaneCrossing(Plane const &plane, Args &&...args) {
    GlobalVector u = plane.normalVector();
    constexpr float small = 1.e-6;  // for orientation of planes

    if (std::abs(u.z()) < small) {
      // barrel plane:
      // instantiate HelixBarrelPlaneCrossing,
      if constexpr (videbug) std::cout  <<"OptCross using Barrel" << std::endl; 
      new (get()) HelixBarrelPlaneCrossingByCircle(args...);
    } else if ((std::abs(u.x()) < small) && (std::abs(u.y()) < small)) {
      // forward plane:
      // instantiate HelixForwardPlaneCrossing
      if constexpr (videbug) std::cout  <<"OptCross using Forward" << std::endl;
      new (get()) HelixForwardPlaneCrossing(args...);
    } else {
      // arbitrary plane:
      // instantiate HelixArbitraryPlaneCrossing
      if constexpr (videbug) std::cout  <<"OptCross using Arbitrary " << typeid(F).name()  << std::endl;
      new (get()) HelixArbitraryPlaneCrossingT<F>(args...);
    }
  }

  ~OptimalHelixPlaneCrossing() { get()->~Base(); }

  Base &operator*() { return *get(); }
  Base const &operator*() const { return *get(); }

private:
  Base *get() { return (Base *)&mem; }
  Base const *get() const { return (Base const *)&mem; }

  union Tmp {
    HelixBarrelPlaneCrossingByCircle a;
    HelixForwardPlaneCrossing b;
    HelixArbitraryPlaneCrossingT<F> c;
  };
  using aligned_union_t = typename std::aligned_storage<sizeof(Tmp), alignof(Tmp)>::type;
  aligned_union_t mem;
};

#endif
