#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"




#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public  MagneticField {
    explicit M5T(double br) :  m(br,br,5.){}
    virtual GlobalVector inTesla (const GlobalPoint&) const {
      return m;
    }

    GlobalVector m;
  };

}

#include "FWCore/Utilities/interface/HRRealTime.h"
void st(){}
void en(){}

#include<iostream>

int main(int argc, char** argv) {
  double br=0.;
  if (argc>1) br=0.1;
  M5T const m(br);

  Basic3DVector<float>  axis(0.5,1.,1);
  
  Surface::RotationType orot(axis,0.5*M_PI);
  std::cout << orot << std::endl;

  Surface::PositionType pos( 0., 0., 0.);

  Plane oplane(pos,orot);
  LocalTrajectoryParameters tpl(-1./3.5, 1.,1., 0.,0.,1.);
  GlobalVector mg = oplane.toGlobal(tpl.momentum());
  GlobalTrajectoryParameters tpg(pos,mg,-1., &m);
  double curv =   tpg.transverseCurvature();
  std::cout << "c/p " << curv << " " <<  mg.mag() << std::endl;
  std::cout << "pos/mom @1 " << tpg.position() << " " << tpg.momentum() << std::endl;
  std::cout << "pt/p " << tpg.momentum().perp() << ' ' << tpg.momentum().mag() << std::endl;

  GlobalTrajectoryParameters tpg0(tpg);

  GlobalPoint zero(0.,0.,0.); 
  std::cout << std::endl;

  
{
    std::cout << "BARREL PLANE" << std::endl;
    GlobalVector h = tpg.magneticFieldInInverseGeV(tpg.position());
    Surface::RotationType rot(-1.,0.,0, 0.,0.,1., 0.,1.,0.);
    std::cout << "h " << h << std::endl;

    Plane lplane(zero,rot);
      HelixBarrelPlaneCrossingByCircle prop(tpg.position(), tpg.momentum(), curv);
      GlobalPoint one(0.,1.,0.);
      Plane newplane(one ,rot);
      auto [ok, st] = prop.pathLength(newplane);
      if(ok) std::cout << "step " << st << std::endl;
      GlobalPoint x(prop.position(st));
      GlobalVector dir(prop.direction(st));
      std::cout <<  dir.mag() << std::endl;
      LocalTrajectoryParameters lpg(lplane.toLocal(tpg.position()),
                                    lplane.toLocal(tpg.momentum()),tpg.charge());


      GlobalTrajectoryParameters tpg2(x,dir,
                       tpg.charge(), &m);

      std::cout << "pos/mom/curv @2 "<< tpg2.position() << " " << tpg2.momentum() << " " << tpg2.transverseCurvature() << std::endl;
      std::cout << "pt/p " << tpg2.momentum().perp() << ' ' << tpg2.momentum().mag() << std::endl;
      AnalyticalCurvilinearJacobian full;
      AnalyticalCurvilinearJacobian delta;
      full.computeFullJacobian(tpg,tpg2.position(),tpg2.momentum(),h,st);
      delta.computeInfinitesimalJacobian(tpg,tpg2.position(),tpg2.momentum(),h,st);
      std::cout <<  "full\n" << full.jacobian() << std::endl;
      std::cout << std::endl;
      std::cout << "delta\n" << delta.jacobian() << std::endl;
      std::cout << std::endl;

      JacobianLocalToCurvilinear jlc(lplane,lpg,m);
      LocalTrajectoryParameters newlpg(newplane.toLocal(tpg2.position()),
                                       newplane.toLocal(tpg2.momentum()),tpg.charge());

      std::cout << "lpg " << lpg.vector() << std::endl;
      std::cout << "newlpg " << newlpg.vector() << std::endl;

  JacobianCurvilinearToLocal jcl(newplane,newlpg,m);
  AlgebraicMatrix55 fjacobianL2L = jcl.jacobian()*full.jacobian()*jlc.jacobian();
  AlgebraicMatrix55 djacobianL2L = jcl.jacobian()*delta.jacobian()*jlc.jacobian();

  std::cout << "full  p2p\n" << fjacobianL2L << std::endl;
  std::cout << "delta p2p\n" << djacobianL2L << std::endl;
  {
       // let's play the Delta game
       auto lp = lplane.toLocal(tpg.position());
       lp += LocalVector(0.1,0.1,0.);
       auto ndir = tpg.momentum();
       std::cout << "app delta " << lp << " " << ndir << std::endl;
       HelixBarrelPlaneCrossingByCircle prop2(lplane.toGlobal(lp), ndir, curv);
       auto [ok2, st2] = prop2.pathLength(newplane);
       if(ok2) std::cout << "step " << st2 << std::endl;
       GlobalPoint x2(prop2.position(st2));
       GlobalVector dir2(prop2.direction(st2));
       std::cout <<  dir2.mag() << std::endl;  
       std::cout << " new pos/dir " << newplane.toLocal(x2) << ' ' << dir2 << std::endl;
       std::cout << "deltaX " << newplane.toLocal(x2)- newplane.toLocal(x) << std::endl;
       std::cout << "delatP " << dir2-dir << std::endl;
  }
  {
       // let's play the Delta game
       auto lp = lplane.toLocal(tpg.position());
       // lp += LocalVector(0.1,0.1,0.);
       auto ndir = 1.05*tpg.momentum();
       std::cout << "app delta " << lp << " " << ndir << std::endl;
       HelixBarrelPlaneCrossingByCircle prop2(lplane.toGlobal(lp), ndir, curv/1.05);
       auto [ok2, st2] = prop2.pathLength(newplane);
       if(ok2) std::cout << "step " << st2 << std::endl;
       GlobalPoint x2(prop2.position(st2));
       GlobalVector dir2(prop2.direction(st2));
       std::cout <<  dir2.mag() << std::endl;
       std::cout << " new pos/dir " << newplane.toLocal(x2) << ' ' << dir2 << std::endl;
       std::cout << "deltaX " << newplane.toLocal(x2)- newplane.toLocal(x) << std::endl;
       std::cout << "delatP " << dir2-dir << std::endl;
  }
}

{
    std::cout << "FORWARD PLANE" << std::endl;      
    GlobalVector h = tpg.magneticFieldInInverseGeV(tpg.position());
    Surface::RotationType rot(Basic3DVector<float>(h),0);

    // here local and global is the same...)
    Plane lplane(zero,rot);
    GlobalPoint one(0.,0.,1.);
    Plane newplane(one ,rot);
    HelixForwardPlaneCrossing::PositionType  a(tpg.position());
    HelixForwardPlaneCrossing::DirectionType p(tpg.momentum());
      double lcurv =   -h.mag()/p.perp()*tpg.charge();
      std::cout << "c/p "<< lcurv << " " <<  p.mag() << std::endl;
      HelixForwardPlaneCrossing prop(a, p, curv);
      auto [ok, st] = prop.pathLength(newplane);
      if(ok) std::cout << "step " << st << std::endl;
      GlobalPoint x(prop.position(st));
      GlobalVector dir(prop.direction(st));
      std::cout <<  dir.mag() << std::endl;
      LocalTrajectoryParameters lpg(lplane.toLocal(tpg.position()),
                                    lplane.toLocal(tpg.momentum()),tpg.charge());
      

      GlobalTrajectoryParameters tpg2( x, (tpg.momentum().mag()/dir.mag())*dir, 
				       tpg.charge(), &m);
      
      std::cout << "pos/mom/curv @2 "<< tpg2.position() << " " << tpg2.momentum() << " " << tpg2.transverseCurvature() << std::endl;
      std::cout << "pt/p " << tpg2.momentum().perp() << ' ' << tpg2.momentum().mag() << std::endl;
      AnalyticalCurvilinearJacobian full;
      AnalyticalCurvilinearJacobian delta;
      full.computeFullJacobian(tpg,tpg2.position(),tpg2.momentum(),h,st);
      delta.computeInfinitesimalJacobian(tpg,tpg2.position(),tpg2.momentum(),h,st);
      std::cout <<  "full\n" << full.jacobian() << std::endl;
      std::cout << std::endl;
      std::cout << "delta\n" << delta.jacobian() << std::endl;
      std::cout << std::endl;

      JacobianLocalToCurvilinear jlc(lplane,lpg,m);
      LocalTrajectoryParameters newlpg(newplane.toLocal(tpg2.position()),
                                       newplane.toLocal(tpg2.momentum()),tpg.charge());
       
      std::cout << "lpg " << lpg.vector() << std::endl;
      std::cout << "newlpg " << newlpg.vector() << std::endl;

  JacobianCurvilinearToLocal jcl(newplane,newlpg,m);
  AlgebraicMatrix55 fjacobianL2L = jcl.jacobian()*full.jacobian()*jlc.jacobian();
  AlgebraicMatrix55 djacobianL2L = jcl.jacobian()*delta.jacobian()*jlc.jacobian();

  std::cout << "full  p2p\n" << fjacobianL2L << std::endl;
  std::cout << "delta p2p\n" << djacobianL2L << std::endl;

}

  return 0;
}
