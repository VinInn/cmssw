#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossing2OrderLocal.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing2Order.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "DataFormats/Math/test/rdtscp.h"


#include <algorithm>
#include <cmath>
#include<tuple>

#include<iostream>

  typedef Surface::GlobalPoint    GlobalPoint;
  typedef Surface::GlobalVector   GlobalVector;
  typedef Surface::LocalPoint     LocalPoint;
  typedef Surface::LocalVector    LocalVector;


std::pair<LocalPoint,LocalVector> secondOrderAccurate(
						     const GlobalPoint& startingPos,
						     const GlobalVector& startingDir,
						     double rho, const Plane& plane)
{

  typedef Basic2DVector<float>  Vector2D;

  // translate problem to local frame of the plane
  LocalPoint lPos = plane.toLocal(startingPos);
  LocalVector lDir = plane.toLocal(startingDir);

  LocalVector yPrime = plane.toLocal( GlobalVector(0,0,1.f));
  float sinPhi=0, cosPhi=0;
  Vector2D pos;
  Vector2D dir;

  sinPhi = yPrime.y();
  cosPhi = yPrime.x();
  pos = Vector2D( lPos.x()*cosPhi + lPos.y()*sinPhi,
		  -lPos.x()*sinPhi + lPos.y()*cosPhi);
  dir = Vector2D( lDir.x()*cosPhi + lDir.y()*sinPhi,
		    -lDir.x()*sinPhi + lDir.y()*cosPhi);

  double d = -lPos.z();
  double x = pos.x() + dir.x()/lDir.z()*d - 0.5*rho*d*d;
  double y = pos.y() + dir.y()/lDir.z()*d;


  LocalPoint thePos( x*cosPhi - y*sinPhi,
		     x*sinPhi + y*cosPhi, 0);
  float px = dir.x()+rho*d;
  LocalVector theDir( px*cosPhi - dir.y()*sinPhi,
		     px*sinPhi + dir.y()*cosPhi, lDir.z());
  
  return std::pair<LocalPoint,LocalVector>(thePos,theDir);

}

void crossing1() {
  std::cout << std::endl;


  GlobalPoint startingPos(-7.79082,-47.6418,9.18163);
  GlobalVector startingDir(-0.553982,-5.09198,1.02212);

  GlobalPoint  pos(-2.95456,-48.2127,3.1033);
  
  float rho = 0.00223254;
  
  Surface::RotationType rot(0.995292,0.0969201,0.000255868,
			    8.57131e-06,0.00255196,-0.999997,
			    -0.0969205,0.995289,0.00253912);
  
  std::cout << rot << std::endl;

  Plane plane(pos,rot);

  bool cross; double s;

  StraightLinePlaneCrossing  line(startingPos.basicVector(), startingDir.basicVector(), alongMomentum);
  std::tie(cross,s) = line.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(line.position(s))) << " " << std::endl;


  HelixArbitraryPlaneCrossing arbitrary(startingPos.basicVector(), startingDir.basicVector(), rho,alongMomentum);
  HelixArbitraryPlaneCrossing2Order arbitrary2O(startingPos.basicVector(), startingDir.basicVector(), rho,alongMomentum);

  std::tie(cross,s) = arbitrary.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(arbitrary.position(s))) << " " <<  plane.toLocal(GlobalVector(arbitrary.direction(s))) << std::endl;

  std::tie(cross,s) = arbitrary2O.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(arbitrary2O.position(s))) << " " <<  plane.toLocal(GlobalVector(arbitrary2O.direction(s))) << std::endl;



  HelixBarrelPlaneCrossingByCircle precise(startingPos, startingDir, rho,alongMomentum);
  std::tie(cross,s) = precise.pathLength(plane);

  HelixBarrelPlaneCrossing2OrderLocal crossing(startingPos, startingDir, rho, plane);

  std::cout << s << ' ' << plane.toLocal(GlobalPoint(precise.position(s))) << " " <<  plane.toLocal(GlobalVector(precise.direction(s))) << std::endl;
  std::cout << HelixBarrelPlaneCrossing2OrderLocal::positionOnly(startingPos, startingDir, rho, plane) << ' ';
  std::cout << crossing.position() << ' ' << crossing.direction() << std::endl;

  LocalPoint thePos; LocalVector theDir;
  std::tie(thePos,theDir) = secondOrderAccurate(startingPos, startingDir, rho, plane);
  std::cout << thePos << ' ' << theDir << std::endl;



}

void crossing2() {

  std::cout << std::endl;

  GlobalPoint startingPos(-8.12604,-50.829,9.82116);   
  GlobalVector startingDir(-0.517536,-5.09581,1.02212);

  GlobalPoint  pos(-2.96723,-51.4573,14.8322);
  
  Surface::RotationType rot(0.995041,0.0994701,0.000124443,
			    0.000108324,-0.00233467,0.999997,
			    0.0994701,-0.995038,-0.00233387);
  std::cout << rot << std::endl;
  
  Plane plane(pos,rot);
  
  float rho = 0.00223254;
  
  bool cross; double s;


  StraightLinePlaneCrossing  line(startingPos.basicVector(), startingDir.basicVector(), alongMomentum);
  std::tie(cross,s) = line.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(line.position(s))) << " " << std::endl;
 

  HelixArbitraryPlaneCrossing arbitrary(startingPos.basicVector(), startingDir.basicVector(), rho,alongMomentum);
  HelixArbitraryPlaneCrossing2Order arbitrary2O(startingPos.basicVector(), startingDir.basicVector(), rho,alongMomentum);

  std::tie(cross,s) = arbitrary.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(arbitrary.position(s))) << " " <<  plane.toLocal(GlobalVector(arbitrary.direction(s))) << std::endl;

  std::tie(cross,s) = arbitrary2O.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(arbitrary2O.position(s))) << " " <<  plane.toLocal(GlobalVector(arbitrary2O.direction(s))) << std::endl;




  HelixBarrelPlaneCrossingByCircle precise(startingPos, startingDir, rho,alongMomentum);
  std::tie(cross,s) = precise.pathLength(plane);

  HelixBarrelPlaneCrossing2OrderLocal crossing(startingPos, startingDir, rho, plane);

  std::cout << s<< ' ' << plane.toLocal(GlobalPoint(precise.position(s))) << " " <<  plane.toLocal(GlobalVector(precise.direction(s))) << std::endl;
  std::cout << HelixBarrelPlaneCrossing2OrderLocal::positionOnly(startingPos, startingDir, rho, plane) << ' ';
  std::cout << crossing.position() << ' ' << crossing.direction() << std::endl;

  LocalPoint thePos; LocalVector theDir;
  std::tie(thePos,theDir) = secondOrderAccurate(startingPos, startingDir, rho, plane);

  std::cout << thePos << ' ' << theDir << std::endl;



}



void crossing3() {

  std::cout << std::endl;

  GlobalPoint startingPos(-8.12604,-50.829,9.82116);
  GlobalVector startingDir(-0.517536,-5.09581,1.02212);

    startingPos-= 10*startingDir;


  GlobalPoint  pos(-2.96723,-51.4573,14.8322);

  Surface::RotationType rot(0.995041,0.0994701,0.000124443,
                            0.000108324,-0.00233467,0.999997,
                            0.0994701,-0.995038,-0.00233387);
  std::cout << rot << std::endl;

  Plane plane(pos,rot);

  float rho = 0.00223254;

  bool cross; double s;


  StraightLinePlaneCrossing  line(startingPos.basicVector(), startingDir.basicVector(), alongMomentum);
  std::tie(cross,s) = line.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(line.position(s))) << " " << std::endl;
 

  HelixArbitraryPlaneCrossing arbitrary(startingPos.basicVector(), startingDir.basicVector(), rho,alongMomentum);
  HelixArbitraryPlaneCrossing2Order arbitrary2O(startingPos.basicVector(), startingDir.basicVector(), rho,alongMomentum);

  std::tie(cross,s) = arbitrary.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(arbitrary.position(s))) << " " <<  plane.toLocal(GlobalVector(arbitrary.direction(s))) << std::endl;

  std::tie(cross,s) = arbitrary2O.pathLength(plane);
  std::cout << s << ' ' << plane.toLocal(GlobalPoint(arbitrary2O.position(s))) << " " <<  plane.toLocal(GlobalVector(arbitrary2O.direction(s))) << std::endl;

 


  HelixBarrelPlaneCrossingByCircle precise(startingPos, startingDir, rho,alongMomentum);
  std::tie(cross,s) = precise.pathLength(plane);
  
  HelixBarrelPlaneCrossing2OrderLocal crossing(startingPos, startingDir, rho, plane);

  std::cout << s<< ' ' << plane.toLocal(GlobalPoint(precise.position(s))) << " " <<  plane.toLocal(GlobalVector(precise.direction(s))) << std::endl;
  std::cout << HelixBarrelPlaneCrossing2OrderLocal::positionOnly(startingPos, startingDir, rho, plane) << ' ';
  std::cout << crossing.position() << ' ' << crossing.direction() << std::endl;
                            
  LocalPoint thePos; LocalVector theDir;
  std::tie(thePos,theDir) = secondOrderAccurate(startingPos, startingDir, rho, plane);
  
  std::cout << thePos << ' ' << theDir << std::endl;
  
  

}



void timing() {

  GlobalPoint startingPos(-8.12604,-50.829,9.82116);
  GlobalVector startingDir(-0.517536,-5.09581,1.02212);

  // startingPos-= 10*startingDir;


  GlobalPoint  pos(-2.96723,-51.4573,14.8322);
  
  Surface::RotationType rot(0.995041,0.0994701,0.000124443,
                            0.000108324,-0.00233467,0.999997,
                            0.0994701,-0.995038,-0.00233387);
  std::cout << rot << std::endl;

  Plane plane(pos,rot);
  
  double rho = 0.00223254;

  HelixArbitraryPlaneCrossing arbitrary(startingPos.basicVector(), startingDir.basicVector(), rho,alongMomentum);

  bool cross; double s;
 
  double stot=0;

  auto const & hp = plane.hessianPlaneDouble();
  // auto hp = new decltype(plane.hessianPlaneDouble())(plane.hessianPlaneDouble());
  unsigned long long t = -rdtsc();
  for (int i=0; i<1000000; ++i) {
    std::tie(cross,s) = arbitrary.pathLength(hp);
    stot+=s;
  }
  t+=rdtsc();


  std::cout << "time " << double(t)/1000000 << ' ' << stot << std::endl;

}



void testHelixBarrelPlaneCrossing2OrderLocal() {

  crossing1();
  crossing2();
  crossing3();

  timing();

}



int main() {


  testHelixBarrelPlaneCrossing2OrderLocal();

  return 0;
}
