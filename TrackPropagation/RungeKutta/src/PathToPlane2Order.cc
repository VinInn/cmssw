#include "PathToPlane2Order.h"
#include "RKLocalFieldProvider.h"
#include "frameChanger.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

#include <iostream>

std::pair<bool,double> 
PathToPlane2Order::operator()( const HessianPlane<float>& plane, 
			       const Vector3D& pos,
			       const Vector3D& momentum,
			       float charge,
			       const PropagationDirection propDir) const
{
    auto fpos = theField.toGlobal( Frame::LocalPoint(pos));

    auto theFieldFrame = &theField;
    // access to the field in field frame local coordinates
    auto B = inTesla(pos);

     auto localZ = theField.fieldInTesla(fpos).unit();
    //auto localZ = theFieldFrame->toGlobal( Frame::LocalVector( B.unit())); // local Z along field
    auto localY = localZ.cross( Frame::GlobalVector( 1,0,0));
    if (localY.mag2() < 0.01) {
	localY = localZ.cross( Frame::GlobalVector(0,1,0)).unit();
    }
    else {
	localY = localY.unit();
    }
    auto localX = localY.cross(localZ);

    Frame::RotationType frot( localX, localY, localZ);
    // frame in which the field is along Z
    Frame frame( fpos, frot);
    
    //    cout << "PathToPlane2Order frame " << frame.position() << endl << frame.rotation() << endl;
    
    // transform the position and direction to that frame
    Frame::LocalPoint localPos(0,0,0);  // same as  frame.toLocal( fpos); 

    //transform momentum from field frame to new frame via global frame
    Frame::GlobalVector gmom( theFieldFrame->toGlobal( Frame::LocalVector(momentum)));
    Frame::LocalVector localMom = frame.toLocal( gmom); 

    // transform the plane to the same frame
    auto localPlane =  frameChanger::transform( plane, frame);

/*
     cout << "PathToPlane2Order input plane       " << plane.position() << endl 
 	 << plane.rotation() << endl;
     cout << "PathToPlane2Order transformed plane " << localPlane->position() << endl 
 	 << localPlane->rotation() << endl;
*/
    
    constexpr float k = 2.99792458e-3;
    float curvature = -k * charge * std::sqrt(B.mag2() / localMom.perp2());

/*
     cout << "PathToPlane2Order curvature " << curvature << endl;
     cout << "transverseMomentum " << transverseMomentum << endl;
     cout << "B.mag() " << B.mag() << endl;
     cout << "localZ " << localZ << endl;
     cout << "pos      " << pos << endl;
     cout << "momentum " << momentum << endl;
     cout << "localPos " << localPos << endl;
     cout << "localMom " << localMom << endl;
*/
/*
    cout << "PathToPlane2Order: local pos " << localPos << " mom " << localMom 
	 << " curvature " << curvature << endl;
    cout << "PathToPlane2Order: local plane pos " << localPlane->position() 
	 << " normal " << localPlane->normalVector() << endl;
*/
    HelixArbitraryPlaneCrossing crossing( localPos.basicVector(), localMom.basicVector(), 
					  curvature, propDir);
    std::pair<bool,double> res = crossing.pathLength(localPlane);

    return res;
}
