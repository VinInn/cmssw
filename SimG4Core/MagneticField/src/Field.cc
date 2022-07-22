#include "MagneticField/Engine/interface/MagneticField.h"
#include "SimG4Core/MagneticField/interface/Field.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

using namespace sim;

Field::Field(const MagneticField *f, double d) : G4MagneticField(), theCMSMagneticField(f), theDelta(d) {
  for (int i = 0; i < 3; ++i) {
    oldp[i] = 1.0e12;
    oldb[i] = 0.0;
  }
  oldp[3] = 1.0e12;
}

Field::~Field() {}

void Field::GetFieldValue(const G4double xyz[4], G4double bfield[3]) const {
  using GPD = Basic3DVector<double>;
  GPD pi(xyz[0],xyz[1],xyz[2],xyz[3]);
  auto delta = oldp-pi;
  for (int i=0; i<4; ++i) delta[i]=std::abs(delta[i]);

  if ( (delta[0] > theDelta) | (delta[1] > theDelta) |
       (delta[2] > theDelta) ) {
    constexpr float lunit = (1.0 / CLHEP::cm);
    GlobalPoint ggg(pi);
    ggg.mathVector() =  ggg.mathVector()*lunit;
    GlobalVector v = theCMSMagneticField->inTesla(ggg);

    constexpr float btesla = CLHEP::tesla;
    v*=btesla;
    oldb[0] = v.x();
    oldb[1] = v.y();
    oldb[2] = v.z();
    oldp = pi;
  }

  bfield[0] = oldb[0];
  bfield[1] = oldb[1];
  bfield[2] = oldb[2];
}
