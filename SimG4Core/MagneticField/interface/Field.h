#ifndef SimG4Core_Field_H
#define SimG4Core_Field_H

#include "G4MagneticField.hh"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class MagneticField;

namespace sim {
  class Field : public G4MagneticField {
  public:
    using GPD = Basic3DVector<double>;

    Field(const MagneticField *f, double d);
    ~Field() override;
    void GetFieldValue(const G4double p[4], G4double b[3]) const override;

  private:
    const MagneticField *theCMSMagneticField;
    double theDelta;

    mutable GPD oldp;
    mutable double oldb[3];
  };
};  // namespace sim
#endif
