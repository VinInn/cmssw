/** \file
 *  A simple program to print field value.
 *
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "DataFormats/GeometryVector/interface/Pi.h"
//#include "DataFormats/GeometryVector/interface/CoordinateSets.h"


#include <iostream>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>
#include "benchmark.h"

using namespace edm;
using namespace Geom;
using namespace std;

class TrackInField final : public edm::EDAnalyzer {
 public:

  TrackInField(const edm::ParameterSet&){}

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override{
   ESHandle<MagneticField> magfield;
   setup.get<IdealMagneticFieldRecord>().get(magfield);

   field = magfield.product();

   cout << "Field Nominal Value: " << field->nominalValue() << endl;

  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

  long long n=0;
  int nsteps=1200.;
  for (int kk=0; kk<1000; ++kk)
  for (float dz=0; dz<10.f; dz+=0.1f) {
    float x=0,y=0,z=0;
    for (int i=0; i<nsteps; ++i) {
     x+=1.f; y+=1.f; z+=dz;
    
     GlobalPoint g(x,y,z);

     delta -= (chrono::high_resolution_clock::now()-start);
     benchmark::touch(g);
     auto f = field->inTesla(g);
     benchmark::keep(f);
     delta += (chrono::high_resolution_clock::now()-start);
     ++n;
     if (z>1100.f) break;
     if (f.z()==0) break;
//     cout << "At R=" << g.perp() << " phi=" << g.phi()<< " z " << g.z() << " B=" << f << endl;
    }
   }
   std::cout << " query took "
              << std::chrono::duration_cast<chrono::milliseconds>(delta).count()/double(n)
              << " ms" << std::endl;
   

  }
   
 private:
  const MagneticField* field;
};


DEFINE_FWK_MODULE(TrackInField);

