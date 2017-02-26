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

class queryField : public edm::EDAnalyzer {
 public:
  queryField(const edm::ParameterSet& pset) {    
  }

  ~queryField(){}

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) {
   ESHandle<MagneticField> magfield;
   setup.get<IdealMagneticFieldRecord>().get(magfield);

   field = magfield.product();

   cout << "Field Nominal Value: " << field->nominalValue() << endl;

   float x,y,z;

  auto start = std::chrono::high_resolution_clock::now();
  auto delta = start - start;

  long long n=0;
   while (1) {
     
     cout << "Enter X Y Z (cm): ";

     if (!(cin >> x >>  y >>  z)) exit(0);
    
     GlobalPoint g(x,y,z);

     delta -= (chrono::high_resolution_clock::now()-start);
     benchmark::touch(g);
     auto f = field->inTesla(g);
     benchmark::keep(f);
     delta += (chrono::high_resolution_clock::now()-start);
     ++n;

     cout << "At R=" << g.perp() << " phi=" << g.phi()<< " B=" << f << endl;
   }
   std::cout << " query took "
              << std::chrono::duration_cast<chrono::milliseconds>(delta).count()/double(n)
              << " ms" << std::endl;
   

  }
   
 private:
  const MagneticField* field;
};


DEFINE_FWK_MODULE(queryField);

