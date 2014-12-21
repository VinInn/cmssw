//valid in r<1.15m and |z|<2.80m

#include "MagneticField/ParametrizedEngine/src/TkBfield.cc"

#include<iostream>

// performance test
#include "DataFormats/Math/test/rdtscp.h"


int main() {

  TkBfield f;

  for (float r=0; r<1.20f; r+=0.1f)
   for (float z=-3.0f; z<3.0f; z+=0.2f) {
     float x[3] = {0.f,r,z}; float b[3];
     f.getBxyz(x,b);
     std::cout << r <<',' << z << " : " << b[1] << ',' << b[2] << std::endl;  
   } 


  float btot[3] = {};
  unsigned long long t = -rdtsc();
  for (float r=0; r<1.20f; r+=0.001f)
   for (float z=-3.0f; z<3.0f; z+=0.002f) {
     float x[3] = {0.f,r,z}; float b[3];
     f.getBxyz(x,b); btot[1]+=b[1]; btot[2]+=b[2]; 
   }
  t += rdtsc();
  std::cout << "time " << t << ' ' << btot[1] <<','<<btot[2]<< std::endl;  

  return 0;


}
