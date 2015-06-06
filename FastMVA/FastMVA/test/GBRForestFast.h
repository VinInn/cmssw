
#ifndef EGAMMAOBJECTS_GBRForestFast
#define EGAMMAOBJECTS_GBRForestFast

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GBRForest                                                            //
//                                                                      //
// A fast minimal implementation of Gradient-Boosted Regression Trees   //
// which has been especially optimized for size on disk and in memory.  //                                                                  
//                                                                      //
// Designed to be built from TMVA-trained trees, but could also be      //
// generalized to otherwise-trained trees, classification,              //
//  or other boosting methods in the future                             //
//                                                                      //
//  Josh Bendavid - MIT                                                 //
//////////////////////////////////////////////////////////////////////////

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include "GBRTreeFast.h"
#include <cmath>
#include <cstdio>

  namespace TMVA {
    class MethodBDT;
  }

  class GBRForestFast {

    public:

       GBRForestFast();
       explicit GBRForestFast(const TMVA::MethodBDT *bdt);
       virtual ~GBRForestFast();
       
       float GetResponse(const short * vector) const;
       float GetClassifier(const short * vector) const;
       
       void SetInitialResponse(float response) { fInitialResponse = response; }
       
       std::vector<GBRTreeFast> &Trees() { return fTrees; }
       const std::vector<GBRTreeFast> &Trees() const { return fTrees; }
       
    protected:
      float               fInitialResponse;
      std::vector<GBRTreeFast> fTrees;  
      
  
  COND_SERIALIZABLE;
};

//_______________________________________________________________________
inline float GBRForestFast::GetResponse(const short * vector) const {
  auto response = fInitialResponse;
  auto s =  fTrees.size();
  float ss[4]={0.f};
  short r[4] ={0}; 
  for (auto i=0U; i<s; i+=4) {
    for (auto j=0; j<4; ++j)
       r[j]=fTrees[i+j].GetResponse(vector);
    float x[4]; tof32(r,x,4);
    for (auto j=0; j<4; ++j) ss[j]+=x[j];
  }
  return response+ss[0]+ss[1]+ss[2]+ss[3];
  // return response+(r[0]+r[1]+r[2]+r[3]);
}

//_______________________________________________________________________
inline float GBRForestFast::GetClassifier(const short * vector) const {
  float response = GetResponse(vector);
  return 2.0f/(1.0f+std::exp(-2.0f*response))-1.f; //MVA output between -1 and 1
}

#endif
