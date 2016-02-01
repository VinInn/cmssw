
#ifndef EGAMMAOBJECTS_GBRForest
#define EGAMMAOBJECTS_GBRForest

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
#include "GBRTree.h"
#include <math.h>
#include <stdio.h>

  namespace TMVA {
    class MethodBDT;
  }

  class GBRForest {

    public:

       GBRForest();
       explicit GBRForest(const TMVA::MethodBDT *bdt);
       virtual ~GBRForest();

#ifdef VECTOR_TMVA
       TMVA_out GetResponseV(TMVA_in const * vector) const;
       TMVA_out GetGradBoostClassifierV(TMVA_in const * vector) const;
#endif
       double GetResponse(const float* vector) const;
       double GetGradBoostClassifier(const float* vector) const;
       double GetAdaBoostClassifier(const float* vector) const { return GetResponse(vector); }
       
       //for backwards-compatibility
       double GetClassifier(const float* vector) const { return GetGradBoostClassifier(vector); }
       
       void SetInitialResponse(double response) { fInitialResponse = response; }
       
       std::vector<GBRTree> &Trees() { return fTrees; }
       const std::vector<GBRTree> &Trees() const { return fTrees; }
       
    protected:
      double               fInitialResponse;
      std::vector<GBRTree> fTrees;  
      
  
  COND_SERIALIZABLE;
};

//_______________________________________________________________________
#ifdef VECTOR_TMVA
inline TMVA_out GBRForest::GetResponseV(TMVA_in const * vector) const {
   FVect response = {0.}; response+=float(fInitialResponse);
    for (std::vector<GBRTree>::const_iterator it=fTrees.begin(); it!=fTrees.end(); ++it) {
    response += it->GetResponseV(vector);
   }
  return response;
}
#endif

inline double GBRForest::GetResponse(const float* vector) const {
  double response = fInitialResponse;
  for (std::vector<GBRTree>::const_iterator it=fTrees.begin(); it!=fTrees.end(); ++it) {
    response += it->GetResponse(vector);
  }
  return response;
}

//_______________________________________________________________________

#ifdef VECTOR_TMVA
inline TMVA_out GBRForest::GetGradBoostClassifierV(TMVA_in const * vector) const {
  auto response = GetResponseV(vector);
  for (int i=0; i<8; ++i) response[i] = 2.0f/(1.0f+expf(-2.0f*response[i]))-1.0f;  
  return response;
}
#endif
inline double GBRForest::GetGradBoostClassifier(const float* vector) const {
  double response = GetResponse(vector);
  return 2.0/(1.0+exp(-2.0*response))-1; //MVA output between -1 and 1
}

#endif
