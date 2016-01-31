
#ifndef EGAMMAOBJECTS_GBRTree
#define EGAMMAOBJECTS_GBRTree

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

// The decision tree is implemented here as a set of two arrays, one for
// intermediate nodes, containing the variable index and cut value, as well
// as the indices of the 'left' and 'right' daughter nodes.  Positive indices
// indicate further intermediate nodes, whereas negative indices indicate
// terminal nodes, which are stored simply as a vector of regression responses


#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#ifdef VECTOR_TMVA
#include<array>
#include "nativeVector.h"
using namespace nativeVector;
using TMVA_out = FVect;
using TMVA_in = std::array<float const *,8>;
#endif

  namespace TMVA {
    class DecisionTree;
    class DecisionTreeNode;
  }

  class GBRTree {

    public:

       GBRTree();
       explicit GBRTree(const TMVA::DecisionTree *tree, double scale, bool useyesnoleaf, bool adjustboundary);
       virtual ~GBRTree();

#ifdef VECTOR_TMVA       
       TMVA_out GetResponseV(TMVA_in const & vector) const;
#endif
       double GetResponse(const float* vector) const;
       int TerminalIndex(const float *vector) const;
       
       std::vector<float> &Responses() { return fResponses; }       
       const std::vector<float> &Responses() const { return fResponses; }
       
       std::vector<int> &CutIndices() { return fCutIndices; }
       const std::vector<int> &CutIndices() const { return fCutIndices; }
       
       std::vector<float> &CutVals() { return fCutVals; }
       const std::vector<float> &CutVals() const { return fCutVals; }
       
       std::vector<int> &LeftIndices() { return fLeftIndices; }
       const std::vector<int> &LeftIndices() const { return fLeftIndices; } 
       
       std::vector<int> &RightIndices() { return fRightIndices; }
       const std::vector<int> &RightIndices() const { return fRightIndices; }
       

       
    protected:      
        unsigned int CountIntermediateNodes(const TMVA::DecisionTreeNode *node);
        unsigned int CountTerminalNodes(const TMVA::DecisionTreeNode *node);
      
        void AddNode(const TMVA::DecisionTreeNode *node, double scale, bool isregression, bool useyesnoleaf, bool adjustboundary);
        
	std::vector<int> fCutIndices;
	std::vector<float> fCutVals;
	std::vector<int> fLeftIndices;
	std::vector<int> fRightIndices;
	std::vector<float> fResponses;  
        
  
  COND_SERIALIZABLE;
};

#include<cassert>
//_______________________________________________________________________
#ifdef VECTOR_TMVA
inline  TMVA_out GBRTree::GetResponseV(TMVA_in const & vector) const {
   IVect index = {0};
   IVect rindex = {0}; rindex+=10000;
   IVect zero = {0};

  do {
    // for (int i=0; i<8; ++i) assert(index[i]>=0);
    auto ci = gather(&fCutIndices.front(),index);
    FVect v;
    for (int i=0; i<8; ++i) v[i]  = vector[i][ci[i]];
    auto c = gather(&fCutVals.front(),index);
    auto r = gather(&fRightIndices.front(),index);
    auto l = gather(&fLeftIndices.front(),index);
    index = v>c ? r : l;
    rindex = rindex<=0 ? rindex : index;
    index = rindex<0 ? zero : rindex;  
  } while(!testz(0<rindex));
  // std::cout << rindex << std::endl;
  // for (int i=0; i<8; ++i) assert(rindex[i]<=0);
  return gather(&fResponses.front(),-rindex);

}
#endif

//_______________________________________________________________________
inline double GBRTree::GetResponse(const float* vector) const {
  return fResponses[TerminalIndex(vector)];

/*
  int index = 0;
  
  unsigned char cutindex = fCutIndices[0];
  float cutval = fCutVals[0];
  
  while (true) {
     
    if (vector[cutindex] > cutval) {
      index = fRightIndices[index];
    }
    else {
      index = fLeftIndices[index];
    }
    
    if (index>0) {
      cutindex = fCutIndices[index];
      cutval = fCutVals[index];
    }
    else {
      return fResponses[-index];
    }
    
  }
  */

}

//_______________________________________________________________________
inline int GBRTree::TerminalIndex(const float* vector) const {
  
  

  int index = 0;
  
  do {
   int v,c;
   memcpy(&v,&vector[fCutIndices[index]],4);
   memcpy(&c,&fCutVals[index],4);
   c = v>c; c=-c;
   index = (c&int(fRightIndices[index])) | ((~c)&int(fLeftIndices[index]));

  } while(index>0);
  return -index;  

}
  
#endif
