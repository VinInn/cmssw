
#ifndef EGAMMAOBJECTS_GBRTreeFast
#define EGAMMAOBJECTS_GBRTreeFast

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
// terminal nodes, which are stored simply as a array of regression responses


#include "CondFormats/Serialization/interface/Serializable.h"

#include <x86intrin.h>
#include <cstring>
#include<cassert>

inline
void tof16(float const * f32, short * f16, unsigned int n) {
//  assert(0==n%4);

  __m128 vf32;
  for (auto i=0U; i<n; i+=4) {
    ::memcpy(&vf32,f32+i,sizeof(vf32));
    auto vf16 = _mm_cvtps_ph (vf32,0) ;
    ::memcpy(f16+i,&vf16,sizeof(long long));
  }
			 
}

inline
void tof32(short const * f16, float * f32, unsigned int n) {
  // assert(0==n%4);

  __m128i vf16;
  for (auto i=0U; i<n; i+=4) {
    ::memcpy(&vf16,f16+i,sizeof(long long));
    auto vf32 = _mm_cvtph_ps (vf16) ;
    ::memcpy(f32+i, &vf32,sizeof(vf32));
  }
			 
}

inline
void tof16(float f32, short & f16) {

  __m128 vf32;
    ::memcpy(&vf32,&f32,sizeof(float));
    auto vf16 = _mm_cvtps_ph (vf32,0) ;
    ::memcpy(&f16,&vf16,sizeof(f16));
}


inline
void tof32(short f16, float & f32) {

  __m128i vf16;
   ::memcpy(&vf16,&f16,sizeof(f16));
    auto vf32 = _mm_cvtph_ps (vf16) ;
    ::memcpy(&f32, &vf32,sizeof(f32));
			 
}


#include <array>

  namespace TMVA {
    class DecisionTree;
    class DecisionTreeNode;
  }

  class GBRTreeFast {

    public:

       static constexpr int NMAX=8;

       GBRTreeFast();
       explicit GBRTreeFast(const TMVA::DecisionTree *tree);
       virtual ~GBRTreeFast();
       
       short GetResponse(const short * array) const;
       short TerminalIndex(const short  *array) const;
       
       std::array<short,NMAX> &Responses() { return fResponses; }       
       const std::array<short,NMAX> &Responses() const { return fResponses; }
       
       std::array<unsigned char,NMAX> &CutIndices() { return fCutIndices; }
       const std::array<unsigned char,NMAX> &CutIndices() const { return fCutIndices; }
       
       std::array<short,NMAX> &CutVals() { return fCutVals; }
       const std::array<short,NMAX> &CutVals() const { return fCutVals; }
       
       std::array<short,NMAX> &LeftIndices() { return fLeftIndices; }
       const std::array<short,NMAX> &LeftIndices() const { return fLeftIndices; } 
       
       std::array<short,NMAX> &RightIndices() { return fRightIndices; }
       const std::array<short,NMAX> &RightIndices() const { return fRightIndices; }
       
       unsigned int size() const { return m_size;}
       void setSize(int s) { m_size =s;}
       unsigned int rsize() const { return m_rsize;}
       void setRsize(int s) { m_rsize =s;}

       
    protected:      
        unsigned int CountIntermediateNodes(const TMVA::DecisionTreeNode *node);
        unsigned int CountTerminalNodes(const TMVA::DecisionTreeNode *node);
      
        void AddNode(const TMVA::DecisionTreeNode *node);
        
	std::array<unsigned char,NMAX> fCutIndices;
	std::array<short,NMAX> fCutVals;
	std::array<short,NMAX> fLeftIndices;
	std::array<short,NMAX> fRightIndices;
	std::array<short,NMAX> fResponses;  
        
        unsigned int  m_size=0; unsigned int  m_rsize=0;
  
  COND_SERIALIZABLE;
};

//_______________________________________________________________________
inline short GBRTreeFast::GetResponse(const short * array) const {
  return fResponses[TerminalIndex(array)];
  
  /*  
  int index = 0;
  
  int s = size();
  short d[NMAX];
  short val[NMAX]; // ={0};
  for (auto i=0; i<s; ++i) val[i]=array[fCutIndices[i]];
  for (auto i=0; i<int(NMAX); ++i) d[i]=val[i]>fCutVals[i];

  while (true) {
    index =  d[index] ?
        fRightIndices[index] :
        fLeftIndices[index];
    
    if (index<=0) 
      return fResponses[-index];
    
  }
  */

}

//_______________________________________________________________________
inline short GBRTreeFast::TerminalIndex(const short * array) const {
  
  short index = 0;
  
  do {
    short c = array[fCutIndices[index]] > fCutVals[index]; c=-c;
    index =  (c&fRightIndices[index]) | ((~c)&fLeftIndices[index]);
  } while (index>0);
  return -index;

}

/*
inline int GBRTreeFast::oldTerminalIndex(const short * array) const {

  int index = 0;

  unsigned char cutindex = fCutIndices[0];
  short cutval = fCutVals[0];

  while (true) {
    if (array[cutindex] > cutval) {
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
      return (-index);
    }

  }


}
*/
  
#endif
