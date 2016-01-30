#include "GBRForestFast.h"
//#include <iostream>
#include "TMVA/DecisionTree.h"
#include "TMVA/MethodBDT.h"



//_______________________________________________________________________
GBRForestFast::GBRForestFast() : 
  fInitialResponse(0.)
{

}

//_______________________________________________________________________
GBRForestFast::~GBRForestFast() 
{
}

//_______________________________________________________________________
GBRForestFast::GBRForestFast(const TMVA::MethodBDT *bdt)
{
  
  if (bdt->DoRegression()) {
    fInitialResponse = bdt->GetBoostWeights().front();
  }
  else {
    fInitialResponse = 0.;
  }
  
  const std::vector<TMVA::DecisionTree*> &forest = bdt->GetForest();
  fTrees.reserve(forest.size());
  for (auto const & t : forest) {
    fTrees.push_back(GBRTreeFast(t));
  }
  
}




