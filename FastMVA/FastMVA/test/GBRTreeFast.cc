#include "GBRTreeFast.h"

#include "TMVA/DecisionTreeNode.h"
#include "TMVA/DecisionTree.h"
#include<cassert>

//_______________________________________________________________________
GBRTreeFast::GBRTreeFast()
{

}

//_______________________________________________________________________
GBRTreeFast::GBRTreeFast(const TMVA::DecisionTree *tree)
{
  
  //printf("boostweights size = %i, forest size = %i\n",bdt->GetBoostWeights().size(),bdt->GetForest().size());
  int nIntermediate = CountIntermediateNodes((TMVA::DecisionTreeNode*)tree->GetRoot());
  int nTerminal = CountTerminalNodes((TMVA::DecisionTreeNode*)tree->GetRoot());
  
  //special case, root node is terminal
  if (nIntermediate==0) nIntermediate = 1;
  assert(nIntermediate<=NMAX);
  assert(nTerminal<=NMAX);

  
  AddNode((TMVA::DecisionTreeNode*)tree->GetRoot());

  //special case, root node is terminal, create fake intermediate node at root
  if (size()==0) {
    m_size=1;
    fCutIndices[0]=0;;
    fCutVals[0]=0;
    fLeftIndices[0]=0;
    fRightIndices[0]=0;
  }

}


//_______________________________________________________________________
GBRTreeFast::~GBRTreeFast() {

}

//_______________________________________________________________________
unsigned int GBRTreeFast::CountIntermediateNodes(const TMVA::DecisionTreeNode *node) {
  
  if (!node->GetLeft() || !node->GetRight() || node->IsTerminal()) {
    return 0;
  }
  else {
    return 1 + CountIntermediateNodes((TMVA::DecisionTreeNode*)node->GetLeft()) + CountIntermediateNodes((TMVA::DecisionTreeNode*)node->GetRight());
  }
  
}

//_______________________________________________________________________
unsigned int GBRTreeFast::CountTerminalNodes(const TMVA::DecisionTreeNode *node) {
  
  if (!node->GetLeft() || !node->GetRight() || node->IsTerminal()) {
    return 1;
  }
  else {
    return 0 + CountTerminalNodes((TMVA::DecisionTreeNode*)node->GetLeft()) + CountTerminalNodes((TMVA::DecisionTreeNode*)node->GetRight());
  }
  
}


//_______________________________________________________________________
void GBRTreeFast::AddNode(const TMVA::DecisionTreeNode *node) {

  if (!node->GetLeft() || !node->GetRight() || node->IsTerminal()) {
    tof16(node->GetResponse(),fResponses[rsize()]);
    ++m_rsize;
    return;
  }
  else {    
    int thisidx = size();
    fCutIndices[m_size] = node->GetSelector();
    tof16(node->GetCutValue(),fCutVals[m_size]);
    fLeftIndices[m_size] = 0;   
    fRightIndices[m_size] = 0;
    ++m_size;

    TMVA::DecisionTreeNode *left;
    TMVA::DecisionTreeNode *right;
    if (node->GetCutType()) {
      left = (TMVA::DecisionTreeNode*)node->GetLeft();
      right = (TMVA::DecisionTreeNode*)node->GetRight();
    }
    else {
      left = (TMVA::DecisionTreeNode*)node->GetRight();
      right = (TMVA::DecisionTreeNode*)node->GetLeft();
    }
    
    
    if (!left->GetLeft() || !left->GetRight() || left->IsTerminal()) {
      fLeftIndices[thisidx] = -rsize();
    }
    else {
      fLeftIndices[thisidx] = size();
    }
    AddNode(left);
    
    if (!right->GetLeft() || !right->GetRight() || right->IsTerminal()) {
      fRightIndices[thisidx] = -rsize();
    }
    else {
      fRightIndices[thisidx] = size();
    }
    AddNode(right);    
    
  }
  
}
