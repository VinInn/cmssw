#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <limits>

#include "getBestVertex.h"

#include "TFile.h"



#include <array>
#include <memory>
#include <atomic>
#include <cassert>


struct NodePtr {
  int aba=0;
  int ptr=-1;
  operator bool() const { return ptr>=0;}
};
inline bool operator==(NodePtr a, NodePtr b) { return a.aba==b.aba && a.ptr==b.ptr; }

template<typename T>
class concurrent_stack {
public:
  using Item=T;
  
  struct Node {
    template<typename ...Arguments>
    Node(Arguments&&...args) : t(args...){}
    T & operator*() {return t;}
    T t;
    NodePtr prev;
  };
  
 
  concurrent_stack() : nel(0), ns(0) {}

  ~concurrent_stack() {    
    assert(ns==nel);
    while(pop().ptr>=0); assert(nel==0);
  }
    
  
  NodePtr pop() {
    NodePtr lhead = head;
    //if (lhead.ptr<0) return NodePtr();
    //NodePtr prev = node(lhead)->prev;
    //while (!head.compare_exchange_weak(lhead,prev)) { if(lhead.ptr<0) return NodePtr(); prev = node(lhead)->prev;}
    while (lhead && !head.compare_exchange_weak(lhead,nodes[lhead.ptr]->prev));
    if (lhead.ptr>=0) {
#ifdef VICONC_DEBUG      
      assert(node(lhead)->prev.ptr!=lhead.ptr);
      // assert(node(lhead)->prev==prev);  // usually fails in case of aba
      node(lhead)->prev=NodePtr(); // not needed
#endif
      nel-=1;
#ifdef VICONC_DEBUG      
      // assert(nel>=0);  // may fail 
      verify[lhead.ptr].v-=1;
      assert(0==verify[lhead.ptr].v);
#endif
    }
    return lhead;
  }

  Node * node(NodePtr p) const { return p.ptr>=0 ? nodes[p.ptr].get() : nullptr;}

  void push(NodePtr np) {
#ifdef VICONC_DEBUG      
    verify[np.ptr].v+=1;
    assert(1==verify[np.ptr].v);
#endif
    auto n = node(np);
    np.aba +=1;  // remove this to see aba in action!
    n->prev = head;
    while (!head.compare_exchange_weak(n->prev,np));
    nel+=1;
  }

  unsigned int size() const { return nel;}
  unsigned int nalloc() const { return ns;}

  
  template<typename ...Arguments>
  NodePtr make(Arguments&&...args) {
    int e = ns;
    while (!ns.compare_exchange_weak(e,e+1));
    nodes[e] = std::make_unique<Node>(args...);
#ifdef VICONC_DEBUG      
    verify[e].v=0;
#endif
    return NodePtr{0,e};
  }

  std::atomic<NodePtr> head;
  std::atomic<int> nel;

  std::atomic<int> ns;
  std::array<std::unique_ptr<Node>,1024> nodes;

#ifdef VICONC_DEBUG
  struct alignas(64) IntA64 { int v;};
  std::array<IntA64,1024> verify;
#endif
};

template<typename T>
struct NodePtrGard {
  using Stack = T;
  using Item = typename Stack::Item;
  
  NodePtrGard(Stack & s) : stack(s){
    while(ptr=stack.pop());
  }

  template<typename ...Arguments>
  NodePtrGard(Stack & s,
	      Arguments&&...args
	      ) : stack(s){
    auto a = stack.pop();
    if (!a) a = stack.make(args...);
    assert(a);
    ptr=a;
  }

  ~NodePtrGard() { stack.push(ptr); }

  Item & operator*() {return **stack.node(ptr);}

    
  Stack & stack;
  NodePtr ptr;
};

std::atomic<unsigned int> ni(0);
struct Stateful {

  Stateful(int i) : id(i) {if (i>=0) ni+=1;}
  ~Stateful() { std::cout << "Stateful " << id << '/'<<count << ' ' << ni << std::endl; }


  void operator()() {
    ++count;
    
    assert(0==verify);
    ++verify;
    assert(1==verify);
    verify *=2;
    assert(2==verify);
    --verify;
    assert(1==verify);
    --verify;
    assert(0==verify);

  }
    
  long long count=0;
  int id=-1;

  int verify=0;
};

namespace {
  using Stack = concurrent_stack<Stateful>;
  Stack stack;

  std::atomic<long long> ncall=0;
}

namespace {
  
template<bool PROMPT>
struct mva {
  mva(const edm::ParameterSet &cfg):
    forestLabel_    ( cfg.getParameter<std::string>("GBRForestLabel") ),
    dbFileName_     ( cfg.getParameter<std::string>("GBRForestFileName") ),
    useForestFromDB_( (!forestLabel_.empty()) & dbFileName_.empty())
  {}

  void beginStream() {
    if(!dbFileName_.empty()){
      TFile gbrfile(dbFileName_.c_str());
      forestFromFile_.reset((GBRForest*)gbrfile.Get(forestLabel_.c_str()));
    }
  }

  void initEvent(const edm::EventSetup& es) {
    forest_ = forestFromFile_.get();
    if(useForestFromDB_){
      edm::ESHandle<GBRForest> forestHandle;
      es.get<GBRWrapperRcd>().get(forestLabel_,forestHandle);
      forest_ = forestHandle.product();
    }
  }

  float operator()(reco::Track const & trk,
		   reco::BeamSpot const & beamSpot,
		   reco::VertexCollection const & vertices) const {

    auto tmva_pt_ = trk.pt();
    auto tmva_ndof_ = trk.ndof();
    auto tmva_nlayers_ = trk.hitPattern().trackerLayersWithMeasurement();
    auto tmva_nlayers3D_ = trk.hitPattern().pixelLayersWithMeasurement()
        + trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
    auto tmva_nlayerslost_ = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    float chi2n =  trk.normalizedChi2();
    float chi2n_no1Dmod = chi2n;
    
    int count1dhits = 0;
    for (auto ith =trk.recHitsBegin(); ith!=trk.recHitsEnd(); ++ith) {
      const auto & hit = *(*ith);
      if (hit.dimension()==1) ++count1dhits;
    }
    
    if (count1dhits > 0) {
      float chi2 = trk.chi2();
      float ndof = trk.ndof();
      chi2n = (chi2+count1dhits)/float(ndof+count1dhits);
    }
    auto tmva_chi2n_ = chi2n;
    auto tmva_chi2n_no1dmod_ = chi2n_no1Dmod;
    auto tmva_eta_ = trk.eta();
    auto tmva_relpterr_ = float(trk.ptError())/std::max(float(trk.pt()),0.000001f);
    auto tmva_nhits_ = trk.numberOfValidHits();
    int lostIn = trk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    int lostOut = trk.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
    auto tmva_minlost_ = std::min(lostIn,lostOut);
    auto tmva_lostmidfrac_ = static_cast<float>(trk.numberOfLostHits()) / static_cast<float>(trk.numberOfValidHits() + trk.numberOfLostHits());
   
    float gbrVals_[PROMPT ? 16 : 12];
    gbrVals_[0] = tmva_pt_;
    gbrVals_[1] = tmva_lostmidfrac_;
    gbrVals_[2] = tmva_minlost_;
    gbrVals_[3] = tmva_nhits_;
    gbrVals_[4] = tmva_relpterr_;
    gbrVals_[5] = tmva_eta_;
    gbrVals_[6] = tmva_chi2n_no1dmod_;
    gbrVals_[7] = tmva_chi2n_;
    gbrVals_[8] = tmva_nlayerslost_;
    gbrVals_[9] = tmva_nlayers3D_;
    gbrVals_[10] = tmva_nlayers_;
    gbrVals_[11] = tmva_ndof_;

    if (PROMPT) {
      auto tmva_absd0_ = std::abs(trk.dxy(beamSpot.position()));
      auto tmva_absdz_ = std::abs(trk.dz(beamSpot.position()));
      Point bestVertex = getBestVertex(trk,vertices);
      auto tmva_absd0PV_ = std::abs(trk.dxy(bestVertex));
      auto tmva_absdzPV_ = std::abs(trk.dz(bestVertex));
      
      gbrVals_[12] = tmva_absd0PV_;
      gbrVals_[13] = tmva_absdzPV_;
      gbrVals_[14] = tmva_absdz_;
      gbrVals_[15] = tmva_absd0_;
    }


    NodePtrGard<Stack> n(stack,++ncall);
    (*n)();

    return forest_->GetClassifier(gbrVals_);

  }

  static const char * name();

  static void fillDescriptions(edm::ParameterSetDescription & desc) {
    desc.add<std::string>("GBRForestLabel",std::string());
    desc.add<std::string>("GBRForestFileName",std::string());
  }
  
  std::unique_ptr<GBRForest> forestFromFile_;
  const GBRForest *forest_ = nullptr; // owned by somebody else
  const std::string forestLabel_;
  const std::string dbFileName_;
  const bool useForestFromDB_;
};

  using TrackMVAClassifierDetached = TrackMVAClassifier<mva<false>>;
  using TrackMVAClassifierPrompt = TrackMVAClassifier<mva<true>>;
  template<>
  const char * mva<false>::name() { return "TrackMVAClassifierDetached";}
  template<>
  const char * mva<true>::name() { return "TrackMVAClassifierPrompt";}
  
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackMVAClassifierDetached);
DEFINE_FWK_MODULE(TrackMVAClassifierPrompt);
