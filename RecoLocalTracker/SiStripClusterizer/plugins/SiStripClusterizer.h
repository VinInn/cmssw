#ifndef RecoLocalTracker_SiStripClusterizer_h
#define RecoLocalTracker_SiStripClusterizer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"


#include <vector>
#include <memory>

class SiStripClusterizer : public edm::stream::EDProducer<>  {

public:

  explicit SiStripClusterizer(const edm::ParameterSet& conf);
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  template<class T> bool findInput(const edm::EDGetTokenT<T>&, edm::Handle<T>&, const edm::Event&);
  const std::vector<edm::InputTag> inputTags;
  std::auto_ptr<StripClusterizerAlgorithm> algorithm;
  typedef edm::EDGetTokenT< edm::DetSetVector<SiStripDigi> > token_t;
  typedef std::vector<token_t> token_v;
  token_v inputTokens;

  // Common Mode for HIP detection
  using CMContainer=edm::DetSetVector<SiStripRawDigi>;
  edm::EDGetTokenT<CMContainer> theCommonMode;

  
};

#endif
