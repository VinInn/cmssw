#ifndef PixelTrackProducer_h
#define PixelTrackProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackReconstruction.h"

namespace edm { class Event; class EventSetup; class ParameterSet; class ConfigurationDescriptions; }
class TrackerTopology;

class PixelTrackProducer :  public edm::stream::EDProducer<> {

public:
  explicit PixelTrackProducer(const edm::ParameterSet& conf);

  ~PixelTrackProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& ev, const edm::EventSetup& es) override;

private:
  PixelTrackReconstruction theReconstruction;
};

#endif // PixelTrackProducer_h
