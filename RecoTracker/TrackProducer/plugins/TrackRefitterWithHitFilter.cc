/** \class TrackRefitterWithHitFilter
 *  Refit Tracks: Produce Tracks from TrackCollection. It performs a new final fit on a TrackCollection.
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"
#include "HitRemoverFromTrack.h"

#include "DataFormats/Common/interface/fakearray.h"

namespace {

  class TrackRefitterWithHitFilter final : public KfTrackProducerBase, public edm::stream::EDProducer<> {
    using FakeHitList = std::vector<fakearray<int,4>>;
    using HitList = HitRemoverFromTrack::HitList;
  public:
    
    /// Constructor
    explicit TrackRefitterWithHitFilter(const edm::ParameterSet& iConfig);
    
    /// Implementation of produce method
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    
  private:
    TrackProducerAlgorithm<reco::Track> theAlgo;
    edm::EDGetToken m_hitList;
    HitRemoverFromTrack m_hitRemover;
  };
}

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"


namespace {

  TrackRefitterWithHitFilter::TrackRefitterWithHitFilter(const edm::ParameterSet& iConfig):
    KfTrackProducerBase(iConfig.getParameter<bool>("TrajectoryInEvent"),
			iConfig.getParameter<bool>("useHitsSplitting")),
    theAlgo(iConfig),
    m_hitList(consumes<FakeHitList>(iConfig.getParameter<edm::InputTag>( "hitList" )))
  {
    setConf(iConfig);
    setSrc( consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>( "src" )), 
	    consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>( "beamSpot" )),
	    consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>( "MeasurementTrackerEvent") ));
    setAlias( iConfig.getParameter<std::string>( "@module_label" ) );
    
    //register your products
    produces<reco::TrackCollection>().setBranchAlias( alias_ + "Tracks" );
    produces<reco::TrackExtraCollection>().setBranchAlias( alias_ + "TrackExtras" );
    produces<TrackingRecHitCollection>().setBranchAlias( alias_ + "RecHits" );
    produces<std::vector<Trajectory> >() ;
    produces<std::vector<int> >() ;
    produces<TrajTrackAssociationCollection>();
    
  }
  
  void TrackRefitterWithHitFilter::produce(edm::Event& theEvent, const edm::EventSetup& setup)
  {
    LogDebug("TrackRefitterWithHitFilter") << "Analyzing event number: " << theEvent.id() << "\n";
    //
    // create empty output collections
    //
    std::unique_ptr<TrackingRecHitCollection>   outputRHColl (new TrackingRecHitCollection);
    std::unique_ptr<reco::TrackCollection>      outputTColl(new reco::TrackCollection);
    std::unique_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);
    std::unique_ptr<std::vector<Trajectory>>    outputTrajectoryColl(new std::vector<Trajectory>);
    std::unique_ptr<std::vector<int>>           outputIndecesInputColl(new std::vector<int>);
    
    //
    //declare and get stuff to be retrieved from ES
    //
    edm::ESHandle<TrackerGeometry> theG;
    edm::ESHandle<MagneticField> theMF;
    edm::ESHandle<TrajectoryFitter> theFitter;
    edm::ESHandle<Propagator> thePropagator;
    edm::ESHandle<MeasurementTracker>  theMeasTk;
    edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
    getFromES(setup,theG,theMF,theFitter,thePropagator,theMeasTk,theBuilder);
    
    edm::ESHandle<TrackerTopology> httopo;
    setup.get<TrackerTopologyRcd>().get(httopo);
    
    //
    //declare and get TrackCollection to be retrieved from the event
    //
    AlgoProductCollection algoResults;
    reco::BeamSpot bs;
    
    edm::Handle<edm::View<reco::Track>> theTCollection;
    getFromEvt(theEvent,theTCollection,bs);
    
    LogDebug("TrackRefitterWithHitFilter") << "TrackRefitterWithHitFilter::produce(none):Number of Trajectories:" << (*theTCollection).size();
    
    if (bs.position()==math::XYZPoint(0.,0.,0.) && bs.type() == reco::BeamSpot::Unknown) {
      edm::LogError("TrackRefitterWithHitFilter") << " BeamSpot is (0,0,0), it is probably because is not valid in the event";
    }
    
    if (theTCollection.failedToGet()){
      edm::EDConsumerBase::Labels labels;
      labelsForToken(src_, labels);
      edm::LogError("TrackRefitterWithHitFilter")<<"could not get the reco::TrackCollection." << labels.module;
    }
    LogDebug("TrackRefitterWithHitFilter") << "run the algorithm" << "\n";
    
    try {
      edm::Handle<FakeHitList> hitListHandle;
      theEvent.getByToken(m_hitList, hitListHandle);
      m_hitRemover.setHits((HitList const *)hitListHandle.product());
      
      theAlgo.runWithTrack(m_hitRemover, theG.product(), theMF.product(), *theTCollection, 
			   theFitter.product(), thePropagator.product(), 
			   theBuilder.product(), bs, algoResults);
    }catch (cms::Exception &e){ edm::LogError("TrackProducer") << "cms::Exception caught during theAlgo.runWithTrack." << "\n" << e << "\n"; throw; }

    std::cout << "TrackRefit " << "Number of Tracks found: " << algoResults.size() << std::endl;

    
    //put everything in th event
    putInEvt(theEvent, thePropagator.product(), theMeasTk.product(), outputRHColl, outputTColl, outputTEColl, outputTrajectoryColl,  outputIndecesInputColl, algoResults,theBuilder.product(), httopo.product());
    LogDebug("TrackRefitterWithHitFilter") << "end" << "\n";
  }
  
}

DEFINE_FWK_MODULE(TrackRefitterWithHitFilter);
