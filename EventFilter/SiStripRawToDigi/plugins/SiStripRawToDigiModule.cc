
#include "SiStripRawToDigiModule.h"
#include "SiStripRawToDigiUnpacker.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstdlib>

#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include<random>
namespace {

  struct RandomCM {

    bool operator()(float frac) { return rgen(eng) < frac; }
    std::mt19937 eng;
    std::uniform_real_distribution<float> rgen = std::uniform_real_distribution<float>(0.,1.);

  };

  thread_local RandomCM randomCM;



  struct Comp {
     bool operator()(SiStripDigi const & d, unsigned int i) const { return d.strip()<i;}
     bool operator()(unsigned int i, SiStripDigi const & d) const { return i < d.strip();} 
  };

  constexpr float probTIB[4] = {22*0.0072,20*0.0050,20*0.0041, 20*0.0027};
  constexpr float probTOB[6] = {18*0.0185, 16*0.0138, 10*0.0101, 10*0.0077, 10*0.0040, 10*0.0031};
  constexpr float probTID[3] = {22*0.0072,20*0.0050,20*0.0041};
  constexpr float probTEC[7] = {22*0.0072,20*0.0050,20*0.0041,10*0.0040,10*0.0040,10*0.0040,10*0.0040};
  constexpr int	napvTEC[7] {6,6,4,4,6,4,4};


  void kill(edm::DetSet<SiStripDigi>  & ds) {
     auto id = DetId(ds.detId()).subdetId()-3;
     float frac=0; int napv=4;
     if (id==1 || id==3) {
       auto l = id==1 ? TIDDetId(ds.detId()).ring() : TECDetId(ds.detId()).ring();
       l--;
       frac =  (id==1) ?  probTID[l] : probTEC[l];
       napv= napvTEC[l];
     } else {
       auto l = TIBDetId(ds.detId()).layer()-1;
       frac =  (id==0) ?  probTIB[l] : probTOB[l];
       if (id==2 && l>3) napv= 6; if (id==1 && l<2) napv= 6;
     }
     for (int i=0; i<napv; ++i) {
        if ( !randomCM(frac) ) continue;
        auto b=i*128;
        auto e=b+128;
        auto fi = std::lower_bound(ds.begin(),ds.end(),b,Comp()); 
        auto la  = std::lower_bound(ds.begin(),ds.end(),e,Comp());
        for (;fi!=la; ++fi) (*fi) = SiStripDigi(fi->strip(),0);       	
     }

  }

}


namespace sistrip {

  RawToDigiModule::RawToDigiModule( const edm::ParameterSet& pset ) :
    rawToDigi_(0),
    cabling_(0),
    cacheId_(0),
    extractCm_(false),
    doFullCorruptBufferChecks_(false),
    doAPVEmulatorCheck_(true)
  {
    if ( edm::isDebugEnabled() ) {
      LogTrace("SiStripRawToDigi")
	<< "[sistrip::RawToDigiModule::" << __func__ << "]"
	<< " Constructing object...";
    }
    
    token_ = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("ProductLabel"));
    int16_t appended_bytes = pset.getParameter<int>("AppendedBytes");
    int16_t trigger_fed_id = pset.getParameter<int>("TriggerFedId");
    bool legacy_unpacker = pset.getParameter<bool>("LegacyUnpacker");
    bool use_daq_register = pset.getParameter<bool>("UseDaqRegister");
    bool using_fed_key = pset.getParameter<bool>("UseFedKey");
    bool unpack_bad_channels = pset.getParameter<bool>("UnpackBadChannels");
    bool mark_missing_feds = pset.getParameter<bool>("MarkModulesOnMissingFeds");

    int16_t fed_buffer_dump_freq = pset.getUntrackedParameter<int>("FedBufferDumpFreq",0);
    int16_t fed_event_dump_freq = pset.getUntrackedParameter<int>("FedEventDumpFreq",0);
    bool quiet = pset.getUntrackedParameter<bool>("Quiet",true);
    extractCm_ = pset.getParameter<bool>("UnpackCommonModeValues");
    doFullCorruptBufferChecks_ = pset.getParameter<bool>("DoAllCorruptBufferChecks");
    doAPVEmulatorCheck_ = pset.getParameter<bool>("DoAPVEmulatorCheck");

    uint32_t errorThreshold = pset.getParameter<unsigned int>("ErrorThreshold");

    rawToDigi_ = new sistrip::RawToDigiUnpacker( appended_bytes, fed_buffer_dump_freq, fed_event_dump_freq, trigger_fed_id, using_fed_key, unpack_bad_channels, mark_missing_feds, errorThreshold);
    rawToDigi_->legacy(legacy_unpacker);
    rawToDigi_->quiet(quiet);
    rawToDigi_->useDaqRegister( use_daq_register ); 
    rawToDigi_->extractCm(extractCm_);
    rawToDigi_->doFullCorruptBufferChecks(doFullCorruptBufferChecks_);
    rawToDigi_->doAPVEmulatorCheck(doAPVEmulatorCheck_);

    produces< SiStripEventSummary >();
    produces< edm::DetSetVector<SiStripRawDigi> >("ScopeMode");
    produces< edm::DetSetVector<SiStripRawDigi> >("VirginRaw");
    produces< edm::DetSetVector<SiStripRawDigi> >("ProcessedRaw");
    produces< edm::DetSetVector<SiStripDigi> >("ZeroSuppressed");
    produces<DetIdCollection>();
    if ( extractCm_ ) produces< edm::DetSetVector<SiStripRawDigi> >("CommonMode");
  
  }

  RawToDigiModule::~RawToDigiModule() {
    if ( rawToDigi_ ) { delete rawToDigi_; }
    if ( cabling_ ) { cabling_ = 0; }
    if ( edm::isDebugEnabled() ) {
      LogTrace("SiStripRawToDigi")
	<< "[sistrip::RawToDigiModule::" << __func__ << "]"
	<< " Destructing object...";
    }
  }

  void RawToDigiModule::beginRun( const edm::Run& run, const edm::EventSetup& setup ) {
    updateCabling( setup );
  }  


  
  /** 
      Retrieves cabling map from EventSetup and FEDRawDataCollection
      from Event, creates a DetSetVector of SiStrip(Raw)Digis, uses the
      SiStripRawToDigiUnpacker class to fill the DetSetVector, and
      attaches the container to the Event.
  */
  void RawToDigiModule::produce( edm::Event& event, const edm::EventSetup& setup ) {
  
    updateCabling( setup );
  
    // Retrieve FED raw data (by label, which is "source" by default)
    edm::Handle<FEDRawDataCollection> buffers;
    event.getByToken( token_, buffers ); 

    // Populate SiStripEventSummary object with "trigger FED" info
    std::auto_ptr<SiStripEventSummary> summary( new SiStripEventSummary() );
    rawToDigi_->triggerFed( *buffers, *summary, event.id().event() ); 

    // Create containers for digis
    edm::DetSetVector<SiStripRawDigi>* sm = new edm::DetSetVector<SiStripRawDigi>();
    edm::DetSetVector<SiStripRawDigi>* vr = new edm::DetSetVector<SiStripRawDigi>();
    edm::DetSetVector<SiStripRawDigi>* pr = new edm::DetSetVector<SiStripRawDigi>();
    edm::DetSetVector<SiStripDigi>* zs = new edm::DetSetVector<SiStripDigi>();
    DetIdCollection* ids = new DetIdCollection();
    edm::DetSetVector<SiStripRawDigi>* cm = new edm::DetSetVector<SiStripRawDigi>();
  
    // Create digis
    if ( rawToDigi_ ) { rawToDigi_->createDigis( *cabling_,*buffers,*summary,*sm,*vr,*pr,*zs,*ids,*cm ); }


    for (auto & ds : *zs) kill(ds);
  
    // Create auto_ptr's of digi products
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > sm_dsv(sm);
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > vr_dsv(vr);
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > pr_dsv(pr);
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > zs_dsv(zs);
    std::auto_ptr< DetIdCollection > det_ids(ids);
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > cm_dsv(cm);
  
    // Add to event
    event.put( summary );
    event.put( sm_dsv, "ScopeMode" );
    event.put( vr_dsv, "VirginRaw" );
    event.put( pr_dsv, "ProcessedRaw" );
    event.put( zs_dsv, "ZeroSuppressed" );
    event.put( det_ids );
    if ( extractCm_ ) event.put( cm_dsv, "CommonMode" );
  
  }

  void RawToDigiModule::updateCabling( const edm::EventSetup& setup ) {

    uint32_t cache_id = setup.get<SiStripFedCablingRcd>().cacheIdentifier();

    if ( cacheId_ != cache_id ) {
    
      edm::ESHandle<SiStripFedCabling> c;
      setup.get<SiStripFedCablingRcd>().get( c );
      cabling_ = c.product();
    
      if ( edm::isDebugEnabled() ) {
	if ( !cacheId_ ) {
	  std::stringstream ss;
	  ss << "[sistrip::RawToDigiModule::" << __func__ << "]"
	     << " Updating cabling for first time..." << std::endl
	     << " Terse print out of FED cabling:" << std::endl;
	  cabling_->terse(ss);
	  LogTrace("SiStripRawToDigi") << ss.str();
	}
      }
    
      if ( edm::isDebugEnabled() ) {
	std::stringstream sss;
	sss << "[sistrip::RawToDigiModule::" << __func__ << "]"
	    << " Summary of FED cabling:" << std::endl;
	cabling_->summary(sss);
	LogTrace("SiStripRawToDigi") << sss.str();
      }
      cacheId_ = cache_id;
    }
  }

}

