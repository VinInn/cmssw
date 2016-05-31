
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


class SwapAverage {
public:
  using Container = std::vector<unsigned short>;
  
  Container & toFill() { return m_ave[0];}
  Container const & ave() const { return m_ave[1];}
  Container & noiseFill() { return m_noise[0];}
  Container const & noise() const { return m_noise[1];}


 void reset(int n) {
   m_ave[0].clear();
   m_ave[1].clear();
   m_ave[0].resize(n,0);
   m_ave[1].resize(n,0);
   m_noise[0].clear();
   m_noise[1].clear();
   m_noise[0].resize(n,0);
   m_noise[1].resize(n,0);

   first=false;
   init=false;
   nEv=0;
 }

  void finalize() {
    ++nEv;
    if (maxEv==nEv) {
      for ( auto & val : toFill()) val /=nEv;
      for ( auto & val : noiseFill()) {val /=nEv; val=std::max(int(std::sqrt(float(val))),1)+1;}

      if (init)
      for (auto k=0U; k<ave().size(); ++k) {
        if ( std::abs(toFill()[k]-ave()[k]) > noiseFill()[k]) 
          std::cout << "cm dif " << k << ' ' << toFill()[k] << ' ' << ave()[k] << ' ' << noiseFill()[k]<<std::endl;
      } 
   
      std::swap(m_ave[0],m_ave[1]);
      std::swap(m_noise[0],m_noise[1]);
      for ( auto & val : toFill()) val=0;
      for ( auto & val : noiseFill()) val=0;
      nEv=0;
      init=true;
     {
      unsigned short mi=6000; unsigned short ma=0; int s=0; int n=0;
      for (auto val : ave()) {
        if (val<10) continue;
        ++n; s+=val;
        mi=std::min(mi,val); ma=std::max(ma,val);
      }
      std::cout << "cm mn " << mi << ' ' << ma <<' '<< s/n << std::endl;
     }
     {
      unsigned short mi=6000; unsigned short ma=0; int s=0; int n=0;
      for (auto val : noise()) {
        if (0==val || val>50) continue;
        ++n; s+=val;
        mi=std::min(mi,val); ma=std::max(ma,val);
      }
      std::cout << "noise mn " << mi << ' ' << ma <<' '<< s/n << std::endl;
     }

   }
  }

  const int maxEv=100;
  int nEv=0;
  bool init=false;
  bool first=true;

  Container m_ave[2];
  Container m_noise[2];

};


thread_local SwapAverage m_cmave;
thread_local std::unique_ptr< edm::DetSetVector<SiStripRawDigi> > cm_prev; // previous ev

#include<iostream>
namespace {
  struct Stat {
    Stat() { for (auto & x :tots) x=0;for (auto & x :zeros) x=0;for (auto & x :hips) x=0; for (auto & x :over) x=0;  }
    std::atomic<long long> tots[4];
    std::atomic<long long> zeros[4];
    std::atomic<long long> hips[4];
    std::atomic<long long> over[4];

    ~Stat() {
      std::cout << "Zeros Hips/Ev ";
      for (int i=0;i<4;++i) std::cout << zeros[i]/double(tots[i])<<'/';std::cout << ' ';
      for (int i=0;i<4;++i) std::cout << hips[i]/double(tots[i])<<'/';std::cout << ' ';
      for (int i=0;i<4;++i) std::cout << over[i]/double(tots[i])<<'/';
      std::cout<<std::endl;
    }

  };
  Stat stat;
}


// #define RANDOMCM


#ifdef RANDOMCM
#include<random>
namespace {

  struct RandomCM {

    static constexpr float frac = 1000./50000.;
    bool operator()() { return rgen(eng) < frac; }
    std::mt19937 eng;
    std::uniform_real_distribution<float> rgen = std::uniform_real_distribution<float>(0.,1.);

  };

  thread_local RandomCM randomCM;

}
#endif


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
  
    // Create auto_ptr's of digi products
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > sm_dsv(sm);
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > vr_dsv(vr);
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > pr_dsv(pr);
    std::auto_ptr< edm::DetSetVector<SiStripDigi> > zs_dsv(zs);
    std::auto_ptr< DetIdCollection > det_ids(ids);
    std::auto_ptr< edm::DetSetVector<SiStripRawDigi> > cm_dsv(cm);


#ifdef RANDOMCM
    for (auto & ds : (*cm_dsv))
      for (auto & cm : ds) cm = SiStripRawDigi(randomCM() ? 0 : 128);
#endif

    // mind false sharing (and mfence storms...)
    int tots[4]={0}, zeros[4]={0}, hips[4]={0}, over[4]={0};
    for (auto & ds : (*cm_dsv)) {
      auto id = DetId(ds.detId()).subdetId()-3;
      tots[id]+=ds.size();
      for (auto & cm : ds) {
	if (cm.adc()<1) ++zeros[id];
	if (cm.adc()<40)++hips[id];
        if (cm.adc()>128+40)++over[id];
      }
    }
    for (int i=0;i<4;++i) {
      stat.tots[i]+=tots[i];
      stat.zeros[i]+=zeros[i];
      stat.hips[i]+=hips[i];
      stat.over[i]+=over[i];
    }

      
     /*
     unsigned int nn=0;
     decltype((*cm_dsv).begin()) prev;
     if (cm_prev) prev = (*cm_prev).begin();
     for (auto const & ds : (*cm_dsv)) {
        nn+=ds.size();
        if (ds.empty()) std::cout << "cm empty " << ds.detId() << std::endl;
        if (cm_prev && ds.size()!= (*prev).size() ) std::cout << "cm differs " << ds.detId() << ' ' << ds.size() <<"!=" <<(*prev).size() << std::endl;
        ++prev;
     }
     if (m_cmave.first) m_cmave.reset(6*cm_dsv->size());
     std::cout << " cms0 " << nn << std::endl;
    
      std::cout << " cm " << cm_dsv->size() << std::endl;
      unsigned int nc=0;
      for (auto const & ds : (*cm_dsv)) {
        int k=0;
        for (auto const & cm : ds) { 
          m_cmave.toFill()[nc+k]+=cm.adc(); 
          auto d = std::abs(cm.adc()-m_cmave.ave()[nc+k]);
          m_cmave.noiseFill()[nc+k]+=d*d;++k;}
        nc+=6;
      }
      std::cout << " cms " << nc << std::endl;
      assert(nc==m_cmave.toFill().size());


     if (m_cmave.init) {
      unsigned int nc=0;
      int nHip=0; int nZero=0;
      for (auto const & ds : (*cm_dsv)) {
        int k=0;
        for (auto const & cm : ds) {
          if ( cm.adc() ==0 && m_cmave.ave()[nc+k]!=0 ) ++nZero; 
          if ( cm.adc() < m_cmave.ave()[nc+k]-5*m_cmave.noise()[nc+k]){
           std::cout << "HIP " << ds.detId() << ' ' << k 
               << ": " << cm.adc() << ' ' <<  m_cmave.ave()[nc+k] << ' ' << m_cmave.noise()[nc+k]
               << std::endl;
           ++nHip;
          }

        ++k;}
        nc+=6;
      }
      std::cout << "nHIP " << nHip << ' ' << nZero << std::endl;
     }

      m_cmave.finalize();
      cm_prev = std::make_unique<edm::DetSetVector<SiStripRawDigi>>(*cm);

      */

    /*
    std::cout << "bad det " << det_ids->size() << std::endl;
    for (auto const & ds : (*cm_dsv)) 
      for (auto const & cm : ds)
        if ( cm.adc() < 40 ) { det_ids->push_back(ds.detId()); break;}
    std::cout << "bad det after cm " << det_ids->size() << std::endl;
    */


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

