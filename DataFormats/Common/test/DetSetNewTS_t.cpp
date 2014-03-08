#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#define private public
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetAlgorithm.h"
#include "DataFormats/Common/interface/DetSet2RangeMap.h"
#undef private

#include "FWCore/Utilities/interface/EDMException.h"

#include<vector>
#include<algorithm>

#include<omp.h>
#include <mutex>
typedef std::mutex Mutex;
// typedef std::lock_guard<std::mutex> Lock;
typedef std::unique_lock<std::mutex> Lock;

namespace global {
  // control cout....
  Mutex coutLock;
}

#include <iostream>
#include <atomic>
#include <thread>


template<typename T>
inline void spinlock(std::atomic<T> const & lock, T val) {
  while (lock.load(std::memory_order_acquire)!=val){}
}



template<typename T>
inline void spinlockSleep(std::atomic<T> const & lock, T val) {
  while (lock.load(std::memory_order_acquire)!=val){nanosleep(0,0);}
}

// syncronize all threads in a parallel section (for testing purposes)
void sync(std::atomic<int> & all) {
  auto sum = omp_get_num_threads(); sum = sum*(sum+1)/2;
  all.fetch_add(omp_get_thread_num()+1,std::memory_order_acq_rel);
  spinlock(all,sum);

}



struct B{
  virtual ~B(){}
  virtual B * clone() const=0;
};

struct T : public B {
  T(int iv=0) : v(iv){}
  int v;
  bool operator==(T t) const { return v==t.v;}

  virtual T * clone() const { return new T(*this); }
};

bool operator==(T const& t, B const& b) {
  T const * p = dynamic_cast<T const *>(&b);
  return p && p->v==t.v;
}

bool operator==(B const& b, T const& t) {
  return t==b;
}

typedef edmNew::DetSetVector<T> DSTV;
typedef edmNew::DetSet<T> DST;
typedef edmNew::det_id_type det_id_type;
typedef DSTV::FastFiller FF;
typedef DSTV::TSFastFiller TSFF;


class TestDetSet: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestDetSet);
  CPPUNIT_TEST(infrastructure);
  CPPUNIT_TEST(fillSeq);
  CPPUNIT_TEST(fillPar);

  CPPUNIT_TEST_SUITE_END();

public:
  TestDetSet();
  ~TestDetSet() {}
  void setUp() {}
  void tearDown() {}

  void infrastructure();
  void fillSeq();
  void fillPar();

public:
  std::vector<DSTV::data_type> sv;

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDetSet);

TestDetSet::TestDetSet() : sv(10){
  DSTV::data_type v[10] = {0,1,2,3,4,5,6,7,8,9};
  std::copy(v,v+10,sv.begin());
}

void read(DSTV const & detsets, bool all=false) {
  int i=0;
  for (auto di = detsets.begin(false); di!=detsets.end(false); ++di) {
    auto ds = *di;
    // if (detsets.onDemand()) CPPUNIT_ASSERT(!detsets.rcu().unique()); can be unique if ROU in action...
    auto id = ds.id();
    std::cout << id <<' ';
    // if (all) CPPUNIT_ASSERT(int(id)==20+i);
    if (ds.isValid())
    {
      CPPUNIT_ASSERT(ds[0]==100*(id-20)+3);
      CPPUNIT_ASSERT(ds[1]==-(100*(id-20)+3));
    }    
    ++i;
  }
  std::cout << std::endl;
}


void TestDetSet::infrastructure() {
  std::cout << std::endl;
  for (int i=0; i<10; i++) {
    int a=0;
    std::atomic<int> b(0);
    std::atomic<int> lock(0);
    std::atomic<int> nt(0);
#pragma omp parallel 
    {
      nt = omp_get_num_threads();
      sync(lock);
      a++;
      b.fetch_add(1,std::memory_order_acq_rel);;
    }
    
    if (i==5) std::cout << lock << " " << a << ' ' << b << std::endl;
    CPPUNIT_ASSERT(b==nt);
    a=0; b=0;
    
#pragma omp parallel 
    {
      a++;
      b.fetch_add(1,std::memory_order_acq_rel);
    }
    if (i==5) std::cout << lock << " " << a << ' ' << b << std::endl;
  }




}

void TestDetSet::fillSeq() {
  std::cout << std::endl;

  DSTV detsets(2);
  // unsigned int ntot=0;
  
  std::atomic<int> lock(0);
  std::atomic<int> idet(0);
  std::atomic<int> trial(0);
  int maxDet=100;
#pragma omp parallel 
  {
    sync(lock);
    while(true) {
      int ldet = idet.load(std::memory_order_acquire);
      if (!(ldet<maxDet)) break;
      while(!idet.compare_exchange_weak(ldet,ldet+1,std::memory_order_acq_rel));
      if (ldet>=maxDet) break;
      unsigned int id=20+ldet;
      bool done=false;
      while(!done) {
	try {
	  {
	    FF ff(detsets, id); // serialize
	    ff.push_back(100*ldet+3);
	    CPPUNIT_ASSERT(detsets.m_data.back().v==(100*ldet+3));
	    ff.push_back(-(100*ldet+3));
	    CPPUNIT_ASSERT(detsets.m_data.back().v==-(100*ldet+3));
	  }
	  // read(detsets);  // cannot read in parallel while filling in this case
	  done=true;
	} catch (edm::Exception const&) {
	  trial.fetch_add(1,std::memory_order_acq_rel);
	  //read(detsets);
	}
      }
    }
    // read(detsets);
  }

  std::cout << idet << ' ' << detsets.size() << std::endl;
  read(detsets,true);
  CPPUNIT_ASSERT(int(detsets.size())==maxDet);
  std::cout << trial << std::endl;
}


  struct Getter final : public DSTV::Getter {
    Getter(TestDetSet * itest):ntot(0), test(*itest){}

    void fill(TSFF& ff) override {
      int n=ff.id()-20;
      CPPUNIT_ASSERT(n>=0);
      CPPUNIT_ASSERT(ff.size()==0);
      ff.push_back((100*n+3));
      CPPUNIT_ASSERT(ff.size()==1);
      CPPUNIT_ASSERT(ff[0]==100*n+3);
      ff.push_back(-(100*n+3));
      CPPUNIT_ASSERT(ff.size()==2);
      CPPUNIT_ASSERT(ff[1]==-(100*n+3));
      ntot.fetch_add(1,std::memory_order_acq_rel);
    }

    std::atomic<unsigned int> ntot;
    TestDetSet & test;
  };

void TestDetSet::fillPar() {
  std::cout << std::endl;
  boost::shared_ptr<Getter> pg(new Getter(this));
  Getter & g = *pg;
  int maxDet=100;
  std::vector<unsigned int> v(maxDet); int k=20;for (auto &i:v) i=k++;
  DSTV detsets(pg,v,2);
  CPPUNIT_ASSERT(g.ntot==0);
  CPPUNIT_ASSERT(detsets.onDemand());
  CPPUNIT_ASSERT(detsets.rcu().unique());
  CPPUNIT_ASSERT(maxDet==int(detsets.size()));

  
  std::atomic<int> lock(0);
  std::atomic<int> idet(0);
  std::atomic<int> trial(0);

  std::atomic<int> count(0);
  

  DST df31 = detsets[31];

#pragma omp parallel 
  {
    sync(lock);
    if (omp_get_thread_num()%2==0) {
      DST df = detsets[25]; // everybody!
      CPPUNIT_ASSERT(df.id()==25);
      CPPUNIT_ASSERT(df.size()==2);
      CPPUNIT_ASSERT(df[0]==100*(25-20)+3);
      CPPUNIT_ASSERT(df[1]==-(100*(25-20)+3));
      if(!(*df.m_rcu).empty()) CPPUNIT_ASSERT(&(*df.m_rcu).front() == df.m_data);

      count = std::max(int(detsets.rcu().use_count()),int(count.load(std::memory_order_acquire))); 
    }
    while(true) {
      if (omp_get_thread_num()==0) read(detsets);
      int ldet = idet.load(std::memory_order_acquire);
      if (!(ldet<maxDet)) break;
      while(!idet.compare_exchange_weak(ldet,ldet+1,std::memory_order_acq_rel));
      if (ldet>=maxDet) break;
      unsigned int id=20+ldet;
      {
	DST df = *detsets.find(id);
	CPPUNIT_ASSERT(df.id()==id);
	CPPUNIT_ASSERT(df.size()==2);
	CPPUNIT_ASSERT(df[0]==100*(id-20)+3);
	CPPUNIT_ASSERT(df[1]==-(100*(id-20)+3));
	if(!(*df.m_rcu).empty()) CPPUNIT_ASSERT(&(*df.m_rcu).front() == df.m_data);

      }
      if (omp_get_thread_num()==1) read(detsets);
    }
  }

  CPPUNIT_ASSERT(df31.id()==31);
  CPPUNIT_ASSERT(df31.size()==2);
  CPPUNIT_ASSERT(df31[0]==100*(31-20)+3);
  CPPUNIT_ASSERT(df31[1]==-(100*(31-20)+3));
  if(!(*df31.m_rcu).empty()) { 
    std::cout << "RCU in action" << std::endl;
    CPPUNIT_ASSERT(&(*df31.m_rcu).front() == df31.m_data);
  }

  CPPUNIT_ASSERT(detsets.rcu().unique()); // not necessarely true due to df31...
  std::cout << "summary " << idet << ' ' << detsets.size() << ' ' << g.ntot  << ' ' << count << std::endl;
  read(detsets,true);
  CPPUNIT_ASSERT(int(g.ntot)==maxDet);
  CPPUNIT_ASSERT(int(detsets.size())==maxDet);
  CPPUNIT_ASSERT(detsets.rcu().unique());
}
