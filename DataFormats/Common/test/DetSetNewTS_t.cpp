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
  T(float iv=0) : v(iv){}
  float v;
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


class TestDetSet: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestDetSet);
  CPPUNIT_TEST(infrastructure);
  CPPUNIT_TEST(fill);

  CPPUNIT_TEST_SUITE_END();

public:
  TestDetSet();
  ~TestDetSet() {}
  void setUp() {}
  void tearDown() {}

  void infrastructure();
  void fill();

public:
  std::vector<DSTV::data_type> sv;

};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDetSet);

TestDetSet::TestDetSet() : sv(10){
  DSTV::data_type v[10] = {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.};
  std::copy(v,v+10,sv.begin());
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

void TestDetSet::fill() {
  std::cout << std::endl;

  DSTV detsets(2);
  // unsigned int ntot=0;
  
  std::atomic<int> lock(0);
  std::atomic<int> idet(0);
  std::atomic<int> trial(0);
  int maxDet=20;
#pragma omp parallel 
  {
    sync(lock);
    while(true) {
      int ldet = idet.fetch_add(1,std::memory_order_acq_rel);;
      if (ldet>maxDet) break;
      unsigned int id=20+ldet;
      try {
	FF ff(detsets, id);
	ff.push_back(ldet+3.14);
	CPPUNIT_ASSERT(detsets.m_data.back().v==ldet+3.14f);
	ff.push_back(-(ldet+3.14));
	CPPUNIT_ASSERT(detsets.m_data.back().v==-(ldet+3.14f));
      } catch (edm::Exception const&) {
	trial.fetch_add(1,std::memory_order_acq_rel);
      }

    }
  }
  std::cout << trial << std::endl;
}
