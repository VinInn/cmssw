#define VECTOR_TMVA
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "GBRForestFast.h"
using Vin = TMVA_in;

#include "TFile.h"
#include<memory>
#include<algorithm>

#include<random>

#include<iostream>
#include<cassert>

#include <x86intrin.h>

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}


template<typename A, typename B>
void copyIn(A const & a, B & b) {
  std::copy(a.begin(),a.end(),b.begin());
}

template<typename A, typename B>
void copyf16(A const & a, B & b) {
  std::transform(a.begin(),a.end(),b.begin(),[](float x) { short y; tof16(x,y); return y;});
}



void copy(GBRForest const & f, GBRForestFast & ff) {
  ff.Trees().resize(f.Trees().size());
  int kk=0;
  for (auto const & t : f.Trees() ) {
    auto & tf = ff.Trees()[kk++];
    copyf16(t.Responses(),tf.Responses());
    copyIn(t.CutIndices(),tf.CutIndices());
    copyf16(t.CutVals(),tf.CutVals());
    copyIn(t.LeftIndices(),tf.LeftIndices());
    copyIn(t.RightIndices(),tf.RightIndices());
    tf.setSize(t.CutIndices().size());
    tf.setRsize(t.Responses().size());
    assert(tf.size()<=tf.CutIndices().size());
    assert(tf.rsize()<=tf.Responses().size());

  }

}


template<typename V>
long long vsize(V const & v) { return v.size()*sizeof(v[0]);}


long long t1=0;
long long t2=0;
long long t3=0;


template<typename FF>
float eval(FF const & forest, Vin const & x, long long & t1) {

    Vin gbr;
    std::array<std::array<float,16>,8> gbrVals_;
    for (int i=0; i<8; ++i) {
    gbrVals_[i][0] = 0.2f + 20.f*x[i][0];
    gbrVals_[i][1] = x[i][1];
    gbrVals_[i][2] = int(3.f*x[i][2]);
    gbrVals_[i][3] = int(20.f*x[i][3]);
    gbrVals_[i][4] = 0.1f*x[i][4];
    gbrVals_[i][5] = 2.4f+4.8d*x[i][5];
    gbrVals_[i][6] = 8.f*x[i][6];
    gbrVals_[i][7] = 8.f*x[i][6];
    gbrVals_[i][8] = int(3.f*x[i][7]);
    gbrVals_[i][9] = int(7.f*x[i][3]);
    gbrVals_[i][10] = int(12.f*x[i][3]);
    gbrVals_[i][11] = int(20.f*x[i][3])-5;
    gbrVals_[i][12] = 0.01*x[i][8];
    gbrVals_[i][13] = 15.f*x[i][9];
    gbrVals_[i][14] = x[i][10];
    gbrVals_[i][15] = 0.01*x[i][8];;
    gbr[i] = &gbrVals_[i][0];
    }
    t3 -= rdtsc();
//    auto ret = forest.GetClassifier(gbrVals_);
    auto ret = forest.GetResponseV(gbr);
    float out=0;
    for (int i=0; i<8; ++i) out+=ret[i];
    t3 += rdtsc();
    return out;;
}

template<typename FF>
float eval(FF const & forest, float const *  x, long long & t1) {

    float gbrVals_[16];

    gbrVals_[0] = 0.2f + 20.f*x[0];
    gbrVals_[1] = x[1];
    gbrVals_[2] = int(3.f*x[2]);
    gbrVals_[3] = int(20.f*x[3]);
    gbrVals_[4] = 0.1f*x[4];
    gbrVals_[5] = 2.4f+4.8d*x[5];
    gbrVals_[6] = 8.f*x[6];
    gbrVals_[7] = 8.f*x[6];
    gbrVals_[8] = int(3.f*x[7]);
    gbrVals_[9] = int(7.f*x[3]);
    gbrVals_[10] = int(12.f*x[3]);
    gbrVals_[11] = int(20.f*x[3])-5;
    gbrVals_[12] = 0.01*x[8];
    gbrVals_[13] = 15.f*x[9];
    gbrVals_[14] = x[10];
    gbrVals_[15] = 0.01*x[8];;


    t1 -= rdtsc();
//    auto ret = forest.GetClassifier(gbrVals_);
    auto ret = forest.GetResponse(gbrVals_);
    t1 += rdtsc();
    return ret;

}


template<>
float eval(GBRForestFast const & forest, float const *  x, long long & t1) {

    float gbrVals_[16];
    gbrVals_[0] = 0.2f + 20.f*x[0];
    gbrVals_[1] = x[1];
    gbrVals_[2] = int(3.f*x[2]);
    gbrVals_[3] = int(20.f*x[3]);
    gbrVals_[4] = 0.1f*x[4];
    gbrVals_[5] = 2.4f+4.8d*x[5];
    gbrVals_[6] = 8.f*x[6];
    gbrVals_[7] = 8.f*x[6];
    gbrVals_[8] = int(3.f*x[7]);
    gbrVals_[9] = int(7.f*x[3]);
    gbrVals_[10] = int(12.f*x[3]);
    gbrVals_[11] = int(20.f*x[3])-5;
    gbrVals_[12] = 0.01*x[8];
    gbrVals_[13] = 15.f*x[9];
    gbrVals_[14] = x[10];
    gbrVals_[15] = 0.01*x[8];;


    t1 -= rdtsc();
    short xx[16];
    tof16(gbrVals_,xx,16);
    auto ret = forest.GetResponse(xx);
    t1 += rdtsc();
    return ret;
}



void go(const char * fname) {

     TFile gbrfile(fname);
     auto forest = (GBRForest*)gbrfile.Get("GBRForest");
     std::cout << "size " << forest->Trees().size() << std::endl;
     long long s=0; int mi=0;
     for (auto const & t : forest->Trees() ) {
        s += vsize(t.Responses())
           + vsize(t.CutIndices())
           + vsize(t.CutVals())
           + vsize(t.LeftIndices())
           + vsize(t.RightIndices());
        mi = std::max(mi,int(*std::max_element(t.CutIndices().begin(),t.CutIndices().end())));
        /*
        std::cout << "max ind " << int(*std::max_element(t.CutIndices().begin(),t.CutIndices().end())) << std::endl;
        std::cout << "tree sizes rs,ci,cv,li,ri" 
                  << ' ' << t.Responses().size()  
                  << ' ' << t.CutIndices().size()
                  << ' ' << t.CutVals().size()
                  << ' ' << t.LeftIndices().size()
                  << ' ' << t.RightIndices().size()
                  << std::endl;
        */
     }
     std::cout << "size, max ind " << s <<  ' ' << mi << std::endl; 

   GBRForestFast  ff;
   copy(*forest, ff);

   std::mt19937 eng;
   std::uniform_real_distribution<float> rgen(0.,1.);

   std::array<std::array<float,11>,8> x;
   std::array<float const *,8> xp;
   for (int i=0; i<8; ++i) xp[i] = &x[i][0];
   t1=0;
   t2=0;
   t3=0;
   long long ntot=0;
   double sum1=0;
   double sum2=0;
   double sum3=0;
   for (int kk=0; kk<1000; ++kk) {
     for (int i=0; i<8; ++i)  for (auto & y : x[i]) y= rgen(eng);
     for (int j=0; j<160; j+=8) {
     for (int i=0; i<8; ++i) {
       auto k = x[i][(j+i)%8];
       x[i][(j+i)%8] = x[i][(j+i)%10]; x[i][(j+i)%10]=k;
       auto val1 = eval(*forest, xp[i],t1);
       sum1 +=val1;
       auto val2 = eval(ff, xp[i],t2);
       sum2 +=val2;
      }
      auto val3 = eval(*forest, xp,t3);
      sum3 +=val3;

       ++ntot;
   }}
   std::cout << "old " << sum1/ntot << ' ' << double(t1)/ntot << std::endl;
   std::cout << "new " << sum2/ntot << ' ' << double(t2)/ntot << std::endl;
   std::cout << "vec " << sum3/ntot << ' ' << double(t3)/ntot << std::endl;
}


int main(int argc, char ** argv) {

  if (argc < 2) return 0;


  for (int i=1; i!=argc; ++i)
     go(argv[i]);


   return 0;


}
