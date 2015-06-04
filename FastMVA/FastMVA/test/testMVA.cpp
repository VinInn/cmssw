#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "GBRForestFast.h"

#include "TFile.h"
#include<memory>
#include<algorithm>

#include<random>

#include<iostream>

#include <x86intrin.h>

unsigned int taux=0;
inline volatile unsigned long long rdtsc() {
 return __rdtscp(&taux);
}


template<typename A, typename B>
void copyIn(A const & a, B & b) {
  b.resize(a.size());
  std::copy(a.begin(),a.end(),b.begin());
}

void copy(GBRForest const & f, GBRForestFast & ff) {
  ff.Trees().resize(f.Trees().size());
  int kk=0;
  for (auto const & t : f.Trees() ) {
    auto & tf = ff.Trees()[kk++];
    copyIn(t.Responses(),tf.Responses());
    copyIn(t.CutIndices(),tf.CutIndices());
    copyIn(t.CutVals(),tf.CutVals());
    copyIn(t.LeftIndices(),tf.LeftIndices());
    copyIn(t.RightIndices(),tf.RightIndices());
  }

}


template<typename V>
long long vsize(V const & v) { return v.size()*sizeof(v[0]);}


long long t1=0;
long long t2=0;


template<typename FF>
float eval(FF const & forest, float *  x, long long & t1) {

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
    auto ret = forest.GetClassifier(gbrVals_);
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

   float x[11];
   t1=0;
   t2=0;
   long long ntot=0;
   double sum1=0;
   double sum2=0;
   for (int kk=0; kk<1000; ++kk) {
     for (auto & y : x) y= rgen(eng);
     for (int i=0; i<100; ++i) {
       auto k = x[i%8];
       x[i%8] = x[i%10]; x[i%10]=k;
       auto val1 = eval(*forest, x,t1);
       sum1 +=val1;
       auto val2 = eval(ff, x,t2);
       sum2 +=val2;

       ++ntot;
   }}
   std::cout << sum1/ntot << ' ' << double(t1)/ntot << std::endl;
   std::cout << sum2/ntot << ' ' << double(t2)/ntot << std::endl;
}


int main(int argc, char ** argv) {

  if (argc < 2) return 0;


  for (int i=1; i!=argc; ++i)
     go(argv[i]);


   return 0;


}
