// c++ -pthread -fPIC -O3 -std=c++11 -Wall test_tfmydnn.cpp -I/data/vin/tensorflow/ test_graph_tfmydnn_tfcompile_function.o /data/vin/tensorflow/bazel-bin/tensorflow/compiler/tf2xla/libxla_compiled_cpu_function.so /data/vin/tensorflow/bazel-bin/tensorflow/compiler/aot/libruntime.so

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseDim.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusEllipseSigma.h"


#include <array>
#include <algorithm>

#include <thread>
#include <functional>
#include<vector>

#include <iostream>
/*
usecols=(n.index('isBarrel'), n.index('layer'), n.index('x'),n.index('y'),
                                n.index('dx'), n.index('dy'), n.index('l2'),
                                n.index('sx'), n.index('sy'),
)
*/


void  go4() {

  float tot=0;
  int N=10000000/100;
  std::array<ClusEllipseDim,100> dnnD;
  std::array<ClusEllipseSigma,100> dnnS;
  std::cout <<"running 2*" << dnnD.size() << " dnns" <<std::endl;
 
  for (int i=0; i<N; ++i) {
  for (int j=0; j<100; ++j) {
  dnnS[j].arg0_data()[0] = dnnD[j].arg0_data()[0] = (i%2);
  dnnS[j].arg0_data()[1] = dnnD[j].arg0_data()[1] = (i%2) ? (j%3) : (j%4);
  dnnS[j].arg0_data()[2] = dnnD[j].arg0_data()[2] = float(i)/N;
  dnnS[j].arg0_data()[3] = dnnD[j].arg0_data()[3] = float(j)/100;
  dnnS[j].arg0_data()[4] = dnnD[j].arg0_data()[4] = 4*((j%3) ? (i%5) : -(j%10));
  dnnS[j].arg0_data()[5] = dnnD[j].arg0_data()[5] = (j%10);
  dnnS[j].arg0_data()[6] = dnnD[j].arg0_data()[6] = 0.5;
  dnnS[j].arg0_data()[7] = dnnD[j].arg0_data()[7] = (i%5);     
  dnnS[j].arg0_data()[8] = dnnD[j].arg0_data()[8] = 4*(j%10); 


  dnnD[j].Run();
  dnnS[j].Run();
  tot+= dnnD[j].result0_data()[0];
  }
  }
  std::cout << tot << ' ' << dnnD[0].result0_data()[0] << ' ' << dnnD[0].result0_data()[1] 
                   << ' ' << dnnS[0].result0_data()[0] << ' ' << dnnS[0].result0_data()[1] 
            << std::endl;

};



int main() {
  typedef std::thread Thread;
  typedef std::vector<std::thread> ThreadGroup;


  
  const int NUMTHREADS=8;

  ThreadGroup threads;
  threads.reserve(NUMTHREADS);
  for (int i=0; i<NUMTHREADS; ++i) {
    threads.push_back(Thread(go4));
  }
  
  std::for_each(threads.begin(),threads.end(), 
		std::bind(&Thread::join,std::placeholders::_1));

  

  return 0;
}
