#include <thread>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include<cassert>

void theRealMain(int tid);



int main(int argc, char ** argv) {

   if (argc<2 || atoi(argv[1])<=0 ) {
     theRealMain(-1);
     return 0;
   }

  int nThreads = atoi(argv[1]);
  assert(nThreads>=0);

  std::atomic<int> tid(nThreads);
  std::atomic<bool> wait(true);
  auto wrapper = [&]() {
    std::cout <<'.';
    int me = --tid;
    if (0==me) { wait=false; std::cout << std::endl;}
    while(wait){} 
    theRealMain(me);
  };

  std::vector<std::threads> pool(nThreads,wrapper);

  for ( auto & t : pool) t.join();

  return 0;

}
