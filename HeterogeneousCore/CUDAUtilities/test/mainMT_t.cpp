#include "mainMT.h"
#include <sstream>
#include <mutex>


std::mutex lock;

void theRealMain(int tid) {

   std::ostringstream sout;

   sout << "running " << tid;
   
   {
     std::lock_guard guard(lock);
     std::cout << sout.str() << std::endl;
   }

}
