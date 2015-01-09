#include "VITest/ExprEval/include/ExprEval.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/GetEnvironmentVariable.h"

#include "popenCPP.h"
#include <fstream>
#include <iostream>
#include <dlfcn.h>



namespace {
  std::string generateName() {
    auto s1 = popenCPP("uuidgen | sed 's/-//g'");
    char c; std::string n1;
    while (s1->get(c)) n1+=c;
    n1.pop_back();
    return n1;
  }


  std::string cxxflags() {
    std::string ret = 
      "-DGNU_GCC -D_GNU_SOURCE -DPROJECT_NAME='\"CMSSW\"' -DBOOST_SPIRIT_THREADSAFE -DPHOENIX_THREADSAFE -O2 -pthread -pipe -Werror=main -Werror=pointer-arith -Werror=overlength-strings -Wno-vla -Werror=overflow -Wstrict-overflow -std=c++11 -msse3 -ftree-vectorize -Wno-strict-overflow -Werror=array-bounds -Werror=format-contains-nul -Werror=type-limits -fvisibility-inlines-hidden -fno-math-errno --param vect-max-version-for-alias-checks=50 -fipa-pta -Wa,--compress-debug-sections -felide-constructors -fmessage-length=0 -ftemplate-depth-300 -Wall -Wno-non-template-friend -Wno-long-long -Wreturn-type -Wunused -Wparentheses -Wno-deprecated -Werror=return-type -Werror=missing-braces -Werror=unused-value -Werror=address -Werror=format -Werror=sign-compare -Werror=write-strings -Werror=delete-non-virtual-dtor -Werror=maybe-uninitialized -Werror=strict-aliasing -Werror=narrowing -Werror=uninitialized -Werror=unused-but-set-variable -Werror=reorder -Werror=unused-variable -Werror=conversion-null -Werror=return-local-addr -Werror=switch -fdiagnostics-show-option -Wno-unused-local-typedefs -Wno-attributes -Wno-psabi -DBOOST_DISABLE_ASSERTS -fPIC ";
    ret += "-DCMSSW_GIT_HASH='"+ edm::getReleaseVersion() + "' -DPROJECT_VERSION='" + edm::getReleaseVersion() + "' ";
    return ret;
  }

}

ExprEval::ExprEval(const char * pkg, const char * iname, const char * iexpr) :
  m_name("VI_"+generateName())
{

  
  std::string factory = "factory" + m_name;

  std::string quote("\"");
  std::string source = std::string("#include ")+quote+pkg +"/src/precompile.h"+quote+"\n";
  source+="struct "+m_name+" final : public "+iname + "{\n";
  source+=iexpr;
  source+="\n};\n";


  source += "extern " + quote+'C'+quote+' ' + std::string(iname) + "* "+factory+"() {\n";
  source += "static "+m_name+" local;\n";
  source += "return &local;\n}\n";


  std::cout << source << std::endl;

  std::string sfile = "/tmp/"+m_name+".cc";
  std::string ofile = "/tmp/"+m_name+".so";

  {
    std::ofstream tmp(sfile.c_str());
    tmp<<source << std::endl;
  }

  auto arch = edm::getEnvironmentVariable("SCRAM_ARCH");
  auto baseDir = edm::getEnvironmentVariable("CMSSW_BASE");

  std::string cpp = "c++ -H -Wall -shared -Winvalid-pch "; cpp+=cxxflags();
  cpp += "-I" + baseDir + "/include/" + arch; 
  cpp +=  " -o " + ofile + ' ' + sfile+" 2>&1\n";

  std::cout << cpp << std::endl;

  try{
    auto ss = popenCPP(cpp);
    char c;
    while (ss->get(c)) std::cout << c;
    std::cout << std::endl;
  }catch(...) { std::cout << "error in popen" << cpp << std::endl;}

  void * dl = dlopen(ofile.c_str(),RTLD_LAZY);
  if (!dl) {
    std::cout << dlerror() <<std::endl;
    return;
  }

  m_expr = dlsym(dl,factory.c_str());

}


ExprEval::~ExprEval(){
  std::string sfile = "/tmp/"+m_name+".cc";
  std::string ofile = "/tmp/"+m_name+".so";

  std::string rm="rm -f "; rm+=sfile+' '+ofile;

  system(rm.c_str());

}
