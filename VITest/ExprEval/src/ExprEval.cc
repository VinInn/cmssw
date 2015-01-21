#include "VITest/ExprEval/include/ExprEval.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/GetEnvironmentVariable.h"
#include "FWCore/Utilities/interface/Exception.h"


#include "popenCPP.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <dlfcn.h>



namespace {
  std::string generateName() {
    auto n1 = execSysCommand("uuidgen | sed 's/-//g'");
    n1.pop_back();
    return n1;
  }

  /*
  std::string cxxflags() {
    std::string ret = 
      "-DGNU_GCC -D_GNU_SOURCE -DPROJECT_NAME='\"CMSSW\"' -DBOOST_SPIRIT_THREADSAFE -DPHOENIX_THREADSAFE -O2 -pthread -pipe -Werror=main -Werror=pointer-arith -Werror=overlength-strings -Wno-vla -Werror=overflow -Wstrict-overflow -std=c++11 -msse3 -ftree-vectorize -Wno-strict-overflow -Werror=array-bounds -Werror=format-contains-nul -Werror=type-limits -fvisibility-inlines-hidden -fno-math-errno --param vect-max-version-for-alias-checks=50 -fipa-pta -Wa,--compress-debug-sections -felide-constructors -fmessage-length=0 -ftemplate-depth-300 -Wall -Wno-non-template-friend -Wno-long-long -Wreturn-type -Wunused -Wparentheses -Wno-deprecated -Werror=return-type -Werror=missing-braces -Werror=unused-value -Werror=address -Werror=format -Werror=sign-compare -Werror=write-strings -Werror=delete-non-virtual-dtor -Werror=maybe-uninitialized -Werror=strict-aliasing -Werror=narrowing -Werror=uninitialized -Werror=unused-but-set-variable -Werror=reorder -Werror=unused-variable -Werror=conversion-null -Werror=return-local-addr -Werror=switch -fdiagnostics-show-option -Wno-unused-local-typedefs -Wno-attributes -Wno-psabi -DBOOST_DISABLE_ASSERTS -fPIC ";
    ret += "-DCMSSW_GIT_HASH='"+ edm::getReleaseVersion() + "' -DPROJECT_VERSION='" + edm::getReleaseVersion() + "' ";
    return ret;
  }
  */
  
}

ExprEval::ExprEval(const char * pkg, const char * iname, const char * iexpr) :
  m_name("VI_"+generateName())
{

  
  std::string factory = "factory" + m_name;

  std::string pch = pkg; pch += "/src/precompile.h";


  std::string quote("\"");
  std::string source = std::string("#include ")+quote+ pch +quote+"\n";
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
  auto relDir = edm::getEnvironmentVariable("CMSSW_RELEASE_BASE");

  std::string incDir = "/include/" + arch + "/";
  std::string cxxf;
  {
    std::string file = baseDir + incDir + pch + ".cxxflags";
    std::ifstream ss(file.c_str());
    std::cout << file << std::endl;
    if (ss) {
      std::getline(ss,cxxf);
      incDir = baseDir + incDir;
    } else {
       std::string file = relDir + incDir + pch + ".cxxflags";
       std::ifstream ss(file.c_str());
       if (!ss) throw  cms::Exception("ExprEval", file + " file not found!");
       std::getline(ss,cxxf);
       incDir = relDir + incDir;
    }


    // auto ss = popenCPP(std::string("sed 's/-I[^ ]*//g' ") + baseDir + "/include/" + arch + "/" + pkg +"/src/precompile.h.cxxflags");
    { std::regex rq("-I[^ ]+"); cxxf = std::regex_replace(cxxf,rq,std::string("")); }
    { std::regex rq("=\""); cxxf = std::regex_replace(cxxf,rq,std::string("='\"")); }
    { std::regex rq("\" "); cxxf = std::regex_replace(cxxf,rq,std::string("\"' ")); }
    std::cout << '|' << cxxf << "|\n" << std::endl;

  }

  std::string cpp = "c++ -H -Wall -shared -Winvalid-pch "; cpp+=cxxf;
  cpp += " -I" + incDir; 
  cpp += " -o " + ofile + ' ' + sfile+" 2>&1\n";

  std::cout << cpp << std::endl;

 
  auto ss = execSysCommand(cpp);
  std::cout << ss << std::endl;

  void * dl = dlopen(ofile.c_str(),RTLD_LAZY);
  if (!dl) {
     throw  cms::Exception("ExprEval",  cpp + ss + "dlerror " + dlerror());
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
