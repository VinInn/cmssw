#ifndef VIPlotsBaseHistos_H
#define VIPlotsBaseHistos_H

#include<vector>
#include<algorithm>
#include<memory>
#include<TF1.h>
#include<TH1F.h>
#include<TFile.h>


namespace viplots {


  class Histos {
  public:
    Histos(){}
    explicit Histos(const char * fname) : outFile(TFile::Open(fname,"RECREATE")){}
    ~Histos() { if (outFile) { outFile->Write(); outFile->Close();} else for (auto h : m_histos) delete h;}
    
    
    void put(size_t i, TH1 * h) {
      h->SetDirectory(outFile);
      m_histos.resize(std::max(m_histos.size(),i+1),nullptr);
      m_histos[i] = h;
    }

    void add(Histos const & hs) {
      int k=0;
      for (auto h : hs.m_histos)
	if (h) m_histos[k++]->Add(h);
    }
    

    TH1 & operator[](unsigned int i) { return *m_histos[i]; }
    
    template<typename H>
    H & get(unsigned i) { return *reinterpret_cast<H*>(m_histos[i]); }
    
    
    std::vector<TH1 * > m_histos;
    TFile * outFile=nullptr;


  };


}
#endif //  VIPlotsBaseHistos_H
