#include "VIPlots/Base/interface/Histos.h"

#include<iostream>


int main() {

  enum Hist { one, two };

  
  viplots::Histos histos1, histos2;

  auto book = [&](viplots::Histos & histos) {
    histos.put(one, new TH1F("one","one",10.,0.,10.));
    histos.put(two, new TH1F("two","two",10.,0.,10.));
  };

  book(histos1); book(histos2);

  histos1[one].Fill(2.5);
  histos2[one].Fill(2.5);
  histos2[one].Fill(4.5);
  histos1[one].Fill(0.5);

  {

    viplots::Histos histos("htest.root");
    book(histos);
    histos.add(histos1);
    histos.add(histos2);
  
    auto const & h = histos[one];

    for (int i=0; i<10; ++i) std::cout << h.GetBinContent(i) <<',';
    std::cout << std::endl;
  }

  {
    auto f =  TFile::Open("htest.root");
    auto h =  (TH1F const *)(f->Get("one"));
    for (int i=0; i<10; ++i) std::cout << h->GetBinContent(i) <<',';
    std::cout << std::endl;

  }

  return 0;

}
