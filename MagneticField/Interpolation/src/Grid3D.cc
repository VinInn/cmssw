#include "Grid3D.h"
#include <iostream>
#include<atomic>

namespace {
  struct Stat {
    Stat() : ng(0), stot(0){}
    ~Stat() { 
       std::cout << "MF Grid stats: ngrids, tot size, zip size " << ng << ' ' << stot << ' ' << ztot << std::endl;
     }
    std::atomic<long long> ng, stot, ztot;
  };

 Stat stat;

}


Grid3D::Grid3D( const Grid1D& ga, const Grid1D& gb, const Grid1D& gc,
          std::vector<BVector>& data) :
    grida_(ga), gridb_(gb), gridc_(gc),
    zipX(gridc_.nodes(),gridb_.nodes(),grida_.nodes(),6),
    zipY(gridc_.nodes(),gridb_.nodes(),grida_.nodes(),6),
    zipZ(gridc_.nodes(),gridb_.nodes(),grida_.nodes(),6)
    {
     stride1_ = gridb_.nodes() * gridc_.nodes();
     stride2_ = gridc_.nodes();

//     std::cout << data.size() << '/' << zipX.size() << std::endl;
//     std::cout << grida_.nodes()*gridb_.nodes() * gridc_.nodes()<< std::endl;

     std::vector<float> tmp(data.size());
     for (unsigned int i=0; i<data.size(); ++i) tmp[i] = data[i][0];
     zipX.set(&tmp.front());
     for (unsigned int i=0; i<data.size(); ++i) tmp[i] =	data[i][1];
     zipY.set(&tmp.front());
     for (unsigned int i=0; i<data.size(); ++i) tmp[i] =	data[i][2];
     zipZ.set(&tmp.front());

     data_.swap(data);

     ++stat.ng; stat.stot+=sizeof(BVector)*data_.size();  stat.ztot+= zipX.compressed_size()+zipY.compressed_size()+zipZ.compressed_size();
}


/*
Grid3D::Grid3D( const Grid1D& ga, const Grid1D& gb, const Grid1D& gc,
		std::vector<ValueType> const & data) : 
  grida_(ga), gridb_(gb), gridc_(gc) {
  data_.reserve(data.size());
  //FIXME use a std algo
  for (size_t i=0; i<=data.size(); ++i)
    data_.push_back(ValueType(data[i].x(),data[i].y(),data[i].z()));
  stride1_ = gridb_.nodes() * gridc_.nodes();
  stride2_ = gridc_.nodes();
}
*/

void Grid3D::dump() const
{
  for (int j=0; j<gridb().nodes(); ++j) {
    for (int k=0; k<gridc().nodes(); ++k) {
      for (int i=0; i<grida().nodes(); ++i) {
        std::cout << grida().node(i) << " " << gridb().node(j) << " " << gridc().node(k) << " " 
		  << operator()(i,j,k) << std::endl;
      }
    }
  }
}


