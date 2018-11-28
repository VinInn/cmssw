#include "RecoPixelVertexing/PixelTrackFitting/interface/FitResult.h"


using Rfit::Vector5d;
using Rfit::Matrix5d;


Vector5d transf(Vector5d  p) {
  p(2) = 1/p(2);
  return p;
}

Matrix5d transfFast(Matrix5d const & cov) {

  return cov;

}

Matrix5d Jacobian(Vector5d const &  p) {

  Matrix5d J = Matrix5d::Identity();
  J(2,2) = -1/(p(2)*p(2));

  return J;
}

Matrix5d transf(Matrix5d const & cov, Matrix5d const& J) {

   return J*cov*J.transpose();

}  

Matrix5d loadCov(Vector5d const & e) {

  Matrix5d cov = Matrix5d::Zero();
  for (int i=0; i<5; ++i) cov(i,i) = e(i)*e(i);
  return cov;
}


#include<iostream>
int main() {

  //!<(phi,Tip,pt,cotan(theta)),Zip)
  Vector5d par0; par0 << 0.2,0.1,3.5,0.2,0.1;
  Vector5d del0; del0 << 0.01,0.01,0.035,0.01,0.01;

  Matrix5d J = Jacobian(par0);


  Vector5d par1 = transf(par0);
  Vector5d par2 = transf(par0+del0);
  Vector5d del1 = par2-par1; 

  Matrix5d cov0 = loadCov(del0);
  Matrix5d cov1 = transf(cov0,J);
  Matrix5d cov2 = transfFast(cov0);

  // don't ask: guess
  std::cout << "par0 " << par0.transpose() << std::endl;
  std::cout << "del0 " << del0.transpose() << std::endl;


  std::cout << "par1 " << par1.transpose() << std::endl;
  std::cout << "del1 " << del1.transpose() << std::endl;
  std::cout << "del1^2 " << del1*del1.transpose() << std::endl;
  std::cout << std::endl;
  std::cout << "J\n" << J << std::endl;
  
  std::cout << "cov0\n" << cov0 << std::endl;
  std::cout << "cov1\n" << cov1 << std::endl;
  std::cout << "cov2\n" << cov2 << std::endl;


  return 0;


}
