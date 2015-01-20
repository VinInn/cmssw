#include "VITest/ExprEval/include/ExprEval.h"

#include "VITest/ExprEval/include/MyExpr.h"
#include "VITest/ExprEval/include/vcut.h"


#include <iostream>

int main() {


  MyExpr::Coll c;
  for (int i=5; i<15; ++i) { c.emplace_back(new Cand(i,1,1)); }
  MyExpr::Res r;

  std::string expr = "void eval(Coll const & c, Res & r) override{ r.resize(c.size()); std::transform(c.begin(),c.end(),r.begin(), [](Coll::value_type const & c){ return (*c).pt()>10;}); }";

  ExprEval parser("VITest/ExprEval", "MyExpr",expr.c_str());

  auto func = parser.expr<MyExpr>();

  func->eval(c,r);

  std::cout << r.size()  << ' '  <<  std::count(r.begin(),r.end(),true) << std::endl;


  std::string cut = "bool eval(int i, int j) override { return i<10&& j<5; }";

  ExprEval parser2("VITest/ExprEval","vcut",cut.c_str());

  auto mcut = parser2.expr<vcut>();

  std::cout << mcut->eval(2,7) << ' ' << mcut->eval(3, 4) << std::endl;

  return 0;

}
