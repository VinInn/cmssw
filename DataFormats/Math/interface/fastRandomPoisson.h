#ifndef DataFormatsMathfastRandomPoisson_H
#define DataFormatsMathfastRandomPoisoon_H

#include "DataFormats/Math/interface/fastRandomTwoSingle.h"
#include "CLHEP/Random/RandomEngine.h"

#include <tuple>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fastRandom {

  template <typename T>
  class PoissonHist {
  public:
    explicit PoissonHist(T mu) { reset(mu); }
    PoissonHist(CLHEP::HepRandomEngine& engine, T mu) : m_gen(engine) { reset(mu); }

    int operator()() { return findSlow(m_gen()); }

    void reset(T mu) {
      m_cumulative.clear();
      auto poiss = std::exp(-mu);
      m_cumulative.push_back(poiss);
      for (;;) {
        poiss *= mu / T(size());
        if (poiss <= std::numeric_limits<T>::epsilon())
          break;
        auto val = m_cumulative.back() + poiss;
        if (val >= T(1.))
          break;
        m_cumulative.push_back(val);
      }
      m_quantile[0] = findSlow(0.25);
      m_quantile[1] = findSlow(0.5);
      m_quantile[2] = findSlow(0.75);
      m_quantile[3] = findSlow(0.95);
      m_quantile[4] = findSlow(0.99);
    }

    int findSlow(T x) const {
      auto bin = std::lower_bound(m_cumulative.begin(), m_cumulative.end(), x);
      return bin - m_cumulative.begin();
    }

    int find(T x) const {
      int min = x < T(0.5) ? 0 : m_quantile[1];
      int max = x < T(0.5) ? m_quantile[1] + 1 : size();
      /*
                    ? (x < T(0.25) ? 0 : m_quantile[0])
                    : (x < T(0.75) ? m_quantile[1]
                                   : (x < T(0.95) ? m_quantile[2] : (x < T(0.99) ? m_quantile[3] : m_quantile[4])));
      int max = x < T(0.5) ? (x < T(0.25) ? m_quantile[0] : m_quantile[1])
                           : (x < T(0.75) ? m_quantile[2]
                                          : (x < T(0.95) ? m_quantile[3] : (x < T(0.99) ? m_quantile[4] : size() - 1)));
      max++;
      */
      //     assert(min==0 || x>=m_cumulative[min-1]);
      //     assert(max==size() || x<m_cumulative[max]);
      auto bin = std::lower_bound(m_cumulative.begin() + min, m_cumulative.begin() + max, x);
      return bin - m_cumulative.begin();
    }

    auto size() const { return m_cumulative.size(); }

  private:
    OneFloat m_gen;
    std::vector<T> m_cumulative;
    int m_quantile[5] = {0, 0, 0, 0, 0};
  };

  class PoissonKnuth {
  public:
    explicit PoissonKnuth(float av) : m_expAvm(std::exp(-av)) {}
    PoissonKnuth(CLHEP::HepRandomEngine& engine, float av) : m_gen(engine), m_expAvm(std::exp(-av)) {}

    void reset(float av) { m_expAvm = std::exp(-av); }

    void reset(CLHEP::HepRandomEngine& engine) { m_gen.reset(engine); }

    int operator()() {
      int em = -1;
      float t = 1.0f;
      do {
        em += 1;
        t *= m_gen();
      } while (t > m_expAvm);
      return em;
    }

    int operator()(CLHEP::HepRandomEngine& engine) {
      // for safety we assume it is a new engine
      reset(engine);
      return (*this)();
    }

  private:
    OneFloat m_gen;
    float m_expAvm;
  };

}  // namespace fastRandom
#endif
