#ifndef DataFormatsMathfastRandomPoisson_H
#define DataFormatsMathfastRandomPoisoon_H

#include "DataFormats/Math/interface/fastRandomTwoSingle.h"
#include "CLHEP/Random/RandomEngine.h"

#include <tuple>
#include <cmath>

namespace fastRandom {
  class Poisson {
  public:
    explicit Poisson(float av) : m_engine(nullptr), m_expAvm(std::exp(-av)) {}
    Poisson(CLHEP::HepRandomEngine& engine, float av) : m_engine(&engine), m_expAvm(std::exp(-av)) {}

    void reset(float av) { m_expAvm = std::exp(-av); }

    void reset(CLHEP::HepRandomEngine& engine) {
      m_ready = false;
      m_engine = &engine;
    }

    int operator()() {
      int em = -1;
      float t = 1.0f;
      do {
        em += 1;
        t *= nextRN();
      } while (t > m_expAvm);
      return em;
    }

    int operator()(CLHEP::HepRandomEngine& engine) {
      // for safety we assume it is a new engine
      reset(engine);
      return (*this)();
    }

  private:
    float nextRN() {
      if (m_ready) {
        m_ready = false;
        return m_rn.second;
      }
      m_rn = twoSingle(*m_engine);
      m_ready = true;
      return m_rn.first;
    }

    CLHEP::HepRandomEngine* m_engine;
    float m_expAvm;
    std::pair<float, float> m_rn;
    bool m_ready = false;
  };

}  // namespace fastRandom
#endif
