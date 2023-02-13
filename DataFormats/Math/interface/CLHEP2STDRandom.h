#ifndef DataFormatsMathCLHEP2STDRandom_H
#define DataFormatsMathCLHEP2STDRandom_H

#include <cstdint>
#include <limits>
#include <cstring>
#include "CLHEP/Random/RandomEngine.h"

// a wrapper to make a HepRandomEngine  std conformant
class CLHEP2STDRandom {
public:
  using result_type = uint64_t;

  CLHEP2STDRandom(CLHEP::HepRandomEngine& engine) : m_engine(engine) {}

  static constexpr result_type u53 = std::numeric_limits<result_type>::max() >> (64 - 53);

  result_type operator()() {
    double d = m_engine;
    d += 1.;
    uint64_t u;
    std::memcpy(&u, &d, sizeof(uint64_t));
    return u & u53;
  }

  static constexpr result_type min() { return 0; }
  static constexpr result_type max() { return u53; }

private:
  CLHEP::HepRandomEngine& m_engine;
};
#endif
