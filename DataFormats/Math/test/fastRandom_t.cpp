#include "DataFormats/Math/interface/fastRandomTwoSingle.h"
#include "DataFormats/Math/interface/fastRandomNormal.h"

#include "DataFormats/Math/interface/CLHEP2STDRandom.h"
#include "CLHEP/Random/MixMaxRng.h"

#include "CLHEP/Random/RandGaussQ.h"
#include <random>
// when C++20    will be    used
// #include <bits>

#include "benchmark.h"

#include <iostream>
#include <cassert>
namespace {
  union FInt {
    float f;
    uint32_t i;
  };
}  // namespace

void fillBits(int* v, float x) {
  FInt fi;
  fi.f = x;
  fi.f += 1.f;
  for (int i = 0; i < 23; ++i) {
    v[i] += fi.i & 1;
    fi.i = fi.i >> 1;
  }
}

int main() {
  CLHEP::MixMaxRng engine;

  {
    // test wrapper
    CLHEP2STDRandom wrapper(engine);
    std::poisson_distribution<> poisson(0.5);
    std::cout << "test wrapper " << poisson(wrapper) << " bits " << __builtin_popcountll(CLHEP2STDRandom::max()) << ' '
              << std::log2(CLHEP2STDRandom::max() + 1) << std::endl;
    // when C++20 will be used
    //          << std::popcount(CLHEP2STDRandom::max()) << ' ' << countl_zero(CLHEP2STDRandom::max()) <<    ' ' <<    ' ' << countr_one(CLHEP2STDRandom::max()) << std::endl;
  }

  using namespace fastRandom;

  int N = 2 * 1000 * 1000;
  int vx[23], vy[23];
  for (int i = 0; i < 23; ++i)
    vx[i] = vy[i] = 0;
  for (int i = 0; i < N; ++i) {
    auto [x, y] = twoSingle(engine);
    assert(x >= 0 && x < 1.f);
    assert(y >= 0 && y < 1.f);
    fillBits(vx, x);
    fillBits(vy, y);
  }
  auto t = 1000 * 1000;
  for (int i = 0; i < 23; ++i) {
    assert(vx[i] > 0);
    assert(vy[i] > 0);
    if (std::abs(vx[i] - t) > 3000)
      std::cout << i << ' ' << vx[i] << std::endl;
    if (std::abs(vy[i] - t) > 3000)
      std::cout << i << ' ' << vy[i] << std::endl;
  }

  double dv[128];
  float dummy[1];
  auto clhep = [&](float const*, float* r, int) {
    for (int k = 0; k < 256; ++k) {
      r[k] = CLHEP::RandGaussQ::shoot(&engine);
    }
  };

  auto gen = [&](float const*, float* r, int) {
    for (auto& d : dv)
      d = engine;
    for (int k = 0; k < 128; ++k) {
      auto [x, y] = normal(dv[k]);
      r[k] = x;
      r[k + 128] = y;
    }
  };

  for (int m = 0; m < 3; ++m) {
    benchmark::TimeIt bench;
    // fill in batch of 256
    float rv[256];
    double n = 0;
    double ave = 0;
    double sigma = 0;
    double q3 = 0, q4 = 0, q5 = 0, q6 = 0;
    for (int i = 0; i < 10 * N; ++i) {
      bench(gen, dummy, rv, 128);
      for (auto r : rv) {
        n++;
        ave += r;
        sigma += r * r;
        if (std::abs(r) > 3.f)
          q3 += 1;
        if (std::abs(r) > 4.f)
          q4 += 1;
        if (std::abs(r) > 5.f)
          q5 += 1;
        if (std::abs(r) > 6.f)
          q6 += 1;
      }
    }
    std::cout << "fast normal" << std::endl;
    std::cout << n << ' ' << q3 << ' ' << q4 << ' ' << q5 << ' ' << q6 << std::endl;
    std::cout << n << ' ' << ave / n << ' ' << (n * sigma - ave * ave) / (n * n) << ' ' << (q3 / n) / 0.002699796063
              << ' ' << (q4 / n) / 0.000063342484 << ' ' << (q5 / n) / 0.000000573303 << ' '
              << (q6 / n) / 0.000000001973 << std::endl;
    std::cout << "duration " << bench.lap() << std::endl;
  }

  for (int m = 0; m < 3; ++m) {
    benchmark::TimeIt bench;
    // fill in batch of 256
    float rv[256];
    double n = 0;
    double ave = 0;
    double sigma = 0;
    double q3 = 0, q4 = 0, q5 = 0, q6 = 0;
    for (int i = 0; i < 10 * N; ++i) {
      bench(clhep, dummy, rv, 256);
      for (auto r : rv) {
        n++;
        ave += r;
        sigma += r * r;
        if (std::abs(r) > 3.f)
          q3 += 1;
        if (std::abs(r) > 4.f)
          q4 += 1;
        if (std::abs(r) > 5.f)
          q5 += 1;
        if (std::abs(r) > 6.f)
          q6 += 1;
      }
    }
    std::cout << "clhep gaussQ" << std::endl;
    std::cout << n << ' ' << q3 << ' ' << q4 << ' ' << q5 << ' ' << q6 << std::endl;
    std::cout << n << ' ' << ave / n << ' ' << (n * sigma - ave * ave) / (n * n) << ' ' << (q3 / n) / 0.002699796063
              << ' ' << (q4 / n) / 0.000063342484 << ' ' << (q5 / n) / 0.000000573303 << ' '
              << (q6 / n) / 0.000000001973 << std::endl;
    std::cout << "duration " << bench.lap() << std::endl;
  }

  return 0;
}
