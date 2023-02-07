#include "DataFormats/Math/interface/fastRandomTwoSingle.h"

#include "CLHEP/Random/MixMaxRng.h"

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

  return 0;
}
