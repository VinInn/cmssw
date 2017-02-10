#ifndef edmfakearray_h
#define edmfakearray_h
// FIXME: REMOVEME. waiting for ROOT to swallow std::array in all possible sauces
template<typename T, int N> class fakearray { T v[N];};
#endif
