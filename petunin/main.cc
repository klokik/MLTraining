#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <random>
#include <utility>
#include <vector>


using real_t = float;

using Sample = std::vector<real_t>;
using Samples = std::vector<Sample>;
using PairFMap = std::map<std::pair<int, int>, real_t>;
using PairFFMap = std::map<std::pair<int, int>, std::pair<real_t, real_t>>;

#define MP std::make_pair

Samples getSamples(real_t _mean, real_t _dist, size_t _ssize, size_t _nsize) {
  Samples sets(_nsize);

  std::random_device rd;
  std::mt19937 rng(rd());

  std::normal_distribution<real_t> distrib(_mean, _dist);

  for (auto &sample : sets) {
    for (size_t j = 0; j < _ssize; ++j)
      sample.push_back(distrib(rng));

    std::sort(sample.begin(), sample.end());
  }

  return sets;
}

real_t getL(PairFFMap &_pij, PairFMap &_pijn, size_t _n) {
  real_t l {0};

  for (size_t i = 0; i < _n-1; ++i)
    for (size_t j = i + 1; j < _n; ++j)
      if (_pij[MP(i, j)].second > _pijn[{i, j}])
        l += 1.;

  return l;
}

PairFMap getHinN(Sample &_a, Sample &_b) {
  PairFMap hijn;
  auto n = _a.size();

  for (size_t i = 0; i < n - 1; ++i)
    for (size_t j = i + 1; j < n; ++j)
      hijn[MP(i, j)] = 1. * std::count_if(_b.begin(), _b.end(),
        [&_a, i, j](auto x) {return (_a[i] < x && _a[j] > x);});

  return hijn;
}

PairFFMap getPij(PairFMap &_hijn, real_t _m, real_t _g) {
  PairFFMap pij;

  for (auto &ijh : _hijn) {
    auto ijhv = ijh.second;
    auto h1 = (ijhv*_m + _g*_g/2 - _g*std::sqrt(ijhv*(1-ijhv)*_m + _g*_g/4)) / (_m + _g*_g);
    auto h2 = (ijhv*_m + _g*_g/2 + _g*std::sqrt(ijhv*(1-ijhv)*_m + _g*_g/4)) / (_m + _g*_g);

    pij[MP(ijh.first.first, ijh.first.second)] = MP(h1, h2);
  }

  return pij;
}

PairFMap getPijN(size_t _n) {
  PairFMap pijn;

  for (size_t i = 0; i < _n-1; ++i)
    for (size_t j = i + 1; j < _n; ++j)
      pijn[MP(i, j)] = 1. * (j - i) / (_n + 1);

  return pijn;
}

real_t getPetuninProximity(Sample &_a, Sample &_b) {
  auto N = _a.size()*(_a.size()-1) / 2.;

  constexpr real_t g = 3.;

  auto hijn = getHinN(_a, _b);
  auto pij = getPij(hijn, _b.size(), g);
  auto pijn = getPijN(_a.size());

  auto L = getL(pij, pijn, _a.size());

  return L / N;
}

real_t getAvgPetuninProximity(Samples &_a, Samples &_b) {
  assert(_a.size() == _b.size());
  assert(_a.size() != 0);

  real_t sum = 0;
  for (size_t i = 0; i < _a.size(); ++i)
    sum += getPetuninProximity(_a[i], _b[(i+7) % _b.size()]);

  return sum / _a.size();
}

int main() {
  size_t ssize = 100;
  size_t nsize = 100;

  std::cout << "Generate twice " << nsize << " samples of size " << ssize 
            << " with normal distributions (0, 1) and (3, 1);\n" << std::endl;

  Samples setA = getSamples(0, 1, ssize, nsize);
  Samples setB = getSamples(3, 1, ssize, nsize);

  std::cout << "Petunin proximity of sets A and A is " << getAvgPetuninProximity(setA, setA) << std::endl;
  std::cout << "Petunin proximity of sets A and B is " << getAvgPetuninProximity(setA, setB) << std::endl;
  std::cout << "Petunin proximity of sets B and B is " << getAvgPetuninProximity(setB, setB) << std::endl;

  std::cout << "\nDone" << std::endl;

  return 0;
}