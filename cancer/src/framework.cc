#include <cassert>


#include <algorithm>
#include <exception>
#include <iostream>
#include <iterator>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <type_traits>
#include <typeinfo>

#include "framework.hh"


v_t dot(const vec &_a, const vec &_b) {
  assert(_a.size() == _b.size());

  v_t sum = 0;
  for (size_t i = 0; i < _a.size(); ++i)
    sum += _a[i] * _b[i];

  return sum;
}

v_t norm(const vec &_a) {
  return std::sqrt(dot(_a, _a));
}

vec operator + (const vec &_a, const vec &_b) {
  assert(_a.size() == _b.size());

  vec c;
  for (size_t i = 0; i < _a.size(); ++i)
    c[i] = _a[i] + _b[i];

  return c;
}

vec operator - (const vec &_a, const vec &_b) {
  assert(_a.size() == _b.size());

  vec c;
  for (size_t i = 0; i < _a.size(); ++i)
    c[i] = _a[i] - _b[i];

  return c;
}

vec operator * (const vec &_a, const vec &_b) {
  assert(_a.size() == _b.size());

  vec c;
  for (size_t i = 0; i < _a.size(); ++i)
    c[i] = _a[i] * _b[i];

  return c;
}

vec operator * (const v_t _l, const vec &_a) {
  vec c;
  for (size_t i = 0; i < _a.size(); ++i)
    c[i] = _l * _a[i];

  return c;
}

vec operator * (const vec &_a, const v_t _l) {
  return _l * _a;
}

vec operator / (const vec &_a, const v_t _l) {
  vec c;
  for (size_t i = 0; i < _a.size(); ++i)
    c[i] = _a[i] / _l;

  return c;
}

bool operator == (const vec &_a, const vec &_b) {
  assert(_a.size() == _b.size());

  for (size_t i = 0; i < _a.size(); ++i)
    if (_a[i] != _b[i])
      return false;

  return true;
}

bool operator != (const vec &_a, const vec &_b) {
  return !(_a == _b);
}

vec project(const vec &_v, const vec &_u) {
  return _u*(dot(_v, _u)/dot(_u, _u));
}

vec project(const vec &_v, const LinearSpan &_span) {
  vec c{};

  for (const auto &u : _span)
    c = c + project(_v, u);

  return c;
}

vec orth(const vec &_v, const LinearSpan &_span) {
  return _v - project(_v, _span);
}

std::pair<TrainingData, TrainingData> split(TrainingData &_data, float _rate, float _position) {
  auto n = _data.size();
  size_t n1 = n*_rate;
  size_t offset = (n-n1)*_position;
  
  TrainingData slice_1, slice_2;

  for (size_t i = 0; i < _data.size(); ++i)
    if (i >= offset && i < offset+n1)
      slice_1.push_back(_data[i]);
    else
      slice_2.push_back(_data[i]);

  return {slice_1, slice_2};
}

std::tuple<v_t, v_t> levelTopBottom(TrainingData &_data, size_t _axis, float _rate) {
  auto i = _axis;
  std::sort(_data.begin(), _data.end(),
    [i] (auto &a, auto &b) { return a.first[i] < b.first[i]; });

  auto n = _data.size();
  auto level = (1 - _rate)/2;

  auto lower = _data[n*level].first[_axis];
  auto upper = _data[n*(1-level)].first[_axis];

  return {lower, upper};
}

std::pair<vec, vec> preprocessData(TrainingData &_data) {
  std::vector<uint8_t> mask(_data.size(), true);

  for (size_t i = 0; i < _data.begin()->first.size(); ++i) {
    v_t bottom, top;
    std::tie(bottom, top) = levelTopBottom(_data, i, 0.9);
    for (size_t j = 0; j < _data.size(); ++j)
      mask[j] &= (_data[j].first[i] >= bottom && _data[j].first[i] <= top);
  }

  // remove masked elements
  TrainingData new_data;
  for (size_t i = 0; i < _data.size(); ++i) {
    if (mask[i])
      new_data.push_back(_data[i]);
  }
  _data = std::move(new_data);

  // scale to [-1, 1] range
  vec lower, upper;
  for (size_t i = 0; i < lower.size(); ++i) {
    auto min_max = std::minmax_element(_data.begin(), _data.end(),
      [i] (auto &a, auto &b) {
        return a.first[i] < b.first[i];
      });

    lower[i] = min_max.first->first[i];
    upper[i] = min_max.second->first[i];

    std::for_each(_data.begin(), _data.end(),
      [i, &lower, &upper](auto &a) { a.first[i] = (a.first[i] - lower[i])*2/(upper[i] - lower[i]) - 1; });
  }

  return {lower, upper};
}

vec scaleDataPoint(vec _v, std::pair<vec, vec> _range) {
  vec result;
  for (size_t i = 0; i < _v.size(); ++i) {
    result[i] = (_v[i] - _range.first[i])*2/(_range.second[i] - _range.first[i]) - 1;
  }

  return result;
}

size_t teach(Classifier &_classifier, TrainingData _data) {
  size_t steps = 0;
  bool learned = true;

  do {
    learned = true;
    if (_classifier.isOnline())
      for (auto item : _data)
        learned &= _classifier.learn(item.first, item.second);
    else
      _classifier.batchLearn(_data);

    steps++;

    if (steps % 1000 == 0)
      std::cout << "\tlearning step " << steps << std::endl;

    // if (steps % 1000 == 0)
      // std::cout << _classifier.dumpSettings() << std::endl;

    if (steps >= 10000)
      break;
  } while (!learned);

  return steps;
}

std::tuple<float, float, float> getClassificationScore(Classifier &_classifier, TrainingData &_data) {
  float true_positive = 0;
  float true_negative = 0;

  size_t positive = 0;
  size_t negative = 0;

  for (auto item : _data) {
    auto tag = _classifier.classify(item.first);
    // sum += std::pow(_classifier.error(tag, item.second), 2);
    // std::cout << sum << std::endl;
    if (tag == item.second) {
      if (tag == 0)
        true_negative += 1;
      else
        true_positive += 1;
    }
    // std::cout << tag << " ";

    if (item.second == 0)
      ++negative;
    else
      ++positive;
  }

  float accuracy = (true_positive+true_negative) / _data.size();

  return {true_positive / positive, true_negative / negative, accuracy};
}

float runExperiment(Classifier &_classifier, TrainingData &_training_data, TrainingData &_validation_data) {
  std::cout << "Learning: ..." << std::endl;

  auto steps = teach(_classifier, _training_data);

  std::cout << "Done in " << steps << " steps" << std::endl;
  std::cout << _classifier.dumpSettings() << std::endl;

  std::cout << "Validating ..." << std::endl;

  float sensitivity, specificity, accuracy;
  std::tie(sensitivity, specificity, accuracy) = getClassificationScore(_classifier, _validation_data);

  std::cout << "Classification accuracy:\t" << accuracy
            << "\n\t\tsensitivity:\t" << sensitivity
            << "\n\t\tspecificity:\t" << specificity << std::endl << std::endl;

  return accuracy;
}

TrainingData readData(std::string _filename) {
  std::ifstream ifs(_filename);

  if (!ifs.is_open())
    throw std::runtime_error("File 'training_data.data' not found");

  TrainingData data;

  while(ifs.good()) {
    std::string line;
    std::getline(ifs, line);

    std::istringstream iss(line);

    vec v;
    Tag tag;

    for (auto &x : v)
      iss >> x;

    iss >> tag;

    data.push_back({v, tag});
  }

  ifs.close();

  // std::random_device rd;
  std::mt19937 rgn(1337);//rd());

  std::shuffle(data.begin(), data.end(), rgn);

  return data;
}

