#include <cassert>
#include <cmath>

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <iterator>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>


constexpr auto cv_t = CV_32FC1;
constexpr auto cv_tag_t = CV_32SC1;
using v_t = float;
constexpr size_t v_len = 15;
using vec = std::array<v_t, v_len>;

using LinearSpan = std::vector<vec>;


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

inline vec project(const vec &_v, const vec &_u) {
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

vec operator * (cv::Mat &_mtx, vec &_a) {
  //static_assert(std::is_same<v_t, float>::value, "Assumption that vector has 'float' components does not hold");

  cv::Mat x(_a.size(), 1, cv_t, &_a[0], sizeof(v_t));

  x = _mtx * x;

  vec result;
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = x.at<v_t>(i, 0);

  return result;
}

using Tag = int;
using TrainingData = std::vector<std::pair<vec, Tag>>;

std::pair<TrainingData, TrainingData> split(TrainingData &_data, float _rate, float _position = 0) {
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

class Classifier
{
  public: virtual bool learn(vec _v, Tag _tag) = 0;
  public: virtual Tag classify(vec _v) = 0;

  public: virtual bool batchLearn(TrainingData &_data) {
    throw std::runtime_error("Not implemented");
  }

  public: virtual bool isOnline() {
    return true;
  }

  public: virtual float error(Tag _observed, Tag _expected) {
    return std::abs(static_cast<int>(_expected) - static_cast<int>(_observed));
  }

  public: virtual std::string dumpSettings() {
    return "Unknown classifier";
  }

  public: Classifier() = default;
  public: virtual ~Classifier() = default;
};

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

    if (steps % 100 == 0)
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

TrainingData readData() {
  std::ifstream ifs("training_data.data");

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

class DummyClassifier: public Classifier
{
  public: virtual bool learn(vec _v, Tag _tag) override {
    // nop )
    return true;
  }

  public: virtual Tag classify(vec _v) override {
    return 0;
  }

  public: virtual std::string dumpSettings() override {
    return "Dummy classifier";
  }
};

class NearestNClassifier: public Classifier
{
  public: virtual bool learn(vec _v, Tag _tag) override {
    this->memory.push_back({_v, _tag});

    return true;
  }

  public: virtual Tag classify(vec _v) override {
    assert(!this->memory.empty());

    using vec_t = std::pair<vec, Tag>;
    auto it = std::min_element(this->memory.begin(), this->memory.end(),
      [_v](const vec_t &a, const vec_t &b) { return norm(a.first - _v) < norm(b.first - _v);});

    return it->second;
  }

  public: virtual std::string dumpSettings() override {
    std::stringstream sstr;
    sstr << "Nearest neighbor classifier\n\t"
         << this->memory.size() << " elements in memory";
    return sstr.str();
  }

  private: std::vector<std::pair<vec, Tag>> memory;
};

class HardSVMClassifier: public Classifier
{
  public: HardSVMClassifier(size_t _steps, float _max_err, float _learn_rate):
    steps(_steps), max_err(_max_err), learn_rate(_learn_rate) {
    for (auto &item : w)
      item = 0.1f;
  }

  public: virtual bool isOnline() override {
    return false;
  }

  public: virtual bool learn(vec _v, Tag _tag) override {
    throw std::runtime_error("Not implemented");
    return false;
  }

  public: virtual bool batchLearn(TrainingData &_data) override {
    size_t steps = 0ul;

    while (steps < this->steps) {
      for (auto &item : _data) {
        auto &x = item.first;
        auto pred = (dot(w, x) - b > 0 ? 1. : -1.);
        auto corr = (item.second ? 1. : -1.);

        auto err = pred - corr;
        if (steps % 100000 == 0)
          std::cout << dot(w, x) - b << " " << b << std::endl;

        this->w = w - x*2*err*learn_rate;
        this->b = b - 2*err*learn_rate;

        if (steps++ >= this->steps)
          break;
      }
    }

    return true;
  }

  public: virtual Tag classify(vec _v) override {
    return (dot(w, _v) - b > 0 ? 1 : 0);
  }

  public: virtual std::string dumpSettings() override {
    std::stringstream sstr;
    sstr << "Hard SVM classifier\n\t"
         << "w = {";
    for (auto wi : w)
      sstr << wi << ",\t";
    sstr << "}\tb = " << b;

    return sstr.str();
  }

  protected: vec w {};
  protected: v_t b = 0;

  protected: size_t steps;
  protected: float max_err;
  protected: float learn_rate;
};

class SoftSVMCVClassifier: public Classifier
{
  public: SoftSVMCVClassifier() {
    this->svm = cv::ml::SVM::create();
    this->svm->setType(cv::ml::SVM::C_SVC);
    this->svm->setKernel(cv::ml::SVM::RBF);
    this->svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10000, 1e-3));
    this->svm->setC(1);
    this->svm->setNu(0.5);
    this->svm->setGamma(2);
    this->svm->setDegree(5);
  }

  public: virtual bool isOnline() override {
    return false;
  }

  public: virtual bool learn(vec _v, Tag _tag) override {
    throw std::runtime_error("Not implemented");
    return false;
  }

  public: virtual bool batchLearn(TrainingData &_data) override {
    cv::Mat training_data(_data.size(), v_len, cv_t);
    cv::Mat training_tags(_data.size(), 1, cv_tag_t);

    int i = 0;
    for (auto &item : _data) {
      memcpy(training_data.ptr(i), &item.first[0], sizeof(v_t)*v_len);
      training_tags.at<v_t>(i) = (item.second == 0 ? -1 : 1); 
      ++i;
    }

    cv::Ptr<cv::ml::TrainData> tagged_training_data =
      cv::ml::TrainData::create(training_data, cv::ml::ROW_SAMPLE, training_tags);

    this->svm->train(tagged_training_data);
/*    this->svm->trainAuto(tagged_training_data,
      10,                                               // kFold
      cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C),      // Cgrid
      cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA),  // gammaGrid
      cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P),      // pGrid
      cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU),     // nuGrid
      cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF),   // coeffGrid
      cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE), // degreeGrid
      true);                                            // balanced*/

    return true;
  }

  public: virtual Tag classify(vec _v) override {
    cv::Mat input(1, v_len, cv_t, &_v[0]);

    auto response = this->svm->predict(input);

    return (response > 0);
/*    if (response == 1)
      return 1;
    else if (response == -1)
      return 0;*/

    std::cout << "Invalid response " << response << std::endl;
    throw std::runtime_error("Invalid state: failed to classify data sample");
  }

  public: virtual std::string dumpSettings() override {
    return "Soft SVM (OCV) classifier";
  }

  protected: cv::Ptr<cv::ml::SVM> svm;
};

struct Ellipse {
  LinearSpan axes;
  vec origin;
};

v_t eqForEllipse(vec _x, Ellipse &_ell) {
  v_t sum = 0;

  for (auto axis : _ell.axes) {
    auto p_x = project(_x-_ell.origin, {axis});
    sum += std::pow(norm(p_x)/norm(axis), 2);
  }

  return sum;
}

class EllipseRanking1Classifier: public Classifier
{
  public: virtual bool learn(vec _v, Tag _tag) override {
    throw std::runtime_error("Not implemented");
  }

  public: virtual bool isOnline() override {
    return false;
  }

  public: virtual bool batchLearn(TrainingData &_data) override {
    rankss[0].clear();
    rankss[1].clear();

    std::vector<vec> datas[2];

    for (auto &item : _data)
      if (item.second == 0)
        datas[0].push_back(item.first);
      else
        datas[1].push_back(item.first);

    for (int i = 0; i < 2; ++i) {
      auto &data = datas[i];
      int rank = 0;
      while (data.size() >= 2) {
        std::cout << "\trank " << rank++ << std::endl;

        Ellipse ell;
        std::vector<v_t> dists;
        std::tie(ell, dists) = buildEllipseEx(data);

        // find the furtherest point from center
        v_t dist = *std::max_element(dists.begin(), dists.end());

        // upscale the ellipse's span
        for (auto &item : ell.axes)
          item = item * dist;

        this->rankss[i].push_back(ell);

        auto it = std::max_element(data.begin(), data.end(),
          [&ell](auto _a, auto _b) {
            return eqForEllipse(_a, ell) < eqForEllipse(_b, ell);
          });

        data.erase(it);
      }
    }

    return true;
  }

  protected: virtual std::pair<Ellipse, std::vector<v_t>> buildEllipseEx(std::vector<vec> &_data) {
    LinearSpan span;
    vec origin;

    v_t dist_max = 0;
    vec a, b;
    for (size_t i = 0; i < _data.size()-1; ++i)
      for (size_t j = i+1; j < _data.size(); ++j) {
        auto dist = norm(_data[i] - _data[j]);
        if (dist > dist_max)
          std::tie(dist_max, a, b) = {dist, _data[i], _data[j]};
      }

    span.push_back((b - a)/2);
    origin = (b + a)/2;

    std::vector<vec> new_data;
    for (auto item : _data)
      new_data.push_back(item - origin);

    while (span.size() != origin.size()) {
      // find the next furtherest point from the linear span
      auto it = std::max_element(new_data.begin(), new_data.end(),
        [&span](vec a, vec b) {
          auto h_a = orth(a, span);
          auto h_b = orth(b, span);
          return norm(h_a) < norm(h_b);
        });

      assert(it != new_data.end());
      auto h_c = orth(*it, span);
      if (norm(h_c) < 0.1)
        break; // otherwise matrix will be singular

//      assert(std::abs(norm(project(h_c, span))) < 0.1f);
      // find the most distant point from the opposite side of the lin-span
      auto jt = std::min_element(new_data.begin(), new_data.end(),
        [&span, h_c](vec a, vec b) {
          auto h_a = orth(a, span);
          auto h_b = orth(b, span);
          return dot(h_c, h_a) < dot(h_c, h_b);
        });

      auto h_d = project(orth(*jt, span), h_c);
      // assert(dot(h_d, h_c) <= 1e-1);

      //span.push_back(h_c);
      span.push_back(h_c*(norm(h_c - h_d)/norm(h_c)));

      // shift all the pionts so that their median
      // on the corresponding hyperplane intersects current span
      auto offset = (h_c + h_d)/2;
      for (auto &item : new_data)
        item = item - offset;
      origin = origin + offset;
    }

    // convert data frame to [-1,1]^n hypercube using matrix S
    cv::Mat S(origin.size(), span.size(), cv_t);
    for (int i = 0; i < S.cols; ++i)
      for (int j = 0; j < S.rows; ++j)
        S.at<v_t>(j, i) = span[i][j];
    auto res = cv::invert(S, S, cv::DECOMP_SVD);
    assert(res != 0);

    std::vector<vec> scaled_data;
    for (auto &item : new_data) {
      // clamp vector to [-1, 1] cube
      auto sitem = S * item;
      for (auto &xi : sitem)
        xi = (std::abs(xi) < 1 ? xi : 0);
      scaled_data.push_back(sitem);
    }

    std::vector<v_t> dists;
    for (auto &item : scaled_data) {
      auto dist = norm(item);
      assert(dist < 10);
      dists.push_back(dist);
    }

    return {{span, origin}, dists};
  }

  public: virtual Tag classify(vec _v) override {
    auto pred = [&_v] (Ellipse &el) { return eqForEllipse(_v, el) < 1;};

    auto it_0 = std::find_if(rankss[0].rbegin(), rankss[0].rend(), pred);
    auto it_1 = std::find_if(rankss[1].rbegin(), rankss[1].rend(), pred);

    auto inv_rank0 = std::distance(rankss[0].rbegin(), it_0)/static_cast<float>(rankss[0].size());
    auto inv_rank1 = std::distance(rankss[1].rbegin(), it_1)/static_cast<float>(rankss[1].size());

    return inv_rank1 < inv_rank0;
  }

  public: virtual std::string dumpSettings() override {
    return "Ellipse Ranking Type1 classifier";
  }

  protected: std::vector<Ellipse> rankss[2];
};

class EllipseRanking2Classifier: public EllipseRanking1Classifier
{
  public: virtual bool batchLearn(TrainingData &_data) override {
    rankss[0].clear();
    rankss[1].clear();

    std::vector<vec> datas[2];

    for (auto &item : _data)
      if (item.second == 0)
        datas[0].push_back(item.first);
      else
        datas[1].push_back(item.first);

    for (int i = 0; i < 2; ++i) {
      auto &data = datas[i];
        Ellipse ell;
        std::vector<v_t> dists;

        std::tie(ell, dists) = buildEllipseEx(data);

        std::sort(dists.begin(), dists.end(), [](auto a, auto b) {return a > b;});

        for (auto dist : dists) {
          Ellipse sc_ell;

          sc_ell.origin = ell.origin;

          // upscale the ellipse's span
          for (auto &item : ell.axes)
            sc_ell.axes.push_back(item * dist);

          this->rankss[i].push_back(sc_ell);
        }
    }

    return true;
  }

  public: virtual std::string dumpSettings() override {
    return "Ellipse Ranking Type2 classifier";
  }
};

class RosenblatClassifier: public Classifier
{
  public: virtual bool learn(vec _v, Tag _tag) override {
    auto cur_tag = classify(_v);
    if (cur_tag == 0 && _tag == 1) {
      this->w = this->w + _v;
      this->w0 += 1;
    }
    else if (cur_tag == 1 && _tag == 0) {
      this->w = this->w - _v;
      this->w0 -= 1;
    }
    else
      return true;

    return false;
  }

  public: virtual Tag classify(vec _v) override {
    return dot(this->w, _v) + this->w0 > 0;
  }

  public: virtual std::string dumpSettings() override {
    std::stringstream sstr;
    sstr << "Single-layer rosenblat classifier\n\t" << "Weights: ";
    std::copy(this->w.begin(), this->w.end(), std::ostream_iterator<v_t>(sstr, " "));
    sstr << " " << this->w0;

    return sstr.str();
  }

  private: vec w{};
  private: v_t w0;
};

class MeansClassifier: public Classifier
{
  public: virtual bool learn(vec _v, Tag _tag) override {
    throw std::runtime_error("Not implemented");
  }

  public: virtual bool isOnline() override {
    return false;
  }

  public: virtual bool batchLearn(TrainingData &_data) override {
    means.clear();

    vec sums[2]{};
    size_t nums[] = {0, 0};

    for (auto &item : _data) {
      sums[item.second] = sums[item.second] + item.first;
      nums[item.second]++;
    }

    assert(nums[0] != 0);
    assert(nums[1] != 0);

    this->means.push_back(sums[0]/nums[0]);
    this->means.push_back(sums[1]/nums[1]);

    return true;
  }

  public: virtual Tag classify(vec _v) override {
    return norm(means[1] - _v) > norm(means[0] - _v);
  }

  public: virtual std::string dumpSettings() override {
    return "Means classifier";
  }

  protected: std::vector<vec> means;
};

class Voting2PlaneClassifier: public Classifier
{
  public: virtual bool isOnline() override {
    return false;
  }

  public: virtual bool learn(vec _v, Tag _tag) override {
    throw std::runtime_error("Not implemented");
    return false;
  }

  public: virtual bool batchLearn(TrainingData &_data) override {
    bool done = true;

    for (size_t i = 0; i < v_len-1; ++i)
      for (size_t j = i+1; j < v_len; ++j)
        done &= this->batchLearnInPlane(i, j, _data);

    return done;
  }

  public: virtual Tag classify(vec _v) override {
    size_t votes[] = {0, 0};

    for (size_t i = 0; i < v_len-1; ++i)
      for (size_t j = i+1; j < v_len; ++j)
        votes[classifyIn2Plane(i, j, _v)] += 1;

    return (votes[1] > votes[0] ? 1 : 0);
  }

  public: virtual bool batchLearnInPlane(size_t _i, size_t _j,
    TrainingData &_data) = 0;

  public: virtual Tag classifyIn2Plane(size_t _i, size_t _j,
    vec _v) = 0;

  public: virtual std::string dumpSettings() override {
    return "Voting Abstract classifier";
  }
};

class FisherLinearClassifier2Plane: public Voting2PlaneClassifier
{
  public: virtual bool batchLearnInPlane(size_t _i, size_t _j,
    TrainingData &_data) override {

    cv::Vec2f x1, x2;
    size_t n1{0}, n2{0};

    for (auto &item : _data) {
      cv::Vec2f pr {item.first[_i], item.first[_j]};

      if (item.second == 0) {
        x1 += pr;
        n1++;
      } else {
        x2 += pr;
        n2++;
      }
    }

    assert(n1 > 0);
    assert(n2 > 0);

    x1 = x1 * (1.f/n1);
    x2 = x2 * (1.f/n2);

    cv::Matx22f s1 { 0, 0, 0, 0 };
    cv::Matx22f s2 { 0, 0, 0, 0 };

    for (auto &item : _data) {
      cv::Vec2f pr {item.first[_i], item.first[_j]};

      if (item.second == 0) {
        s1 = s1 + (pr - x1)*(pr - x1).t();
      } else {
        s2 = s2 + (pr - x2)*(pr - x2).t();
      }
    }
    s1 = s1*(1./(n1-1));
    s2 = s2*(1./(n2-1));

    cv::Vec2f si_w = ((s1*(n1-1.f) + s2*(n2-1.f))*(1./(n1+n2-2.f))).inv()*(x1-x2);

    float y1 {0}, y2 {0};
    for (auto &item : _data) {
      cv::Vec2f pr {item.first[_i], item.first[_j]};

      auto y = si_w.dot(pr);
      if (item.second == 0)
        y1 += y;
      else
        y2 += y;
    }
    y1 = y1*(1.f/n1);
    y2 = y2*(1.f/n2);
    v_t margin = (y1+y2) / 2;

    if (y2 < y1) {
      si_w = -si_w;
      margin = -margin;
    }

    this->ws[_i*v_len+_j] = std::make_pair(si_w, margin);

    return true;
  }

  public: virtual Tag classifyIn2Plane(size_t _i, size_t _j,
    vec _v) override {
    cv::Vec2f pr {_v[_i], _v[_j]};

    cv::Vec2f w;
    v_t margin;

    std::tie(w, margin) = this->ws.at(_i*v_len+_j);

    return (w.dot(pr) >= margin ? 1 : 0);
  }

  public: virtual std::string dumpSettings() override {
    return "Fisher linear classifier";
  }

  private: std::map<size_t, std::pair<cv::Vec2f, float>> ws; // w and margin level
};


int main(int argc, char **argv) {
  TrainingData data = readData();
  // data.resize(100);

  TrainingData training_data, validation_data;
  std::tie(training_data, validation_data) = split(data, 0.8f);
 
#if 0
  auto range = preprocessData(training_data);
  for (auto &item : validation_data)
    item.first = scaleDataPoint(item.first, range);
  // validation_data = training_data;
#endif

#if 0
  for (auto item : validation_data)
    std::cout << item.second << " ";
  std::cout << std::endl;
#endif

  std::cout << "Have " << data.size() << " data samples:\n\t"
            << training_data.size() << " for training,\t"
            << validation_data.size() << " for validataion,\t"
            << data.size() - training_data.size() - validation_data.size() << " dropped"
            << std::endl << std::endl;

  // DummyClassifier dummy_cl;
  // runExperiment(dummy_cl, training_data, validation_data);

  NearestNClassifier nearest_cl;
  runExperiment(nearest_cl, training_data, validation_data);

  HardSVMClassifier hsvm_cl(1000000, 1e-3, 1e-5);
  runExperiment(hsvm_cl, training_data, validation_data);

  SoftSVMCVClassifier ssvm_cl;
  runExperiment(ssvm_cl, training_data, validation_data);

  // RosenblatClassifier ros_cl;
  // runExperiment(ros_cl, training_data, validation_data);

  // EllipseRanking1Classifier ellr1_cl;
  // runExperiment(ellr1_cl, training_data, validation_data);

  // EllipseRanking2Classifier ellr2_cl;
  // runExperiment(ellr2_cl, training_data, validation_data);

  MeansClassifier means_cl;
  runExperiment(means_cl, training_data, validation_data);

  FisherLinearClassifier2Plane flc2p_cl;
  runExperiment(flc2p_cl, training_data, validation_data);

  return 0;
}
