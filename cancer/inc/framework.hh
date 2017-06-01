#pragma once

#include <cassert>
#include <cmath>

#include <array>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>


constexpr size_t v_len = 15;
using v_t = float;
using vec = std::array<v_t, v_len>;

using LinearSpan = std::vector<vec>;

v_t dot(const vec &_a, const vec &_b);
v_t norm(const vec &_a);
vec operator + (const vec &_a, const vec &_b);
vec operator - (const vec &_a, const vec &_b);
vec operator * (const vec &_a, const vec &_b);
vec operator * (const v_t _l, const vec &_a);
vec operator * (const vec &_a, const v_t _l);
vec operator / (const vec &_a, const v_t _l);
vec project(const vec &_v, const vec &_u);
vec project(const vec &_v, const LinearSpan &_span);
vec orth(const vec &_v, const LinearSpan &_span);
bool operator == (const vec &_a, const vec &_b);
bool operator != (const vec &_a, const vec &_b);

using Tag = int;
using TrainingData = std::vector<std::pair<vec, Tag>>;

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

using ClassifierPtr = std::shared_ptr<Classifier>;
using ClCtor = std::function<ClassifierPtr()>;

class GlobalClassifierFactory
{
  public: GlobalClassifierFactory() = default;

  public: int registerClassifier(const char *_name,
    ClCtor _ctor) {
    registered_classifiers[_name] = _ctor;

    return registered_classifiers.size();
  }

  public: ClassifierPtr makeClassifier(const char *_name) {
    return registered_classifiers.at(_name)();
  }

  private: static std::map<std::string, ClCtor> registered_classifiers;
};

extern GlobalClassifierFactory data_classification_factory;

#define TOKENPASTE1(x, y) x ## y
#define TOKENPASTE2(x, y) TOKENPASTE1(x, y)
#define REGISTER_CLASSIFIER(CL) static int \
  TOKENPASTE2(reg_decl_unused, __LINE__) = \
    ::data_classification_factory.registerClassifier(\
    #CL, [](){return std::static_pointer_cast<Classifier>\
      (std::make_shared<CL>());})

#define MK_CLASSIFIER(CL) (::data_classification_factory.makeClassifier(#CL))

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

std::pair<TrainingData, TrainingData> split(TrainingData &_data,
  float _rate, float _position = 0);

std::pair<vec, vec> preprocessData(TrainingData &_data);
std::tuple<v_t, v_t> levelTopBottom(TrainingData &_data, size_t _axis, float _rate);
vec scaleDataPoint(vec _v, std::pair<vec, vec> _range);

size_t teach(Classifier &_classifier, TrainingData _data);

std::tuple<float, float, float> getClassificationScore(
  Classifier &_classifier, TrainingData &_data);

float runExperiment(Classifier &_classifier,
  TrainingData &_training_data, TrainingData &_validation_data);

TrainingData readData(std::string _filename);