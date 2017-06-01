#include <algorithm>

#include "framework.hh"


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
