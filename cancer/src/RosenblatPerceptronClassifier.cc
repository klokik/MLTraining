#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include "framework.hh"


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

REGISTER_CLASSIFIER(RosenblatClassifier);
