#include "framework.hh"


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

REGISTER_CLASSIFIER(DummyClassifier);
