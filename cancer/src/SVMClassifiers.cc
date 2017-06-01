#include "framework.hh"
#include "cv_common.hh"


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

class SoftSVMClassifier: public Classifier
{
  public: SoftSVMClassifier(size_t _steps, float _max_err,
    float _learn_rate, float _regularization):
    steps(_steps), max_err(_max_err), learn_rate(_learn_rate),
    regularization(_regularization) {
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
      vec grad_w {};
      v_t grad_b = 0;

      for (auto &item : _data) {
        auto &x = item.first;
        auto pred = (dot(w, x) - b);
        auto corr = (item.second ? 1. : -1.);

        auto gtz = (1.f - pred*corr > 0 ? 1.f : 0.f);
        auto err_i = x * (corr * gtz);

        grad_w = grad_w + err_i;
        grad_b = grad_b + corr*gtz;
      }

      grad_w = grad_w * (1.f/_data.size()) + w * regularization;
      grad_b = grad_b / _data.size() + b * regularization;

      this->w = w - grad_w*learn_rate;
      this->b = b - grad_b*learn_rate;

      if (steps % 1000 == 0)
        std::cout << norm(grad_w) << " " << grad_b << std::endl;
      steps++;
    }

    return true;
  }

  public: virtual Tag classify(vec _v) override {
    return (dot(w, _v) - b > 0 ? 1 : 0);
  }

  public: virtual std::string dumpSettings() override {
    std::stringstream sstr;
    sstr << "Soft SVM classifier\n\t"
         << "w = {";
    for (auto wi : w)
      sstr << wi << ",\t";
    sstr << "}\tb = " << b;

    return sstr.str();
  }

  protected: vec w {};
  protected: v_t b = 0.1;

  protected: size_t steps;
  protected: float max_err;
  protected: float learn_rate;
  protected: float regularization;
};

REGISTER_CLASSIFIER(SoftSVMCVClassifier);
//REGISTER_CLASSIFIER(SoftSVMClassifier);
//REGISTER_CLASSIFIER(HardSVMClassifier);
