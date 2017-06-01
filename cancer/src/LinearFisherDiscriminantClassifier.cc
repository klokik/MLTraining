#include <map>


#include "framework.hh"
#include "cv_common.hh"


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
