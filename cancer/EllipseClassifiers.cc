#include "framework.hh"
#include "cv_common.hh"

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

vec operator * (cv::Mat &_mtx, vec &_a) {
  //static_assert(std::is_same<v_t, float>::value, "Assumption that vector has 'float' components does not hold");

  cv::Mat x(_a.size(), 1, cv_t, &_a[0], sizeof(v_t));

  x = _mtx * x;

  vec result;
  for (size_t i = 0; i < v_len; ++i)
    result[i] = (static_cast<int>(i) < _mtx.rows ? x.at<v_t>(i, 0) : 0);

  return result;
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
        if (rank++ % 50 == 0 )
          std::cout << "\trank " << rank << std::endl;

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

      // assert(std::abs(norm(project(h_c, span))) < 0.1f);
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
