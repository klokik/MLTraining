#include <iostream>

#include "framework.hh"


int main(int argc, char **argv) {
  TrainingData data = readData("data/training_data.data");
  // data.resize(100);

  TrainingData training_data, validation_data;
  std::tie(training_data, validation_data) = split(data, 0.8f);
 
#if 0
  auto range = preprocessData(training_data);
  for (auto &item : validation_data)
    item.first = scaleDataPoint(item.first, range);
  // validation_data = training_data;
#endif

  std::cout << "Have " << data.size() << " data samples:\n\t"
            << training_data.size() << " for training,\t"
            << validation_data.size() << " for validataion,\t"
            << data.size() - training_data.size() - validation_data.size() << " dropped"
            << std::endl << std::endl;

#if defined(CROSS_VALIDATION_EX)
  Cross validation across whole set in chunks
  int chunks = 10;
  for (int i = 0; i < chunks; ++i) {
    TrainingData training_data, validation_data;
    std::tie(training_data, validation_data) = split(data, 1.f/chunks, static_cast<float>(i)/chunks);
    auto dummy_cl = MK_CLASSIFIER(DummyClassifier);
    runExperiment(*dummy_cl, training_data, validation_data);
  }
#endif

  auto dummy_cl = MK_CLASSIFIER(DummyClassifier);
  runExperiment(*dummy_cl, training_data, validation_data);

  auto nearest_cl = MK_CLASSIFIER(NearestNClassifier);
  runExperiment(*nearest_cl, training_data, validation_data);

  auto hsvm_cl = MK_CLASSIFIER(HardSVMClassifier);
  runExperiment(*hsvm_cl, training_data, validation_data);

  auto ssvm_cl = MK_CLASSIFIER(SoftSVMClassifier);
  runExperiment(*ssvm_cl, training_data, validation_data);

  auto ssvmcv_cl = MK_CLASSIFIER(SoftSVMCVClassifier);
  runExperiment(*ssvmcv_cl, training_data, validation_data);

  auto ros_cl = MK_CLASSIFIER(RosenblatClassifier);
  runExperiment(*ros_cl, training_data, validation_data);

  auto ellr1_cl = MK_CLASSIFIER(EllipseRanking1Classifier);
  runExperiment(*ellr1_cl, training_data, validation_data);

  auto ellr2_cl = MK_CLASSIFIER(EllipseRanking2Classifier);
  runExperiment(*ellr2_cl, training_data, validation_data);

  auto means_cl = MK_CLASSIFIER(MeansClassifier);
  runExperiment(*means_cl, training_data, validation_data);

  auto flc2p_cl = MK_CLASSIFIER(FisherLinearClassifier2Plane);
  runExperiment(*flc2p_cl, training_data, validation_data);

  return 0;
}
