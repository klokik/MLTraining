#include <iostream>

#include "framework.hh"


int main(int argc, char **argv) {
  TrainingData data = readData("training_data.data");
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

  // int chunks = 10;
  // for (int i = 0; i < chunks; ++i) {
  //   TrainingData training_data, validation_data;
  //   std::tie(training_data, validation_data) = split(data, 1.f/chunks, static_cast<float>(i)/chunks);
  //   DummyClassifier dummy_cl;
  //   runExperiment(dummy_cl, training_data, validation_data);
  // }

  return 0;

  // NearestNClassifier nearest_cl;
  // runExperiment(nearest_cl, training_data, validation_data);

  // HardSVMClassifier hsvm_cl(1000000, 1e-3, 1e-5);
  // runExperiment(hsvm_cl, training_data, validation_data);

  // SoftSVMClassifier ssvm_cl(10000, 1e-3, 1e-5, 0.5);
  // runExperiment(ssvm_cl, training_data, validation_data);

  // SoftSVMCVClassifier ssvmcv_cl;
  // runExperiment(ssvmcv_cl, training_data, validation_data);

  // RosenblatClassifier ros_cl;
  // runExperiment(ros_cl, training_data, validation_data);

  // EllipseRanking1Classifier ellr1_cl;
  // runExperiment(ellr1_cl, training_data, validation_data);

  // EllipseRanking2Classifier ellr2_cl;
  // runExperiment(ellr2_cl, training_data, validation_data);

  // MeansClassifier means_cl;
  // runExperiment(means_cl, training_data, validation_data);

  // FisherLinearClassifier2Plane flc2p_cl;
  // runExperiment(flc2p_cl, training_data, validation_data);

  return 0;
}