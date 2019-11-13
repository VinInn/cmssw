#ifndef RecoLocalCalo_HcalRecAlgos_EigenMatrixTypes_h
#define RecoLocalCalo_HcalRecAlgos_EigenMatrixTypes_h

#include <Eigen/Dense>

constexpr int MaxSVSize = 10;
constexpr int MaxFSVSize = 15;
constexpr int MaxPVSize = 8;

typedef Eigen::Matrix<float, 8, 1> SampleVector;
typedef Eigen::Matrix<float, 8, 1> PulseVector;
typedef Eigen::Matrix<int, 8, 1> BXVector;

typedef Eigen::Matrix<float, Eigen::Dynamic, 1, 0, MaxFSVSize, 1> FullSampleVector;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, 0, MaxFSVSize, MaxFSVSize> FullSampleMatrix;

typedef Eigen::Matrix<float, 8, 8> SampleMatrix;
typedef Eigen::Matrix<float, 8, 8> PulseMatrix;
typedef Eigen::Matrix<float, 8, 8> SamplePulseMatrix;

typedef Eigen::LLT<SampleMatrix> SampleDecompLLT;

#endif
