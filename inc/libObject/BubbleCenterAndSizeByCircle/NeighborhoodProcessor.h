//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// NeighborhoodProcessor.h
//
// Code generation for function 'NeighborhoodProcessor'
//

#ifndef NEIGHBORHOODPROCESSOR_H
#define NEIGHBORHOODPROCESSOR_H

// Include files
#include "coder_array.h"
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>
#include <omp.h>


// Type Definitions
namespace coder {
namespace images {
namespace internal {
namespace coder {
class NeighborhoodProcessor {
public:
  static void computeParameters(const int imSize[2], const bool nhConn[9],
                                int loffsets[9], int linds[9], int soffsets[18],
                                int interiorStart[2], int interiorEnd[2]);
  void process2D(const ::coder::array<double, 2U> &in,
                 ::coder::array<bool, 2U> &out,
                 const ::coder::array<bool, 2U> &previous) const;
  bool Neighborhood[9];
  int ImageSize[2];
  int InteriorStart[2];
  int InteriorEnd[2];
  int ImageNeighborLinearOffsets[9];
  double Padding;
  double PadValue;
  bool ProcessBorder;
  double NeighborhoodCenter;
  int NeighborLinearIndices[9];
  int NeighborSubscriptOffsets[18];
};

} // namespace coder
} // namespace internal
} // namespace images
} // namespace coder

#endif
// End of code generation (NeighborhoodProcessor.h)
