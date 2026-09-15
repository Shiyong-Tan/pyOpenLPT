#ifndef OPENLPT_GPU_EXACT_MORPHOLOGY_H
#define OPENLPT_GPU_EXACT_MORPHOLOGY_H

// Run the exact Boolean fixed-point propagation used by the generated
// NeighborhoodProcessor. The input array is column-major (MATLAB layout).
// Returns false when CUDA is unavailable or an execution error occurs; callers
// must then execute the complete CPU implementation.
bool openlptGpuExactMorphology(const double *values, int rows, int cols,
                              bool *output);

#endif
