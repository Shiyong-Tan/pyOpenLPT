//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: CircleIdentifier.cpp
//
// MATLAB Coder version            : 23.2
// C/C++ source code generated on  : 03-Jul-2025 19:27:50
//
#include "CircleIdentifier.h"
#ifdef OPENLPT_ENABLE_GPU_EXACT_MORPHOLOGY
#include "GpuExactMorphology.h"
#endif

// Function Declarations
static double rt_atan2d_snf(double u0, double u1);

static double rt_hypotd_snf(double u0, double u1);

// Function Definitions
static double rt_atan2d_snf(double u0, double u1)
{
    double y;
    if (std::isnan(u0) || std::isnan(u1)) {
        y = rtNaN;
    }
    else if (std::isinf(u0) && std::isinf(u1)) {
        int b_u0;
        int b_u1;
        if (u0 > 0.0) {
            b_u0 = 1;
        }
        else {
            b_u0 = -1;
        }
        if (u1 > 0.0) {
            b_u1 = 1;
        }
        else {
            b_u1 = -1;
        }
        y = std::atan2(static_cast<double>(b_u0), static_cast<double>(b_u1));
    }
    else if (u1 == 0.0) {
        if (u0 > 0.0) {
            y = RT_PI / 2.0;
        }
        else if (u0 < 0.0) {
            y = -(RT_PI / 2.0);
        }
        else {
            y = 0.0;
        }
    }
    else {
        y = std::atan2(u0, u1);
    }
    return y;
}

static double rt_hypotd_snf(double u0, double u1)
{
    double a;
    double y;
    a = std::abs(u0);
    y = std::abs(u1);
    if (a < y) {
        a /= y;
        y *= std::sqrt(a * a + 1.0);
    }
    else if (a > y) {
        y /= a;
        y = a * std::sqrt(y * y + 1.0);
    }
    else if (!std::isnan(y)) {
        y = a * 1.4142135623730951;
    }
    return y;
}

std::vector<double> CircleIdentifier::BubbleCenterAndSizeByCircle(
    std::vector<Pt2D> &center,
    std::vector<double> &radius,
    double rmin, double rmax, double sense)
{
  coder::array<double, 2U> centers;
  coder::array<double, 2U> radii;
  coder::array<double, 2U> metrics;
  coder::images::internal::coder::NeighborhoodProcessor np;
  coder::array<struct_T, 1U> s;
  coder::array<creal_T, 2U> accumMatrix;
  coder::array<creal_T, 1U> e_x;
  coder::array<double, 2U> Hd;
  coder::array<double, 2U> accumMatrixRe;
  coder::array<double, 2U> b_accumMatrixRe;
  coder::array<double, 2U> b_centers;
  coder::array<double, 2U> b_metrics;
  coder::array<double, 2U> d_x;
  coder::array<double, 2U> f_x;
  coder::array<double, 2U> metric;
  coder::array<double, 1U> c_x;
  coder::array<double, 1U> cenPhase;
  coder::array<float, 2U> gradientImg;
  coder::array<int, 2U> iidx;
  coder::array<int, 1U> ii;
  coder::array<bool, 2U> b_x;
  coder::array<bool, 2U> bwpre;
  coder::array<bool, 2U> expl_temp_bw;
  coder::array<bool, 1U> x;
  double radiusRange_data[2];
  int b_np[2];
  int radiusRange_size[2];
  int exponent;
  int nrows;
  bool c_np[9];
  bool continuePropagation;
  bool exitg1;
  if (rmin == rmax) {
      radiusRange_size[0] = 1;
      radiusRange_size[1] = 1;
      radiusRange_data[0] = rmin;
  }
  else {
      radiusRange_size[0] = 1;
      radiusRange_size[1] = 2;
      radiusRange_data[0] = rmin;
      radiusRange_data[1] = rmax;
  }
  centers.set_size(0, 0);
  radii.set_size(0, 0);
  metrics.set_size(0, 0);
  coder::chaccum(img, radiusRange_data, radiusRange_size, accumMatrix,
      gradientImg);
  continuePropagation = false;
  nrows = 0;
  exitg1 = false;
  while ((!exitg1) &&
      (nrows + 1 <= accumMatrix.size(0) * accumMatrix.size(1))) {
      if (((accumMatrix[nrows].re == 0.0) && (accumMatrix[nrows].im == 0.0)) ||
          (std::isnan(accumMatrix[nrows].re) ||
              std::isnan(accumMatrix[nrows].im))) {
          nrows++;
      }
      else {
          continuePropagation = true;
          exitg1 = true;
      }
  }
  if (continuePropagation) {
      double absx;
      int i;
      int i1;
      int idx;
      int loop_ub;
      int nx;
      centers.set_size(0, 0);
      metrics.set_size(0, 0);
      nx = accumMatrix.size(0) * accumMatrix.size(1);
      accumMatrixRe.set_size(accumMatrix.size(0), accumMatrix.size(1));
      for (nrows = 0; nrows < nx; nrows++) {
          accumMatrixRe[nrows] =
              rt_hypotd_snf(accumMatrix[nrows].re, accumMatrix[nrows].im);
      }
      x.set_size(accumMatrixRe.size(0) * accumMatrixRe.size(1));
      loop_ub = accumMatrixRe.size(0) * accumMatrixRe.size(1);
      for (i = 0; i < loop_ub; i++) {
          x[i] = (accumMatrixRe[i] == accumMatrixRe[0]);
      }
      continuePropagation = true;
      nrows = 1;
      exitg1 = false;
      while ((!exitg1) && (nrows <= x.size(0))) {
          if (!x[nrows - 1]) {
              continuePropagation = false;
              exitg1 = true;
          }
          else {
              nrows++;
          }
      }
      if (!continuePropagation) {
          if (static_cast<unsigned int>(accumMatrixRe.size(0)) >
              static_cast<unsigned int>(accumMatrixRe.size(1))) {
              i = accumMatrixRe.size(1);
          }
          else {
              i = accumMatrixRe.size(0);
          }
          if (i > 5) {
              if ((accumMatrixRe.size(0) == 0) || (accumMatrixRe.size(1) == 0)) {
                  Hd.set_size(accumMatrixRe.size(0), accumMatrixRe.size(1));
                  loop_ub = accumMatrixRe.size(0) * accumMatrixRe.size(1);
                  for (i = 0; i < loop_ub; i++) {
                      Hd[i] = accumMatrixRe[i];
                  }
              }
              else {
                  b_accumMatrixRe.set_size(accumMatrixRe.size(0),
                      accumMatrixRe.size(1));
                  loop_ub = accumMatrixRe.size(0) * accumMatrixRe.size(1) - 1;
                  for (i = 0; i <= loop_ub; i++) {
                      b_accumMatrixRe[i] = accumMatrixRe[i];
                  }
                  coder::images::internal::coder::optimized::medfilt2(b_accumMatrixRe,
                      Hd);
              }
          }
          else {
              Hd.set_size(accumMatrixRe.size(0), accumMatrixRe.size(1));
              loop_ub = accumMatrixRe.size(0) * accumMatrixRe.size(1);
              for (i = 0; i < loop_ub; i++) {
                  Hd[i] = accumMatrixRe[i];
              }
          }
          absx = std::abs(1.0 - sense);
          if ((!std::isinf(absx)) && (!std::isnan(absx))) {
              if (absx <= 2.2250738585072014E-308) {
                  absx = 4.94065645841247E-324;
              }
              else {
                  frexp(absx, &exponent);
                  absx = std::ldexp(1.0, exponent - 53);
              }
          }
          else {
              absx = rtNaN;
          }
          b_accumMatrixRe.set_size(Hd.size(0), Hd.size(1));
          loop_ub = Hd.size(0) * Hd.size(1) - 1;
          for (i = 0; i <= loop_ub; i++) {
              b_accumMatrixRe[i] = Hd[i];
          }
          coder::imhmax(b_accumMatrixRe, std::fmax((1.0 - sense) - absx, 0.0), Hd);
          np.ImageSize[0] = Hd.size(0);
          np.ImageSize[1] = Hd.size(1);
          np.Padding = 1.0;
          np.ProcessBorder = true;
          np.NeighborhoodCenter = 1.0;
          np.PadValue = 0.0;
          for (exponent = 0; exponent < 9; exponent++) {
              np.Neighborhood[exponent] = true;
              np.ImageNeighborLinearOffsets[exponent] = 0;
              np.NeighborLinearIndices[exponent] = 0;
          }
          std::memset(&np.NeighborSubscriptOffsets[0], 0, 18U * sizeof(int));
          expl_temp_bw.set_size(Hd.size(0), Hd.size(1));
          loop_ub = Hd.size(0) * Hd.size(1);
          for (i = 0; i < loop_ub; i++) {
              expl_temp_bw[i] = true;
          }
          continuePropagation = true;
          bwpre.set_size(expl_temp_bw.size(0), expl_temp_bw.size(1));
          auto *morph_current = &expl_temp_bw;
          auto *morph_next = &bwpre;
#ifdef OPENLPT_ENABLE_GPU_EXACT_MORPHOLOGY
          const bool gpu_morphology_complete = openlptGpuExactMorphology(
              Hd.data(), Hd.size(0), Hd.size(1), expl_temp_bw.data());
#else
          const bool gpu_morphology_complete = false;
#endif
          if (!gpu_morphology_complete) {
            while (continuePropagation) {
              bool p;
              b_np[0] = np.ImageSize[0];
              b_np[1] = np.ImageSize[1];
              for (int i2{ 0 }; i2 < 9; i2++) {
                  c_np[i2] = np.Neighborhood[i2];
              }
              coder::images::internal::coder::NeighborhoodProcessor::
                  computeParameters(b_np, c_np, np.ImageNeighborLinearOffsets,
                      np.NeighborLinearIndices,
                      np.NeighborSubscriptOffsets, np.InteriorStart,
                      np.InteriorEnd);
              np.process2D(Hd, *morph_next, *morph_current);
              p = false;
              if ((morph_current->size(0) == morph_next->size(0)) &&
                  (morph_current->size(1) == morph_next->size(1))) {
                  p = true;
              }
              if (p && ((morph_current->size(0) != 0) &&
                        (morph_current->size(1) != 0)) &&
                  ((morph_next->size(0) != 0) &&
                   (morph_next->size(1) != 0))) {
                  nrows = 0;
                  exitg1 = false;
                  while ((!exitg1) &&
                      (nrows <= morph_current->size(0) *
                                    morph_current->size(1) - 1)) {
                      if ((*morph_current)[nrows] != (*morph_next)[nrows]) {
                          p = false;
                          exitg1 = true;
                      }
                      else {
                          nrows++;
                      }
                  }
              }
              continuePropagation = p;
              continuePropagation = !continuePropagation;
              if (continuePropagation) {
                  auto *morph_tmp = morph_current;
                  morph_current = morph_next;
                  morph_next = morph_tmp;
              }
            }
            if (morph_next != &expl_temp_bw) {
              expl_temp_bw = *morph_next;
            }
          }
          coder::regionprops(expl_temp_bw, accumMatrixRe, s);
          if (s.size(0) != 0) {
              centers.set_size(s.size(0), 2);
              i = s.size(0);
              for (idx = 0; idx < i; idx++) {
                  centers[idx] = s[idx].WeightedCentroid[0];
                  centers[idx + centers.size(0)] = s[idx].WeightedCentroid[1];
              }
              i = centers.size(0) - 1;
              i1 = static_cast<int>(
                  ((-1.0 - static_cast<double>(centers.size(0))) + 1.0) / -1.0);
              for (idx = 0; idx < i1; idx++) {
                  nx = i - idx;
                  if (std::isnan(centers[nx]) ||
                      std::isnan(centers[nx + centers.size(0)])) {
                      d_x.set_size(centers.size(0), 2);
                      loop_ub = centers.size(0) * 2;
                      for (exponent = 0; exponent < loop_ub; exponent++) {
                          d_x[exponent] = centers[exponent];
                      }
                      nrows = centers.size(0) - 1;
                      for (loop_ub = 0; loop_ub < 2; loop_ub++) {
                          for (exponent = nx + 1; exponent <= nrows; exponent++) {
                              d_x[(exponent + d_x.size(0) * loop_ub) - 1] =
                                  d_x[exponent + d_x.size(0) * loop_ub];
                          }
                      }
                      if (1 > centers.size(0) - 1) {
                          loop_ub = 0;
                      }
                      else {
                          loop_ub = centers.size(0) - 1;
                      }
                      for (exponent = 0; exponent < 2; exponent++) {
                          for (nx = 0; nx < loop_ub; nx++) {
                              d_x[nx + loop_ub * exponent] = d_x[nx + d_x.size(0) * exponent];
                          }
                      }
                      d_x.set_size(loop_ub, 2);
                      centers.set_size(loop_ub, 2);
                      nrows = loop_ub << 1;
                      for (exponent = 0; exponent < nrows; exponent++) {
                          centers[exponent] = d_x[exponent];
                      }
                  }
              }
              if (centers.size(0) != 0) {
                  loop_ub = centers.size(0);
                  c_x.set_size(centers.size(0));
                  for (i = 0; i < loop_ub; i++) {
                      c_x[i] = centers[i + centers.size(0)];
                  }
                  i = centers.size(0) - 1;
                  for (nrows = 0; nrows <= i; nrows++) {
                      c_x[nrows] = std::round(c_x[nrows]);
                  }
                  loop_ub = centers.size(0);
                  cenPhase.set_size(centers.size(0));
                  for (i = 0; i < loop_ub; i++) {
                      cenPhase[i] = centers[i];
                  }
                  i = centers.size(0) - 1;
                  for (nrows = 0; nrows <= i; nrows++) {
                      cenPhase[nrows] = std::round(cenPhase[nrows]);
                  }
                  nrows = Hd.size(0);
                  loop_ub = c_x.size(0);
                  for (i = 0; i < loop_ub; i++) {
                      c_x[i] = static_cast<int>(c_x[i]) +
                          nrows * (static_cast<int>(cenPhase[i]) - 1);
                  }
                  metric.set_size(c_x.size(0), 1);
                  loop_ub = c_x.size(0);
                  for (i = 0; i < loop_ub; i++) {
                      metric[i] = Hd[static_cast<int>(c_x[i]) - 1];
                  }
                  coder::internal::sort(metric, iidx);
                  metrics.set_size(metric.size(0), 1);
                  loop_ub = metric.size(0);
                  for (i = 0; i < loop_ub; i++) {
                      metrics[i] = metric[i];
                  }
                  b_centers.set_size(iidx.size(0), 2);
                  loop_ub = iidx.size(0);
                  for (i = 0; i < 2; i++) {
                      for (i1 = 0; i1 < loop_ub; i1++) {
                          b_centers[i1 + b_centers.size(0) * i] =
                              centers[(iidx[i1] + centers.size(0) * i) - 1];
                      }
                  }
                  centers.set_size(b_centers.size(0), 2);
                  loop_ub = b_centers.size(0) * 2;
                  for (i = 0; i < loop_ub; i++) {
                      centers[i] = b_centers[i];
                  }
              }
          }
      }
      if ((centers.size(0) != 0) && (centers.size(1) != 0)) {
          b_x.set_size(metrics.size(0), metrics.size(1));
          loop_ub = metrics.size(0) * metrics.size(1);
          for (i = 0; i < loop_ub; i++) {
              b_x[i] = (metrics[i] >= 1.0 - sense);
          }
          nx = b_x.size(0) * b_x.size(1);
          idx = 0;
          ii.set_size(nx);
          nrows = 0;
          exitg1 = false;
          while ((!exitg1) && (nrows <= nx - 1)) {
              if (b_x[nrows]) {
                  idx++;
                  ii[idx - 1] = nrows + 1;
                  if (idx >= nx) {
                      exitg1 = true;
                  }
                  else {
                      nrows++;
                  }
              }
              else {
                  nrows++;
              }
          }
          if (nx == 1) {
              if (idx == 0) {
                  ii.set_size(0);
              }
          }
          else {
              if (1 > idx) {
                  idx = 0;
              }
              ii.set_size(idx);
          }
          nrows = centers.size(1) - 1;
          b_centers.set_size(ii.size(0), centers.size(1));
          for (i = 0; i <= nrows; i++) {
              loop_ub = ii.size(0);
              for (i1 = 0; i1 < loop_ub; i1++) {
                  b_centers[i1 + b_centers.size(0) * i] =
                      centers[(ii[i1] + centers.size(0) * i) - 1];
              }
          }
          centers.set_size(b_centers.size(0), b_centers.size(1));
          loop_ub = b_centers.size(0) * b_centers.size(1);
          for (i = 0; i < loop_ub; i++) {
              centers[i] = b_centers[i];
          }
          nrows = metrics.size(1) - 1;
          b_metrics.set_size(ii.size(0), metrics.size(1));
          for (i = 0; i <= nrows; i++) {
              loop_ub = ii.size(0);
              for (i1 = 0; i1 < loop_ub; i1++) {
                  b_metrics[i1] = metrics[ii[i1] - 1];
              }
          }
          metrics.set_size(b_metrics.size(0), b_metrics.size(1));
          loop_ub = b_metrics.size(0) * b_metrics.size(1);
          for (i = 0; i < loop_ub; i++) {
              metrics[i] = b_metrics[i];
          }
          if (ii.size(0) == 0) {
              centers.set_size(0, 0);
              metrics.set_size(0, 0);
          }
          else if (radiusRange_size[1] == 1) {
              metric.set_size(ii.size(0), 1);
              nrows = ii.size(0);
              for (nx = 0; nx < nrows; nx++) {
                  metric[nx] = radiusRange_data[0];
              }
              radii.set_size(metric.size(0), 1);
              loop_ub = metric.size(0);
              for (i = 0; i < loop_ub; i++) {
                  radii[i] = metric[i];
              }
          }
          else {
              loop_ub = ii.size(0);
              c_x.set_size(ii.size(0));
              for (i = 0; i < loop_ub; i++) {
                  c_x[i] = centers[i + centers.size(0)];
              }
              i = ii.size(0) - 1;
              for (nrows = 0; nrows <= i; nrows++) {
                  c_x[nrows] = std::round(c_x[nrows]);
              }
              loop_ub = ii.size(0);
              cenPhase.set_size(ii.size(0));
              for (i = 0; i < loop_ub; i++) {
                  cenPhase[i] = centers[i];
              }
              i = ii.size(0) - 1;
              for (nrows = 0; nrows <= i; nrows++) {
                  cenPhase[nrows] = std::round(cenPhase[nrows]);
              }
              nrows = accumMatrix.size(0);
              loop_ub = c_x.size(0);
              for (i = 0; i < loop_ub; i++) {
                  c_x[i] = static_cast<int>(c_x[i]) +
                      nrows * (static_cast<int>(cenPhase[i]) - 1);
              }
              e_x.set_size(c_x.size(0));
              loop_ub = c_x.size(0);
              for (i = 0; i < loop_ub; i++) {
                  e_x[i] = accumMatrix[static_cast<int>(c_x[i]) - 1];
              }
              nx = c_x.size(0);
              cenPhase.set_size(c_x.size(0));
              for (nrows = 0; nrows < nx; nrows++) {
                  cenPhase[nrows] = rt_atan2d_snf(e_x[nrows].im, e_x[nrows].re);
              }
              f_x.set_size(1, 2);
              for (i = 0; i < 2; i++) {
                  f_x[i] = radiusRange_data[i];
              }
              f_x[0] = std::log(f_x[0]);
              f_x[1] = std::log(f_x[1]);
              absx = f_x[1] - f_x[0];
              loop_ub = cenPhase.size(0);
              for (i = 0; i < loop_ub; i++) {
                  cenPhase[i] =
                      (cenPhase[i] + 3.1415926535897931) / 6.2831853071795862 * absx +
                      f_x[0];
              }
              nx = cenPhase.size(0);
              for (nrows = 0; nrows < nx; nrows++) {
                  cenPhase[nrows] = std::exp(cenPhase[nrows]);
              }
              radii.set_size(cenPhase.size(0), 1);
              loop_ub = cenPhase.size(0);
              for (i = 0; i < loop_ub; i++) {
                  radii[i] = cenPhase[i];
              }
          }
      }
  }

  // output
  int n_center = *centers.size();
  std::vector<double> metric_out(n_center);
  center = std::vector<Pt2D>(n_center);
  radius = std::vector<double>(n_center);
  for (int i = 0; i < n_center; i++)
  {
    center[i][0] = centers[i] - 1;
    center[i][1] = centers[n_center + i] - 1;
    radius[i] = radii[i];
    metric_out[i] = metrics[i];
  }
  return metric_out;
}

//
// Arguments    : void
// Return Type  : void
//
CircleIdentifier::CircleIdentifier(Image const& img_input)
{
  int rows = img_input.getDimRow();
  int cols = img_input.getDimCol();
  img.set_size(rows, cols);
  double img_max = -1;
  for (int i = 0; i < rows; i ++){
    for (int j = 0; j < cols; j ++){
      img_max = img_input(i,j) > img_max? img_input(i,j) : img_max;
    }
  }
  
  for (int idx0 = 0; idx0 < img.size(0); idx0++)
  {
    for (int idx1 = 0; idx1 < img.size(1); idx1++)
    {
      img[idx0 + img.size(0) * idx1] = img_input(idx0, idx1) / img_max * 255;
    }
  }
//   omp_init_nest_lock(&emlrtNestLockGlobal);
//   extern void emlrtLockInitOnce();  // 在 data.cpp 里声明/放到公共头里也可
  emlrtLockInitOnce();
}

//
// Arguments    : void
// Return Type  : void
//
CircleIdentifier::~CircleIdentifier()
{
//   omp_destroy_nest_lock(&emlrtNestLockGlobal);
}

//
// File trailer for CircleIdentifier.cpp
//
// [EOF]
//
