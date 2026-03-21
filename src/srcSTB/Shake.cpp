#include "nanoflann.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>  // ★ floor, ceil, exp, cos, sin, fabs, round
#include <limits> // ★ std::numeric_limits
#include <omp.h>

#include "BubbleRefImg.h"
#include "BubbleResize.h"
#include "CircleIdentifier.h"
#include "Shake.h"

#define CORR_INIT -100

namespace {
inline bool isInvalidProjectedCenter(const Pt2D &center) {
  constexpr double kProjectFailSentinel = -1e9;
  const double x = center[0];
  const double y = center[1];
  if (!std::isfinite(x) || !std::isfinite(y)) {
    return true;
  }
  return (x <= kProjectFailSentinel || y <= kProjectFailSentinel);
}
} // namespace

// ---------------------------------------------------
Shake::Shake(const std::vector<std::shared_ptr<Camera>> &camera_models,
             const ObjectConfig &obj_cfg)
    : _cam_list(camera_models), _obj_cfg(obj_cfg)

{
  // Ensure strategy exists (construct elsewhere or plug in before calling)
  // Resolve by ObjectConfig::kind() only
  switch (_obj_cfg.kind()) {
  case ObjectKind::Tracer:
    _strategy = std::make_unique<TracerShakeStrategy>(_cam_list, _obj_cfg);
    break;
  case ObjectKind::Bubble:
    _strategy = std::make_unique<BubbleShakeStrategy>(_cam_list, _obj_cfg);
    break;
  default:
    THROW_FATAL(
        ErrorCode::UnsupportedType,
        "Shake::ensureStrategy: unsupported ObjectKind in ObjectConfig.");
  }
}

std::vector<ObjFlag>
Shake::runShake(std::vector<std::unique_ptr<Object3D>> &objs,
                const std::vector<Image> &img_orig) {
  // Basic sanity checks
  const int n_cam = static_cast<int>(_cam_list.size());
  assert(n_cam == static_cast<int>(img_orig.size()) &&
         "cams/img_orig size mismatch");

  const size_t n_obj = objs.size();
  std::vector<ObjFlag> flags(n_obj, ObjFlag::None);

  _img_res_list.clear();
  // Allocate residuals and per-object scores
  _img_res_list.assign(n_cam, Image{});
  _score_list.assign(n_obj, 1.0);

  // Shake schedule
  double delta = _obj_cfg._shake_param._shake_width; // initial Δ (mm)
  double dmin = _obj_cfg._shake_param._shakewidth_min;
  const int n_loop = _obj_cfg._shake_param._n_shake_loop;

  dmin = (dmin > 0) ? dmin : delta / 20;

  // 1) Ensure each object has up-to-date 2D projections for all cameras.
  //    NOTE: Replace the call below with your actual signature if different.
  for (auto &up : objs) {
    if (!up)
      continue;
    up->projectObject2D(_cam_list);
  }

  for (int it = 0; it < n_loop; ++it) {
    // 1) Build residual images for this iteration (Jacobi-style baseline)
    //    Residual fusion must follow strategy->fuseMode(): Overwrite or Min.
    calResidueImage(objs, img_orig);

// OpenMP parallel region (fallback to serial if OpenMP is not enabled)
#pragma omp parallel for if (!omp_in_parallel())
    for (std::ptrdiff_t i_obj = 0; i_obj < static_cast<std::ptrdiff_t>(n_obj);
         ++i_obj) {
      size_t i = static_cast<size_t>(i_obj);
      if (!objs[i])
        continue;
      // 2.1 Build per-object Aug (ROI = residual ROI + this object's
      // projection).
      std::vector<ROIInfo> roi = buildROIInfo(*objs[i], img_orig);
      // 2.2 Shake one object with current Δ (search {-Δ,0,+Δ} per axis, quad
      // fit, refine).
      //     Implementation detail: inside, only consider cameras that are:
      //     active && ROI non-empty && selected by
      //     strategy->selectShakeCam(...).
      // first, we need to determine which camera is used for shaking
      std::vector<bool> shake_cam =
          _strategy->selectShakeCam(*objs[i], roi, img_orig);
      int shake_cam_count =
          std::count(shake_cam.begin(), shake_cam.end(), true);
      if (shake_cam_count < 2) {
        _score_list[i] = 0.0;
        continue;
      } // need at least 2 cameras to shake

      (void)shakeOneObject(*objs[i], roi, delta, shake_cam);
      // 2.3 Compute object score (cross-camera aggregation inside your
      // calObjectScore).
      double score = calObjectScore(*objs[i], roi, shake_cam);
      _score_list[i] *= score;
    }
    // 4) Δ schedule (halve each loop; clamp to delta_min)
    delta *= 0.5;
    if (delta < dmin)
      delta = dmin;
  }
  // 5) Post-processing — mark repeated
  std::vector<bool> is_repeated = markRepeatedObj(objs);
  for (size_t i = 0; i < is_repeated.size(); ++i) {
    if (is_repeated[i])
      flags[i] |= ObjFlag::Repeated;
  }

  // 6) Post-processing — mark ghosts
  //    Here we threshold by relative-to-mean rule: score < score_min *
  //    mean(score)
  double sum = 0.0;
  size_t cnt = 0;
  for (size_t i = 0; i < _score_list.size(); ++i) {
    if (is_repeated[i])
      continue;
    sum += _score_list[i];
    ++cnt;
  }
  double mean_score = (cnt ? sum / cnt : 0.0);

  const double percent_ghost = _obj_cfg._shake_param._thred_ghost;

  for (size_t i = 0; i < n_obj; ++i) {
    if (!objs[i])
      continue;
    if (_score_list[i] < mean_score * percent_ghost || _score_list[i] == 0.0) {
      flags[i] |= ObjFlag::Ghost;
    }
  }

  // // 1) collect valid scores (exclude repeated and non-finite)
  // std::vector<double> valid_score; valid_score.reserve(_score_list.size());
  // for (size_t i = 0; i < _score_list.size(); ++i) {
  //     if (i < is_repeated.size() && is_repeated[i]) continue;
  //     const double s = _score_list[i];
  //     if (std::isfinite(s)) valid_score.push_back(s);
  // }

  // if (!valid_score.empty()) {
  //     const double percent_ghost = _obj_cfg._shake_param._thred_ghost;
  //     // 2) compute low_cut by KDE modal HDR + guards (grid size is chosen
  //     adaptively inside) const double low_cut =
  //     myMATH::computeLowCutKDE(valid_score, 1 - percent_ghost);

  //     // 3) apply: ONLY delete left side (score < low_cut)
  //     for (size_t i = 0; i < _score_list.size(); ++i) {
  //         const double s = _score_list[i];
  //         if (!std::isfinite(s)) continue;
  //         if (s < low_cut) {
  //             flags[i] |= ObjFlag::Ghost;
  //         }
  //     }
  // }

  return flags;
}

void Shake::calResidueImage(const std::vector<std::unique_ptr<Object3D>> &objs,
                            const std::vector<Image> &img_orig, bool output_ipr,
                            const std::vector<ObjFlag> *flags) {
  const int n_cam = static_cast<int>(_cam_list.size());
  // std::cout << n_cam << std::endl;

  REQUIRE(n_cam == static_cast<int>(img_orig.size()),
          ErrorCode::InvalidArgument,
          "Shake::calResidueImage: cams/img_orig size mismatch.");
  const bool use_mask = (flags != nullptr);
  if (use_mask) {
    REQUIRE(objs.size() == flags->size(), ErrorCode::InvalidArgument,
            "The number of objects doesn't match the number of object flags!");
  }

  // 2) Copy originals to residual buffers (one per camera)
  _img_res_list = img_orig; // deep copy; if your Image doesn't support
                            // operator= as deep copy, use explicit copy

// 4) Parallelize across cameras (each thread owns one residual image)
//    You can also parallelize rows inside the camera loop if Images are big.
#pragma omp parallel for if (!omp_in_parallel())
  for (int k = 0; k < n_cam; ++k) {
    // 4.1 Skip inactive cameras, but keep slot alignment (residual stays as
    // original) for IPR output, we need to calculate all cameras
    if (!_cam_list[k]->is_active && !output_ipr)
      continue;

    Image &res = _img_res_list[k]; // get the reference
    const Image &orig = img_orig[k];

    // 4.2 For each object, subtract its projection over its ROI only
    for (size_t id_obj = 0; id_obj < objs.size(); ++id_obj) {
      if (use_mask) {
        if ((*flags)[id_obj] != ObjFlag::None)
          continue; // skip ghost and repeated objects (flagged objects)
      }
      const Object3D *obj = objs[id_obj].get();

      // --- Obtain ROI center from object's 2D projection; size from strategy
      // ---
      REQUIRE(obj->_obj2d_list[k] != nullptr, ErrorCode::InvalidArgument,
              "No 2D projection in object.");
      Pt2D pt_center = obj->_obj2d_list[k]->_pt_center;
      double cx = pt_center[0], cy = pt_center[1];

      // Strategy returns (dx, dy) = half width/height (in pixels) for the ROI
      // If your calROISize returns std::vector<double>, read [0],[1]; or change
      // to a struct.
      const auto sz = _strategy->calROISize(*obj, k);
      double dx = sz.dx, dy = sz.dy;

      // Compute and clamp ROI bounds to the image
      // projection size: one object size
      const PixelRange roi = calROIBound(k, cx, cy, dx, dy);
      if (roi.row_max < roi.row_min || roi.col_max < roi.col_min)
        continue; // empty ROI

      // 4.2.1 Iterate pixels in ROI and fuse projection into residual
      //       NOTE: replace res.at(r,c) / orig.at(r,c) / projection accessor
      //       with your actual image API.
      for (int r = roi.row_min; r < roi.row_max; ++r) {
        for (int c = roi.col_min; c < roi.col_max; ++c) {
          const double p = _strategy->project2DInt(*obj, k, r, c);
          if (p == 0.0)
            continue; // cheap skip
          // "Min" fusion for all types:
          // residual := min(current residual, orig - projection_of_this_object)
          double &rr = res(r, c);
          const double o = orig(r, c);
          const double cand = o - p;
          if (cand < rr)
            rr = cand;
          if (output_ipr && rr < 0)
            rr = 0.0; // IPR output: clamp negative to 0
        }
      }
    } // end for each object
  } // end per-camera loop
}

PixelRange Shake::calROIBound(int id_cam, double cx, double cy, double dx,
                              double dy) const {
  const int H = _cam_list[id_cam]->getNRow();
  const int W = _cam_list[id_cam]->getNCol();

  if (H <= 0 || W <= 0 || dx <= 0.0 || dy <= 0.0 || cx <= 0 || cy <= 0 ||
      cx >= W || cy >= H) {        // for points that are out of range
    return PixelRange{1, 0, 1, 0}; // empty range
  }

  const int width = static_cast<int>(std::round(2.0 * dx));
  const int height = static_cast<int>(std::round(2.0 * dy));

  int cmin = static_cast<int>(
      std::round(cx - (width - 1) / 2.0)); // this formula guaranttee cx is
                                           // closest to the center of the range
  int rmin = static_cast<int>(std::round(
      cy - (height - 1) / 2.0)); // height is int, 2 must be written 2.0 to let
                                 // (H - 1)/2.0 becomes double
  int cmax = cmin + width;       // [cmin, cmax)
  int rmax = rmin + height;      //[rmin, rmax)

  rmin = std::max(0, rmin);
  cmin = std::max(0, cmin);
  rmax = std::min(H, rmax);
  cmax = std::min(W, cmax);

  if (rmax < rmin || cmax < cmin) {
    return PixelRange{1, 0, 1, 0};
  }
  return PixelRange{rmin, rmax, cmin, cmax}; // [min, max)
}

std::vector<ROIInfo>
Shake::buildROIInfo(const Object3D &obj,
                    const std::vector<Image> &img_orig) const {
  const int n_cam = static_cast<int>(_cam_list.size());
  std::vector<ROIInfo> roi_info;
  roi_info.resize(n_cam);

  for (int k = 0; k < n_cam; ++k) {
    // keep slot alignment
    if (!_cam_list[k]->is_active || k >= static_cast<int>(obj._obj2d_list.size()) ||
        !obj._obj2d_list[k]) {
      continue; // do not initialize ROI info
    }

    const Pt2D &ctr = obj._obj2d_list[k]->_pt_center; // (x=col, y=row)

    // ROI half size from strategy (假设已改成 ROISize {dx,dy})
    const auto sz = _strategy->calROISize(obj, k);
    const double dx = std::max(0.0, sz.dx);
    const double dy = std::max(0.0, sz.dy);

    double ratio_region = _obj_cfg._shake_param._ratio_augimg;
    const PixelRange roi =
        calROIBound(k, ctr[0], ctr[1], dx * ratio_region, dy * ratio_region);
    roi_info[k]._ROI_range = roi;

    if (roi.row_max < roi.row_min || roi.col_max < roi.col_min) {
      continue;
    }

    // allocate Aug & corr map
    roi_info[k].allocAugImg();
    roi_info[k].allocCorrMap();

    // creating augmented image
    // project back the object but within the range of project (one object size)
    const PixelRange range_project = calROIBound(k, ctr[0], ctr[1], dx, dy);
    for (int r = roi.row_min; r < roi.row_max; ++r)
      for (int c = roi.col_min; c < roi.col_max; ++c) {
        double aug = _img_res_list[k](r, c);
        if (r >= range_project.row_min && r < range_project.row_max &&
            c >= range_project.col_min && c < range_project.col_max)
          aug += _strategy->project2DInt(obj, k, r, c);

        roi_info[k].aug_img(r, c) =
            std::max(0.0, aug); // augmented image should be non-negative
      }
  }
  return roi_info;
}

/*
shake one object, input delta is the shake width, return the score for updating,
obj is also updated double Shake::shakeOneObject(Object3D& obj,
std::vector<ROIInfo>& ROI_info, double delta, const std::vector<bool>&
shake_cam) const
{
    std::vector<double> delta_list = { -delta, 0.0, +delta };
    std::vector<double> array_list(4, 0.0);          // x0,x1,x2,x*
    std::vector<double> array_list_fit(3, 0.0);      // for polyfit (x0,x1,x2)
    std::vector<double> coeff(3, 0.0);               // quad coeffs
    std::vector<double> residue_list(4, 0.0);        // f0,f1,f2,f*
    std::vector<double> residue_list_fit(3, 0.0);    // for polyfit

    // shaking on x,y,z direction
    double residue = 0.0;

    CreateArgs args;
    args._proto = &obj;
    std::unique_ptr<Object3D> obj3d_temp =
_obj_cfg.creatObject3D(std::move(args)); // create a temporary object for
shaking

    for (int i = 0; i < 3; i ++)
    {
        for (int j = 0; j < 3; j ++)
        {
            array_list[j] = obj._pt_center[i] + delta_list[j];
            array_list_fit[j] = array_list[j];

            obj3d_temp->_pt_center[i] = array_list[j];

            // update 2D information
            obj3d_temp->projectObject2D(_cam_list);

            residue_list[j] = _strategy->calShakeResidue(*obj3d_temp, ROI_info,
shake_cam); residue_list_fit[j] = residue_list[j];
        }

        // residue = coeff[0] + coeff[1] * x + coeff[2] * x^2
        myMATH::polyfit(coeff, array_list_fit, residue_list_fit, 2);

        // 计算区间内的顶点 x*
        bool has_star = false;
        if (coeff[2] != 0.0) {
            array_list[3] = - coeff[1] / (2.0 * coeff[2]);
            if (array_list[3] > array_list[0] && array_list[3] < array_list[2])
{ obj3d_temp->_pt_center[i] = array_list[3]; obj3d_temp->projectObject2D(_cam_list);
                residue_list[3] = _strategy->calShakeResidue(*obj3d_temp,
ROI_info, shake_cam); has_star = true; } else { residue_list[3] =
std::numeric_limits<double>::infinity(); // safer than "+1"
            }
        } else {
            residue_list[3] = std::numeric_limits<double>::infinity();
        }

        // 选最小残差的位置（含 x* 如有效）
        int min_id = 0;
        double min_val = residue_list[0];
        int t_max = (has_star ? 4 : 3);
        for (int t = 1; t < t_max; ++t) {
            if (residue_list[t] < min_val) { min_val = residue_list[t]; min_id =
t; }
        }

        // update obj
        obj3d_temp->_pt_center[i] = array_list[min_id];
        obj._pt_center[i] = array_list[min_id];
        residue = residue_list[min_id];
    }

    // TODO: remove camera with residue > thredshold, and redo shakeoneobject

    // update 2D information for next loop of shaking
    obj.projectObject2D(_cam_list);

    return residue;
}
*/

double Shake::shakeOneObject(Object3D &obj, std::vector<ROIInfo> &ROI_info,
                             double delta,
                             const std::vector<bool> &shake_cam) const {
  // ---------- small helpers ----------
  auto median = [](std::vector<double> &v) -> double {
    if (v.empty())
      return 0.0;
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];
    if ((n & 1) == 0) {
      std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
      m = 0.5 * (m + v[mid - 1]);
    }
    return m;
  };
  auto inROI = [&](int k, double u, double v) -> bool {
    const auto &rr = ROI_info[k]._ROI_range;
    return (rr.col_min <= u && u < rr.col_max && rr.row_min <= v &&
            v < rr.row_max);
  };

  // Reusable temp object for function evaluations
  CreateArgs args;
  args._proto = &obj;
  std::unique_ptr<Object3D> obj_tmp = _obj_cfg.creatObject3D(std::move(args));

  // Return the best residue on the last axis (keeps your original behavior)
  double residue_axis = 0.0;

  // ---------- process X, Y, Z axes ----------
  for (int ax = 0; ax < 3; ++ax) {
    const double x0 =
        obj._pt_center[ax]; // absolute world coordinate on this axis
    const double DL = -delta, D0 = 0.0, DR = +delta;
    const double sample_rel[3] = {DL, D0, DR};

    // Cache: three samples' residues and per-camera 2D points (for Jx /
    // margins)
    double f[3] = {0, 0, 0};

    struct UV {
      double u = 0, v = 0;
      bool ok = false;
    };
    std::vector<UV> uv[3]; // uv[0]=at -Δ, uv[1]=at 0, uv[2]=at +Δ
    for (int j = 0; j < 3; ++j)
      uv[j].resize(_cam_list.size());

    // ---- evaluate (-Δ, 0, +Δ) in a tight loop (no repetition) ----
    for (int j = 0; j < 3; ++j) {
      const double x_abs = x0 + sample_rel[j];
      obj_tmp->_pt_center[ax] = x_abs;
      obj_tmp->projectObject2D(_cam_list);
      f[j] = _strategy->calShakeResidue(*obj_tmp, ROI_info, shake_cam);

      // record UV for each camera at this sample (used to estimate Jx and
      // margins later)
      for (size_t k = 0; k < _cam_list.size(); ++k) {
        if (!shake_cam[k])
          continue;
        const auto &o2d = obj_tmp->_obj2d_list[k];
        const double u = o2d->_pt_center[0];
        const double v = o2d->_pt_center[1];
        uv[j][k].u = u;
        uv[j][k].v = v;
        uv[j][k].ok = inROI((int)k, u, v);
      }
    }

    // ==========================  adaptive shadth width adjustment
    // ===========================//
    // ---- basic stats from the 3 samples ----
    const double fL = f[0], f0 = f[1], fR = f[2];
    const double tol = 1e-3 * std::max(1.0, f0);

    const bool mono_dec =
        (fL > f0 + tol) && (f0 > fR + tol); // monotone decreasing along +x
    const bool mono_inc =
        (fR > f0 + tol) && (f0 > fL + tol); // monotone increasing along +x
    const bool mono = mono_dec || mono_inc;
    int s = mono_dec ? +1 : (mono_inc ? -1 : 0);

    // second difference and central slope (relative-x form)
    const double kappa = (fL - 2.0 * f0 + fR);                 // = a * Δ^2
    const double g = (fR - fL) / std::max(2.0 * delta, 1e-12); // ≈ f'(0)

    // candidate pool (relative-x) and absolute positions
    std::vector<double> cand_x = {DL, 0.0, DR};
    std::vector<double> cand_f = {fL, f0, fR};
    std::vector<double> cand_abs{x0 + DL, x0, x0 + DR};

    // ---------- Gate: if minimum is clearly inside [-Δ,Δ], DO NOT EXPAND
    // ----------
    bool skip_expand = false;
    bool allow_eval_xhat =
        true; // see if it is needed to calculate mimimum point at final step

    if (kappa > 0.0) {                                     // convex only
      const double x3 = delta * (fL - fR) / (2.0 * kappa); // relative
      const double beta_in = 0.95; // clearly inside [-Δ,Δ]
      if (std::abs(x3) <=
          beta_in * delta) { // minimum lies within [-Δ,Δ] skip_expand = true;
        // predict the relative improvment   rho_pred = g^2 / (2 a max(f0,1)), a
        // = kappa/Δ^2
        const double a = kappa / (delta * delta + 1e-12);
        const double rho_pred = (g * g) / (2.0 * a * std::max(f0, 1.0));
        const double eta_small = 8e-4;

        skip_expand = true;

        // improvment is trivial, and calculation for minimum point is not need
        if (rho_pred < eta_small) {
          allow_eval_xhat = false;
        }
      }
    }

    // ---------- Single expansion (only if not gated AND monotonic) ----------
    if (!skip_expand && mono) {
      // Jx is when the particle moves 1 mm, how many pixels it moves on image
      // Estimate Jx (px/mm) from ±Δ samples (reuse uv cache); estimate pixel
      // margin at center.
      std::vector<double> Jvals, margins;
      Jvals.reserve(_cam_list.size());
      margins.reserve(_cam_list.size());
      for (size_t k = 0; k < _cam_list.size(); ++k) {
        if (!shake_cam[k])
          continue;

        if (uv[0][k].ok && uv[2][k].ok) {
          const double du = uv[2][k].u - uv[0][k].u;
          const double dv = uv[2][k].v - uv[0][k].v;
          const double Jk = std::hypot(du, dv) / std::max(2.0 * delta, 1e-12);
          Jvals.push_back(Jk);
        }
        if (uv[1][k].ok) {
          const auto &rr = ROI_info[k]._ROI_range;
          const double mu =
              std::min(uv[1][k].u - rr.col_min, rr.col_max - 1.0 - uv[1][k].u);
          const double mv =
              std::min(uv[1][k].v - rr.row_min, rr.row_max - 1.0 - uv[1][k].v);
          margins.push_back(std::min(mu, mv));
        }
      }
      double Jx = median(Jvals);
      double mpx = median(margins);
      const double Delta_roi = (Jx > 0.0 && mpx > 0.0) ? (0.5 * mpx / Jx) : 0.0;

      // Choose Δ_try:
      //  A) convex & |x*|>Δ -> direction by sign(x*), with "1px cap"
      //  B) otherwise -> 1px step along monotonic direction
      double Delta_try = 0.0;
      if (kappa > 0.0) {
        const double x3 = delta * (fL - fR) / (2.0 * kappa);
        if (std::abs(x3) > delta) {
          s = (x3 > 0.0 ? +1 : -1);
          if (Jx > 0.0 && std::abs(x3) > (1.0 / Jx)) {
            Delta_try = 1.0 / Jx; // cap by 1 pixel (world units)
          } else {
            const double alpha = 1.15;
            Delta_try = alpha * std::abs(x3);
          }
        }
      }
      if (Delta_try <= 0.0 && Jx > 0.0) {
        Delta_try = 1.0 / Jx; // default 1px
      }

      // Apply ROI limit and evaluate expansion point at most once
      const double Delta_prime = (Delta_try > 0.0 && Delta_roi > 0.0)
                                     ? std::min(Delta_try, Delta_roi)
                                     : 0.0;
      if (Delta_prime > 0.0) {
        const double dx_exp = (s >= 0 ? +Delta_prime : -Delta_prime);
        obj_tmp->_pt_center[ax] = x0 + dx_exp;
        obj_tmp->projectObject2D(_cam_list);
        bool out = false;
        for (size_t k = 0; k < _cam_list.size(); ++k) {
          if (!shake_cam[k])
            continue;
          const auto &o2d = obj_tmp->_obj2d_list[k];
          const double u = o2d->_pt_center[0], v = o2d->_pt_center[1];
          if (!inROI((int)k, u, v)) {
            out = true;
            break;
          }
        }
        if (!out) {
          const double fexp =
              _strategy->calShakeResidue(*obj_tmp, ROI_info, shake_cam);
          cand_x.push_back(dx_exp);
          cand_f.push_back(fexp);
          cand_abs.push_back(x0 + dx_exp);
        }
      }
    }
    // else: non-monotonic or gated -> no expansion; go to final step.

    // ---------- Unified final step: LS quadratic over evaluated candidates
    // ----------
    std::vector<double> coeff(3, 0.0);
    myMATH::polyfit(coeff, cand_x, cand_f,
                    2); // coeff[0] + coeff[1] x + coeff[2] x^2

    if (allow_eval_xhat && coeff[2] > 0.0) {             // only if convex
      const double x_hat = -coeff[1] / (2.0 * coeff[2]); // relative-x
      double xmin = cand_x[0], xmax = cand_x[0];
      for (double t : cand_x) {
        xmin = std::min(xmin, t);
        xmax = std::max(xmax, t);
      }
      if (x_hat >= xmin && x_hat <= xmax) {
        obj_tmp->_pt_center[ax] = x0 + x_hat;
        obj_tmp->projectObject2D(_cam_list);
        bool out = false;
        for (size_t k = 0; k < _cam_list.size(); ++k) {
          if (!shake_cam[k])
            continue;
          const auto &o2d = obj_tmp->_obj2d_list[k];
          const double u = o2d->_pt_center[0], v = o2d->_pt_center[1];
          if (!inROI((int)k, u, v)) {
            out = true;
            break;
          }
        }
        if (!out) {
          const double fhat =
              _strategy->calShakeResidue(*obj_tmp, ROI_info, shake_cam);
          cand_x.push_back(x_hat);
          cand_f.push_back(fhat);
          cand_abs.push_back(x0 + x_hat);
        }
      }
    }

    // ---------- pick the best evaluated candidate ----------
    int best_id = 0;
    double best_f = cand_f[0];
    for (int t = 1; t < (int)cand_f.size(); ++t) {
      if (cand_f[t] < best_f) {
        best_f = cand_f[t];
        best_id = t;
      }
    }

    // Update this axis and refresh 2D before next axis
    obj._pt_center[ax] = cand_abs[best_id];
    residue_axis = best_f;
    obj.projectObject2D(_cam_list);
  }

  // Final projection for next outer iteration
  obj.projectObject2D(_cam_list);
  return residue_axis;
}

// calculate score based on the intensity
double Shake::calObjectScore(Object3D &obj, std::vector<ROIInfo> &ROI_info,
                             const std::vector<bool> &shake_cam) const {
  // check whether object is present in all selected cameras
  if (!_strategy->additionalObjectCheck(obj, ROI_info, shake_cam)) {
    return 0.0;
  }
  const int n_cam = static_cast<int>(_cam_list.size());
  constexpr double kTiny = 1e-12; // avoid 0
  constexpr double kDown =
      kTiny; // if camera number is less than 2, return this

  // per-camera
  std::vector<PixelRange> score_rect(
      n_cam); // region for evaluating the intensity
  std::vector<int> use_cam(
      n_cam, 0); // evaluate cameras only without the highest intensity
  std::vector<double> peak(n_cam, 0.0);

  int n_used = 0;
  for (int k = 0; k < n_cam; ++k) {
    if (!_cam_list[k]->is_active)
      continue;

    // region: one object size
    const auto sz = _strategy->calROISize(obj, k);
    const double dx = std::max(0.0, sz.dx);
    const double dy = std::max(0.0, sz.dy);
    if (dx <= 0.0 || dy <= 0.0)
      continue;

    const Pt2D &ctr = obj._obj2d_list[k]->_pt_center; // (x=col, y=row)
    score_rect[k] = calROIBound(k, ctr[0], ctr[1], dx, dy);

    const bool empty = (score_rect[k].row_max < score_rect[k].row_min) ||
                       (score_rect[k].col_max < score_rect[k].col_min);
    if (empty)
      continue; // if no region found on camera, skip this camera

    use_cam[k] = 1;
    ++n_used;

    // get the highest peak intensity for this camera
    for (int r = score_rect[k].row_min; r < score_rect[k].row_max; ++r) {
      for (int c = score_rect[k].col_min; c < score_rect[k].col_max; ++c) {
        const double val =
            ROI_info[k].inRange(r, c)
                ? ROI_info[k].aug_img(r, c)
                : _img_res_list[k](r, c); // residual may contain negative
                                          // value, but it doesn't matter here
        if (val > peak[k])
          peak[k] = val;
      }
    }
  }

  // if less than 2 camera
  if (n_used < 2)
    return kDown;

  // remove one of the camera with highest intensity, if camera number > 2
  if (n_used > 2) {
    int max_id = -1;
    double max_v = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < n_cam; ++k) {
      if (use_cam[k] && peak[k] > max_v) {
        max_v = peak[k];
        max_id = k;
      }
    }
    if (max_id >= 0) {
      use_cam[max_id] = 0;
      --n_used;
    }
  }

  // calculate intensity ratio for each camera r_k = |sum_measured / sum_model|
  double total_numer = 0.0;
  double total_denom = 0.0;

  for (int k = 0; k < n_cam; ++k) {
    if (!use_cam[k])
      continue;
    const auto &R = score_rect[k];

    double numer = 0.0; // sum of measured intensity
    double denom = 0.0; // sum of predicted intensity

    for (int r = R.row_min; r < R.row_max; ++r) {
      for (int c = R.col_min; c < R.col_max; ++c) {
        // measured：ROI 内用 Aug，外用 residual（可能为负）
        double meas = ROI_info[k].inRange(r, c) ? ROI_info[k].aug_img(r, c)
                                                : _img_res_list[k](r, c);
        meas = std::max(0.0, meas); // since residual can be negative due to
                                    // overlap particle, we set it 0
        numer += meas;

        // predict intensity, using same projection model as in calResidualImage
        const double pred = _strategy->project2DInt(obj, k, r, c);
        denom += pred;
      }
    }
    total_numer += numer;
    total_denom += denom;
  }

  // Calculate aggregated ratio
  double ratio = (total_denom > kTiny) ? (total_numer / total_denom) : 1.0;

  // Paper: "capping the intensity ratio at 3/2 and 2/3"
  ratio = std::max(2.0 / 3.0, std::min(1.5, ratio));

  // Paper: "Taking the root of the intensity ratio proved to dampen intensity
  // oscillations"
  return std::sqrt(ratio);
}

// mark repeated objects
std::vector<bool>
Shake::markRepeatedObj(const std::vector<std::unique_ptr<Object3D>> &objs) {
  int n_obj = objs.size();
  std::vector<bool> is_repeated;
  is_repeated.resize(n_obj, false);
  int n_repeated = 0;

  // Build KD tree for fast neighbor search
  using KDTreeObj3d = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, Obj3dCloud, double,
                                   size_t>, // 显式写 DistanceType 和 IndexType
      Obj3dCloud, 3, size_t>;

  Obj3dCloud obj3d_cloud(objs);
  nanoflann::KDTreeSingleIndexAdaptorParams params(10); // 显式构造
  KDTreeObj3d tree_obj3d(3, obj3d_cloud, params);
  tree_obj3d.buildIndex();

  // mark repeated objects
  double tol_3d = _obj_cfg._sm_param.tol_3d_mm;
  double repeat_thres_2 = tol_3d * tol_3d;
  for (int i = 0; i < n_obj - 1; i++) {
    if (is_repeated[i]) {
      continue;
    }

    std::vector<nanoflann::ResultItem<size_t, double>> indices_dists;
    nanoflann::RadiusResultSet<double, size_t> resultSet(repeat_thres_2,
                                                         indices_dists);
    tree_obj3d.findNeighbors(resultSet, objs[i]->_pt_center.data(),
                             nanoflann::SearchParameters());

    for (int j = 1; j < resultSet.size(); j++) {
      is_repeated[indices_dists[j].first] = 1;
      n_repeated++;
    }
  }
  return is_repeated;
}

//------------------------------ROIINfo----------------------------//
// allocate augmented image for cam id_cam
void ROIInfo::allocAugImg() {
  const int roi_h = _ROI_range.getNumOfRow();
  const int roi_w = _ROI_range.getNumOfCol();
  _ROI_augimg = Image(roi_h, roi_w, 0.0);
}

void ROIInfo::allocCorrMap() {
  const int roi_h = _ROI_range.getNumOfRow();
  const int roi_w = _ROI_range.getNumOfCol();
  _ROI_corrmap =
      Image(roi_h, roi_w, CORR_INIT); // set value to #define CORR_INIT -100
}

bool ROIInfo::inRange(int row, int col) const {
  if (_ROI_range.getNumOfRow() <= 0 || _ROI_range.getNumOfCol() <= 0)
    return false;

  // 包含判断（左闭右开）
  if (row < _ROI_range.row_min || row >= _ROI_range.row_max ||
      col < _ROI_range.col_min || col >= _ROI_range.col_max) {
    return false;
  }

  return true;
}

bool ROIInfo::mapToLocal(int row, int col, int &i, int &j) const {
  if (!inRange(row, col))
    return false;

  // 全局(row,col) → 局部(i,j)
  i = row - _ROI_range.row_min;
  j = col - _ROI_range.col_min;
  return true;
}

double &ROIInfo::aug_img(int row, int col) {
  int i = -1, j = -1;
  const bool inside = mapToLocal(row, col, i, j);
  assert(inside &&
         "aug_img(): (row,col) is outside ROI for this camera or ROI is empty");

  return _ROI_augimg(i, j);
}

double ROIInfo::aug_img(int row, int col) const {
  int i = -1, j = -1;
  const bool inside = mapToLocal(row, col, i, j);
  assert(inside && "aug_img() const: (row,col) is outside ROI for this camera "
                   "or ROI is empty");

  return _ROI_augimg(i, j);
}

double &ROIInfo::corr_map(int row, int col) {
  int i = -1, j = -1;
  const bool inside = mapToLocal(row, col, i, j);
  assert(
      inside &&
      "corr_map(): (row,col) is outside ROI for this camera or ROI is empty");

  return _ROI_corrmap(i, j);
}

double ROIInfo::corr_map(int row, int col) const {
  int i = -1, j = -1;
  const bool inside = mapToLocal(row, col, i, j);
  assert(inside && "corr_map() const: (row,col) is outside ROI for this camera "
                   "or ROI is empty");

  return _ROI_corrmap(i, j);
}

//----------------------------TracerShakeStrategy---------------------------
double TracerShakeStrategy::gaussIntensity(
    int x, int y, Pt2D const &pt2d,
    std::vector<double> const &otf_param) const {
  double dx = x - pt2d[0];
  double dy = y - pt2d[1];

  const double A = otf_param[0], B = otf_param[1], C = otf_param[2];
  const double cs = otf_param[3], sn = otf_param[4]; // cos(theta), sin(theta)

  const double xx = dx * cs + dy * sn;
  const double yy = -dx * sn + dy * cs;
  double value = A * std::exp(-(B * xx * xx + C * yy * yy));
  return std::max(0.0, value);
}

double TracerShakeStrategy::project2DInt(const Object3D &obj, int id_cam,
                                         int row, int col) const {
  const auto &tr_cfg = static_cast<const TracerConfig &>(_obj_cfg);

  // std::vector<double> otf_param = tr_cfg._otf.getOTFParam(id_cam,
  // obj._pt_center); // calculate OTF param for each pixel is too slow

  // save the otf_param in thread local storage for reuse
  struct Cache {
    const Object3D *obj = nullptr;
    int cam = -1;
    double center3d[3]{0, 0, 0};
    std::vector<double> otf;
  };
  static thread_local Cache cache; // get the current thread's cache

  // check if need to update the cache
  auto center_changed = [&] {
    const double *p = obj._pt_center.data();
    return p[0] != cache.center3d[0] || p[1] != cache.center3d[1] ||
           p[2] != cache.center3d[2];
  };

  if (cache.obj != &obj || cache.cam != id_cam || center_changed()) {
    cache.obj = &obj;
    cache.cam = id_cam;
    const double *p = obj._pt_center.data();
    cache.center3d[0] = p[0];
    cache.center3d[1] = p[1];
    cache.center3d[2] = p[2];

    cache.otf = static_cast<const TracerConfig &>(_obj_cfg)._otf.getOTFParam(
        id_cam, obj._pt_center);
  }

  return gaussIntensity(col, row, obj._obj2d_list[id_cam]->_pt_center,
                        cache.otf);
}

ShakeStrategy::ROISize TracerShakeStrategy::calROISize(const Object3D &obj,
                                                       int id_cam) const {
  ROISize roi_size;
  Object2D *obj2d = obj._obj2d_list[id_cam]
                        .get(); // must get the raw pointer from the unique_ptr
                                // because unique_ptr cannot be casted directly
  auto *tr = static_cast<Tracer2D *>(obj2d);
  roi_size.dx = tr->_r_px;
  roi_size.dy = tr->_r_px;

  return roi_size;
}

double
TracerShakeStrategy::calShakeResidue(const Object3D &obj_candidate,
                                     std::vector<ROIInfo> &ROI_info,
                                     const std::vector<bool> &shake_cam) const {
  // For Tracer: per-camera MSE on the fixed ROI, then arithmetic mean over
  // usable cameras. Usable camera = ROI non-empty (cameras already filtered
  // upstream by _is_active when building ROI). Read measured intensity from ROI
  // Aug via roi_info.aug_img(cam, row, col) using FULL image indices. Predicted
  // intensity comes from project2DInt(obj_candidate, cam, row, col) (≥0; 3σ
  // cutoff inside). If no usable camera exists, return a large sentinel cost.

  const int n_cam = static_cast<int>(_cam_list.size());
  int cams_used = 0;
  double cost_acc = 0.0;

  for (int cam = 0; cam < n_cam; ++cam) {
    if (!_cam_list[cam]->is_active)
      continue;
    if (!shake_cam[cam])
      continue;
    const PixelRange &ROI_range = ROI_info[cam]._ROI_range;
    const int nrows = ROI_range.getNumOfRow();
    const int ncols = ROI_range.getNumOfCol();

    // Skip empty ROI
    if (nrows <= 0 || ncols <= 0)
      continue;

    double sse = 0.0;
    std::size_t pix = 0;

    // this is used for seting range of calculating intensity
    Object2D *obj2d = obj_candidate._obj2d_list[cam].get();
    auto *tr = static_cast<Tracer2D *>(obj2d);
    Pt2D obj2d_center = tr->_pt_center;
    if (isInvalidProjectedCenter(obj2d_center)) {
      continue;
    }
    double r_px = tr->_r_px;
    int pred_row_min = std::floor(obj2d_center[1] - r_px),
        pred_row_max = std::ceil(obj2d_center[1] + r_px + 1);
    int pred_col_min = std::floor(obj2d_center[0] - r_px),
        pred_col_max = std::ceil(obj2d_center[0] + r_px + 1);

    // Iterate over the ROI in FULL image coordinates (left-closed, right-open:
    // [min, max))
    for (int row = ROI_range.row_min; row < ROI_range.row_max; ++row) {
      for (int col = ROI_range.col_min; col < ROI_range.col_max; ++col) {
        // Measured Aug value from ROI (the accessor maps full (row,col) to
        // local (i,j) with checks)
        const double meas = ROI_info[cam].aug_img(row, col);

        double pred = 0.0;
        // Forward model prediction for this pixel (Gaussian PSF; returns 0
        // outside object size)
        if (row >= pred_row_min && row < pred_row_max && col >= pred_col_min &&
            col < pred_col_max)
          pred = project2DInt(obj_candidate, cam, row, col);

        const double d = meas - pred;
        sse += d * d;
        ++pix;
      }
    }

    // Guard against degenerate ROIs (should not happen if ranges are valid)
    if (pix > 0) {
      // Per-camera MSE to avoid scale drift with different ROI sizes
      const double mse = sse / static_cast<double>(pix);
      cost_acc += mse;
      ++cams_used;
    }
  }

  // Aggregate across cameras: arithmetic mean; if none usable, return a large
  // penalty
  return (cams_used >
          1) // must have at least 2 cameras to calculate the residue
             ? (cost_acc / static_cast<double>(cams_used))
             : std::numeric_limits<double>::infinity();
}

//----------------------------BubbleShakeStrategy---------------------------

double BubbleShakeStrategy::project2DInt(const Object3D &obj, int id_cam,
                                         int row, int col) const {
  Object2D *obj2d = obj._obj2d_list[id_cam].get();
  auto *bb2d = static_cast<Bubble2D *>(obj2d);
  double xc = bb2d->_pt_center[0], yc = bb2d->_pt_center[1];
  double r_px = bb2d->_r_px + 1; // +1 to include the edge pixels

  double dist = (xc - col) * (xc - col) + (yc - row) * (yc - row);
  double int_val = 0.0;
  if (dist < r_px * r_px) {
    int_val = _cam_list[id_cam]->max_intensity;
  }

  return int_val;
}

ShakeStrategy::ROISize BubbleShakeStrategy::calROISize(const Object3D &obj,
                                                       int id_cam) const {
  Object2D *obj2d = obj._obj2d_list[id_cam].get();
  auto *bb2d = static_cast<Bubble2D *>(obj2d);
  double r_px = bb2d->_r_px;
  ROISize roi_size;
  roi_size.dx = r_px;
  roi_size.dy = r_px;
  return roi_size;
}

std::vector<bool>
BubbleShakeStrategy::selectShakeCam(const Object3D &obj,
                                    const std::vector<ROIInfo> &roi_info,
                                    const std::vector<Image> &imgOrig) const {
  const int n_cam = static_cast<int>(_cam_list.size());
  std::vector<bool> shake_cam;
  shake_cam.assign(n_cam, true);

  for (int cam = 0; cam < n_cam; ++cam) {
    if (!_cam_list[cam]->is_active) {
      shake_cam[cam] = false;
      continue;
    }

    const PixelRange &region = roi_info[cam]._ROI_range;

    // 1) ROI validity (half-open [min, max))
    const int n_row = region.getNumOfRow();
    const int n_col = region.getNumOfCol();
    if (n_row <= 0 || n_col <= 0) {
      shake_cam[cam] = false;
      continue;
    }

    // 2) Extract original subimage (ROI)
    Image imgOrig_sub(n_row, n_col, 0.0);
    for (int i = 0; i < n_row; ++i) {
      for (int j = 0; j < n_col; ++j) {
        imgOrig_sub(i, j) =
            imgOrig[cam](region.row_min + i, region.col_min + j);
      }
    }

    // 3) Get bubble 2D info on this camera
    const Object2D *base = obj._obj2d_list[cam].get();
    const auto *bb2d = dynamic_cast<const Bubble2D *>(base);
    const double r_px = bb2d->_r_px;

    // 4) Run circle detection (small-radius branch uses centered square crop +
    // 2x upsampling)
    std::vector<Pt2D> centers;
    std::vector<double> radii;
    std::vector<double> metrics;
    double rmin = 2.0, rmax = 0.0, sense = 0.95;

    if (r_px < 5.0) {
      // 4.a) center-crop to a square: size = min(n_row, n_col)
      const int npix = std::min(n_row, n_col);
      Image img_ref(npix, npix, 0.0);

      const int r0 = (n_row - npix) / 2;
      const int c0 = (n_col - npix) / 2;
      for (int y = 0; y < npix; ++y)
        for (int x = 0; x < npix; ++x)
          img_ref(y, x) = imgOrig_sub(r0 + y, c0 + x);

      // 4.b) 2x upsample
      const int img_size = 2 * npix;
      BubbleResize bb_resizer;
      const Image img_up =
          bb_resizer.ResizeBubble(img_ref, img_size,
                                  _cam_list[cam]->max_intensity);

      // 4.c) detect circles on upsampled image
      CircleIdentifier circle_id(img_up);
      rmin = 2.0;
      rmax = std::ceil(r_px) * 2.0 + 6.0; // keep your original heuristic
      sense = 0.95;
      metrics = circle_id.BubbleCenterAndSizeByCircle(centers, radii, rmin,
                                                      rmax, sense);

      // 4.d) restore to original scale
      for (size_t i = 0; i < centers.size(); ++i) {
        centers[i] *= 0.5;
        radii[i] *= 0.5;
      }
    } else {
      // large-radius branch: detect directly on ROI subimage
      CircleIdentifier circle_id(imgOrig_sub);
      rmin = std::max(2.0, r_px * 0.5);
      rmax = std::ceil(r_px) + 3.0; // keep your original heuristic
      sense = 0.95;
      metrics = circle_id.BubbleCenterAndSizeByCircle(centers, radii, rmin,
                                                      rmax, sense);
    }

    // 5) Decide validity
    const int n_bb = static_cast<int>(radii.size());
    bool use_cam = false;

    if (n_bb == 0) {
      // Fallback: consider the case of "filled by a big bubble".
      // Keep your original mean-intensity gate using reference intensity.
      double int_orig = 0.0;
      for (int i = 0; i < n_row; ++i)
        for (int j = 0; j < n_col; ++j)
          int_orig += imgOrig_sub(i, j);
      int_orig /= static_cast<double>(n_row * n_col);

      auto &bb_cfg = static_cast<const BubbleConfig &>(_obj_cfg);
      const BubbleRefImg &bb_refimg = bb_cfg._bb_ref_img;
      const double intRef = bb_refimg.getIntRef(cam);
      if (int_orig < intRef * 1.2 && int_orig > intRef * 0.8) {
        use_cam = true; // within [0.8, 1.2] * intRef
      }
    } else {
      // At least one detected bubble close to expected radius and with
      // sufficient metric
      for (int ci = 0; ci < n_bb; ++ci) {
        const double cr = radii[ci];
        const bool ok = (std::fabs(cr - r_px) < std::min(0.3 * r_px, 2.0)) &&
                        (metrics[ci] > 0.1);
        if (ok) {
          use_cam = true;
          break;
        }
      }
    }

    shake_cam[cam] = use_cam;
  }

  return shake_cam;
}

double BubbleShakeStrategy::getImgCorr(ROIInfo &roi_info, const int x,
                                       const int y,
                                       const Image &ref_img) const {
  auto computeCorr = [&]() -> double {
    const double cx = static_cast<double>(x - roi_info._ROI_range.col_min);
    const double cy = static_cast<double>(y - roi_info._ROI_range.row_min);
    return myMATH::imgCrossCorrAtPt(roi_info.getAugImg(), ref_img, cx, cy);
  };

  double corr = 0.0;
  if (roi_info.inRange(y, x)) {
    // if inside the correlation map, then we can update or get from corr_map
    if (roi_info.corr_map(y, x) < -1) {
      // location on augmented image
      corr = computeCorr();
      roi_info.corr_map(y, x) = corr; // update corr_map
    } else {
      corr = roi_info.corr_map(y, x);
    }
  } else {
    // if outside the map, we have to calculate it and can't update to corr_map
    // location on augmented image
    corr = computeCorr();
  }

  return corr;
}

double
BubbleShakeStrategy::calShakeResidue(const Object3D &obj_candidate,
                                     std::vector<ROIInfo> &roi_info,
                                     const std::vector<bool> &shake_cam) const {
  const int n_cam = static_cast<int>(_cam_list.size());
  int cams_used = 0;
  double residue = 0;

  for (int cam = 0; cam < n_cam; ++cam) {
    if (!_cam_list[cam]->is_active)
      continue;
    if (!shake_cam[cam])
      continue;

    Object2D *obj2d = obj_candidate._obj2d_list[cam].get();
    const auto *bb2d = static_cast<const Bubble2D *>(obj2d);
    double xc = bb2d->_pt_center[0], yc = bb2d->_pt_center[1];
    double r_px = bb2d->_r_px;
    if (isInvalidProjectedCenter(bb2d->_pt_center)) {
      continue;
    }
    const PixelRange &range_corrmap = roi_info[cam]._ROI_range;

    // the candidate bubble can be partly within the augmented image
    // however, if the bubble is total out of the augmented image, then residue
    // = 1
    // ---- Early-out: bubble completely outside augmented image ----
    // Check overlap between the ref patch (centered at (xc,yc)) and the ROI
    // image. Use ROI-local center for the test.
    const double cx_local = xc - range_corrmap.col_min;
    const double cy_local = yc - range_corrmap.row_min;
    const double safe_factor = 0.8;
    const double half_w =
        r_px * safe_factor; // geometric half-size (nearest-neighbor NCC)
    const double half_h = r_px * safe_factor;

    const bool no_overlap =
        (cx_local <= -half_w) ||
        (cx_local >= range_corrmap.getNumOfCol() + half_w) ||
        (cy_local <= -half_h) ||
        (cy_local >= range_corrmap.getNumOfRow() + half_h);

    if (no_overlap)
      continue; // skip this camera

    // calcualte correlation of 4 pixel around xc, yc and do interpolation for
    // sub-pixel accuracy
    int x_low = std::floor(xc);
    int x_high = x_low + 1;
    int y_low = std::floor(yc);
    int y_high = y_low + 1;

    // get reference image
    const int r_int = std::round(r_px);
    int npix = r_int * 2 +
               1; // guarantee there is only a whole center pixel on ref_img
    const auto &bb_cfg = static_cast<const BubbleConfig &>(
        _obj_cfg); // to get the bubble reference image
    BubbleResize bb_resizer;
    const Image ref_img =
        bb_resizer.ResizeBubble(bb_cfg._bb_ref_img[cam], npix,
                                _cam_list[cam]->max_intensity);

    // calculate cross-correlation
    std::vector<double> corr_interp(4, 0);
    corr_interp[0] = getImgCorr(roi_info[cam], x_low, y_low, ref_img);
    corr_interp[1] = getImgCorr(roi_info[cam], x_high, y_low, ref_img);
    corr_interp[2] = getImgCorr(roi_info[cam], x_high, y_high, ref_img);
    corr_interp[3] = getImgCorr(roi_info[cam], x_low, y_high, ref_img);

    // bilinear interpolation
    AxisLimit grid_limit(x_low, x_high, y_low, y_high, 0, 0);
    std::vector<double> center = {xc, yc};
    double res = 1 - myMATH::bilinearInterp(grid_limit, corr_interp, center);
    residue += res;
    cams_used++;
  }

  residue = cams_used > 1
                ? residue / cams_used
                : 2; // must have at least 2 cameras to calculate the residue
  return residue;
}

// BubbleShakeStrategy.cpp
bool BubbleShakeStrategy::additionalObjectCheck(
    const Object3D &obj, std::vector<ROIInfo> &roi_info,
    const std::vector<bool> &shake_cam) const {
  constexpr double CORR_TH = 0.1;

  for (size_t cam = 0; cam < shake_cam.size(); ++cam) {
    if (!_cam_list[cam]->is_active)
      continue;
    if (!shake_cam[cam])
      continue;

    Object2D *obj2d = obj._obj2d_list[cam].get();
    const auto *bb2d = static_cast<const Bubble2D *>(obj2d);
    if (isInvalidProjectedCenter(bb2d->_pt_center)) {
      return false;
    }
    double xc = bb2d->_pt_center[0], yc = bb2d->_pt_center[1];
    double r_px = bb2d->_r_px;
    const PixelRange &range_corrmap = roi_info[cam]._ROI_range;

    const int r_int = std::round(r_px);
    int npix = r_int * 2 +
               1; // guarantee there is only a whole center pixel on ref_img
    const auto &bb_cfg = static_cast<const BubbleConfig &>(
        _obj_cfg); // to get the bubble reference image
    BubbleResize bb_resizer;
    const Image ref_img =
        bb_resizer.ResizeBubble(bb_cfg._bb_ref_img[cam], npix,
                                _cam_list[cam]->max_intensity);

    // ====== 你原来 calShakeResidue 里那套插值 ======
    int x_low = static_cast<int>(std::floor(xc));
    int y_low = static_cast<int>(std::floor(yc));
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    // calculate cross-correlation
    std::vector<double> corr_interp(4, 0);
    corr_interp[0] = getImgCorr(roi_info[cam], x_low, y_low, ref_img);
    corr_interp[1] = getImgCorr(roi_info[cam], x_high, y_low, ref_img);
    corr_interp[2] = getImgCorr(roi_info[cam], x_high, y_high, ref_img);
    corr_interp[3] = getImgCorr(roi_info[cam], x_low, y_high, ref_img);

    AxisLimit grid_limit(x_low, x_high, y_low, y_high, 0, 0);
    std::vector<double> center = {xc, yc};

    double corr = myMATH::bilinearInterp(grid_limit, corr_interp, center);

    // ✅ 你的规则：任意一个 < 0.1 → false
    if (corr < CORR_TH) {
      return false;
    }
  }

  return true;
}
