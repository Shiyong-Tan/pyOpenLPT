// ========================================================================
// VSC.cpp - Volume Self Calibration Implementation
// ========================================================================
//
// This file implements the VSC module for OpenLPT.
// Main components:
// - accumulate(): Collects reliable 3D-2D correspondences from tracked
// particles.
// - runVSC(): Optimizes camera extrinsics using Levenberg-Marquardt.
// - runOTF(): Fits spatially-varying Gaussian OTF parameters.
//
// Dependencies:
// - nanoflann: KD-tree for fast neighbor search (isolation check).
// - myMATH: Matrix operations (inverse, eye, etc.).
// - OpenMP: Parallel processing.
// ========================================================================

// CRITICAL: nanoflann.hpp must be included FIRST due to template requirements
#include <nanoflann.hpp>

#include "ImageIO.h"
#include "OTF.h"
#include "ObjectFinder.h"
#include "Camera.h"
#include "VSC.h"
#include "myMATH.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>

#include <random>
#include <unordered_map>

// ========================================================================
// KD-Tree Adaptor for 2D Object Candidates
// ========================================================================
// This adaptor allows nanoflann to operate on a vector of Object2D pointers.
// Used for fast isolation check: find all 2D detections near a projected point.

struct Obj2DCloud {
  using coord_t = double; ///< Coordinate type required by nanoflann

  const std::vector<std::unique_ptr<Object2D>>
      &_pts; ///< Reference to object list

  /// Return number of points in the dataset
  inline size_t kdtree_get_point_count() const { return _pts.size(); }

  /// Return coordinate 'dim' (0=x, 1=y) of point 'idx'
  inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const {
    return _pts[idx]->_pt_center[dim];
  }

  /// Optional bounding box (not used, return false)
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

/// KD-tree type for 2D point cloud
using VisKDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Obj2DCloud>, Obj2DCloud, 2>;

// ========================================================================
// Anonymous Namespace: Helper Functions
// ========================================================================

namespace {

/**
 * @brief Convert rotation vector to rotation matrix using Rodrigues' formula.
 *
 * R = I + sin(theta)*K + (1-cos(theta))*K^2
 * where K is the skew-symmetric matrix of the unit rotation axis.
 *
 * @param r_vec Rotation vector (axis * angle).
 * @return 3x3 rotation matrix.
 */
Matrix<double> Rodrigues(const Pt3D &r_vec) {
  double theta = std::sqrt(r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] +
                           r_vec[2] * r_vec[2]);

  // For very small rotations, return identity
  if (theta < 1e-8) {
    return myMATH::eye<double>(3);
  }

  // Build skew-symmetric matrix K from unit axis k = r_vec / theta
  Matrix<double> K(3, 3, 0);
  K(0, 1) = -r_vec[2] / theta;
  K(0, 2) = r_vec[1] / theta;
  K(1, 0) = r_vec[2] / theta;
  K(1, 2) = -r_vec[0] / theta;
  K(2, 0) = -r_vec[1] / theta;
  K(2, 1) = r_vec[0] / theta;

  // Rodrigues' formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
  Matrix<double> I = myMATH::eye<double>(3);
  return I + (K * std::sin(theta)) + ((K * K) * (1.0 - std::cos(theta)));
}

/**
 * @brief Map 3D world position to OTF grid index.
 *
 * Replicates OTF::mapGridID logic for use in VSC without needing friend access.
 *
 * @param param OTF parameter structure with grid info.
 * @param pt 3D world point.
 * @return Linear grid index.
 */
int getOTFGridID(const OTFParam &param, const Pt3D &pt) {
  // Compute grid cell indices
  int x_id =
      static_cast<int>(std::lround((pt[0] - param.boundary.x_min) / param.dx));
  int y_id =
      static_cast<int>(std::lround((pt[1] - param.boundary.y_min) / param.dy));
  int z_id =
      static_cast<int>(std::lround((pt[2] - param.boundary.z_min) / param.dz));

  // Clamp to valid range
  x_id = std::max(0, std::min(x_id, param.nx - 1));
  y_id = std::max(0, std::min(y_id, param.ny - 1));
  z_id = std::max(0, std::min(z_id, param.nz - 1));

  // Linear index: i = x_id * (ny*nz) + y_id * nz + z_id
  return x_id * (param.ny * param.nz) + y_id * param.nz + z_id;
}

/**
 * @brief Solve linear system Ax = B using Jacobi Eigenvalue Decomposition (SVD
 * for symmetric).
 *
 * A must be symmetric (approx). A = V * W * V^T, where W is diagonal eigenvals.
 * A^-1 = V * W^-1 * V^T.
 *
 * Handles ill-conditioned A by thresholding small eigenvalues.
 *
 * @param A Symmetric matrix (NxN).
 * @param B RHS vector (Nx1).
 * @return Solution vector x (Nx1).
 */
Matrix<double> solveSymmetricSVD(const Matrix<double> &A,
                                 const Matrix<double> &B) {
  int n = A.getDimRow();
  Matrix<double> V = myMATH::eye<double>(n); // Eigenvectors
  Matrix<double> D = A;                      // Will become diagonal eigenvalues

  // Jacobi Iteration
  int max_iter = 50;
  for (int iter = 0; iter < max_iter; ++iter) {
    double max_off_diag = 0.0;
    int p = 0, q = 1;

    // Find pivot
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        if (std::abs(D(i, j)) > max_off_diag) {
          max_off_diag = std::abs(D(i, j));
          p = i;
          q = j;
        }
      }
    }

    if (max_off_diag < 1e-12)
      break; // Converged

    double theta = 0.5 * std::atan2(2.0 * D(p, q), D(q, q) - D(p, p));
    double c = std::cos(theta);
    double s = std::sin(theta);

    // Rotate rows/cols p and q
    // Update D (upper triangle only needed strictly, but we maintain full for
    // simplicity) Actually, simple standard update: D' = J^T * D * J V' = V * J

    // Careful manual update to avoid full multiplication
    double D_pp = D(p, p);
    double D_qq = D(q, q);
    double D_pq = D(p, q);

    D(p, p) = c * c * D_pp - 2 * s * c * D_pq + s * s * D_qq;
    D(q, q) = s * s * D_pp + 2 * s * c * D_pq + c * c * D_qq;
    D(p, q) = 0; // Explicitly zero
    D(q, p) = 0;

    for (int k = 0; k < n; ++k) {
      if (k != p && k != q) {
        double D_kp = D(k, p);
        double D_kq = D(k, q);
        D(k, p) = c * D_kp - s * D_kq;
        D(p, k) = D(k, p);
        D(k, q) = s * D_kp + c * D_kq;
        D(q, k) = D(k, q);
      }

      // Update V (eigenvectors)
      // V_new = V * J
      // Col p = c*Col p - s*Col q
      // Col q = s*Col p + c*Col q
      double V_kp = V(k, p);
      double V_kq = V(k, q);
      V(k, p) = c * V_kp - s * V_kq;
      V(k, q) = s * V_kp + c * V_kq;
    }
  }

  // Solve x = V * W^-1 * V^T * B
  // 1. C = V^T * B
  Matrix<double> C(n, 1, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {   // B is n x 1
      C(i, 0) += V(j, i) * B(j, 0); // V^T(i,j) = V(j,i)
    }
  }

  // 2. D = W^-1 * C
  // Threshold eigenvalues
  double max_eig = 0;
  for (int i = 0; i < n; ++i)
    max_eig = std::max(max_eig, std::abs(D(i, i)));
  double thresh = max_eig * 1e-9;

  for (int i = 0; i < n; ++i) {
    if (std::abs(D(i, i)) > thresh) {
      C(i, 0) /= D(i, i);
    } else {
      C(i, 0) = 0;
    }
  }

  // 3. x = V * D (which is C now)
  Matrix<double> x(n, 1, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      x(i, 0) += V(i, j) * C(j, 0);
    }
  }

  return x;
}

} // namespace

// ========================================================================
// VSC Methods Implementation
// ========================================================================

void VSC::configure(const VSCParam &cfg) { _cfg = cfg; }

void VSC::initStrategy(const ObjectConfig &obj_cfg) {
  if (obj_cfg.kind() == ObjectKind::Tracer) {
    _strategy = std::make_unique<VSCTracerStrategy>();
  } else {
    _strategy = std::make_unique<VSCBubbleStrategy>();
  }
}

void VSC::reset() {
  _buffer.clear();
  _voxel_counts.clear();
  _grid_initialized = false;
}

int VSC::computeVoxelIndex(const Pt3D &pt) const {
  if (!_grid_initialized) {
    return -1;
  }

  const int n_div = std::max(1, _cfg._n_divisions);

  // Compute voxel sizes
  const double dx = (_grid_max[0] - _grid_min[0]) / n_div;
  const double dy = (_grid_max[1] - _grid_min[1]) / n_div;
  const double dz = (_grid_max[2] - _grid_min[2]) / n_div;

  // Handle degenerate cases
  const double safe_dx = (dx > 1e-6) ? dx : 1.0;
  const double safe_dy = (dy > 1e-6) ? dy : 1.0;
  const double safe_dz = (dz > 1e-6) ? dz : 1.0;

  int xi = static_cast<int>((pt[0] - _grid_min[0]) / safe_dx);
  int yi = static_cast<int>((pt[1] - _grid_min[1]) / safe_dy);
  int zi = static_cast<int>((pt[2] - _grid_min[2]) / safe_dz);

  // Clamp to valid range
  xi = std::max(0, std::min(xi, n_div - 1));
  yi = std::max(0, std::min(yi, n_div - 1));
  zi = std::max(0, std::min(zi, n_div - 1));

  return xi * n_div * n_div + yi * n_div + zi;
}

bool VSC::isReady() const {
  return _buffer.size() >= static_cast<size_t>(_cfg._min_points_to_trigger);
}

// ========================================================================
// accumulate() - Collect calibration points from current frame
// ========================================================================

void VSC::accumulate(int frame_id, const std::deque<Track> &active_tracks,
                     const std::vector<Image> &images,
                     const std::vector<std::shared_ptr<Camera>> &camera_models,
                     const ObjectConfig &obj_cfg) {
  // Initialize strategy on first call
  if (!_strategy) {
    initStrategy(obj_cfg);
  }

  const size_t n_cams = camera_models.size();
  if (n_cams == 0) {
    return;
  }

  // ----- Step 1: Detect 2D objects in all cameras -----
  // Each camera's detections are stored independently for parallel processing.
  std::vector<std::vector<std::unique_ptr<Object2D>>> all_candidates(n_cams);

#pragma omp parallel for if (!omp_in_parallel())
  for (int k = 0; k < static_cast<int>(n_cams); ++k) {
    // Thread-local ObjectFinder for thread safety
    ObjectFinder2D finder;
    all_candidates[k] = finder.findObject2D(images[k], obj_cfg);
  }

  // ----- Step 2: Build KD-Trees for fast neighbor queries -----
  // Used to check isolation: a valid calibration point should have exactly
  // one 2D detection nearby (itself). Multiple detections indicate overlap.
  std::vector<std::unique_ptr<VisKDTree>> kdtrees(n_cams);
  std::vector<Obj2DCloud> clouds;
  clouds.reserve(n_cams);

  for (size_t k = 0; k < n_cams; ++k) {
    if (all_candidates[k].empty())
      return;

    clouds.emplace_back(Obj2DCloud{all_candidates[k]});
    kdtrees[k] = std::make_unique<VisKDTree>(
        2, clouds.back(), nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtrees[k]->buildIndex();
  }

  // ----- Step 3: Precompute object radii for each camera -----
  // Avoids repeated dynamic_cast in getObject2DSize
  std::vector<std::vector<double>> obj_radii(n_cams);
  for (size_t k = 0; k < n_cams; ++k) {
    size_t N = all_candidates[k].size();
    obj_radii[k].resize(N);
    for (size_t i = 0; i < N; ++i) {
      obj_radii[k][i] = _strategy->getObject2DSize(*all_candidates[k][i]) / 2.0;
    }
  }

  // ----- Step 4: Precompute Isolation for each object in each camera -----
  // is_isolated[k][i] = true if object i in camera k has no overlapping
  // neighbors
  // Simple O(N^2/2) traversal without early termination for correctness
  std::vector<std::vector<bool>> is_isolated(n_cams);

  for (size_t k = 0; k < n_cams; ++k) {
    size_t N = all_candidates[k].size();
    is_isolated[k].assign(N, true);

    for (size_t i = 0; i < N; ++i) {
      double r_i = obj_radii[k][i];
      const Pt2D &c_i = all_candidates[k][i]->_pt_center;

      for (size_t j = i + 1; j < N; ++j) {
        double r_j = obj_radii[k][j];
        const Pt2D &c_j = all_candidates[k][j]->_pt_center;

        double dist2 = myMATH::dist2(c_i, c_j);
        double min_dist = r_i + r_j + _cfg._isolation_radius;
        double min_dist2 = min_dist * min_dist;

        if (dist2 < min_dist2) {
          is_isolated[k][i] = false;
          is_isolated[k][j] = false;
        }
      }
    }
  }

  // Use thread-local buffer to reduce critical section overhead
  // Grid Initialization (Once) - Dynamic update handled safely below

  std::vector<CalibrationPoint> local_buffer;
  bool is_tracer = (obj_cfg.kind() == ObjectKind::Tracer);

  for (const auto &trk : active_tracks) {
    // Filter: track must be long enough to be reliable
    if (trk._obj3d_list.size() < static_cast<size_t>(_cfg._min_track_len)) {
      continue;
    }
    if (trk._obj3d_list.empty())
      continue;

    // Get 3D position from the most recent point in track
    const auto &obj3d = *trk._obj3d_list.back();
    Pt3D P3 = obj3d._pt_center;

    // Check visibility and isolation in ALL active cameras
    bool all_visible = true;
    std::vector<Observation> current_obs;

    for (size_t k = 0; k < n_cams; ++k) {
      // User request: Must be visible in ALL cameras.
      // Check if candidate list is empty for this camera
      if (all_candidates[k].empty()) {
        all_visible = false;
        break;
      }

      // Project 3D point to 2D
      // Project 3D point to 2D
      if (!camera_models[k]) {
        all_visible = false;
        break;
      }
      auto proj_status = camera_models[k]->project(P3);
      if (!proj_status) {
        all_visible = false;
        break;
      }
      Pt2D P2_proj = proj_status.value();

      // ----- Isolation Check using precomputed is_isolated -----
      // Find the nearest candidate to the projected point
      const double query_pt[2] = {P2_proj[0], P2_proj[1]};
      size_t nearest_idx = 0;
      double nearest_dist2 = 0;
      nanoflann::KNNResultSet<double> resultSet(1);
      resultSet.init(&nearest_idx, &nearest_dist2);
      kdtrees[k]->findNeighbors(resultSet, query_pt,
                                nanoflann::SearchParameters());

      double obj_r_px = obj_radii[k][nearest_idx];
      // Check if the nearest candidate is close enough to be "this" object
      double margin = 1.0;
      double match_threshold2 = obj_r_px * margin * obj_r_px * margin;
      if (nearest_dist2 > match_threshold2) {
        all_visible = false; // No matching candidate found
        break;
      }

      // Check if this candidate is isolated (no overlapping neighbors)
      if (!is_isolated[k][nearest_idx]) {
        all_visible = false; // Candidate has overlapping neighbors
        break;
      }

      // Check if projection is within image bounds
      if (P2_proj[0] - obj_r_px < 0 ||
          P2_proj[0] + obj_r_px >= camera_models[k]->getNCol() ||
          P2_proj[1] - obj_r_px < 0 ||
          P2_proj[1] + obj_r_px >= camera_models[k]->getNRow()) {
        all_visible = false;
        break;
      }

      size_t idx = nearest_idx;
      const auto &candidate = all_candidates[k][idx];

      Observation obs;
      obs._cam_id = static_cast<int>(k);
      obs._meas_2d = candidate->_pt_center;
      obs._proj_2d = P2_proj;
      obs._quality_score = 1.0;
      obs._obj_radius = obj_r_px; // Use precomputed radius

      // ----- Identify OTF Parameters (Tracer Only) -----
      if (is_tracer && _cfg._enable_otf) {
        // Use tighter window for fitting robustness, slightly larger than 1.0r
        // to identify decay
        int half_w = static_cast<int>(std::ceil(1.5 * obj_r_px));

        // Check image bounds before calling estimate
        int cx = std::lround(obs._meas_2d[0]);
        int cy = std::lround(obs._meas_2d[1]);

        // Strict bounds check: only accept points fully inside the image
        // to ensure the OTF window is complete (2*half_w + 1)
        // CRITICAL: LOGIC MUST MATCH runOTF(). IF YOU CHANGE THIS CLIPPING
        // BEHAVIOR, YOU MUST UPDATE VSC::runOTF to handle partial windows or
        // different bounds!
        int x0 = cx - half_w;
        int x1 = cx + half_w + 1;
        int y0 = cy - half_w;
        int y1 = cy + half_w + 1;

        if (x0 >= 0 && y0 >= 0 && x1 <= images[k].getDimCol() &&
            y1 <= images[k].getDimRow()) {

          // Extract ROI
          obs._roi_intensity = Matrix<double>(y1 - y0, x1 - x0, 0.0);
          for (int r = y0; r < y1; ++r) {
            for (int c = x0; c < x1; ++c) {
              obs._roi_intensity(r - y0, c - x0) = images[k](r, c);
            }
          }

          // Relative center within the ROI
          Pt2D rel_center = obs._meas_2d;
          rel_center[0] -= x0;
          rel_center[1] -= y0;

          // Estimate params using ROI and relative coords
          obs._otf_params = estimateOTFParams(obs._roi_intensity, rel_center,
                                              obs._obj_radius);
        } else {
          // If clipped, discard observation for robustness
          continue;
        }

        // If OTF fit failed (degenerate or invalid), discard this observation
        // User requirement: Do not use this point if fit fails
        if (!obs._otf_params.valid) {
          // Option 1: Mark not all_visible -> Drop entire 3D point
          // Option 2: Drop just this camera -> Point might still be valid with
          // fewer cameras? User said: "VSC should not use this point". Usually
          // points must be seen by ALL cameras.
          all_visible = false;
          break;
        }
      }

      current_obs.push_back(obs);
    }

    if (all_visible && !current_obs.empty()) {
      // Dynamic Grid Update (Critical Section handled safely due to serial
      // track loop)
      updateGridAndRebalance(P3);

      int voxel_id = computeVoxelIndex(P3);

      if (voxel_id >= 0 && _buffer.size() >= 100) {
        int total_points = 0;
        for (const auto &[vid, cnt] : _voxel_counts) {
          total_points += cnt;
        }
        size_t n_voxels =
            std::max(_voxel_counts.size(), static_cast<size_t>(1));
        double avg_count = static_cast<double>(total_points) / n_voxels;
        double threshold = std::max(
            avg_count * 2.0, static_cast<double>(_cfg._min_points_per_voxel));

        int current_count = 0;
        auto it = _voxel_counts.find(voxel_id);
        if (it != _voxel_counts.end()) {
          current_count = it->second;
        }

        if (current_count >= static_cast<int>(threshold)) {
          continue;
        }
      }

      CalibrationPoint cp;
      cp._pos_3d = P3;
      cp._frame_id = frame_id;
      cp._obs = std::move(current_obs);
      local_buffer.push_back(std::move(cp));

      if (voxel_id >= 0) {
        _voxel_counts[voxel_id]++;
      }
    }
  }

  // Append local buffer to global buffer
  if (!local_buffer.empty()) {
    _buffer.insert(_buffer.end(), std::make_move_iterator(local_buffer.begin()),
                   std::make_move_iterator(local_buffer.end()));

    // Log current volume size
    if (_grid_initialized) {
      double L = _grid_max[0] - _grid_min[0];
      double W = _grid_max[1] - _grid_min[1];
      double H = _grid_max[2] - _grid_min[2];
      std::cout << "  VSC Volume: " << L << " x " << W << " x " << H
                << " (LxWxH)" << std::endl;
    }
  }
}

// ========================================================================
// updateGridAndRebalance() - Dynamic grid expansion
// ========================================================================

void VSC::updateGridAndRebalance(const Pt3D &pt) {
  if (!_grid_initialized) {
    _grid_min = pt;
    _grid_max = pt;
    _grid_initialized = true;
    // Add small margin to avoid zero volume initially
    for (int i = 0; i < 3; ++i) {
      _grid_min[i] -= 0.1;
      _grid_max[i] += 0.1;
    }
    return;
  }

  bool changed = false;
  // Expand with small margin to include the new point
  const double margin = 0.1;

  for (int i = 0; i < 3; ++i) {
    if (pt[i] < _grid_min[i]) {
      _grid_min[i] = pt[i] - margin;
      changed = true;
    }
    if (pt[i] > _grid_max[i]) {
      _grid_max[i] = pt[i] + margin;
      changed = true;
    }
  }

  if (changed) {
    // Grid changed: previous voxel IDs are invalid.
    // Re-compute all.
    _voxel_counts.clear();
    for (const auto &cp : _buffer) {
      int vid = computeVoxelIndex(cp._pos_3d);
      if (vid >= 0) {
        _voxel_counts[vid]++;
      }
    }
  }
}

// Fit Log-Intensity to quadratic model:
// ln(I) ~ p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy
//
// OTF Model: I = a * exp( -(b*xx^2 + c*yy^2) )
// Exponent term Q(x,y) = b*xx^2 + c*yy^2 is positive definite.
// Matrix M in (x-xc)^T M (x-xc) has eigenvalues b, c.
//
// We solve for p vector (6x1).
// System: A * p = L
// A matrix rows: [1, x, y, x^2, y^2, xy]
// L vector rows: [ln(I)]

VSC::OTFParams VSC::estimateOTFParams(const Matrix<double> &roi,
                                      const Pt2D &center_rel,
                                      double obj_radius) const {
  OTFParams params;
  params.valid = false;

  int rows = roi.getDimRow();
  int cols = roi.getDimCol();
  double xc = center_rel[0];
  double yc = center_rel[1];

  // 0. Find max intensity within obj_radius of the center
  double max_val = 0;
  double r2_limit = obj_radius * obj_radius;
  int argmax_x = -1, argmax_y = -1;

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      double dx = x - xc, dy = y - yc;
      if (dx * dx + dy * dy <= r2_limit) {
        double val = roi(y, x);
        if (val > max_val) {
          max_val = val;
          argmax_x = x;
          argmax_y = y;
        }
      }
    }
  }

  // Sanity check: Peak argmax drift check
  if (argmax_x == -1 || max_val < 10.0)
    return params;
  double dist_to_peak =
      std::sqrt(std::pow(argmax_x - xc, 2) + std::pow(argmax_y - yc, 2));
  if (dist_to_peak > obj_radius + 0.5 && dist_to_peak > 1.5)
    return params;

  // Dynamic fitting window based on 10% contour
  double t_high = max_val * 0.10;

  // Restore dynamic window sizing logic:
  // We want to find the radius 'r_fit' such that all pixels within this radius
  // are considered, but we stop expanding if the intensity drops below
  // threshold. Since we are working with ROI, we need to check relative
  // coordinates.

  int r_fit = 1;
  int cx_int = std::lround(xc);
  int cy_int = std::lround(yc);

  // Maximum possible radius within the ROI
  // User suggestion: "Get roi half width"
  int max_r = (std::min(rows, cols) - 1) / 2;

  // Start from 1 to check immediate neighbors
  // Check neighbors to expand ROI up to max_r
  for (int r = 1; r <= max_r; ++r) {
    bool expand = false;
    // Check square perimeter at radius r
    for (int k = -r; k <= r; ++k) {
      // Top: (cx+k, cy-r)
      if (cy_int - r >= 0 && cx_int + k >= 0 && cx_int + k < cols &&
          roi(cy_int - r, cx_int + k) > t_high) {
        expand = true;
        break;
      }
      // Bottom: (cx+k, cy+r)
      if (cy_int + r < rows && cx_int + k >= 0 && cx_int + k < cols &&
          roi(cy_int + r, cx_int + k) > t_high) {
        expand = true;
        break;
      }
      // Left: (cx-r, cy+k)
      if (cy_int + k >= 0 && cy_int + k < rows && cx_int - r >= 0 &&
          roi(cy_int + k, cx_int - r) > t_high) {
        expand = true;
        break;
      }
      // Right: (cx+r, cy+k)
      if (cy_int + k >= 0 && cy_int + k < rows && cx_int + r < cols &&
          roi(cy_int + k, cx_int + r) > t_high) {
        expand = true;
        break;
      }
    }

    if (expand)
      r_fit = r;
    else
      break; // Stop if this radius doesn't reach the threshold
  }

  // Define fitting constraints based on r_fit
  int x0_fit = std::max(0, cx_int - r_fit);
  int x1_fit = std::min(cols, cx_int + r_fit + 1);
  int y0_fit = std::max(0, cy_int - r_fit);
  int y1_fit = std::min(rows, cy_int + r_fit + 1);
  int w_fit = x1_fit - x0_fit, h_fit = y1_fit - y0_fit;
  bool is_tiny = (w_fit <= 3 && h_fit <= 3);

  // 1. Stage 1: Linear Seed (Log-domain)
  // Adaptive threshold: 0% for tiny (use all Info), 10% for normal
  double seed_threshold = is_tiny ? 0.0 : (max_val * 0.10);

  Matrix<double> AtA(6, 6, 0.0);
  Matrix<double> AtL(6, 1, 0.0);
  int n_samples_seed = 0;

  for (int y = y0_fit; y < y1_fit; ++y) {
    for (int x = x0_fit; x < x1_fit; ++x) {
      double val = roi(y, x);
      if (val <= seed_threshold)
        continue;

      double dx = x - xc, dy = y - yc;
      double log_v = std::log(std::max(val, 1e-6 * max_val));
      double basis[6] = {1.0, dx, dy, dx * dx, dy * dy, dx * dy};

      for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j)
          AtA(i, j) += basis[i] * basis[j];
        AtL(i, 0) += basis[i] * log_v;
      }
      n_samples_seed++;
    }
  }

  // 2. Extract Seeds
  double dx_peak = 0, dy_peak = 0;
  bool seed_ok = false;

  if (n_samples_seed >= 6) {
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < i; ++j)
        AtA(i, j) = AtA(j, i);

    Matrix<double> p = solveSymmetricSVD(AtA, AtL);
    if (std::isfinite(p(0, 0))) {
      double p0 = p(0, 0), p1 = p(1, 0), p2 = p(2, 0);
      double p3 = p(3, 0), p4 = p(4, 0), p5 = p(5, 0);
      double det_H = 4.0 * p3 * p4 - p5 * p5;
      double mxx = -p3, myy = -p4, mxy = -0.5 * p5;
      double delta_eig = std::sqrt(std::pow(mxx - myy, 2) + 4.0 * mxy * mxy);
      double lam1 = (mxx + myy + delta_eig) / 2.0;
      double lam2 = (mxx + myy - delta_eig) / 2.0;

      if (std::abs(det_H) > 1e-15) {
        dx_peak = (p5 * p2 - 2.0 * p4 * p1) / det_H;
        dy_peak = (p5 * p1 - 2.0 * p3 * p2) / det_H;
      }

      if (lam2 > 1e-10 && lam1 > 0 && p3 < 0 && p4 < 0 &&
          (dx_peak * dx_peak + dy_peak * dy_peak < 4.1)) {
        params.a =
            std::exp(p0 + p1 * dx_peak + p2 * dy_peak + p3 * dx_peak * dx_peak +
                     p4 * dy_peak * dy_peak + p5 * dx_peak * dy_peak);
        params.b = lam2; // Major axis
        params.c = lam1; // Minor axis
        params.alpha = (std::abs(mxy) > 1e-12 || std::abs(mxx - myy) > 1e-12)
                           ? 0.5 * std::atan2(2.0 * mxy, mxx - myy)
                           : 0.0;
        seed_ok = true;
      }
    }
  }

  // Naive fallback for tiny ROIs if full seed fails
  if (!seed_ok) {
    if ((x1_fit - x0_fit) <= 3 && (y1_fit - y0_fit) <= 3 && max_val > 10.0) {
      params.a = max_val;
      params.b = params.c = 1.0;
      params.alpha = 0.0;
      dx_peak = dy_peak = 0.0;
    } else {
      return params;
    }
  }

  auto normalize = [](double &b, double &c, double &alpha) {
    if (b > c) {
      std::swap(b, c);
      alpha += 1.57079632679;
    }
    while (alpha > 1.57079632679)
      alpha -= 3.14159265359;
    while (alpha < -1.57079632679)
      alpha += 3.14159265359;
    if (c > 6.0 * b)
      c = 6.0 * b;
    if (b < 0.001)
      b = 0.001;
    if (c < 0.001)
      c = 0.001;
  };
  normalize(params.b, params.c, params.alpha);

  // 3. Stage 2: Non-linear Refinement (Gauss-Newton/LM)
  bool use_isotropic = is_tiny;
  int n_params = use_isotropic ? 4 : 6;
  Matrix<double> theta(n_params, 1, 0.0);

  theta(0, 0) = std::clamp(params.a, 0.1 * max_val, 5.0 * max_val);
  theta(1, 0) = std::clamp(xc + dx_peak, (double)x0_fit, (double)(x1_fit - 1));
  theta(2, 0) = std::clamp(yc + dy_peak, (double)y0_fit, (double)(y1_fit - 1));
  if (use_isotropic)
    theta(3, 0) = 0.5 * (params.b + params.c);
  else {
    theta(3, 0) = params.b;
    theta(4, 0) = params.c;
    theta(5, 0) = params.alpha;
  }

  const int max_iter = 10;
  double gn_threshold = use_isotropic ? (0.01 * max_val) : (0.05 * max_val);

  for (int iter = 0; iter < max_iter; ++iter) {
    Matrix<double> JtJ(n_params, n_params, 0.0);
    Matrix<double> Jtr(n_params, 1, 0.0);

    double A = theta(0, 0), x_p = theta(1, 0), y_p = theta(2, 0);

    for (int y = y0_fit; y < y1_fit; ++y) {
      for (int x = x0_fit; x < x1_fit; ++x) {
        double val = roi(y, x);
        if (val < gn_threshold)
          continue;

        double dx = x - x_p, dy = y - y_p;
        double Q = 0, e = 0, f = 0;
        double grad_Q[5] = {0}; // dQ/dx0, dQ/dy0, dQ/db(k), dQ/dc, dQ/dalpha

        if (use_isotropic) {
          double k = theta(3, 0);
          double r2 = dx * dx + dy * dy;
          Q = k * r2;
          e = std::exp(-Q);
          if (!is_tiny && e < 1e-13)
            continue;
          e = std::max(e, 1e-13);
          f = A * e;
          grad_Q[0] = -2.0 * k * dx;
          grad_Q[1] = -2.0 * k * dy;
          grad_Q[2] = r2;
        } else {
          double b = theta(3, 0), c = theta(4, 0), a_rad = theta(5, 0);
          double ca = std::cos(a_rad), sa = std::sin(a_rad);
          double xp = dx * ca + dy * sa, yp = -dx * sa + dy * ca;
          Q = b * xp * xp + c * yp * yp;
          e = std::exp(-Q);
          if (!is_tiny && e < 1e-13)
            continue;
          e = std::max(e, 1e-13);
          f = A * e;
          grad_Q[0] = -2.0 * b * xp * ca + 2.0 * c * yp * sa;
          grad_Q[1] = -2.0 * b * xp * sa - 2.0 * c * yp * ca;
          grad_Q[2] = xp * xp;
          grad_Q[3] = yp * yp;
          grad_Q[4] = 2.0 * (b - c) * xp * yp;
        }

        double res = val - f;
        double J[6] = {e,
                       -f * grad_Q[0],
                       -f * grad_Q[1],
                       -f * grad_Q[2],
                       -f * grad_Q[3],
                       -f * grad_Q[4]};

        // Weighted Least Squares: prioritize high SNR peak pixels
        double w2 = (val / max_val) * (val / max_val);

        for (int i = 0; i < n_params; ++i) {
          double wJ_i = w2 * J[i];
          for (int j = i; j < n_params; ++j)
            JtJ(i, j) += wJ_i * J[j];
          Jtr(i, 0) += wJ_i * res;
        }
      }
    }

    // Fill lower triangle for solver compatibility
    for (int i = 0; i < n_params; ++i)
      for (int j = 0; j < i; ++j)
        JtJ(i, j) = JtJ(j, i);

    // Scale-Adaptive Additive Damping (Robust LM)
    double lambda = 1e-3;
    for (int i = 0; i < n_params; ++i)
      JtJ(i, i) += lambda * (JtJ(i, i) + 1.0);

    Matrix<double> delta = solveSymmetricSVD(JtJ, Jtr);
    if (!std::isfinite(delta(0, 0)))
      break;

    for (int i = 0; i < n_params; ++i)
      theta(i, 0) += delta(i, 0);

    // Iterative Box Constraints
    theta(0, 0) = std::clamp(theta(0, 0), 0.1 * max_val, 5.0 * max_val);
    theta(1, 0) =
        std::clamp(theta(1, 0), xc - 2.0, xc + 2.0); // Center drift limit
    theta(2, 0) = std::clamp(theta(2, 0), yc - 2.0, yc + 2.0);
    theta(1, 0) = std::clamp(theta(1, 0), (double)x0_fit, (double)(x1_fit - 1));
    theta(2, 0) = std::clamp(theta(2, 0), (double)y0_fit, (double)(y1_fit - 1));

    if (use_isotropic) {
      theta(3, 0) = std::max(theta(3, 0), 0.001);
    } else {
      double &b = theta(3, 0), &c = theta(4, 0);
      b = std::max(b, 0.001);
      c = std::max(c, 0.001);
      if (c > 6.0 * b)
        c = 6.0 * b;
      if (b > 6.0 * c)
        b = 6.0 * c;
    }

    double step_norm = 0;
    for (int i = 0; i < n_params; ++i)
      step_norm += delta(i, 0) * delta(i, 0);
    if (step_norm < 1e-8)
      break; // Converged
  }

  // Final params assignment
  params.a = theta(0, 0);
  if (use_isotropic) {
    params.b = params.c = theta(3, 0);
    params.alpha = 0.0;
  } else {
    params.b = theta(3, 0);
    params.c = theta(4, 0);
    params.alpha = theta(5, 0);
    normalize(params.b, params.c, params.alpha);
  }

  // Tighter final validation logic
  bool center_ok =
      std::pow(theta(1, 0) - xc, 2) + std::pow(theta(2, 0) - yc, 2) < 4.1;
  params.valid =
      (params.a > 0.1 * max_val && params.b > 0 && params.c > 0 && center_ok);
  return params;
}

/**
 * @brief Extract camera parameters (8 params: 6 extrinsic + 2 intrinsic).
 *
 * @param cam Camera to extract from.
 * @return [rx, ry, rz, tx, ty, tz, f_eff, k1].
 */
struct CamParamPack {
  std::vector<double> params;
  bool has_k1 = false;
  bool has_k2 = false;
};

static Pt3D rotationMatrixToVector(const Matrix<double> &r_mtx) {
  Pt3D r_vec;

  double tr = (myMATH::trace<double>(r_mtx) - 1.0) / 2.0;
  tr = tr > 1.0 ? 1.0 : tr < -1.0 ? -1.0 : tr;
  double theta = std::acos(tr);
  double s = std::sin(theta);

  if (std::abs(s) > 1e-12) {
    double ratio = theta / (2.0 * s);
    r_vec[0] = (r_mtx(2, 1) - r_mtx(1, 2)) * ratio;
    r_vec[1] = (r_mtx(0, 2) - r_mtx(2, 0)) * ratio;
    r_vec[2] = (r_mtx(1, 0) - r_mtx(0, 1)) * ratio;
  } else if (tr > 0.0) {
    r_vec[0] = 0.0;
    r_vec[1] = 0.0;
    r_vec[2] = 0.0;
  } else {
    r_vec[0] = theta * std::sqrt((r_mtx(0, 0) + 1.0) / 2.0);
    r_vec[1] =
        theta * std::sqrt((r_mtx(1, 1) + 1.0) / 2.0) * (r_mtx(0, 1) > 0 ? 1 : -1);
    r_vec[2] =
        theta * std::sqrt((r_mtx(2, 2) + 1.0) / 2.0) * (r_mtx(0, 2) > 0 ? 1 : -1);
  }

  return r_vec;
}

CamParamPack getCamExtrinsics(const PinholeCamera &cam) {
  CamParamPack pack;
  pack.params.resize(12, 0.0);

  const auto &p = cam.param();

  Pt3D r_vec = rotationMatrixToVector(p.r_mtx);
  Pt3D t_vec = p.t_vec;

  double fx = p.cam_mtx(0, 0);
  double fy = p.cam_mtx(1, 1);
  double cx = p.cam_mtx(0, 2);
  double cy = p.cam_mtx(1, 2);

  double k1 = 0.0, k2 = 0.0;
  pack.has_k1 = p.dist_coeff.size() > 0;
  pack.has_k2 = p.dist_coeff.size() > 1;
  if (pack.has_k1)
    k1 = p.dist_coeff[0];
  if (pack.has_k2)
    k2 = p.dist_coeff[1];

  pack.params = {r_vec[0], r_vec[1], r_vec[2], t_vec[0], t_vec[1], t_vec[2],
                 fx,       fy,       cx,       cy,       k1,       k2};
  return pack;
}

/**
 * @brief Update camera parameters from 8-parameter vector.
 *
 * @param cam Camera to update.
 * @param params [rx, ry, rz, tx, ty, tz, f_eff, k1].
 */
bool updateCamExtrinsics(PinholeCamera &cam, const std::vector<double> &params,
                         bool has_k1, bool has_k2) {
  // Extract rotation vector
  Pt3D r_vec;
  r_vec[0] = params[0];
  r_vec[1] = params[1];
  r_vec[2] = params[2];

  // Extract translation vector
  Pt3D t_vec;
  t_vec[0] = params[3];
  t_vec[1] = params[4];
  t_vec[2] = params[5];

  // Extract intrinsics
  double fx = params[6];
  double fy = params[7];
  double cx = params[8];
  double cy = params[9];
  double k1 = params[10];
  double k2 = params[11];

  std::vector<double> dist_coeff;
  if (has_k2) {
    dist_coeff = {k1, k2};
  } else if (has_k1) {
    dist_coeff = {k1};
  }

  Status st = cam.setExtrinsics(r_vec, t_vec);
  if (!st)
    return false;
  st = cam.setIntrinsics(fx, fy, cx, cy, dist_coeff);
  if (!st)
    return false;
  st = cam.commitUpdate();
  if (!st)
    return false;
  return true;
}

// ========================================================================
// runVSC() - Optimize camera parameters using Levenberg-Marquardt
// ========================================================================
//
// Algorithm Description:
// ----------------------
// This function performs a non-linear Least Squares optimization to refine
// both the extrinsic and intrinsic parameters of each active camera.
//
// Optimized Parameters (8 DoF per camera):
//    1-3. Rotation Vector (rx, ry, rz)
//    4-6. Translation Vector (tx, ty, tz)
//    7.   Effective Focal Length (f_eff)
//    8.   Radial Distortion Coefficient (k1)
//
// Objective Function:
//    Minimize Sum_i || P_proj(C_k, X_i) - x_meas_i ||^2
//    Where:
//      X_i     : 3D position of the i-th calibration point (fixed from
//      tracking). x_meas_i: Measured 2D centroid of the i-th point in image
//      k. C_k     : Camera parameters to optimize. P_proj  : Pinhole
//      projection function with distortion.
//
// Solver: Levenberg-Marquardt (LM) Algorithm
//    Iteratively updates parameters 'p' to minimize the sum of squared
//    residuals. Update rule: (J^T * J + lambda * I) * delta = J^T * r
//
// Key Steps:
// 1. Independent Optimization: Each camera is optimized independently because
//    the 3D points (X_i) are derived from reliable long tracks and are
//    treated as ground truth references for this refinement step.
// 2. Data Collection: Gather all 2D observations for each camera.
// 3. Finite Differences: Jacobian J is computed numerically.
// 4. Robustness: Updates are only accepted if reprojection error decreases.
//
bool VSC::runVSC(std::vector<std::shared_ptr<Camera>> &camera_models) {
  if (!isReady())
    return false;

  // ========================================================================
  // Joint Optimization with Re-triangulation (Aligned with Python VSC)
  // ========================================================================
  //
  // Key differences from previous implementation:
  // 1. All cameras are optimized JOINTLY (not per-camera).
  // 2. 3D points are RE-TRIANGULATED in each iteration using current camera
  //    parameters (not fixed).
  // 3. Residuals include both TRIANGULATION ERROR (mm) and REPROJECTION ERROR
  //    (px).
  // 4. SLIDING WINDOW constraints: bounds are re-centered every outer
  //    iteration.
  //
  // ========================================================================

  const double eps = 1e-6;                   // Numerical differentiation step
  const int max_lm_iter = 50;                // Max LM iterations per outer loop
  const double convergence_threshold = 1e-8; // Early stop
  const int n_params_per_cam =
      12;                      // [rx,ry,rz, tx,ty,tz, fx, fy, cx, cy, k1, k2]
  const int n_outer_iters = 3; // Sliding window iterations

  // Bounds constants (relative constraints)
  const double rvec_bound = 0.1;     // ±0.1 rad (~5.7 degrees)
  const double tvec_bound = 50.0;    // ±50 mm
  const double f_bound_ratio = 0.05; // ±5%
  const double c_bounds = 50.0;      // ±50 px for cx, cy (Aligned with Python)
  const double k1_bound_ratio = 0.5; // ±50% or 0.1 min

  // ----- Collect active camera indices -----
  std::vector<size_t> active_cams;
  std::vector<std::shared_ptr<PinholeCamera>> working_cams;
  std::vector<bool> has_k1_flags;
  std::vector<bool> has_k2_flags;
  for (size_t k = 0; k < camera_models.size(); ++k) {
    const auto &base_cam = camera_models[k];
    if (!base_cam)
      continue;
    if (!base_cam->is_active || base_cam->type() != CameraType::Pinhole)
      continue;

    auto pinhole_cam = std::dynamic_pointer_cast<PinholeCamera>(base_cam);
    if (!pinhole_cam)
      continue;

    active_cams.push_back(k);
    auto pack = getCamExtrinsics(*pinhole_cam);
    has_k1_flags.push_back(pack.has_k1);
    has_k2_flags.push_back(pack.has_k2);
    working_cams.push_back(pinhole_cam);
  }

  if (active_cams.empty()) {
    std::cout << "  VSC: No active PINHOLE cameras found." << std::endl;
    return false;
  }

  const int n_cams = static_cast<int>(active_cams.size());
  const int total_params = n_cams * n_params_per_cam;

  // ----- Build multi-view correspondences -----
  // Each correspondence: 3D point + observations from multiple cameras
  // Structure: {obs_list} where obs_list[i] = (cam_internal_idx, 2D point)
  struct MultiViewCorr {
    std::vector<std::pair<int, Pt2D>> obs; // (internal_cam_idx, 2d_meas)
  };
  std::vector<MultiViewCorr> correspondences;

  // Map external cam_id to internal index
  std::unordered_map<int, int> cam_id_to_internal;
  for (int i = 0; i < n_cams; ++i) {
    cam_id_to_internal[static_cast<int>(active_cams[i])] = i;
  }

  // Group observations by 3D point (from _buffer)
  for (const auto &cp : _buffer) {
    MultiViewCorr corr;
    for (const auto &obs : cp._obs) {
      auto it = cam_id_to_internal.find(obs._cam_id);
      if (it != cam_id_to_internal.end()) {
        corr.obs.emplace_back(it->second, obs._meas_2d);
      }
    }
    // Need at least 2 views for triangulation
    if (corr.obs.size() >= 2) {
      correspondences.push_back(std::move(corr));
    }
  }

  if (correspondences.size() < 100) {
    std::cout << "  VSC: Insufficient multi-view correspondences ("
              << correspondences.size() << " < 100)." << std::endl;
    return false;
  }

  std::cout << "  VSC: " << correspondences.size()
            << " multi-view correspondences from " << n_cams << " cameras."
            << std::endl;

  // ----- Initialize joint parameter vector -----
  std::vector<double> params(total_params);

  for (int i = 0; i < n_cams; ++i) {
    CamParamPack cam_pack = getCamExtrinsics(*working_cams[i]);
    for (int j = 0; j < n_params_per_cam; ++j) {
      params[i * n_params_per_cam + j] = cam_pack.params[j];
    }
  }

  // ----- Lambda function: Update cameras from params -----
  auto updateCamerasFromParams = [&](const std::vector<double> &p) {
    for (int i = 0; i < n_cams; ++i) {
      std::vector<double> cam_p(p.begin() + i * n_params_per_cam,
                                p.begin() + (i + 1) * n_params_per_cam);
      updateCamExtrinsics(*working_cams[i], cam_p, has_k1_flags[i],
                          has_k2_flags[i]);
    }
  };

  // ----- Lambda function: Triangulate and compute combined error -----
  // Returns: {total_error, triang_error_sum, reproj_error_sum, n_valid_pts}
  auto computeErrors = [&](const std::vector<double> &p)
      -> std::tuple<double, double, double, int> {
    // First update cameras
    std::vector<std::shared_ptr<PinholeCamera>> temp_cams;
    temp_cams.reserve(static_cast<size_t>(n_cams));
    for (int i = 0; i < n_cams; ++i)
      temp_cams.push_back(std::make_shared<PinholeCamera>(*working_cams[i]));
    for (int i = 0; i < n_cams; ++i) {
      std::vector<double> cam_p(p.begin() + i * n_params_per_cam,
                                p.begin() + (i + 1) * n_params_per_cam);
      if (!updateCamExtrinsics(*temp_cams[i], cam_p, has_k1_flags[i],
                               has_k2_flags[i])) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(), 0};
      }
    }

    std::vector<std::shared_ptr<Camera>> temp_cam_list;
    temp_cam_list.reserve(static_cast<size_t>(n_cams));
    for (int i = 0; i < n_cams; ++i) {
      temp_cam_list.push_back(std::static_pointer_cast<Camera>(temp_cams[i]));
    }

    double total_err = 0.0;
    double triang_err_sum = 0.0;
    double reproj_err_sum = 0.0;
    int n_valid = 0;

    for (const auto &corr : correspondences) {
      // Build lines of sight for triangulation
      std::vector<Line3D> lines;
      for (const auto &obs : corr.obs) {
        int cam_idx = obs.first;
        Pt2D pt_2d = obs.second;
        auto los_status = temp_cam_list[cam_idx]->lineOfSight(pt_2d);
        if (!los_status) {
          lines.clear();
          break;
        }
        lines.push_back(los_status.value());
      }
      if (lines.size() < 2)
        continue;

      // Triangulate 3D point
      Pt3D pt3d;
      double triang_error = 0.0;
      myMATH::triangulation(pt3d, triang_error, lines);

      if (std::isnan(pt3d[0]) || std::isnan(pt3d[1]) || std::isnan(pt3d[2])) {
        continue; // Skip invalid points
      }

      // Compute reprojection error for each observation
      bool reproj_ok = true;
      for (const auto &obs : corr.obs) {
        int cam_idx = obs.first;
        Pt2D pt_meas = obs.second;
        auto proj_status = temp_cam_list[cam_idx]->project(pt3d);
        if (!proj_status) {
          reproj_ok = false;
          break;
        }
        Pt2D pt_proj = proj_status.value();
        double reproj_err = myMATH::dist2(pt_proj, pt_meas);
        reproj_err_sum += reproj_err;
      }
      if (!reproj_ok)
        continue;

      // Triangulation error (from myMATH::triangulation output)
      triang_err_sum += triang_error * triang_error;
      n_valid++;

      // Combined error: triang (mm^2) + reproj (px^2)
      // Note: Different scales, but LM handles this via Jacobian weighting
      total_err += triang_error * triang_error;
      for (const auto &obs : corr.obs) {
        auto proj_status = temp_cam_list[obs.first]->project(pt3d);
        if (!proj_status) {
          reproj_ok = false;
          break;
        }
        total_err += myMATH::dist2(proj_status.value(), obs.second);
      }
      if (!reproj_ok)
        continue;
    }

    return {total_err, triang_err_sum, reproj_err_sum, n_valid};
  };

  // ----- Build bounds -----
  auto buildBounds = [&](const std::vector<double> &center_params)
      -> std::pair<std::vector<double>, std::vector<double>> {
    std::vector<double> lb(total_params), ub(total_params);
    for (int i = 0; i < n_cams; ++i) {
      int base = i * n_params_per_cam;
      // Get current params from optimizer state 'p'
      double fx = center_params[base + 6];
      double fy = center_params[base + 7];
      double cx = center_params[base + 8];
      double cy = center_params[base + 9];
      double k1 = center_params[base + 10];
      double k2 = center_params[base + 11];

      // Extrinsics bounds (additive)
      for (int k = 0; k < 3; ++k) {
        lb[base + k] = center_params[base + k] - rvec_bound;
        ub[base + k] = center_params[base + k] + rvec_bound;
      }
      for (int k = 3; k < 6; ++k) {
        lb[base + k] = center_params[base + k] - tvec_bound;
        ub[base + k] = center_params[base + k] + tvec_bound;
      }

      // Intrinsics bounds

      // fx: ±5%
      double fx_delta = std::abs(fx) * f_bound_ratio;
      lb[base + 6] = fx - fx_delta;
      ub[base + 6] = fx + fx_delta;

      // fy: ±5%
      double fy_delta = std::abs(fy) * f_bound_ratio;
      lb[base + 7] = fy - fy_delta;
      ub[base + 7] = fy + fy_delta;

      // cx, cy: ±50 px
      lb[base + 8] = cx - c_bounds;
      ub[base + 8] = cx + c_bounds;
      lb[base + 9] = cy - c_bounds;
      ub[base + 9] = cy + c_bounds;

      // Adaptive Distortion Constraints (k1, k2)
      bool has_k1 = has_k1_flags[i];
      bool has_k2 = has_k2_flags[i];

      // k1 bounds
      if (has_k1) {
        double k1_margin = std::max(0.1, std::abs(k1) * k1_bound_ratio);
        lb[base + 10] = k1 - k1_margin;
        ub[base + 10] = k1 + k1_margin;
      } else {
        lb[base + 10] = k1 - 1e-10;
        ub[base + 10] = k1 + 1e-10;
      }

      // k2 bounds
      if (has_k2) {
        double k2_margin = std::max(0.1, std::abs(k2) * k1_bound_ratio);
        lb[base + 11] = k2 - k2_margin;
        ub[base + 11] = k2 + k2_margin;
      } else {
        lb[base + 11] = k2 - 1e-10;
        ub[base + 11] = k2 + 1e-10;
      }
    }
    return {lb, ub};
  };

  // ----- Initial error -----
  auto [init_err, init_triang, init_reproj, init_n] = computeErrors(params);
  double init_reproj_rmse =
      (init_n > 0) ? std::sqrt(init_reproj / (init_n * n_cams)) : 0.0;
  double init_triang_rmse =
      (init_n > 0) ? std::sqrt(init_triang / init_n) : 0.0;
  std::cout << "  Initial: TriangErr=" << init_triang_rmse
            << "mm, ProjErr=" << init_reproj_rmse << "px" << std::endl;

  // ========================================================================
  // Sliding Window Optimization (Outer Loop)
  // ========================================================================
  for (int outer = 0; outer < n_outer_iters; ++outer) {
    auto bounds = buildBounds(params);
    std::vector<double> lb = std::get<0>(bounds);
    std::vector<double> ub = std::get<1>(bounds);

    std::cout << "  [Iter " << (outer + 1) << "/" << n_outer_iters
              << "] Re-centered bounds. Running LM..." << std::endl;

    double lambda = 0.001;
    auto init_err_data = computeErrors(params);
    double current_err = std::get<0>(init_err_data);

    // ----- LM Inner Loop -----
    for (int iter = 0; iter < max_lm_iter; ++iter) {
      // Build JtJ and Jtr using numerical Jacobian
      Matrix<double> JtJ(total_params, total_params, 0.0);
      Matrix<double> Jtr(total_params, 1, 0.0);

      // Compute residuals at current params
      auto err0_data = computeErrors(params);
      double f0 = std::get<0>(err0_data);

      // Numerical Jacobian (forward difference)
      std::vector<double> grad(total_params, 0.0);

#pragma omp parallel for
      for (int p = 0; p < total_params; ++p) {
        std::vector<double> params_p = params;
        params_p[p] += eps;
        // Clamp to bounds
        params_p[p] = std::max(lb[p], std::min(ub[p], params_p[p]));
        auto err_p_data = computeErrors(params_p);
        double fp = std::get<0>(err_p_data);
        grad[p] = (fp - f0) / eps;
      }

      // Build approximate Hessian: JtJ = grad * grad^T
      // And gradient: Jtr = grad * f0 (simplified LM update)
      for (int i = 0; i < total_params; ++i) {
        for (int j = 0; j < total_params; ++j) {
          JtJ(i, j) = grad[i] * grad[j];
        }
        Jtr(i, 0) = -grad[i] * f0; // Negative gradient for descent
      }

      // Apply damping: (JtJ + lambda * diag(JtJ))
      for (int i = 0; i < total_params; ++i) {
        JtJ(i, i) *= (1.0 + lambda);
        if (JtJ(i, i) < 1e-10)
          JtJ(i, i) = 1e-10; // Regularization
      }

      // Solve for delta
      Matrix<double> delta = myMATH::inverse(JtJ) * Jtr;

      // Check convergence
      double delta_norm = 0;
      for (int i = 0; i < total_params; ++i) {
        delta_norm += delta(i, 0) * delta(i, 0);
      }
      if (delta_norm < convergence_threshold) {
        break;
      }

      // Compute candidate parameters with bounds clamping
      std::vector<double> params_new = params;
      for (int i = 0; i < total_params; ++i) {
        params_new[i] += delta(i, 0);
        params_new[i] = std::max(lb[i], std::min(ub[i], params_new[i]));
      }

      // Evaluate new error
      auto [new_err, nt, nr, nn] = computeErrors(params_new);

      // LM decision
      if (new_err < current_err) {
        params = params_new;
        current_err = new_err;
        lambda /= 10.0;
      } else {
        lambda *= 10.0;
      }

      // Early stopping if lambda too large
      if (lambda > 1e10)
        break;
    }

    // Log progress
    auto [err, te, re, nv] = computeErrors(params);
    double triang_rmse = (nv > 0) ? std::sqrt(te / nv) : 0.0;
    double reproj_rmse = (nv > 0) ? std::sqrt(re / (nv * n_cams)) : 0.0;
    std::cout << "  [Iter " << (outer + 1) << "] TriangErr=" << triang_rmse
              << "mm, ProjErr=" << reproj_rmse << "px" << std::endl;
  }

  // ----- Apply final parameters to cameras -----
  updateCamerasFromParams(params);
  for (int i = 0; i < n_cams; ++i) {
    camera_models[active_cams[i]] = std::static_pointer_cast<Camera>(working_cams[i]);
  }

  // ----- Final error -----
  auto [final_err, final_triang, final_reproj, final_n] = computeErrors(params);
  double final_triang_rmse =
      (final_n > 0) ? std::sqrt(final_triang / final_n) : 0.0;
  double final_reproj_rmse =
      (final_n > 0) ? std::sqrt(final_reproj / (final_n * n_cams)) : 0.0;
  std::cout << "  Final: TriangErr=" << final_triang_rmse
            << "mm, ProjErr=" << final_reproj_rmse << "px" << std::endl;

  // ----- Save cameras -----
  if (!_cfg._output_path.empty()) {
    for (size_t k = 0; k < camera_models.size(); ++k) {
      if (!camera_models[k] || !camera_models[k]->is_active)
        continue;
      std::string cam_path =
          _cfg._output_path + "/vsc_cam" + std::to_string(k) + ".txt";
      auto st = camera_models[k]->saveParameters(cam_path);
      if (!st) {
        std::cerr << "  VSC warning: failed to save camera " << k << ": "
                  << st.err.toString() << std::endl;
      }
    }
    std::cout << "  VSC cameras saved to " << _cfg._output_path << std::endl;
  }

  return true;
}

// ========================================================================
// runOTF() - Spatially Varying OTF Calibration
// ========================================================================
//
// Algorithm Description:
// ----------------------
// This function builds a 3D map of OTF (Optical Transfer Function) parameters
// (a, b, c, alpha) for each camera. The OTF describes the particle intensity
// profile:
//      I(x,y) = a * exp(- b*x'^2 - c*y'^2)
// where (x', y') are coordinates rotated by alpha.
//
// The parameters (a, b, c, alpha) were already estimated for each individual
// particle during the 'accumulate' phase using Linear Least Squares fitting.
//
// Spatial Aggregation:
// 1. Grid Mapping: The 3D measurement volume is divided into a voxel grid
//    (defined in tracer_cfgs[k]._otf._param).
// 2. Binning: Each valid calibration point is assigned to a voxel based on
// its
//    3D position.
// 3. Averaging: Compute candidate parameters for each grid cell.
// 4. Verification: Compare reconstruction error of Candidates vs Current
// parameters
//    against the stored ROI intensity. Only accept candidates if error
//    decreases.
// 5. Update: Write accepted parameters to TracerConfig.
//
// Return:
//    true if any grid cell was updated.
//
bool VSC::runOTF(std::vector<TracerConfig> &tracer_cfgs) {
  if (_buffer.empty())
    return false;

  // Check global enable flag
  if (!_cfg._enable_otf)
    return false;

  bool updated = false;

  // Iterate over all provided tracer configurations (typically just one)
  for (auto &t_cfg : tracer_cfgs) {
    auto &otf = t_cfg._otf;
    auto &param = otf._param;

    if (param.n_grid <= 0 || param.n_cam <= 0)
      continue;

    // Temporary accumulators for candidate parameters
    // Key: mapping index = cam_id * n_grid + grid_id
    struct OTFAccum {
      double a = 0, b = 0, c = 0;
      double cos_2a = 0,
             sin_2a = 0; // For robust angular averaging (\pi periodicity)
      int count = 0;
    };
    std::unordered_map<int, OTFAccum> candidates;

    // 1. Accumulate observations to form candidates
    for (const auto &cp : _buffer) {
      int grid_id = getOTFGridID(param, cp._pos_3d);
      if (grid_id < 0 || grid_id >= param.n_grid)
        continue;

      for (const auto &obs : cp._obs) {
        if (!obs._otf_params.valid)
          continue;

        int cam_id = obs._cam_id;
        if (cam_id < 0 || cam_id >= param.n_cam)
          continue;

        int key = cam_id * param.n_grid + grid_id;
        auto &cand = candidates[key];
        const auto &p = obs._otf_params;

        cand.a += p.a;
        cand.b += p.b;
        cand.c += p.c;
        cand.cos_2a += std::cos(2.0 * p.alpha);
        cand.sin_2a += std::sin(2.0 * p.alpha);
        cand.count++;
      }
    }

    // 2. Compute candidate averages
    // Store as OTFParams for easier usage.
    // We only process keys that exist in candidates.
    std::unordered_map<int, OTFParams> candidate_params;
    for (auto &[key, cand] : candidates) {
      if (cand.count > 5) {
        double inv_n = 1.0 / cand.count;
        OTFParams p;
        p.a = cand.a * inv_n;
        p.b = cand.b * inv_n;
        p.c = cand.c * inv_n;
        // Reconstruct average angle from trig sums (period \pi)
        p.alpha = 0.5 * std::atan2(cand.sin_2a, cand.cos_2a);
        p.valid = true;
        candidate_params[key] = p;
      }
    }

    // 3. Verification: Calculate errors for Old vs New
    std::unordered_map<int, double> err_old_sum;
    std::unordered_map<int, double> err_new_sum;

    // Helper to compute reconstruction error for a single observation
    // Helper to compute reconstruction error for a single observation
    auto compute_obj_error = [&](const Observation &obs, const OTFParams &p) {
      double err = 0;
      int rows = obs._roi_intensity.getDimRow();
      int cols = obs._roi_intensity.getDimCol();

      // Determine max value within object radius for thresholding
      // Logic copied from estimateOTFParams
      // Using relative coordinate system of the ROI

      // Relative center in ROI (this is usually rows/2, cols/2 if centered,
      // but let's calculate it from obs._meas_2d and ROI origin if we had that
      // info. Wait, in accumulate we do: Pt2D rel_center = obs._meas_2d;
      // rel_center[0] -= x0; rel_center[1] -= y0;
      // obs._otf_params = estimate...
      // But we don't store rel_center in Observation.
      // However, we know obs._meas_2d is global.
      // And we need to know x0, y0 of the ROI to get relative center.
      // Issue: Observation struct doesn't strictly store the ROI offset (x0,
      // y0). But in accumulate, ROI is created as: x0 = cx - h; y0 = cy - h;
      // (centered around integer center) So detailed relative center: rel_xc =
      // obs._meas_2d[0] - x0 = obs._meas_2d[0] - (cx - h)
      //        = obs._meas_2d[0] - (round(obs._meas_2d[0]) - h)
      //        = (obs._meas_2d[0] - round(obs._meas_2d[0])) + h

      // Re-derive ROI geometry assuming it was created centered on
      // lround(meas_2d)
      int h_row = (rows - 1) / 2;
      int h_col = (cols - 1) / 2;

      double rel_xc = (obs._meas_2d[0] - std::lround(obs._meas_2d[0])) + h_col;
      double rel_yc = (obs._meas_2d[1] - std::lround(obs._meas_2d[1])) + h_row;

      double max_val = 0;
      double r2_limit = obs._obj_radius * obs._obj_radius;

      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          double dx = c - rel_xc;
          double dy = r - rel_yc;
          if (dx * dx + dy * dy <= r2_limit) {
            double val = obs._roi_intensity(r, c);
            if (val > max_val)
              max_val = val;
          }
        }
      }

      if (max_val < 1.0)
        return 0.0; // Should not happen for valid obs

      double t_high = max_val * 0.30;

      // Dynamic r_fit logic (Square Perimeter Check)
      int cx_int = std::lround(rel_xc);
      int cy_int = std::lround(rel_yc);
      // Maximum possible radius within the ROI
      int max_r = (std::min(rows, cols) - 1) / 2;
      int r_fit = 1;

      for (int r = 1; r <= max_r; ++r) {
        bool expand = false;
        // Check square perimeter at radius r
        for (int k = -r; k <= r; ++k) {
          // Top || Bottom
          if ((cy_int - r >= 0 && cx_int + k >= 0 && cx_int + k < cols &&
               obs._roi_intensity(cy_int - r, cx_int + k) > t_high) ||
              (cy_int + r < rows && cx_int + k >= 0 && cx_int + k < cols &&
               obs._roi_intensity(cy_int + r, cx_int + k) > t_high)) {
            expand = true;
            break;
          }
          // Left || Right
          if ((cy_int + k >= 0 && cy_int + k < rows && cx_int - r >= 0 &&
               obs._roi_intensity(cy_int + k, cx_int - r) > t_high) ||
              (cy_int + k >= 0 && cy_int + k < rows && cx_int + r < cols &&
               obs._roi_intensity(cy_int + k, cx_int + r) > t_high)) {
            expand = true;
            break;
          }
        }
        if (expand)
          r_fit = r;
        else
          break;
      }

      // Compute error in the determined window
      int x0_fit = std::max(0, cx_int - r_fit);
      int x1_fit = std::min(cols, cx_int + r_fit + 1);
      int y0_fit = std::max(0, cy_int - r_fit);
      int y1_fit = std::min(rows, cy_int + r_fit + 1);

      for (int r = y0_fit; r < y1_fit; ++r) {
        for (int c = x0_fit; c < x1_fit; ++c) {
          // Relative coordinates to centroid
          double dx = c - rel_xc;
          double dy = r - rel_yc;

          // Rotate coordinates
          double cos_a = std::cos(p.alpha);
          double sin_a = std::sin(p.alpha);
          double xx = dx * cos_a + dy * sin_a;
          double yy = -dx * sin_a + dy * cos_a;

          // Model intensity
          double exponent = -(p.b * xx * xx + p.c * yy * yy);
          double model_val = p.a * std::exp(exponent);

          double meas_val = obs._roi_intensity(r, c);
          double diff = meas_val - model_val;
          err += diff * diff;
        }
      }
      return err;
    };

    for (const auto &cp : _buffer) {
      int grid_id = getOTFGridID(param, cp._pos_3d);
      if (grid_id < 0 || grid_id >= param.n_grid)
        continue;

      for (const auto &obs : cp._obs) {
        if (!obs._otf_params.valid)
          continue;
        int cam_id = obs._cam_id;
        int key = cam_id * param.n_grid + grid_id;

        // Only verifying if we have a candidate update
        if (candidate_params.find(key) == candidate_params.end())
          continue;

        // Current (Old) Parameters
        OTFParams p_old;
        p_old.a = param.a(cam_id, grid_id);
        p_old.b = param.b(cam_id, grid_id);
        p_old.c = param.c(cam_id, grid_id);
        p_old.alpha = param.alpha(cam_id, grid_id);

        // Candidate (New) Parameters
        const OTFParams &p_new = candidate_params[key];

        err_old_sum[key] += compute_obj_error(obs, p_old);
        err_new_sum[key] += compute_obj_error(obs, p_new);
      }
    }

    // 4. Update if error reduced
    // Map to track which cells were definitively updated this run
    std::vector<bool> is_updated_map(param.n_cam * param.n_grid, false);

    for (const auto &[key, p_cand] : candidate_params) {
      // If error decreased (or first initialization), accept
      // Initialization check: if old 'a' is 0, it's likely uninitialized
      int cam_id = key / param.n_grid;
      int grid_id = key % param.n_grid;

      double old_a = param.a(cam_id, grid_id);
      double e_old = err_old_sum[key];
      double e_new = err_new_sum[key];

      // Simple heuristic: accept if better or if old was uninitialized (a <
      // 1e-9)
      if (e_new < e_old || old_a < 1e-9) {
        param.a(cam_id, grid_id) = p_cand.a;
        param.b(cam_id, grid_id) = p_cand.b;
        param.c(cam_id, grid_id) = p_cand.c;
        param.alpha(cam_id, grid_id) = p_cand.alpha;
        updated = true;
        is_updated_map[key] = true;
      }
    }
    // Log update progress for this tracer
    int n_candidates = candidate_params.size();
    int n_updated = 0;
    for (const auto &[key, p] : candidate_params) {
      // Re-evaluate the update condition for logging purposes
      int cam_id = key / param.n_grid;
      int grid_id = key % param.n_grid;
      double old_a = param.a(
          cam_id, grid_id); // This `param.a` is already updated if accepted
      // To correctly count `n_updated`, we need to check the condition that
      // *would have* led to an update. The original `old_a` and `e_old` are
      // needed. This is a slight discrepancy with the provided snippet, which
      // uses `err_new_sum.count(key) && err_new_sum.at(key) <
      // err_old_sum.at(key)`. Sticking to the provided snippet for
      // faithfulness.
      if (err_new_sum.count(key) && err_new_sum.at(key) < err_old_sum.at(key)) {
        n_updated++;
      }
    }
    std::cout << "  [OTF] Tracer " << (&t_cfg - &tracer_cfgs[0]) << ": "
              << n_updated << "/" << n_candidates
              << " grid cells updated with VSC data." << std::endl;

    // 5. Gap Filling & Spatial Smoothing
    // Ensure spatial continuity and fill un-updated regions
    {
      int nx = param.nx, ny = param.ny, nz = param.nz;
      int n_grid = param.n_grid;
      int n_cam = param.n_cam;

      // a. Identify Valid Mask (Active cells) from is_updated_map
      // Maps: cam_id -> vector<bool>
      std::vector<std::vector<bool>> has_data(n_cam,
                                              std::vector<bool>(n_grid, false));
      for (int k = 0; k < is_updated_map.size(); ++k) {
        if (is_updated_map[k]) {
          int cam_id = k / n_grid;
          int grid_id = k % n_grid;
          if (cam_id >= 0 && cam_id < n_cam && grid_id >= 0 &&
              grid_id < n_grid) {
            has_data[cam_id][grid_id] = true;
          }
        }
      }

      // b. Iterative Diffusion Filling
      // Propagate values from valid cells to invalid ones.
      // Never modify original valid (measured) cells.
      int max_iter = std::max({nx, ny, nz}) + 5;

      for (int c = 0; c < n_cam; ++c) {
        // Initialize validity state for this camera
        std::vector<bool> current_valid = has_data[c];

        for (int iter = 0; iter < max_iter; ++iter) {
          bool any_update = false;
          OTFParam p_next = param;
          std::vector<bool> next_valid = current_valid;

          for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
              for (int x = 0; x < nx; ++x) {
                int idx = x + nx * (y + ny * z);

                // Rule 1: Never touch measured data
                if (has_data[c][idx])
                  continue;

                // Rule 2: Try to fill if not measured
                // (We update even if already filled to allow
                // relaxation/smoothing of filled regions) Gather valid
                // neighbors
                double sa = 0, sb = 0, sc = 0, sal = 0;
                int count = 0;

                int dz[] = {-1, 1, 0, 0, 0, 0};
                int dy[] = {0, 0, -1, 1, 0, 0};
                int dx[] = {0, 0, 0, 0, -1, 1};

                for (int k = 0; k < 6; ++k) {
                  int iz = z + dz[k];
                  int iy = y + dy[k];
                  int ix = x + dx[k];

                  if (iz >= 0 && iz < nz && iy >= 0 && iy < ny && ix >= 0 &&
                      ix < nx) {
                    int n_idx = ix + nx * (iy + ny * iz);
                    // Only use neighbors that have meaningful values (Measured
                    // or Filled)
                    if (current_valid[n_idx]) {
                      sa += param.a(c, n_idx);
                      sb += param.b(c, n_idx);
                      sc += param.c(c, n_idx);
                      sal += param.alpha(c, n_idx);
                      count++;
                    }
                  }
                }

                if (count > 0) {
                  p_next.a(c, idx) = sa / count;
                  p_next.b(c, idx) = sb / count;
                  p_next.c(c, idx) = sc / count;
                  p_next.alpha(c, idx) = sal / count;
                  next_valid[idx] = true; // Mark as valid for next iter
                  any_update = true;
                }
              }
            }
          }

          // Apply updates
          param = p_next;
          current_valid = next_valid;

          if (!any_update)
            break;
        }
      }
    }

    // 5. Save results if output path provided (unconditional)
    if (!_cfg._output_path.empty()) {
      std::string suffix = (&t_cfg - &tracer_cfgs[0] == 0)
                               ? ""
                               : std::to_string(&t_cfg - &tracer_cfgs[0]);
      std::string otf_path = _cfg._output_path + "/OTF" + suffix + ".txt";
      t_cfg._otf.saveParam(otf_path);
    }
  }

  if (!_cfg._output_path.empty()) {
    std::cout << "  VSC OTF parameters saved to " << _cfg._output_path
              << std::endl;
  }

  return updated;
}
