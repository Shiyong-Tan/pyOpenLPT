
/**
 * StereoMatch
 * Pipeline: build (first m active cams) -> check (remaining active cams)
 *           -> prune (global disjoint selection) -> triangulate (final
 * Object3D).
 *
 * Notes:
 * - A 2D observation may be reused across different matches until the prune
 * step.
 * - Per-call inputs are bound at the start of match() via internal pointers.
 *   Do not call match() concurrently on the same instance.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex> // for std::mutex
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "Camera.h" // Camera model interface
#include "Config.h"
#include "Matrix.h"
#include "ObjectInfo.h"
#include "STBCommons.h"
#include "myMATH.h" // Pt2D, Pt3D, Line2D, Line3D

// Forward declarations of types used here
class IDMap;

class StereoMatch {
public:
  // Constructor
  explicit StereoMatch(
      const std::vector<std::shared_ptr<Camera>> &camera_models,
      const std::vector<std::vector<std::unique_ptr<Object2D>>> &obj2d_list,
      const ObjectConfig
          &obj_cfg); // All reference variable should be initialized first

  // Main entry
  std::vector<std::unique_ptr<Object3D>> match() const;

private:
  // ---- bound per-call inputs (read-only during a match) ----
  const std::vector<std::shared_ptr<Camera>> &_cam_list;
  const ObjectConfig &_obj_cfg;
  std::vector<std::vector<const Object2D *>> _obj2d_list;

  // ---- shared accelerators usable by build/check during a match ----
  std::vector<std::unique_ptr<IDMap>>
      _idmaps; // one per camera (nullptr if inactive)
  // _idmaps must be built in the contructor, because match() const cannot
  // change any member variable due to "const".

  // ---- main pipeline pieces ----
  void buildMatch(std::vector<int> &match_candidate_id,
                  std::vector<std::vector<int>> &build_candidates) const;

  // Enumerate candidates on a specific camera, given the 3D lines of sight,
  // chosen cams and chosen pts are used for back-projection check.
  void enumerateCandidatesOnCam(const std::vector<Line3D> &los3d,
                                int target_cam,
                                const std::vector<int> &chosen_cams,
                                const std::vector<Pt2D> &chosen_pts,
                                std::vector<int> &out_candidates) const;

  bool checkMatch(const std::vector<int> &candidate_ids,
                  double &out_e_check)
      const; // out_e_check: the triangulation error obtained during checking

  std::vector<std::vector<int>>
  pruneMatch(const std::vector<std::vector<int>> &match_candidates,
             const std::vector<double> &e_checks)
      const; // e_checks is used to prune matches

  std::vector<std::unique_ptr<Object3D>>
  triangulateMatch(const std::vector<std::vector<int>> &selected_matches) const;

  // ---- camera selection helpers ----
  int selectSecondCameraByLineLength(
      const Line3D &los_ref, const std::vector<int> &remaining_cams) const;

  int selectNextCameraByMaxPairAngle(
      const std::vector<Line3D> &los3d,
      const std::vector<int> &remaining_cams) const;

  // ---- LOS→image line helpers ----
  bool makeLine2DFromLOS3D(int cam_id, const Line3D &los, Line2D &out_line) const;

  bool buildLinesOnCam(const std::vector<Line3D> &los3d, int cam_id,
                       std::vector<Line2D> &out_lines) const;

  // ---- early checks & tolerances (decl only; you已有实现或后续实现) ----
  double computeMinParallaxDeg(const std::vector<Line3D> &los3d) const;
  double calTriangulateTol(double final_tol_3d_mm, int k_selected, int k_target,
                           double min_parallax_deg) const;
  bool TriangulationCheckWithTol(const std::vector<Line3D> &los3d,
                                 double tol_3d_mm) const;
  double triangulationVariance(const std::vector<Line3D> &los) const;

  // project point q_t on target_cam back to all chosen cams, check distance to
  // chosen_pts
  bool checkBackProjection(int target_cam, const Pt2D &q_t,
                           const std::vector<int> &chosen_cams,
                           const std::vector<Pt2D> &chosen_pts) const;

  bool objectEarlyCheck(const std::vector<int> &cams_used,
                        const std::vector<int> &ids_on_used) const;
  bool tracerEarlyCheck(const std::vector<int> &cams_in_path,
                        const std::vector<int> &ids_in_path) const;
  bool bubbleEarlyCheck(const std::vector<int> &cams_in_path,
                        const std::vector<int> &ids_in_path) const;

  FRIEND_DEBUG(StereoMatch); // for debugging private members
};

/**
 * @brief Sparse grid (cell buckets) over an image for fast 2D candidate lookup.
 *        Cells are axis-aligned squares of size _cell_px (in pixels).
 *        Each cell stores the indices (pids) of 2D observations.
 */
// IDMap.hpp

class IDMap {
public:
  struct RowSpan {
    int x_min, x_max;
  }; // invalid if x_min > x_max

  IDMap(int img_rows_px, int img_cols_px, int cell_px);

  // Rebuild the buckets for this camera using a contiguous array of 2D objects.
  // 'objs' must outlive the IDMap while you iterate.
  void rebuild(const std::vector<const Object2D *> &objs);

  // Compute per-row intersection of K LOS strips; result in CELL indices.
  // 'spans' is resized to rowsCell(), each entry holds [cx_min, cx_max]
  // (inclusive).
  void computeStripIntersection(const std::vector<Line2D> &lines_px,
                                double tol_px,
                                std::vector<RowSpan> &spans) const;

  // Enumerate points ONLY inside the final per-row spans, with precise
  // point-to-line checks. Geometry check: for each point q, require
  // distance_to_every_line(q) <= tol_px. Dedup is handled internally (a point
  // may appear in multiple cells).
  void visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                             const std::vector<Line2D> &lines_px, double tol_px,
                             std::vector<int> &out_indices) const;

  // ---- small helpers (inline) ----
  inline int rowsCell() const { return _rows_cell; }
  inline int colsCell() const { return _cols_cell; }
  inline int cellSizePx() const { return _cell_px; }

private:
  // Row-major bucket index
  inline int idx(int cy, int cx) const { return cy * _cols_cell + cx; }

  // Normalize a 2D vector defensively (used inside .cpp)
  static Pt2D normalized(const Pt2D &v);

private:
  int _img_rows_px = 0;
  int _img_cols_px = 0;
  int _cell_px = 1;
  int _rows_cell = 0;
  int _cols_cell = 0;

  // Buckets: one vector<int> per cell (row-major)
  std::vector<std::vector<int>> _buckets;

  // Pointer to external storage of Object2D* for coordinate access during
  // enumeration
  const std::vector<const Object2D *> *_objs = nullptr;
};
