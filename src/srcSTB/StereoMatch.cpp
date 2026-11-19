#include "StereoMatch.h"
#include <array>
#include <iterator>
#include <omp.h>

static std::vector<std::vector<const Object2D *>>
makeView(const std::vector<std::vector<std::unique_ptr<Object2D>>> &src2d) {
  // convert the input obj2d_list (unique_ptr) to view-only constant pointer
  // which can make coding easier
  std::vector<std::vector<const Object2D *>> view(src2d.size());
  for (size_t c = 0; c < src2d.size(); ++c) {
    auto &dst = view[c];
    dst.reserve(src2d[c].size());
    for (auto const &up : src2d[c])
      dst.push_back(up.get());
  }
  return view;
}

StereoMatch::StereoMatch(
    const std::vector<std::shared_ptr<Camera>> &camera_models,
    const std::vector<std::vector<std::unique_ptr<Object2D>>> &obj2d_list,
    const ObjectConfig &obj_cfg)
    : _cam_list(camera_models), _obj_cfg(obj_cfg), _obj2d_list(makeView(obj2d_list)) {
  // pre-check
  const int n_cams = static_cast<int>(_cam_list.size());
  REQUIRE(n_cams > 0, ErrorCode::InvalidArgument, "preCheck: cams is empty.");
  REQUIRE(static_cast<int>(_obj2d_list.size()) == n_cams,
          ErrorCode::InvalidArgument,
          "preCheck: cams and obj2d_list sizes must match.");

  int n_active = 0;
  for (int i = 0; i < n_cams; ++i)
    if (_cam_list[i] && _cam_list[i]->is_active)
      ++n_active;

  REQUIRE(n_active >= 2, ErrorCode::InvalidArgument,
          "preCheck: need at least 2 active cameras.");

  int match_cam_count = _obj_cfg._sm_param.match_cam_count;
  REQUIRE_CTX(match_cam_count >= 2 && match_cam_count <= n_active,
              ErrorCode::InvalidArgument,
              "preCheck: match_cam_count must be within [2," +
                  std::to_string(n_active) + "].",
              "match_cam_count = " + std::to_string(match_cam_count));

  _pinhole_matching = true;
  for (const auto& cam : _cam_list)
    if (cam && cam->is_active && cam->type() != CameraType::Pinhole)
      _pinhole_matching = false;
  _matching_pixels.resize(n_cams);
  _matching_valid.resize(n_cams);
  _valid_counts.resize(n_cams, 0);
  _matching_idmaps.resize(n_cams);
  if (_pinhole_matching) {
    for (int c=0;c<n_cams;++c) {
      if (!_cam_list[c] || !_cam_list[c]->is_active) continue;
      const auto& cam = static_cast<const PinholeCamera&>(*_cam_list[c]);
      const auto& p = cam.param();
      auto finiteMatrix = [](const Matrix<double>& m) {
        if (m.getDimRow()!=3 || m.getDimCol()!=3) return false;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j)
          if(!std::isfinite(m(i,j))) return false;
        return true;
      };
      bool valid = finiteMatrix(p.cam_mtx) && finiteMatrix(p.r_mtx) &&
                   finiteMatrix(p.r_mtx_inv);
      for(int i=0;i<3;++i)
        valid = valid && std::isfinite(p.t_vec[i]) && std::isfinite(p.t_vec_inv[i]);
      if(valid) {
        auto identity=p.r_mtx*p.r_mtx_inv;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j)
          valid=valid && std::abs(identity(i,j)-(i==j?1.:0.))<=1e-8;
      }
      if (valid) valid = p.cam_mtx(0,0)!=0 && p.cam_mtx(1,1)!=0;
      if (p.is_distorted) valid = valid &&
        (p.n_dist_coeff==4 || p.n_dist_coeff==5 || p.n_dist_coeff==8 || p.n_dist_coeff==12) &&
        p.dist_coeff.size()==static_cast<size_t>(p.n_dist_coeff);
      for(double k:p.dist_coeff) valid = valid && std::isfinite(k);
      REQUIRE(valid, ErrorCode::InvalidArgument,
              "StereoMatch: invalid pinhole calibration on camera " + std::to_string(c));
      const size_t n = _obj2d_list[c].size();
      _matching_pixels[c].resize(n);
      _matching_valid[c].resize(n,0);
      for(size_t id=0;id<n;++id) {
        const auto* obj = _obj2d_list[c][id];
        if(obj && cam.matchingPixel(obj->_pt_center,_matching_pixels[c][id])) {
          _matching_valid[c][id]=1;
          ++_valid_counts[c];
        }
      }
      _matching_idmaps[c]=std::make_unique<IDMap>(1,1,_obj_cfg._sm_param.idmap_cell_px);
      _matching_idmaps[c]->rebuildCoordinates(_matching_pixels[c],_matching_valid[c]);
    }
  }

  // ---- (Re)build IDMaps for all active cams once (used by
  // buildMatch/checkMatch). ----
  _idmaps.clear();
  _idmaps.resize(n_cams);
  for (int c = 0; c < n_cams; ++c) {
    if (!_cam_list[c] || !_cam_list[c]->is_active)
      continue;
    // create IDMap for this camera
    _idmaps[c] = std::make_unique<IDMap>(_cam_list[c]->getNRow(), _cam_list[c]->getNCol(),
                                         _obj_cfg._sm_param.idmap_cell_px);

    // build IDMap buckets
    _idmaps[c]->rebuild(_obj2d_list[c]);
  }
} // All reference variable should be initialized first

std::vector<std::unique_ptr<Object3D>> StereoMatch::match() const {
  const int n_cams = static_cast<int>(_cam_list.size());

  // ---- Choose reference camera: among active cams, pick the one with the
  // fewest 2D observations. ----
  int ref_cam = -1;
  {
    size_t best = std::numeric_limits<size_t>::max();
    for (int i = 0; i < n_cams; ++i) {
      if (!_cam_list[i] || !_cam_list[i]->is_active)
        continue;
      const size_t sz = _pinhole_matching ? _valid_counts[i] : _obj2d_list[i].size();
      if (sz < best) {
        best = sz;
        ref_cam = i;
      }
    }
    if (ref_cam < 0) {
      throw std::runtime_error("StereoMatch::match: no active camera found.");
    }
  }

  // ---- Stage A + B: build match candidates per reference observation, then
  // check on remaining cams. ----
  const auto &ref_obs = _obj2d_list[ref_cam];

  // Prepare per-thread buckets (no sharing between threads)
  const int T = std::max(1, omp_get_max_threads());

  // bind match ids and e_check (triangulation error) together, guarantee the
  // same order
  struct MatchWithScore {
    std::vector<int> ids;
    double e_check = 0.0;
  };
  std::vector<std::vector<MatchWithScore>> thread_buckets(T);

#pragma omp parallel num_threads(T)
  {
    const int tid = omp_get_thread_num();
    auto &bucket = thread_buckets[tid]; // thread-local bucket (by index)

#pragma omp for schedule(dynamic, 32)
    for (int id_ref = 0; id_ref < static_cast<int>(ref_obs.size()); ++id_ref) {
      if (_pinhole_matching && !_matching_valid[ref_cam][id_ref]) continue;
      // One candidate initialized with -1 everywhere and ref_cam filled.
      std::vector<int> match_candidate_id(n_cams, -1);
      match_candidate_id[ref_cam] = id_ref;

      // Build across the first m active cams (DFS inside; uses new selection
      // rules).
      std::vector<std::vector<int>> build_candidates;
      buildMatch(match_candidate_id, build_candidates); // must be thread-safe

      // Check on remaining active cams (existence / object-specific light
      // checks).
      std::vector<MatchWithScore> selected; // per-iter local
      selected.reserve(build_candidates.size());
      for (auto &cand : build_candidates) {
        double e_check = std::numeric_limits<double>::quiet_NaN();
        if (!checkMatch(static_cast<const std::vector<int> &>(cand), e_check))
          continue;
        REQUIRE(!std::isnan(e_check), ErrorCode::InvalidArgument,
                "StereoMatch::match: checkMatch returned NaN e_check.");
        selected.emplace_back(MatchWithScore{std::move(cand), e_check});
      }

      // Move this thread's selected candidates into its own bucket (no lock)
      if (!selected.empty()) {
        bucket.insert(bucket.end(), std::make_move_iterator(selected.begin()),
                      std::make_move_iterator(selected.end()));
      }
    }
  } // parallel region ends

  // Flatten buckets sequentially (single-thread).
  std::vector<MatchWithScore> match_score_all;
  {
    size_t total = 0;
    for (auto &b : thread_buckets)
      total += b.size();
    match_score_all.reserve(total);
    for (auto &b : thread_buckets) {
      match_score_all.insert(match_score_all.end(),
                             std::make_move_iterator(b.begin()),
                             std::make_move_iterator(b.end()));
    }
  }
  // make sure the sequence the same everytime running the code
  std::sort(match_score_all.begin(), match_score_all.end(),
            [](const MatchWithScore &a, const MatchWithScore &b) {
              return a.ids < b.ids;
            });

  // ---- Split match_score_all into two parallel arrays for pruning. ----
  std::vector<std::vector<int>> match_candidates;
  std::vector<double> e_checks;
  match_candidates.reserve(match_score_all.size());
  e_checks.reserve(match_score_all.size());
  for (auto &m : match_score_all) {
    match_candidates.push_back(std::move(m.ids));
    e_checks.push_back(m.e_check);
  }

  // ---- Global pruning (disjoint usage of 2D points, pick best by
  // triangulation error, etc.). ----
  std::vector<std::vector<int>> selected =
      pruneMatch(match_candidates, e_checks);

  // ---- Final triangulation and Object3D construction (all active cams
  // participate). ----
  std::vector<std::unique_ptr<Object3D>> out = triangulateMatch(selected);

  return out;
}

// Build across the first m active cameras (iterative DFS).
// Input : match_candidate_id -> size == n_cams; only ref_cam position is set
// (>=0), others -1 Output: build_candidates   -> each element size == n_cams;
// used cams filled with 2D ids, others -1
void StereoMatch::buildMatch(
    std::vector<int> &match_candidate_id,
    std::vector<std::vector<int>> &build_candidates) const {
  build_candidates.clear();

  const int n_cams = static_cast<int>(_cam_list.size());

  // ---- Extract (ref_cam, ref_id) from the incoming pattern (match() already
  // guarantees it exists) ----
  int ref_cam = -1, ref_id = -1;
  for (int c = 0; c < n_cams; ++c) {
    if (!_cam_list[c] || !_cam_list[c]->is_active)
      continue;
    if (match_candidate_id[c] >= 0) {
      ref_cam = c;
      ref_id = match_candidate_id[c];
      break;
    }
  }

  // ---- Collect active cams & pool (exclude ref) in one pass ----
  if (_pinhole_matching && (ref_cam<0 || ref_id<0 ||
      ref_id>=static_cast<int>(_matching_valid[ref_cam].size()) ||
      !_matching_valid[ref_cam][ref_id])) return;
  std::vector<int> active_cams;
  active_cams.reserve(n_cams);
  std::vector<int> pool_cams;
  pool_cams.reserve(n_cams);
  for (int c = 0; c < n_cams; ++c) {
    if (!_cam_list[c] || !_cam_list[c]->is_active)
      continue;
    active_cams.push_back(c);
    if (c != ref_cam)
      pool_cams.push_back(c);
  }

  const int build_cam_count =
      std::min(std::max(2, _obj_cfg._sm_param.match_cam_count),
               static_cast<int>(active_cams.size()));

  // ---- DFS frame definition ----
  struct Frame {
    // Chosen path so far (parallel arrays)
    std::vector<int>
        chosen_cams; // camera ids on this path (order = build order)
    std::vector<int> chosen_ids;  // 2D indices aligned with chosen_cams
    std::vector<Pt2D> chosen_pts; // 2D points aligned with chosen
    std::vector<Line3D> los3d;    // LOS for each chosen pair (same order)

    // Remaining build cameras to choose from
    std::vector<int> rem_cams;

    // The camera we are currently expanding and its candidate list
    int target_cam = -1;
    std::vector<int> candidates; // candidate 2D ids on target_cam
    size_t cand_idx = 0;         // next candidate index to try
  };

  std::vector<Frame> stack;
  stack.reserve(build_cam_count + 4); // small headroom

  // ---- Initialize root frame from (ref_cam, ref_id) ----
  Frame root;
  root.chosen_cams = {ref_cam};
  root.chosen_ids = {ref_id};
  root.rem_cams = pool_cams;

  {
    const Pt2D &p = _obj2d_list[ref_cam][ref_id]->_pt_center;
    auto los_status = _cam_list[ref_cam]->lineOfSight(p);
    if (!los_status)
      return;
    root.los3d.push_back(los_status.value());
    root.chosen_pts = {p};
  }

  // Select the 2nd build camera by "shortest projected LOS segment", then
  // enumerate its candidates.
  if (!root.rem_cams.empty()) {
    root.target_cam =
        selectSecondCameraByLineLength(root.los3d.front(), root.rem_cams);
    if (root.target_cam >= 0) {
      enumerateCandidatesOnCam(root.los3d, root.target_cam, root.chosen_cams,
                               root.chosen_pts, root.candidates);
      root.cand_idx = 0;
    }
  }
  if (root.target_cam < 0 || root.candidates.empty())
    return; // cannot grow from this ref

  // ---- Push the root frame to start DFS ----
  stack.push_back(std::move(root));

  // =========================
  // Iterative DFS main loop
  // =========================
  //
  // Stack mechanics:
  // - stack.back() is the current frame to expand.
  // - Try candidates of frame.target_cam one by one:
  //     * For each candidate → extend path (build a child "nxt"),
  //       run quick triangulation + object early checks.
  //     * If checks fail → CONTINUE (stay on current frame, try next
  //     candidate).
  //     * If checks pass:
  //         - If chosen_cams reached build_cam_count → push terminal child;
  //           it will be recorded (materialized) at the next loop head, then
  //           POP (backtrack).
  //         - Else select next target_cam (max pairwise 2D angle), enumerate
  //         candidates,
  //           then PUSH child (go deeper).
  // - When current frame runs out of candidates → POP (backtrack to its
  // parent).
  //
  /*
      ===============================================================
      DFS (stack-based) – concise, plain-language version
      Focus: per-camera / per-2D iteration with PUSH/POP
      ---------------------------------------------------------------

      [Start]
      |
      [Initialize: candidate=-1 per camera; clear scratch; clear stack]
      |
      [Push root frame (first camera, its 2D list, save scratch sizes)]
      |
      +------------------- While stack is not empty -------------------+
      | [Top = stack.back()]                                           |
      |   |                                                            |
      |   +-- Have we tried all 2D ids for Top.camera? -- Yes --> [POP]|
      |   |      (restore scratch sizes; clear this camera;            |
      |   |       pop the frame; go back to while)                     |
      |   |                                                            |
      |   +-- No --> [Take next 2D id for Top.camera]                  |
      |                |                                               |
      |                [Extend locally: set candidate[Top.camera]=id;  |
      |                 append LOS/flat to scratch]                    |
      |                |                                               |
      |                +-- Do quick checks pass? -- No -->             |
      |                |     [Undo local extend (clear and pop)        |
      |                |      try next 2D in the same frame]           |
      |                |                                               |
      |                +-- Yes --> Do we still need more cameras?      |
      |                           |                                    |
      |                           +-- Yes -->                          |
      |                           |    [PUSH child frame               |
      |                           |     (next camera, its 2D list,     |
      |                           |      save current scratch sizes);  |
      |                           |     descend and continue while]    |
      |                           |                                    |
      |                           +-- No  -->                          |
      |                                [Emit full candidate;           |
      |                                 Undo local extend (stay here); |
      |                                 try next 2D in this frame]     |
      +----------------------------------------------------------------+

      Legend:
      - PUSH: push a new frame (chosen camera, its 2D list, and saved scratch
     sizes).
      - POP : pop current frame, restore scratch to saved sizes, clear that
     camera’s slot.
      - “Undo local extend” = revert only the latest (camera,2D) choice without
     popping the frame.
      ===============================================================
      */

  while (!stack.empty()) {
    Frame &fr = stack.back();

    // Terminal frame: reached build_cam_count cameras on this path
    if (static_cast<int>(fr.chosen_cams.size()) == build_cam_count) {
      // Materialize a cam-aligned result starting from the incoming pattern.
      std::vector<int> ids_norm =
          match_candidate_id; // ref already set; others -1
      for (size_t i = 0; i < fr.chosen_cams.size(); ++i) {
        ids_norm[fr.chosen_cams[i]] = fr.chosen_ids[i];
      }
      build_candidates.push_back(std::move(ids_norm));

      // Done with this terminal node → POP to try siblings at the upper level.
      stack.pop_back();
      continue;
    }

    // already loop all candidates on current target → POP and backtrack
    if (fr.cand_idx >= fr.candidates.size()) {
      stack.pop_back();
      continue;
    }

    // Take the next candidate on current target_cam
    const int pid = fr.candidates[fr.cand_idx++]; // advance for the next loop
    const int cam = fr.target_cam;

    // Prepare child frame (nxt). Push only if sanity checks pass.
    Frame nxt;
    nxt.chosen_cams = fr.chosen_cams;
    nxt.chosen_ids = fr.chosen_ids;
    nxt.chosen_pts = fr.chosen_pts;
    nxt.los3d = fr.los3d;
    nxt.rem_cams = fr.rem_cams;

    // Extend path with (cam, pid)
    nxt.chosen_cams.push_back(cam);
    nxt.chosen_ids.push_back(pid);

    {
      const Pt2D &p = _obj2d_list[cam][pid]->_pt_center;
      auto los_status = _cam_list[cam]->lineOfSight(p);
      if (!los_status)
        continue;
      nxt.los3d.push_back(los_status.value());
      nxt.chosen_pts.push_back(p);
    }

    // Remove this cam from the remaining pool
    if (auto it = std::find(nxt.rem_cams.begin(), nxt.rem_cams.end(), cam);
        it != nxt.rem_cams.end())
      nxt.rem_cams.erase(it);

    // ---- Early pruning BEFORE pushing the child frame ----
    // Quick triangulation check with unified tolerance (>=2 LOS here)
    {
      const int k = static_cast<int>(nxt.los3d.size());
      const double th_min_deg = computeMinParallaxDeg(nxt.los3d);
      const double tol_quickmm = calTriangulateTol(
          _obj_cfg._sm_param.tol_3d_mm, k, build_cam_count, th_min_deg);
      if (!TriangulationCheckWithTol(nxt.los3d, tol_quickmm)) {
        // Reject this candidate; stay on current frame to try its next
        // candidate.
        continue;
      }
    }
    // Object-specific cheap check (e.g., bubble radius coherence)
    if (!objectEarlyCheck(nxt.chosen_cams, nxt.chosen_ids)) {
      // Reject this candidate; try the next one in current frame.
      continue;
    }

    // Reached build count after adding this cam → push terminal child.
    if (static_cast<int>(nxt.chosen_cams.size()) == build_cam_count) {
      nxt.target_cam = -1;
      nxt.candidates.clear();
      nxt.cand_idx = 0;
      stack.push_back(
          std::move(nxt)); // will be materialized and popped at loop head
      continue;
    }

    // Select NEXT build camera (we already have >=2 LOS here):
    nxt.target_cam = selectNextCameraByMaxPairAngle(nxt.los3d, nxt.rem_cams);
    if (nxt.target_cam < 0) {
      // Dead end for this candidate; try the next one in current frame.
      continue;
    }

    // Enumerate candidates ONLY for the selected next camera.
    enumerateCandidatesOnCam(nxt.los3d, nxt.target_cam, nxt.chosen_cams,
                             nxt.chosen_pts, nxt.candidates);
    if (nxt.candidates.empty()) {
      // Dead end for this candidate; try the next one in current frame.
      continue;
    }
    nxt.cand_idx = 0;

    // All early checks passed and we have next-camera candidates → PUSH child,
    // go deeper.
    stack.push_back(std::move(nxt));
  }
}

// Enumerate candidate 2D indices on `target_cam` that lie within the
// intersection of all projected strips. Geometry checks & dedup are done inside
// IDMap::visitPointsInRowSpans.
void StereoMatch::enumerateCandidatesOnCam(
    const std::vector<Line3D> &los3d, int target_cam,
    const std::vector<int> &chosen_cams, const std::vector<Pt2D> &chosen_pts,
    std::vector<int> &out_candidates) const {
  out_candidates.clear();
  if (target_cam < 0 || target_cam >= static_cast<int>(_idmaps.size()))
    return;
  IDMap *idm = (_pinhole_matching ? _matching_idmaps[target_cam] : _idmaps[target_cam]).get();
  if (!idm || los3d.empty())
    return;

  // 1) Project LOS -> 2D image lines on this camera
  std::vector<Line2D> lines_px;
  if (!buildMatchingLines(los3d, target_cam, lines_px) || lines_px.empty())
    return;

  // 2) Compute per-row strip intersection in CELL indices
  std::vector<IDMap::RowSpan> spans;
  idm->computeStripIntersection(lines_px, _obj_cfg._sm_param.tol_2d_px, spans);

  bool any_valid = false;
  for (const auto &s : spans) {
    if (s.x_min <= s.x_max) {
      any_valid = true;
      break;
    }
  }
  if (!any_valid)
    return;

  // 3) Visit-and-collect with precise distance test and dedup done by IDMap
  idm->visitPointsInRowSpans(spans, lines_px, _obj_cfg._sm_param.tol_2d_px,
                             out_candidates);
  if (out_candidates.empty())
    return;

  // 4) check back=projection to chosen cams & points
  int write = 0;
  for (int pid : out_candidates) {
    const Object2D *o = _obj2d_list[target_cam][pid];
    if (!o)
      continue;
    const Pt2D &q_t = o->_pt_center;
    if (_pinhole_matching ? checkMatchingBackProjection(target_cam, q_t, chosen_cams, chosen_pts)
                         : checkBackProjection(target_cam, q_t, chosen_cams, chosen_pts)) {
      out_candidates[write++] = pid;
    }
  }
  out_candidates.resize(write);
}

// Verify: for every remaining active (check) camera, the intersection of all
// projected strips (from already chosen LOS) contains at least one 2D point.
// candidate_ids: size == n_cams, chosen build cams have non-negative ids,
// others -1.
bool StereoMatch::checkMatch(const std::vector<int> &candidate_ids,
                             double &out_e_check) const {
  // out_e_check = 0.0;
  // out_e_check = std::numeric_limits<double>::quiet_NaN();
  const int n_cams = static_cast<int>(_cam_list.size());

  // ---- 1) Gather LOS from chosen (build) cameras ----
  std::vector<Line3D> los3d;
  los3d.reserve(_obj_cfg._sm_param.match_cam_count);
  for (int cam = 0; cam < n_cams; ++cam) {
    const int pid = candidate_ids[cam];
    if (pid < 0)
      continue; // not chosen on this cam
    const Pt2D &q = _obj2d_list[cam][pid]->_pt_center;
    auto los_status = _cam_list[cam]->lineOfSight(q);
    if (!los_status)
      return false;
    los3d.push_back(los_status.value());
  }
  if (los3d.size() < 2) {
    // Should not happen if build stage produced a valid candidate,
    // but guard against malformed input.
    return false;
  }

  int n_active_cam = 0;
  for (int c = 0; c < n_cams; ++c)
    if (_cam_list[c] && _cam_list[c]->is_active)
      ++n_active_cam;
  if (los3d.size() ==
      n_active_cam) { // all active cams are used in build → nothing to check
    Pt3D Xw{};
    myMATH::triangulation(Xw, out_e_check, los3d);
    if (out_e_check > _obj_cfg._sm_param.tol_3d_mm)
      return false;
    if (!(_obj_cfg._sm_param.limit.check(Xw[0], Xw[1], Xw[2])))
      return false; // if out of bounds
    // out_e_check = triangulationVariance(los3d);
    return true;
  }

  // Prepare chosen cams & points for back-projection checks
  std::vector<int> chosen_cams;
  std::vector<Pt2D> chosen_pts;
  chosen_cams.reserve(los3d.size());
  chosen_pts.reserve(los3d.size());
  for (int c = 0; c < n_cams; ++c) {
    const int pid = candidate_ids[c];
    if (pid >= 0) {
      chosen_cams.push_back(c);
      chosen_pts.push_back(_obj2d_list[c][pid]->_pt_center);
    }
  }

  const double tol_3d_mm = _obj_cfg._sm_param.tol_3d_mm;

  // ---- 2) For every active camera that is NOT chosen yet (check cameras) ----
  for (int cam = 0; cam < n_cams; ++cam) {
    if (!_cam_list[cam] || !_cam_list[cam]->is_active)
      continue; // inactive -> ignore
    if (candidate_ids[cam] >= 0)
      continue; // already used in build -> skip

    // obtain points in the intersection of LOS from chosen cameras ----
    std::vector<int> inliers;
    enumerateCandidatesOnCam(los3d, /*target_cam=*/cam, chosen_cams, chosen_pts,
                             inliers);

    // Optional: add object-specific quick filter here (e.g., bubble radius
    // coherence per-cam). if (!inliers.empty() && !objectCheckOnCheckCam(cam,
    // inliers, candidate_ids)) { return false; }

    if (inliers.empty())
      return false; // this check cam has no supporting observation

    // get the best triangulation error among all inliers on this check cam
    // if the best is larger than tol_3d_mm, reject this candidate
    double best_err_c = std::numeric_limits<double>::infinity();

    for (int pid : inliers) {
      const Pt2D &q = _obj2d_list[cam][pid]->_pt_center;
      auto los_status = _cam_list[cam]->lineOfSight(q);
      if (!los_status)
        continue;
      Line3D los_q = los_status.value();

      std::vector<Line3D> los_all = los3d;
      los_all.push_back(los_q);

      Pt3D Xw{};
      double err = 0.0;
      myMATH::triangulation(Xw, err, los_all);

      if (!std::isfinite(err))
        continue;
      if (!(_obj_cfg._sm_param.limit.check(Xw[0], Xw[1], Xw[2])))
        continue; // if out of bounds

      if (err < best_err_c)
        best_err_c = err;
    }

    if (!std::isfinite(best_err_c) || best_err_c > tol_3d_mm)
      return false; // if the best is larger than tol_3d_mm, reject this
                    // candidate
    if (!std::isfinite(out_e_check) || best_err_c > out_e_check)
      out_e_check = best_err_c; // E_check = max(best_err_c) over check cameras
  }

  // ---- 6) All check cameras have ≥1 supporting point ----
  return true;
}

// -----------------------------------------------------------------------------
// StereoMatch::pruneMatch
//
// Greedy selection under 2D-point exclusivity, ordered by:
//    1) pct_score  (mean percentile among competitors on each used 2D point)
//    ASC 2) E_check    (triangulation error from checkMatch) ASC 3) input index
//    (stable) ASC
//
// Notes:
//  - Inputs:
//      * match_candidates: size M; each entry is size n_cams, ids[c] = -1 if
//      unused
//      * e_checks       : size M; e_checks[j] is E_check for candidate j
//  - pct_score ∈ [0,1], lower is better; if a candidate is the "best" on every
//    2D point it uses (under E_check), its pct_score will approach 0.
//  - We use small numeric tolerances (rtol/atol) when comparing floating values
//    to make "ties" robust.
//  - Memory: builds reverse indices point->candidates; large scenes with many
//    candidates and 2D points may use noticeable memory.
// -----------------------------------------------------------------------------
std::vector<std::vector<int>>
StereoMatch::pruneMatch(const std::vector<std::vector<int>> &match_candidates,
                        const std::vector<double> &e_checks) const {
  using std::size_t;

  const size_t M = match_candidates.size();
  const int n_cams = static_cast<int>(_cam_list.size());

  std::vector<std::vector<int>> empty_out;
  if (M == 0)
    return empty_out;

  // Sanity: e_checks must align with candidates
  if (e_checks.size() != M) {
    // Defensive fallback: sizes mismatch → nothing selected
    return empty_out;
  }

  // --- Numeric tolerance for robust comparisons ("ties") ---
  const double rtol = 1e-9;
  const double atol = 1e-12;

  auto eq_eps = [&](double a, double b) -> bool {
    // approximately equal
    const double s = std::max(std::fabs(a), std::fabs(b));
    return std::fabs(a - b) <= (s * rtol + atol);
  };

  // [ADDED] Parameters for local replacement thresholds:
  // If a conflict happens, we allow replacing the already-placed set S
  // when the new candidate is clearly better (in pct_score or, if tied, in
  // e_check).
  const double swap_pct_margin =
      1e-12; // new.pct must be at least this much smaller than avg(S).pct
             // (beyond eq_eps)
  const double swap_err_margin =
      1e-12; // if pct ties, new.e must be smaller than avg(S).e by this

  // -------------------------------------------------------------------------
  // Step 0) Build per-camera prefix sums to flatten (cam, pid) -> flat index.
  // flat index f ∈ [0, total_points)
  // -------------------------------------------------------------------------
  std::vector<size_t> cam_offsets(static_cast<size_t>(n_cams) + 1, 0);
  for (int c = 0; c < n_cams; ++c) {
    cam_offsets[static_cast<size_t>(c) + 1] =
        cam_offsets[static_cast<size_t>(c)] +
        _obj2d_list[static_cast<size_t>(c)].size();
  }
  const size_t total_points = cam_offsets.back();

  auto flatIndexOf = [&](int cam, int pid) -> size_t {
    return cam_offsets[static_cast<size_t>(cam)] + static_cast<size_t>(pid);
  };

  // -------------------------------------------------------------------------
  // Step 1) For each candidate, gather its used 2D points as flat indices.
  //         (No triangulation here; E_check is already provided.)
  // -------------------------------------------------------------------------
  std::vector<std::vector<size_t>> flats_per_candidate(M);

  // Step 1 can be parallelized safely (each j writes to its own slot)
  for (size_t j = 0; j < M; ++j) {
    std::vector<size_t> local_flats;
    local_flats.reserve(8);

    const auto &ids = match_candidates[j];
    for (int c = 0; c < n_cams; ++c) {
      if (!_cam_list[c] || !_cam_list[c]->is_active)
        continue;
      const int pid = (c < (int)ids.size()) ? ids[c] : -1;
      if (pid < 0)
        continue;

      const auto &obs_list = _obj2d_list[(size_t)c];
      if ((size_t)pid >= obs_list.size() || obs_list[(size_t)pid] == nullptr)
        continue;

      local_flats.push_back(flatIndexOf(c, pid));
    }

    flats_per_candidate[j] = std::move(local_flats);
  }

  // -------------------------------------------------------------------------
  // Step 2) Build reverse index: for each flat 2D point, list the candidates
  //         that use it. Later we compute percentiles within each list.
  // -------------------------------------------------------------------------
  std::vector<std::vector<int>> point_to_candidates(total_points);
  for (size_t j = 0; j < M; ++j) {
    for (size_t f : flats_per_candidate[j]) {
      point_to_candidates[f].push_back(static_cast<int>(j));
    }
  }

  // -------------------------------------------------------------------------
  // Step 3) Compute pct_score for each candidate:
  //         For each used point f, consider all candidates that also use f.
  //         Rank them by E_check ASC (stable for ties), convert rank to
  //         percentile in [0,1], then pct_score[j] is the mean percentile
  //         over all points used by candidate j.
  // -------------------------------------------------------------------------
  std::vector<double> pct_sum(M, 0.0);
  std::vector<int> pct_count(M, 0);

  for (size_t f = 0; f < total_points;
       ++f) { // loop over all 2D points on all cameras
    const auto &idxs =
        point_to_candidates[f]; // match candidates using this 2D point
    if (idxs.empty())
      continue;

    // Pool only candidates whose E_check is finite
    std::vector<std::pair<double, int>>
        pool; // pool saves all errors with corresponding match candidate index
    pool.reserve(idxs.size());
    for (int j : idxs) {
      const double e = e_checks[(size_t)j];
      if (std::isfinite(e))
        pool.emplace_back(e, j);
    }
    const int n = (int)pool.size();
    if (n == 0)
      continue;

    if (n == 1) {
      // Single competitor → percentile 0.0
      const int j = pool[0].second;
      pct_sum[(size_t)j] += 0.0;
      pct_count[(size_t)j] += 1;
      continue;
    }

    // Stable sort by E_check; ties keep original order (input index)
    std::stable_sort(
        pool.begin(), pool.end(), [&](const auto &A, const auto &B) {
          const double ea = A.first, eb = B.first;
          if (!eq_eps(ea, eb))
            return ea < eb;
          return A.second <
                 B.second; // tie-break by candidate index for stability
        });

    // Assign average rank to equal-error blocks, then normalize to [0,1]
    int k = 0;
    // for every match candidate in this pool, calculate its percentile score
    // and add to pct_sum
    while (k < n) {
      int k0 = k, k1 = k + 1;
      // if there are equal errors, find the end of this block, and assign them
      // the average rank
      while (k1 < n && eq_eps(pool[k1].first, pool[k0].first))
        ++k1;

      const double avg_rank =
          0.5 * (double)(k0 + (k1 - 1)); // average of integer ranks
      const double denom = (double)(n - 1);
      const double pct = (denom > 0.0) ? (avg_rank / denom) : 0.0;

      // for each match candidate in this block, add its percentile score to
      // pct_sum
      for (int t = k0; t < k1; ++t) {
        const int j =
            pool[(size_t)t].second; // the index of this match candidate
        pct_sum[(size_t)j] += pct;  // add its percentile score to pct_sum
        pct_count[(size_t)j] +=
            1; // count how many 2D points this match candidate uses
      }
      k = k1;
    }
  }

  std::vector<double> pct_score(M, std::numeric_limits<double>::infinity());
  for (size_t j = 0; j < M; ++j) {
    if (pct_count[j] > 0) {
      pct_score[j] =
          pct_sum[j] /
          (double)pct_count[j]; // mean percentile score across all cameras used
    }
    // else: stays +inf → will be sorted to the end / ignored
  }

  // -------------------------------------------------------------------------
  // Step 4) Sort candidate indices by (pct_score ASC, E_check ASC, input idx
  // ASC)
  // -------------------------------------------------------------------------
  std::vector<int> order(M);
  std::iota(order.begin(), order.end(), 0);

  std::stable_sort(order.begin(), order.end(), [&](int aj, int bj) {
    const double pa = pct_score[(size_t)aj];
    const double pb = pct_score[(size_t)bj];
    const bool af = std::isfinite(pa);
    const bool bf = std::isfinite(pb);

    if (af && bf) {
      if (!eq_eps(pa, pb))
        return pa < pb; // lower pct_score first
    } else if (af) {
      return true; // finite beats +inf
    } else if (bf) {
      return false; // +inf goes last
    }

    const double ea = e_checks[(size_t)aj];
    const double eb = e_checks[(size_t)bj];
    const bool aef = std::isfinite(ea);
    const bool bef = std::isfinite(eb);

    if (aef && bef) {
      if (!eq_eps(ea, eb))
        return ea < eb; // lower E_check second
    } else if (aef) {
      return true;
    } else if (bef) {
      return false;
    }

    // final tie-breaker: original index → stable, deterministic
    return aj < bj;
  });

  // -------------------------------------------------------------------------
  // Step 5) Greedy packing with 2D-point exclusivity
  //         A flat 2D point can be owned by at most one selected candidate.
  // -------------------------------------------------------------------------
  std::vector<int> owner(total_points, -1);
  std::vector<int> chosen;
  chosen.reserve(M / 10 + 8);

  // [ADDED] We keep a selected-flag to know which candidates are currently
  // placed.
  std::vector<char> selected(M, 0);

  // [ADDED] Small lambdas to place / clear a candidate's ownership.
  auto place = [&](int cand) {
    for (size_t f : flats_per_candidate[(size_t)cand])
      owner[f] = cand;
    selected[(size_t)cand] = 1;
    chosen.push_back(cand);
  };
  auto clear_owner_of = [&](int cand) {
    for (size_t f : flats_per_candidate[(size_t)cand]) {
      if (owner[f] == cand)
        owner[f] = -1;
    }
    selected[(size_t)cand] = 0;
    // note: we DO NOT remove it from 'chosen' to keep O(1). We'll rebuild
    // output from 'selected' later.
  };

  for (int j : order) {
    // Skip those with undefined pct_score or E_check
    if (!std::isfinite(pct_score[(size_t)j]))
      continue;
    if (!std::isfinite(e_checks[(size_t)j]))
      continue;

    const auto &flats = flats_per_candidate[(size_t)j];
    if (flats.empty())
      continue;

    bool conflict = false;
    for (size_t f : flats) {
      if (owner[f] != -1) {
        conflict = true;
        break;
      }
    }
    if (!conflict) {
      // place
      place(j);
      continue;
    }

    // [ADDED] Local replacement path:
    // Collect the unique set S of already-placed candidates that conflict with
    // j.
    std::vector<int> S;
    S.reserve(flats.size());
    for (size_t f : flats) {
      int o = owner[f];
      if (o == -1)
        continue;
      bool seen = false;
      for (int v : S)
        if (v == o) {
          seen = true;
          break;
        }
      if (!seen)
        S.push_back(o);
    }
    if (S.empty()) {
      // theoretically unreachable since we detected conflict, but keep safe
      // guard
      place(j);
      continue;
    }

    // [ADDED] Compute average pct_score and average e_check over S.
    double avg_pct = 0.0, avg_err = 0.0;
    int cnt = 0;
    for (int o : S) {
      const double po = pct_score[(size_t)o];
      const double eo = e_checks[(size_t)o];
      if (!std::isfinite(po) || !std::isfinite(eo))
        continue;
      avg_pct += po;
      avg_err += eo;
      ++cnt;
    }
    if (cnt == 0) {
      // If S has no finite metrics, be conservative and skip replacement.
      continue;
    }
    avg_pct /= (double)cnt;
    avg_err /= (double)cnt;

    const double pj = pct_score[(size_t)j];
    const double ej = e_checks[(size_t)j];

    // [ADDED] Replacement criterion:
    //  - strictly better pct_score beyond eps, or
    //  - pct_score ties (within eps) but strictly better e_check beyond eps.
    const bool better_pct =
        (!eq_eps(pj, avg_pct)) && ((pj + swap_pct_margin) < avg_pct);
    const bool tie_pct_better_err = eq_eps(pj, avg_pct) &&
                                    (!eq_eps(ej, avg_err)) &&
                                    ((ej + swap_err_margin) < avg_err);

    if (better_pct || tie_pct_better_err) {
      // [ADDED] Replace: clear all owners in S, then place j.
      for (int o : S)
        clear_owner_of(o);
      place(j);
    }
    // else: keep the current placement; skip j
  }

  // -------------------------------------------------------------------------
  // Step 6) Materialize output: return the selected candidates
  // -------------------------------------------------------------------------
  std::vector<std::vector<int>> out;
  out.reserve(chosen.size());
  for (size_t j = 0; j < M; ++j) {
    if (selected[j])
      out.push_back(match_candidates[(size_t)j]);
  }
  return out;
}

// Triangulate all selected matches, drop those with error > tol_3d,
// and build Object3D (Tracer3D / Bubble3D) with cloned 2D observations (camera
// order preserved).
std::vector<std::unique_ptr<Object3D>> StereoMatch::triangulateMatch(
    const std::vector<std::vector<int>> &selected_matches) const {
    const auto &obs2d = _obj2d_list;
  const double tol3d = _obj_cfg._sm_param.tol_3d_mm; // [mm]

  std::vector<std::unique_ptr<Object3D>> out;
  out.reserve(selected_matches.size());

  std::vector<Line3D> los;
  Pt3D pt_world;
  double err = 0.0;

  for (const auto &ids : selected_matches) {
    // 1) Build LOS from this match
    los.clear();
    const int n_cams = static_cast<int>(_cam_list.size());
    los.reserve(n_cams);
    for (int c = 0; c < n_cams; ++c) {
      const int pid = ids[c];
      if (pid < 0)
        continue;
      const Pt2D &q = obs2d[c][pid]->_pt_center;
      auto los_status = _cam_list[c]->lineOfSight(q);
      if (!los_status) {
        los.clear();
        break;
      }
      los.push_back(los_status.value());
    }
    if (los.size() < 2)
      continue;

    // 2) Triangulation
    try {
      myMATH::triangulation(pt_world, err, los);
    } catch (...) {
      continue;
    }

    // 3) Final tolerance gate
    if (err > tol3d)
      continue;

    // 4) create object 2D list
    std::vector<std::unique_ptr<Object2D>> obj2d_list(n_cams);
    for (int c = 0; c < n_cams; ++c) {
      const int pid = ids[c];
      if (pid < 0)
        continue;
      obj2d_list[c] = obs2d[c][pid]->clone();
    }

    // 5) create object
    CreateArgs a;
    a._pt_center = pt_world;
    a._obj2d_ready = std::move(obj2d_list);
    a._cam_list = &_cam_list;
    a._compute_bubble_radius = (_obj_cfg.kind() == ObjectKind::Bubble);

    auto obj3d = _obj_cfg.creatObject3D(
        std::move(a)); // create object according to _obj_cfg
    if (!obj3d)
      continue;
    obj3d->_is_tracked = false;

    out.emplace_back(std::move(obj3d));
  }

  return out;
}

// ---- helper: make a 2D line on camera 'cam_id' from a 3D LOS ----
bool StereoMatch::makeMatchingLine(int cam_id, const Line3D& los, Line2D& out) const {
  if (!_pinhole_matching) return makeLine2DFromLOS3D(cam_id,los,out);
  const auto& p = static_cast<const PinholeCamera&>(*_cam_list[cam_id]).param();
  // Plane through target center and world LOS, represented in target frame.
  Pt3D a = p.r_mtx * (los.pt - p.t_vec_inv);
  Pt3D d = p.r_mtx * los.unit_vector;
  double sa=0, sd=0, position_scale=0;
  for(int i=0;i<3;++i) {
    if(!std::isfinite(a[i]) || !std::isfinite(d[i])) return false;
    sa=std::max(sa,std::abs(a[i])); sd=std::max(sd,std::abs(d[i]));
    position_scale=std::max({position_scale,std::abs(los.pt[i]),std::abs(p.t_vec_inv[i])});
  }
  if(!(sa>64*std::numeric_limits<double>::epsilon()*position_scale) || !(sd>0)) return false;
  auto n=myMATH::cross(a/sa,d/sd);
  // Relative degeneracy guard; no world-unit-dependent +1/+2 samples.
  if(n.norm()<=64*std::numeric_limits<double>::epsilon()) return false;
  if(std::hypot(n[0],n[1])<=64*std::numeric_limits<double>::epsilon()*n.norm()) return false;
  const double A=n[0]/p.cam_mtx(0,0), B=n[1]/p.cam_mtx(1,1);
  const double C=n[2]-A*p.cam_mtx(0,2)-B*p.cam_mtx(1,2);
  const double h=std::hypot(A,B);
  if(!(h>0) || !std::isfinite(h)) return false;
  const double nx=A/h, ny=B/h, distance=-C/h;
  out.pt={nx*distance,ny*distance}; out.unit_vector={-ny,nx};
  return std::isfinite(out.pt[0]) && std::isfinite(out.pt[1]);
}

bool StereoMatch::buildMatchingLines(const std::vector<Line3D>& los, int cam,
                                     std::vector<Line2D>& lines) const {
  lines.clear();
  for(const auto& ray:los) {
    Line2D line; if(!makeMatchingLine(cam,ray,line)) return false;
    lines.push_back(line);
  }
  return true;
}

bool StereoMatch::checkMatchingBackProjection(int cam, const Pt2D& measured,
    const std::vector<int>& chosen_cams, const std::vector<Pt2D>& chosen_measured) const {
  auto ray=_cam_list[cam]->lineOfSight(measured); if(!ray) return false;
  if(chosen_cams.size()!=chosen_measured.size()) return false;
  for(size_t i=0;i<chosen_cams.size();++i) {
    int ci=chosen_cams[i]; Pt2D q; Line2D line;
    if(!static_cast<const PinholeCamera&>(*_cam_list[ci]).matchingPixel(chosen_measured[i],q) ||
       !makeMatchingLine(ci,ray.value(),line) ||
       !(myMATH::dist2(q,line)<=std::pow(_obj_cfg._sm_param.tol_2d_px,2))) return false;
  }
  return true;
}

bool StereoMatch::makeLine2DFromLOS3D(int cam_id, const Line3D &los,
                                              Line2D &out_line) const {
  // TODO(refraction): This workaround avoids projecting los.pt (interface point)
  // to reduce boundary degeneracy in refractive projection. It does NOT address
  // physical invalid branches (e.g., TIR/parallel geometry) and should be
  // replaced by a robust failure-aware LOS-to-2D construction.
  Pt3D p_a{los.pt[0] + los.unit_vector[0], los.pt[1] + los.unit_vector[1],
           los.pt[2] + los.unit_vector[2]};
  Pt3D p_b{los.pt[0] + 2.0 * los.unit_vector[0],
           los.pt[1] + 2.0 * los.unit_vector[1],
           los.pt[2] + 2.0 * los.unit_vector[2]};

  auto a_status = _cam_list[cam_id]->project(p_a);
  if (!a_status)
    return false;
  auto b_status = _cam_list[cam_id]->project(p_b);
  if (!b_status)
    return false;

  const Pt2D a = a_status.value();
  const Pt2D b = b_status.value();
  Line2D L;
  L.pt = a;
  L.unit_vector = myMATH::createUnitVector(a, b); // assumed normalized
  out_line = L;
  return true;
}

// ---- helper: project a set of LOS to 2D lines on a camera (re-uses output
// buffer) ----
bool StereoMatch::buildLinesOnCam(const std::vector<Line3D> &los3d,
                                         int cam_id,
                                         std::vector<Line2D> &out_lines) const {
  out_lines.clear();
  out_lines.reserve(los3d.size());
  for (const auto &L3 : los3d) {
    Line2D line;
    if (!makeLine2DFromLOS3D(cam_id, L3, line))
      return false;
    out_lines.push_back(line);
  }
  return true;
}

// ---- early checks & tolerances (decl only; you已有实现或后续实现) ----
// Compute minimal parallax angle among current LOS set (degrees).
double
StereoMatch::computeMinParallaxDeg(const std::vector<Line3D> &los) const {
  if (los.size() < 2)
    return 180.0;
  double min_deg = 180.0;
  for (size_t i = 0; i < los.size(); ++i) {
    const Pt3D &a = los[i].unit_vector;
    for (size_t j = i + 1; j < los.size(); ++j) {
      const Pt3D &b = los[j].unit_vector;
      double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
      dot = std::max(-1.0, std::min(1.0, dot));
      double deg = std::acos(dot) * 180.0 / M_PI;
      if (deg < min_deg)
        min_deg = deg;
    }
  }
  return min_deg;
}

// Compute quick-stage tolerance based on depth (k), target build count (m) and
// min parallax.
double StereoMatch::calTriangulateTol(double tol_final_mm, int k, int m,
                                      double min_parallax_deg) const {
  // Depth factor ~ sqrt(m / k): information accumulation (variance ~ 1/k, stdev
  // ~ 1/sqrt(k))
  const int kk = std::max(k, 2);
  const int mm = std::max(m, 2);
  const double alpha_depth = std::sqrt(double(mm) / double(kk));

  // Angle factor ~ 1/sin(theta), clamped to [1, gamma_max]
  const double theta_ref_deg = 6.0; // tweakable
  const double theta_min_cap = 1.0; // avoid degeneracy
  const double gamma_max = 2.0;     // upper clamp

  constexpr auto deg2rad = [](double deg) constexpr {
    return deg * M_PI / 180.0;
  };

  const double th = std::max(min_parallax_deg, theta_min_cap);
  const double num = std::sin(deg2rad(theta_ref_deg));
  const double den = std::max(1e-6, std::sin(deg2rad(th)));
  const double alpha_angle = std::max(1.0, std::min(gamma_max, num / den));

  return tol_final_mm * alpha_depth * alpha_angle;
}

// Unified triangulation check with a given tolerance (mm).
bool StereoMatch::TriangulationCheckWithTol(const std::vector<Line3D> &los3d,
                                            double tol_3d_mm) const {
  if (los3d.size() < 2)
    return true; // need >=2 LOS to triangulate
  Pt3D pt;
  double err = 0.0;
  try {
    myMATH::triangulation(pt, err, los3d);
  } catch (...) {
    return false;
  }
  return err <= tol_3d_mm;
}

// compute variance of triangulation from pair-wise LOS
double
StereoMatch::triangulationVariance(const std::vector<Line3D> &los) const {
  const size_t n = los.size();
  if (n < 2)
    return std::numeric_limits<double>::infinity();

  std::vector<Pt3D> midpoints;
  midpoints.reserve(n * (n - 1) / 2);

  // helper: compute closest pair (P,Q) on each line between two 3D lines (A,u)
  // and (B,v)
  auto closest_pair = [](const Pt3D &A, const Pt3D &u, const Pt3D &B,
                         const Pt3D &v, Pt3D &P, Pt3D &Q) {
    const Pt3D w0 = A - B;
    const double a = u * u; // ~1
    const double b = u * v;
    const double c = v * v; // ~1
    const double d = u * w0;
    const double e = v * w0;
    const double denom = a * c - b * b;
    double t = 0.0, s = 0.0;
    if (denom > 1e-12) {
      t = (b * e - c * d) / denom;
      s = (a * e - b * d) / denom;
    } else {
      // 近平行：固定一侧
      t = 0.0;
      s = (c > 0.0) ? (e / c) : 0.0;
    }
    P = A + u * t;
    Q = B + v * s;
  };

  // compute all pair-wise midpoints
  for (size_t i = 0; i < n; i++) {
    Pt3D Ai = los[i].pt;
    Pt3D ui = los[i].unit_vector;
    ui = ui / ui.norm();
    for (size_t j = i + 1; j < n; j++) {
      Pt3D Bj = los[j].pt;
      Pt3D vj = los[j].unit_vector;
      vj = vj / vj.norm();
      Pt3D P, Q;
      closest_pair(Ai, ui, Bj, vj, P, Q);
      midpoints.push_back((P + Q) * 0.5);
    }
  }

  // compute variance of midpoints
  Pt3D ctr(0, 0, 0);
  for (auto &m : midpoints)
    ctr += m;
  ctr = ctr * (1.0 / double(midpoints.size()));

  double acc = 0.0;
  for (auto &m : midpoints) {
    Pt3D dv = m - ctr;
    acc += dv * dv;
  }
  return acc / double(midpoints.size());
}

bool StereoMatch::checkBackProjection(
    int target_cam, const Pt2D &q_t, const std::vector<int> &chosen_cams,
    const std::vector<Pt2D> &chosen_pts) const {
  if (chosen_cams.empty())
    return true;

  auto los_t_status = _cam_list[target_cam]->lineOfSight(q_t);
  if (!los_t_status)
    return false;
  const Line3D los_t = los_t_status.value();

  const double tol2 =
      _obj_cfg._sm_param.tol_2d_px * _obj_cfg._sm_param.tol_2d_px;
  const size_t K = chosen_cams.size();

  for (size_t i = 0; i < K; ++i) {
    const int ci = chosen_cams[i];
    const Pt2D &qi = chosen_pts[i];

    Line2D L;
    if (!makeLine2DFromLOS3D(ci, los_t, L))
      return false;
    if (myMATH::dist2(qi, L) > tol2)
      return false;
  }
  return true;
}

// Object-specific lightweight check hook (build stage). Router + stubs.
bool StereoMatch::objectEarlyCheck(const std::vector<int> &cams_in_path,
                                   const std::vector<int> &ids_in_path) const {
  switch (_obj_cfg.kind()) {
  case ObjectKind::Tracer:
    return tracerEarlyCheck(cams_in_path, ids_in_path);
  case ObjectKind::Bubble:
    return bubbleEarlyCheck(cams_in_path, ids_in_path);
  default:
    return true;
  }
}

// Tracer: no extra early rule in build stage.
bool StereoMatch::tracerEarlyCheck(const std::vector<int> &,
                                   const std::vector<int> &) const {
  return true;
}

// bubble: check radius consistency
bool StereoMatch::bubbleEarlyCheck(const std::vector<int> &cams_in_path,
                                   const std::vector<int> &ids_in_path) const {
  const size_t k = std::min(cams_in_path.size(), ids_in_path.size());
  if (k < 2)
    return true;

  // 1) Collect LOS / cameras / 2D objects (aligned by index)
  std::vector<Line3D> los;
  los.reserve(k);
  std::vector<const Camera *> cams;
  cams.reserve(k);
  std::vector<const Object2D *> obj2d_by_cam;
  obj2d_by_cam.reserve(k);

  for (size_t i = 0; i < k; ++i) {
    const int cam = cams_in_path[i], pid = ids_in_path[i];
    if (cam < 0 || pid < 0)
      continue;
    const Object2D *base = _obj2d_list[cam][pid];
    if (!base)
      continue;

    const Pt2D &q = base->_pt_center; // same domain as calibration
    auto los_status = _cam_list[cam]->lineOfSight(q);
    if (!los_status)
      return true;
    los.push_back(los_status.value());
    cams.push_back(_cam_list[cam].get());
    obj2d_by_cam.push_back(base);
  }
  if (los.size() < 2)
    return true;

  // 2) Triangulate 3D center with ALL LOS
  Pt3D X_world{};
  double tri_err = 0.0; // world-length units
  myMATH::triangulation(X_world, tri_err, los);
  if (!std::isfinite(tri_err) || tri_err < 0.0)
    return true; // permissive

  // 3) Build tol_3d from triangulation error with parallax amplification
  //    tol_3d_eff ≈ tri_err / sin(theta_min), clamp theta to avoid blow-up.
  const double theta_min_deg =
      computeMinParallaxDeg(los); // you already have this
  const double min_deg = 10.0;    // TODO: _cfg.min_parallax_deg if available
  const double theta_use_deg = std::max(theta_min_deg, min_deg);
  const double sin_theta =
      std::max(std::sin(theta_use_deg * M_PI / 180.0), 1e-3);
  const double tol_3d_eff = tri_err / sin_theta; // world-length units

  // 4) 2D reprojection tolerance from calibration (pixels)
  const double tol_2d_px =
      _obj_cfg._sm_param.tol_2d_px; // <=0 to disable if you want

  // 5) Radius consistency gate (PINHOLE effective; POLYNOMIAL ignored for now)
  return Bubble::checkRadiusConsistency(X_world, cams, obj2d_by_cam, tol_2d_px,
                                        tol_3d_eff);
}

// Select the 2nd build camera by minimizing the visible segment length of the
// single projected LOS.
int StereoMatch::selectSecondCameraByLineLength(
    const Line3D &los_ref, const std::vector<int> &remaining_cams) const {
  if (remaining_cams.empty())
    return -1;

  auto segLenOnCam = [&](int cam) -> double {
    Line2D L;
    if (!makeMatchingLine(cam, los_ref, L))
      return std::numeric_limits<double>::infinity();
    // Extreme matching extents use an exact-scan index: the length heuristic
    // has a neutral score, rather than clipping to a fictitious sensor/grid.
    if (_pinhole_matching && _matching_idmaps[cam]->scansAll()) return 0.;
    const double px = L.pt[0], py = L.pt[1];
    const double ux = L.unit_vector[0], uy = L.unit_vector[1];

    const double ox = _pinhole_matching ? _matching_idmaps[cam]->originX() : 0.;
    const double oy = _pinhole_matching ? _matching_idmaps[cam]->originY() : 0.;
    const double W = _pinhole_matching ? _matching_idmaps[cam]->extentX() : double(_cam_list[cam]->getNCol());
    const double H = _pinhole_matching ? _matching_idmaps[cam]->extentY() : double(_cam_list[cam]->getNRow());

    auto in = [](double v, double lo, double hi) { return v >= lo && v <= hi; };
    auto push_if = [&](double x, double y, std::vector<Pt2D> &v) {
      if (in(x, ox, ox+W) && in(y, oy, oy+H))
        v.emplace_back(x, y);
    };

    std::vector<Pt2D> pts;
    pts.reserve(4);
    if (std::fabs(ux) > 1e-12) {
      double t = (ox - px) / ux;
      push_if(ox, py + t * uy, pts);
      t = (ox+W - 1 - px) / ux;
      push_if(ox+W - 1, py + t * uy, pts);
    }
    if (std::fabs(uy) > 1e-12) {
      double t = (oy - py) / uy;
      push_if(px + t * ux, oy, pts);
      t = (oy+H - 1 - py) / uy;
      push_if(px + t * ux, oy+H - 1, pts);
    }
    // de-dup corners
    const double eps = 1e-9;
    std::vector<Pt2D> uniq;
    uniq.reserve(pts.size());
    for (auto &q : pts) {
      bool dup = false;
      for (auto &p : uniq)
        if (std::fabs(p[0] - q[0]) < eps && std::fabs(p[1] - q[1]) < eps) {
          dup = true;
          break;
        }
      if (!dup)
        uniq.push_back(q);
    }
    if (uniq.size() < 2)
      return 0.0;
    const double dx = uniq[0][0] - uniq[1][0], dy = uniq[0][1] - uniq[1][1];
    return std::sqrt(dx * dx + dy * dy);
  };

  int best_cam = -1;
  double best_len = std::numeric_limits<double>::infinity();
  for (int cam : remaining_cams) {
    const double L = segLenOnCam(cam);
    if (L < best_len - 1e-9) {
      best_len = L;
      best_cam = cam;
    }
  }
  return best_cam;
}

// Select next build camera (>=2 LOS already selected) by maximizing the maximum
// pairwise 2D angle. Score = max_{i<j} |sin(theta_ij)| = |u_i x u_j|; ties keep
// the first encountered.
int StereoMatch::selectNextCameraByMaxPairAngle(
    const std::vector<Line3D> &los3d,
    const std::vector<int> &remaining_cams) const {
  if (remaining_cams.empty())
    return -1;

  int best_cam = -1;
  double best_score = -1.0;

  std::vector<Line2D> lines_px; // reused buffer

  for (int cam : remaining_cams) {
    if (!buildMatchingLines(los3d, cam, lines_px))
      continue;

    double score = 0.0;
    if (lines_px.size() >= 2) {
      for (size_t i = 0; i < lines_px.size(); ++i) {
        const Pt2D &ui = lines_px[i].unit_vector; // unit
        for (size_t j = i + 1; j < lines_px.size(); ++j) {
          const Pt2D &uj = lines_px[j].unit_vector;
          const double s = std::fabs(ui[0] * uj[1] - ui[1] * uj[0]);
          if (s > score)
            score = s;
          if (score >= 1.0 - 1e-12)
            break; // ~90°, cannot do better
        }
        if (score >= 1.0 - 1e-12)
          break;
      }
    }
    if (score > best_score) {
      best_score = score;
      best_cam = cam;
    }
  }
  return best_cam;
}

IDMap::IDMap(int img_rows_px, int img_cols_px, int cell_px)
    : _img_rows_px(img_rows_px), _img_cols_px(img_cols_px),
      _cell_px(std::max(1, cell_px)) {
  _rows_cell =
      std::max(1, _img_rows_px / _cell_px + (_img_rows_px % _cell_px ? 1 : 0));
  _cols_cell =
      std::max(1, _img_cols_px / _cell_px + (_img_cols_px % _cell_px ? 1 : 0));
  _buckets.assign(_rows_cell * _cols_cell, {});
}

void IDMap::rebuild(const std::vector<const Object2D *> &objs) {
  _objs = objs;
  if (_coordinate_mode) {
    _rows_cell=std::max(1,_img_rows_px/_cell_px+(_img_rows_px%_cell_px?1:0));
    _cols_cell=std::max(1,_img_cols_px/_cell_px+(_img_cols_px%_cell_px?1:0));
    _buckets.assign(_rows_cell*_cols_cell,{});
  }
  _coordinate_mode = false;
  _scan_all=false; _origin_x=_origin_y=0.; _coordinates.clear();
  // clear buckets
  for (auto &v : _buckets)
    v.clear();

  const int n = static_cast<int>(objs.size());
  for (int pid = 0; pid < n; ++pid) {
    const Object2D *o = objs[pid];
    if (!o || !std::isfinite(o->_pt_center[0]) || !std::isfinite(o->_pt_center[1]) ||
        o->_pt_center[0]<0 || o->_pt_center[1]<0 ||
        o->_pt_center[0]>=double(_cols_cell)*_cell_px ||
        o->_pt_center[1]>=double(_rows_cell)*_cell_px) continue;
    int cx = static_cast<int>(std::floor(o->_pt_center[0] / _cell_px));
    int cy = static_cast<int>(std::floor(o->_pt_center[1] / _cell_px));
    if (cx < 0 || cy < 0 || cx >= _cols_cell || cy >= _rows_cell)
      continue;
    _buckets[idx(cy, cx)].push_back(pid);
  }
}

void IDMap::rebuildCoordinates(const std::vector<Pt2D>& pixels,
                              const std::vector<unsigned char>& valid) {
  REQUIRE(pixels.size()==valid.size(),ErrorCode::InvalidArgument,"IDMap validity size mismatch");
  _coordinate_mode=true; _coordinates=pixels; _objs.clear(); _scan_all=false;
  double xmin=0,ymin=0,xmax=0,ymax=0; bool first=true;
  std::vector<int> ids;
  for(size_t i=0;i<pixels.size();++i) {
    const auto& q=pixels[i];
    if(!valid[i] || !std::isfinite(q[0]) || !std::isfinite(q[1])) continue;
    ids.push_back(static_cast<int>(i));
    if(first) { xmin=xmax=q[0]; ymin=ymax=q[1]; first=false; }
    xmin=std::min(xmin,q[0]); xmax=std::max(xmax,q[0]);
    ymin=std::min(ymin,q[1]); ymax=std::max(ymax,q[1]);
  }
  _origin_x=std::floor(xmin/_cell_px)*_cell_px;
  _origin_y=std::floor(ymin/_cell_px)*_cell_px;
  double cols=std::floor((xmax-_origin_x)/_cell_px)+1;
  double rows=std::floor((ymax-_origin_y)/_cell_px)+1;
  // Guard BEFORE integer conversion or allocation. A sparse extreme outlier
  // never drops unrelated valid IDs: fall back to bounded exact scanning.
  constexpr double max_cells=1000000;
  if(!std::isfinite(rows) || !std::isfinite(cols) || rows<1 || cols<1 ||
     rows>max_cells || cols>max_cells || rows*cols>max_cells) {
    _scan_all=true; _rows_cell=_cols_cell=1;
    _buckets.assign(1,std::move(ids)); return;
  }
  _rows_cell=static_cast<int>(rows); _cols_cell=static_cast<int>(cols);
  _buckets.assign(_rows_cell*_cols_cell,{});
  for(int id:ids) {
    int cx=static_cast<int>(std::floor((pixels[id][0]-_origin_x)/_cell_px));
    int cy=static_cast<int>(std::floor((pixels[id][1]-_origin_y)/_cell_px));
    _buckets[idx(cy,cx)].push_back(id);
  }
}

static inline double hypot2(double x, double y) { return x * x + y * y; }

Pt2D IDMap::normalized(const Pt2D &v) {
  const double nx = v[0], ny = v[1];
  const double n2 = nx * nx + ny * ny;
  if (n2 <= 1e-24) {
    // fallback to +X if zero-length (should not happen if callers ensure unit)
    return Pt2D{1.0, 0.0};
  }
  const double inv = 1.0 / std::sqrt(n2);
  return Pt2D{nx * inv, ny * inv};
}

/**
 * Compute per-row intersection of K LOS strips in CELL index space.
 *
 * For each image row of cells (cell size = _cell_px), this function keeps
 * the inclusive [x_min, x_max] range of cell indices whose cell *centers*
 * are within the widened strip of every input 2D line. The widening makes
 * the test conservative at cell resolution: if *any* point inside a cell
 * lies within the pixel tolerance `tol_px` of a line, the cell will not be
 * falsely rejected just because the center is slightly outside the strip.
 *
 * Key idea (proj_extent):
 * -----------------------
 * Let the line have unit direction u and unit normal n = (-u_y, u_x).
 * A square cell has half-size (hw, hh) = (0.5*cell_px, 0.5*cell_px).
 * Any point inside the cell can deviate from the cell center by Δ = (dx, dy)
 * with |dx| <= hw, |dy| <= hh. The worst-case normal projection is:
 *
 *      max_{Δ in cell} | n · Δ | = |n_x| * hw + |n_y| * hh
 *
 * We add this worst-case term to the half-width of the strip so that the
 * center-based distance test remains conservative at cell granularity.
 */
void IDMap::computeStripIntersection(const std::vector<Line2D> &lines_px,
                                     double tol_px,
                                     std::vector<RowSpan> &spans) const {
  // Start with full-span on every row (inclusive indices)
  spans.assign(_rows_cell, RowSpan{0, _cols_cell - 1});
  if (_scan_all) return;

  const double hw = 0.5 * _cell_px;       // half cell width in pixels
  const double hh = 0.5 * _cell_px;       // half cell height in pixels
  const double to_index = 1.0 / _cell_px; // pixel -> cell-index scale
  const double eps = 1e-12;               // guard for near-zero division
  const double tol_px_grow =
      tol_px + 1e-6;         // tiny safety margin at strip boundary
  const double eps_c = 1e-9; // bias for robust ceil/floor at boundaries

  for (const Line2D &L : lines_px) {
    if (!std::isfinite(L.pt[0]) || !std::isfinite(L.pt[1]) ||
        !std::isfinite(L.unit_vector[0]) || !std::isfinite(L.unit_vector[1]) ||
        !(std::hypot(L.unit_vector[0],L.unit_vector[1])>0)) {
      spans.assign(_rows_cell,RowSpan{1,0}); return;
    }
    // Line in normal form: n · x + d = 0, where n is unit-length
    const Pt2D u = normalized(L.unit_vector); // ensure unit direction
    const Pt2D n{-u[1], u[0]};                // unit normal
    const double d = -(n[0] * (L.pt[0]-_origin_x) + n[1] * (L.pt[1]-_origin_y));
    // Overflow in a new-domain strip estimate cannot justify dropping IDs;
    // retain conservative spans and let exact point checks decide.
    if (_coordinate_mode && !std::isfinite(d)) continue;

    for (int cy = 0; cy < _rows_cell; ++cy) {
      RowSpan &row = spans[cy];
      if (row.x_min > row.x_max)
        continue; // already empty for this row

      // Distance from the cell center (cx+0.5, cy+0.5) to the infinite line,
      // measured along the unit normal n.
      // cx, cy is the cell location, x_c, y_c is the pixel location
      const double y_c = (cy + 0.5) * _cell_px;
      // Normal-form of a 2D line: n · x + d = 0, with unit normal n = (-u_y,
      // u_x). For a fixed image row we set y = y_c (the row's cell-center y).
      // Then the signed perpendicular distance of a cell-center (x, y_c) to the
      // line is
      //     n_x * x + n_y * y_c + d.
      // The term below precomputes the part that is constant across the row
      // (independent of x), so the per-column distance becomes | n_x * x + C |.
      // If |n_x| < eps, the distance does not depend on x (horizontal strip).
      const double C = n[1] * y_c + d; // note: |n * c + d| is the distance to
                                       // the line from point c(cx, cy)

      // --- proj_extent explained ---
      // Worst-case extra distance between any point in the cell and the cell
      // center along the normal direction n. This enlarges the strip so that
      // using the center test is conservative at cell resolution.
      const double proj_extent = std::fabs(n[0]) * hw + std::fabs(n[1]) * hh;

      // Effective half-width: pixel tolerance + worst-case cell deviation +
      // tiny safety
      const double T = tol_px_grow + proj_extent;

      int a = 0, b = _cols_cell - 1;

      if (std::fabs(n[0]) < eps) {
        // Horizontal strip: independent of x
        // Accept whole row if |C| <= T, otherwise invalidate the row
        if (std::fabs(C) > T) {
          row.x_min = 1;
          row.x_max = 0;
          continue;
        }
        // else keep [0, cols-1]
      } else {
        // Allowed center-x (in pixels): x ∈ [(-T - C)/n_x, (T - C)/n_x]
        double x_left = (-T - C) / n[0];
        double x_right = (T - C) / n[0];
        if (x_left > x_right)
          std::swap(x_left, x_right);

        // Map to cell indices: center of cell cx is (cx + 0.5) * cell_px
        // eps_c nudges the ceil/floor to be robust against floating rounding at
        // boundaries. ceil and floor is used because proj_extent has already
        // enlarged the range
        if (_coordinate_mode) {
          double left=std::ceil(x_left * to_index - 0.5 - eps_c);
          double right=std::floor(x_right * to_index - 0.5 + eps_c);
          // New matching index: clamp BEFORE integer conversion. Keep legacy
          // raw indexing arithmetic separate for mixed-model compatibility.
          a = static_cast<int>(std::clamp(left,0.,double(_cols_cell)));
          b = static_cast<int>(std::clamp(right,-1.,double(_cols_cell-1)));
        } else {
          // Preserve the established MSVC legacy overflow sentinel/rejection,
          // without invoking undefined floating-to-integer conversion.
          auto legacyIndex=[](double v) {
            if(!std::isfinite(v) || v<std::numeric_limits<int>::min() ||
               v>std::numeric_limits<int>::max()) return std::numeric_limits<int>::min();
            return static_cast<int>(v);
          };
          a = legacyIndex(std::ceil(x_left * to_index - 0.5 - eps_c));
          b = legacyIndex(std::floor(x_right * to_index - 0.5 + eps_c));
        }

        // Clamp to valid column range
        a = std::max(0, a);
        b = std::min(_cols_cell - 1, b);

        if (a > b) {
          row.x_min = 1;
          row.x_max = 0;
          continue;
        } // no overlap on this row
      }

      // Intersect with the existing span on this row
      row.x_min = std::max(row.x_min, a);
      row.x_max = std::min(row.x_max, b);
    }
  }
}

void IDMap::visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                                  const std::vector<Line2D> &lines_px,
                                  double tol_px,
                                  std::vector<int> &out_indices) const {
  out_indices.clear();

  if (lines_px.empty() || spans.size()!=static_cast<size_t>(_rows_cell))
    return;

  // Precompute unit directions (required by cross-product distance)
  std::vector<Pt2D> U(lines_px.size());
  for (size_t i = 0; i < lines_px.size(); ++i)
    {
      const auto& L=lines_px[i];
      if (!std::isfinite(L.pt[0]) || !std::isfinite(L.pt[1]) ||
          !std::isfinite(L.unit_vector[0]) || !std::isfinite(L.unit_vector[1]) ||
          !(std::hypot(L.unit_vector[0],L.unit_vector[1])>0)) return;
      U[i] = normalized(L.unit_vector);
    }

  const double tol2 = tol_px * tol_px;
  const size_t count = _coordinate_mode ? _coordinates.size() : _objs.size();

  // Dedup bitmap: a point may appear in multiple cells along the row-span
  std::vector<unsigned char> seen(count, 0);

  // Iterate every row of cells
  const int rowsC = _rows_cell;
  for (int cy = 0; cy < rowsC; ++cy) {
    const RowSpan &row = spans[cy];
    if (row.x_min > row.x_max)
      continue; // invalid span -> nothing on this row

    // Iterate each cell in the inclusive [x_min, x_max] range
    for (int cx = std::max(0,row.x_min); cx <= std::min(_cols_cell-1,row.x_max); ++cx) {
      const auto &cell = _buckets[idx(cy, cx)];
      for (int pid : cell) {
        if (pid < 0 || static_cast<size_t>(pid) >= count)
          continue;
        if (seen[pid])
          continue;

        const Pt2D &q = _coordinate_mode ? _coordinates[pid] : _objs[pid]->_pt_center;

        // Precise distance check w.r.t. all lines: |(q - p) x u| <= tol
        for (size_t i = 0; i < lines_px.size(); ++i) {
          const Pt2D &Lp = lines_px[i].pt;
          const Pt2D &u = U[i];
          const double dx = q[0] - Lp[0];
          const double dy = q[1] - Lp[1];
          const double z =
              dy * u[0] -
              dx * u[1]; // signed 2D cross (perp distance if u is unit)
          if (!std::isfinite(z) || z * z > tol2)
            goto reject_point;
        }

        // All distances within tol -> accept once
        seen[pid] = 1;
        out_indices.push_back(pid);

      reject_point:;
      }
    }
  }
}
