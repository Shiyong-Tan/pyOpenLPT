#include "STB.h"
#include <cmath>

namespace fs = std::filesystem;

STB::STB(const BasicSetting &setting, const std::string &type,
         const std::string &obj_cfg_path)
    : _basic_setting(setting) {
  // create folder for output
  fs::create_directories(_basic_setting._output_path);
  fs::create_directories(_basic_setting._output_path + "InitialTrack/");
  fs::create_directories(_basic_setting._output_path + "ConvergeTrack/");

  // read configuration settings of IPR, Tracking, Shaking, etc. for the
  // specified object type
  if (type == "Tracer") {
    auto config = std::make_unique<TracerConfig>();
    if (!config->readConfig(obj_cfg_path, _basic_setting)) {
      std::cerr << "Failed to read Tracer config.\n";
      std::exit(1);
    }
    _obj_config = std::move(config);
  } else if (type == "Bubble") {
    auto config = std::make_unique<BubbleConfig>();
    if (!config->readConfig(obj_cfg_path, _basic_setting)) {
      std::cerr << "Failed to read Bubble config.\n";
      std::exit(1);
    }
    _obj_config = std::move(config);
  } else {
    THROW_FATAL_CTX(ErrorCode::UnsupportedType,
                    "Unsupported object type:", type);
  }
  // 未来支持 BubbleConfig 同理

  // Configure VSC
  _vsc.configure(_obj_config->_vsc_param);
}

void STB::processFrame(int frame_id, std::vector<Image> &img_list) {
  if (frame_id < _basic_setting._frame_start ||
      frame_id > _basic_setting._frame_end) {
    THROW_FATAL_CTX(ErrorCode::OutOfRange, "Frame number is out of range",
                    "frame_id=" + std::to_string(frame_id));
  }

  // Process STB
  clock_t t_start, t_end;
  t_start = clock();

  if (frame_id - _basic_setting._frame_start <
      _obj_config->_stb_param._n_initial_frames) {
    // Initial phase
    std::cout << "Initial phase at frame " << frame_id << std::endl;
    runInitPhase(frame_id, img_list);
  } else {
    // Convergence phase
    std::cout << "Convergence phase at frame " << frame_id << std::endl;

    runConvPhase(frame_id, img_list);
  }

  t_end = clock();
  std::cout << "Total time for frame " << frame_id << ": "
            << (double)(t_end - t_start) / CLOCKS_PER_SEC << std::endl;
  std::cout << std::endl;

  if (frame_id == _basic_setting._frame_end &&
      frame_id % 500 != 0) // save at the end if not already saved
  {
    std::cout << "Finish STB!" << std::endl;
    std::cout << "Frame start: " << _basic_setting._frame_start
              << "; Frame end: " << _basic_setting._frame_end << std::endl;
    std::cout << "Number of initial phase frames: "
              << _obj_config->_stb_param._n_initial_frames << std::endl;

    std::cout << "Output folder: " << _basic_setting._output_path << std::endl;

    std::cout << "Number of active short tracks: " << _short_track_active.size()
              << std::endl;
    std::cout << "Number of active long tracks: " << _long_track_active.size()
              << std::endl;
    std::cout << "Number of inactive long tracks: "
              << _long_track_inactive.size() << std::endl;
    std::cout << "Number of exited tracks: " << _exit_track.size() << std::endl;

    saveTracksAll(_basic_setting._output_path + "ConvergeTrack/", frame_id);
  }
}

void STB::runInitPhase(int frame, std::vector<Image> &img_list) {

  IPR ipr(_basic_setting._cam_list);
  // IPR
  std::vector<std::unique_ptr<Object3D>> obj3d_list =
      ipr.runIPR(*_obj_config, img_list);
  _ipr_candidate.emplace_back(std::move(obj3d_list));

  // Calculate the velocity field if number of processed frames is enough.
  // Predict Field
  if (frame - _basic_setting._frame_start ==
      _obj_config->_stb_param._n_initial_frames - 1) {
    std::cout << std::endl;

    for (int i = 0; i < _obj_config->_stb_param._n_initial_frames; i++) {
      int frame_id = i + _basic_setting._frame_start;
      std::cout << "STB initial phase tracking b/w frames: " << frame_id
                << " & " << frame_id + 1 << "; \n";

      if (i == 0) {
        buildTrackFromPredField(
            frame_id,
            nullptr); // no predicted field for the first frame, use nullptr
      } else {
        PredField pf(*_obj_config);

        const clock_t t_start = clock();
        pf.calPredField(_ipr_candidate[i - 1], _ipr_candidate[i]);
        const clock_t t_end = clock();
        std::cout << "\tCalculated predictive field ("
                  << (double)(t_end - t_start) / CLOCKS_PER_SEC << " s), ";

        pf.saveDispField(_basic_setting._output_path + "PredField_" +
                         std::to_string(frame_id) + "_" +
                         std::to_string(frame_id + 1) + ".csv");
        // Build tracks from predicted field
        buildTrackFromPredField(frame_id, &pf);
      }
    }

    // Move all tracks >= _n_initPhase from _short_track_active to
    // _long_track_active
    const auto need_len =
        static_cast<size_t>(_obj_config->_stb_param._n_initial_frames);

    for (auto it = _short_track_active.begin();
         it != _short_track_active.end();) {
      if (it->_obj3d_list.size() >= need_len) {
        _long_track_active.emplace_back(std::move(*it));
        it = _short_track_active.erase(it); // erase 返回下一个位置
      } else {
        ++it;
      }
    }

    std::cout << "Done with initial phase!" << std::endl;
    std::cout << "ACTIVE SHORT TRACKS: " << _short_track_active.size()
              << "; ACTIVE LONG TRACKS: " << _long_track_active.size()
              << "; EXIT TRACKS: " << _exit_track.size() << std::endl;

    // Save all data
    std::string address = _basic_setting._output_path + "InitialTrack/";

    // save all tracks
    saveTracksAll(address, frame);
  }
}

void STB::buildTrackFromPredField(int frame_id, const PredField *pf) {
  // Link particles b/w two frames
  clock_t t_start, t_end;
  t_start = clock();
  int id_ipr = frame_id - _basic_setting._frame_start;

  if (id_ipr > 0) { // first frame does not have a previous frame
    // build kdtree for the next frame
    Obj3dCloud cloud_obj3d(_ipr_candidate[id_ipr]);
    KDTreeObj3d tree_obj3d(3, cloud_obj3d, {10});
    tree_obj3d.buildIndex();
    // extend all the tracks that are active in current frame
    int n_sa = _short_track_active.size();
    std::vector<int> link_id(n_sa, UNLINKED);

#pragma omp parallel for if (!omp_in_parallel())
    for (int j = 0; j < n_sa; j++) {
      Pt3D vel_curr =
          pf->getDisp(_short_track_active[j]._obj3d_list.back()->_pt_center);
      link_id[j] = findNN(
          tree_obj3d,
          _short_track_active[j]._obj3d_list.back()->_pt_center + vel_curr,
          _obj_config->_stb_param._radius_search_obj);
    }
    // update short track and _is_tracked status for the next frame
    for (int j = n_sa - 1; j > -1;
         j--) // reverse order to avoid invalid index after erasing
    {
      if (link_id[j] != UNLINKED) {
        _ipr_candidate[id_ipr][link_id[j]]->_is_tracked = true;
        // we cannot directly move _ipr_candidate because it is still needed for
        // predictive field calculation in the next frame
        CreateArgs a;
        a._proto = _ipr_candidate[id_ipr][link_id[j]].get();
        std::unique_ptr<Object3D> obj3d =
            _obj_config->creatObject3D(std::move(a));
        obj3d->projectObject2D(
            _basic_setting._cam_list); // get 2D projection for the new object
        _short_track_active[j].addNext(
            std::move(obj3d),
            frame_id); // unique_ptr of object3D can only be moved
      } else {
        _short_track_active.erase(_short_track_active.begin() + j);
      }
    }
  }

  // Start a track for all particles left untracked in current frame
  for (int i = 0; i < _ipr_candidate[id_ipr].size(); i++) {
    if (_ipr_candidate[id_ipr][i] // if the object has been moved,
                                  // _ipr_candidate[id_ipr][i] will null_ptr
        && !_ipr_candidate[id_ipr][i]->_is_tracked) {
      _ipr_candidate[id_ipr][i]->_is_tracked = true;
      // we cannot directly move _ipr_candidate because it is still needed for
      // predictive field calculation in the next frame
      CreateArgs a;
      a._proto = _ipr_candidate[id_ipr][i].get();
      std::unique_ptr<Object3D> obj3d =
          _obj_config->creatObject3D(std::move(a));
      obj3d->projectObject2D(
          _basic_setting._cam_list); // get 2D projection for the new object
      // Start a track for the untracked particle
      Track init_tr(std::move(obj3d), frame_id);
      _short_track_active.emplace_back(std::move(
          init_tr)); // Track can only be moved since it has a unique_ptr member
    }
  }

  t_end = clock();
  std::cout << "Linked tracks (" << (double)(t_end - t_start) / CLOCKS_PER_SEC
            << " s)\n";
}

std::unique_ptr<Object3D> STB::predictNext(const Track &tr) const {
  const auto &list = tr._obj3d_list; // or: tr.objects()
  const size_t n = list.size();
  if (n < 3 || !list.back()) {
    // Not enough samples or last is null
    return nullptr;
  }

  // Polynomial order for LMS/Wiener-like predictor (cap at 5)
  const int order = (n < 6) ? static_cast<int>(n) - 1 : 5;
  const size_t start = n - 1 - static_cast<size_t>(order);

  // Create a same-kind object seeded from the last one (copy type-specific
  // params)
  CreateArgs a;
  a._proto = list.back().get(); // center will be overwritten below
  auto obj3d = _obj_config->creatObject3D(std::move(a));
  if (!obj3d)
    return nullptr;

  std::vector<double> series(static_cast<size_t>(order) + 1);
  std::vector<double> filter(static_cast<size_t>(order), 0.0);

  for (int axis = 0; axis < 3; ++axis) {
    // Collect the last (order+1) samples along this axis
    for (int j = 0; j <= order; ++j) {
      series[static_cast<size_t>(j)] =
          list[start + static_cast<size_t>(j)]->_pt_center[axis];
    }

    // Numerical guard: shift if the target is too close to zero
    bool shifted = false;
    constexpr double kShift = 10.0;
    if (std::fabs(series[static_cast<size_t>(order)]) < 1.0) {
      shifted = true;
      for (double &v : series)
        v += kShift;
    }

    // LMS step size ≈ 1 / ||regressors||^2 (guard against zero)
    double denom = 0.0;
    for (int j = 0; j < order; ++j)
      denom += series[static_cast<size_t>(j)] * series[static_cast<size_t>(j)];
    const double step = (denom > 1e-12) ? (1.0 / denom) : 0.0;

    std::fill(filter.begin(), filter.end(), 0.0);

    // Fit coefficients to predict y_t from [y_{t-1}, ..., y_{t-order}]
    double prediction = 0.0;
    double error = series[static_cast<size_t>(order)] - prediction;

    int iter = 0;
    while (std::fabs(error) > SMALLNUMBER && iter < WIENER_MAX_ITER &&
           step > 0.0) {
      prediction = 0.0;
      for (int j = 0; j < order; ++j) {
        const size_t jj = static_cast<size_t>(j);
        filter[jj] += step * series[jj] * error;
        prediction += filter[jj] * series[jj];
      }
      error = series[static_cast<size_t>(order)] - prediction;
      ++iter;
    }

    // One-step-ahead prediction
    if (order >= 1 && step == 0.0) {
      // Fallback: simple linear extrapolation
      prediction = series[static_cast<size_t>(order)] +
                   (series[static_cast<size_t>(order)] -
                    series[static_cast<size_t>(order - 1)]);
    } else {
      prediction = 0.0;
      for (int j = 0; j < order; ++j)
        prediction +=
            filter[static_cast<size_t>(j)] * series[static_cast<size_t>(j + 1)];
    }

    if (shifted)
      prediction -= kShift;
    obj3d->_pt_center[axis] = prediction; // base-class field
  }

  return obj3d;
}

int STB::findNN(KDTreeObj3d const &tree_obj3d, Pt3D const &pt3d_est,
                double radius) const {
  double min_dist2 = radius * radius;
  int obj_id = UNLINKED;

  size_t ret_index = 0;
  double out_dist_sqr = 0;
  nanoflann::KNNResultSet<double> resultSet(1);
  resultSet.init(&ret_index, &out_dist_sqr);
  tree_obj3d.findNeighbors(resultSet, pt3d_est.data(),
                           nanoflann::SearchParameters());

  if (out_dist_sqr < min_dist2) {
    obj_id = ret_index;
  }

  return obj_id;
}

void STB::runConvPhase(int frame, std::vector<Image> &img_list) {
  // Save original images for VSC (before any modifications)
  std::vector<Image> img_orig = img_list;

  // Initialize some variables
  int n_sa = _short_track_active.size();
  int n_la = _long_track_active.size();
  int n_li = _long_track_inactive.size();
  int n_ex = _exit_track.size();
  int add_sa = 0, add_la = 0, rm_sa = 0, rm_la = 0,
      add_li = 0; // s: short, l: long, a: active, i: inactive

  // -------------------- 1. Prediction for active long tracks
  // ----------------------//
  std::cout << " Prediction: ";
  clock_t t_start, t_end;
  t_start = clock();

  std::vector<std::unique_ptr<Object3D>> obj3d_list_pred(n_la);
  std::vector<int> is_inRange(n_la, 1);

#pragma omp parallel for if (!omp_in_parallel())
  for (int i = 0; i < n_la; i++) {
    // Prediction
    obj3d_list_pred[i] = predictNext(_long_track_active[i]);

    // check whether it is out of view
    is_inRange[i] = _basic_setting._axis_limit.check(
        obj3d_list_pred[i]->_pt_center[0], obj3d_list_pred[i]->_pt_center[1],
        obj3d_list_pred[i]->_pt_center[2]);

    // if less than 2 cameras can see the predicted particle remove it
    if (is_inRange[i]) {
      is_inRange[i] =
          obj3d_list_pred[i]->isReconstructable(_basic_setting._cam_list);
    }
  }

  // Remove out-of-range tracks
  size_t w = 0;
  int n_exit = 0;
  int n_exit_del = 0;
  for (size_t i = 0, n = _long_track_active.size(); i < n; ++i) {
    if (!is_inRange[i]) {
      if (_long_track_active[i]._t_list.size() >= LEN_LONG_TRACK) {
        _exit_track.emplace_back(std::move(_long_track_active[i]));
        ++n_exit;
      } else {
        ++n_exit_del;
      }
      ++rm_la;
    } else {
      if (w != i) {
        _long_track_active[w] = std::move(_long_track_active[i]);
        obj3d_list_pred[w] = std::move(obj3d_list_pred[i]);
      }
      ++w;
    }
  }
  _long_track_active.resize(w);
  obj3d_list_pred.resize(w);

  t_end = clock();
  std::cout << (double)(t_end - t_start) / CLOCKS_PER_SEC << " s. Done!"
            << std::endl;

  //----------------------------------- Shake
  // prediction---------------------------------//
  int n_fail_shaking = 0;
  Shake s(_basic_setting._cam_list, *_obj_config);

  if (obj3d_list_pred.size() > 0) {
    t_start = clock();

    std::vector<ObjFlag> flags = s.runShake(obj3d_list_pred, img_list);

    t_end = clock();
    std::cout << " Shake prediction: "
              << (double)(t_end - t_start) / CLOCKS_PER_SEC << " s.\n ";

    // get the residue image for IPR, not including ghost and repeated objects.
    img_list = s.calResidualImage(obj3d_list_pred, img_list, &flags);

    // connect the prediction
    size_t write = 0;
    for (size_t read = 0; read < obj3d_list_pred.size(); ++read) {
      const bool ok =
          (flags[read] == ObjFlag::None); // neither ghost nor repeated
      if (ok) {
        // add to the corresponding track
        _long_track_active[read].addNext(std::move(obj3d_list_pred[read]),
                                         frame);

        // move the track to the right postion if some tracks ahead it was
        // rmoved
        if (write != read)
          _long_track_active[write] = std::move(_long_track_active[read]);

        ++write;
      } else {
        // Ghost / Repeated / combination
        ++n_fail_shaking;
        ++rm_la;
        if (_long_track_active[read]._t_list.size() >= LEN_LONG_TRACK) {
          _long_track_inactive.push_back(std::move(_long_track_active[read]));
          ++add_li;
        }
      }
    }
    _long_track_active.resize(write);
    obj3d_list_pred.clear();
  }

  //------------------------------IPR on residue
  // images----------------------------------//
  // get the residual image

  IPR ipr(_basic_setting._cam_list);
  std::vector<std::unique_ptr<Object3D>> obj3d_list =
      ipr.runIPR(*_obj_config, img_list);

  // remove the particles that are close to the active long tracks
  t_start = clock();

  std::vector<ObjFlag> is_repeate = checkRepeat(obj3d_list);
  std::vector<std::unique_ptr<Object3D>> kept;
  kept.reserve(obj3d_list.size());
  for (size_t i = 0; i < obj3d_list.size(); ++i) {
    if (is_repeate[i] == ObjFlag::None)
      kept.emplace_back(std::move(obj3d_list[i]));
  }
  obj3d_list.swap(kept);

  t_end = clock();
  std::cout << " Remove overlap: " << (double)(t_end - t_start) / CLOCKS_PER_SEC
            << " s." << std::endl;

  //--------------------- Link each _short_track_active to an obj
  //----------------------//
  int n_obj3d = obj3d_list.size();
  if (n_obj3d > 0) {
    std::cout << " Linking: ";
    t_start = clock();

    std::vector<int> link_id(n_sa, 0);
    std::vector<int> is_obj_used(n_obj3d, 0);

    // build kd tree for obj3d_list
    Obj3dCloud cloud_obj3d(obj3d_list);
    KDTreeObj3d tree_obj3d(3, cloud_obj3d, {10 /* max leaf */});
    tree_obj3d.buildIndex();

    // build kd tree for active long tracks
    TrackCloud cloud_track(_long_track_active);
    KDTreeTrack tree_track(3, cloud_track, {10 /* max leaf */});
    tree_track.buildIndex();

// query the kd tree for each active short track
#pragma omp parallel for if (!omp_in_parallel())
    for (int i = 0; i < n_sa; i++) {
      link_id[i] =
          linkShortTrack(_short_track_active[i], 5, tree_obj3d, tree_track);
    }

    // Resolve conflicts: if multiple short tracks link to the same obj, only
    // keep one link (first-come-first-served) TODO: optimize this part
    const int n_cand = n_obj3d;
    std::vector<int> owner(n_cand, -1);

    for (int j = 0; j < n_sa; ++j) {
      const int lid = link_id[j];
      if (lid == UNLINKED || lid < 0 || lid >= n_cand)
        continue;

      if (owner[lid] == -1) {
        owner[lid] = j;
      } else {
        link_id[j] = UNLINKED;
      }
    }

    // In-place compaction pointer for the short-track deque
    size_t write = 0;

    for (int i = 0; i < n_sa; ++i) {
      const int lid = link_id[i];

      if (lid != UNLINKED && lid >= 0 && lid < n_obj3d && obj3d_list[lid]) {
        // Mark the candidate as taken BEFORE moving it
        obj3d_list[lid]->_is_tracked = true;
        // Append the linked candidate to this short track (move-only)
        _short_track_active[i].addNext(std::move(obj3d_list[lid]), frame);
        // Promote to long track if it has become long enough; otherwise keep it
        // (in place-compacted region)
        if (_short_track_active[i]._t_list.size() >=
            _obj_config->_stb_param._n_initial_frames) {
          // Move the finished short track into the long-track list
          _long_track_active.emplace_back(std::move(_short_track_active[i]));
          ++add_la; // accounting: added to long-active
          ++rm_sa;  // accounting: processed one short-active
        } else {
          // Keep as short-active: move it to the 'write' slot if needed
          if (write != static_cast<size_t>(i))
            _short_track_active[write] = std::move(_short_track_active[i]);
          ++write;
        }
      } else {
        // Not linked this round: drop this short track (do not keep, do not
        // promote)
        ++rm_sa; // accounting: processed one short-active
                 // No push/move; the in-place compaction will skip it
      }
    }

    // Shrink deque to the kept short-active tracks
    _short_track_active.resize(write);
    // Create new short tracks for all candidates that were not used this round.
    // NOTE: We must move the candidate object into the new track.
    for (int i = 0; i < n_obj3d; ++i) {
      if (obj3d_list[i] &&
          !obj3d_list[i]->_is_tracked) // obj3d_list[i] is needed because some
                                       // are moved and become nullptr
      {
        Track tr(std::move(obj3d_list[i]), frame);
        _short_track_active.emplace_back(std::move(tr));
        ++add_sa; // accounting: added to short-active
      }
    }

    t_end = clock();
    std::cout << (double)(t_end - t_start) / CLOCKS_PER_SEC << " s. Done!"
              << std::endl;
  }

  //--------------------------- Prune and arrange the tracks
  //--------------------------- //
  int n_fail_lf = 0;

  t_start = clock();

  const size_t n_la_new = _long_track_active.size();

  // Parallel phase: decide which tracks to erase (read-only)
  std::vector<uint8_t> is_erase(n_la_new, 0);

#pragma omp parallel for if (!omp_in_parallel())
  for (int i = 0; i < static_cast<int>(n_la_new); ++i) {
    if (!checkLinearFit(_long_track_active[static_cast<size_t>(i)])) {
      is_erase[static_cast<size_t>(i)] = 1;
    }
  }

  // Serial phase: in-place compaction + move erased ones to inactive if long
  // enough
  size_t write = 0;
  for (size_t i = 0; i < n_la_new; ++i) {
    if (is_erase[i]) {
      if (_long_track_active[i]._obj3d_list.size() >= LEN_LONG_TRACK) {
        _long_track_inactive.emplace_back(std::move(_long_track_active[i]));
        ++add_li;
      }
      ++rm_la;
      ++n_fail_lf;
    } else {
      if (write != i)
        _long_track_active[write] = std::move(_long_track_active[i]);
      ++write;
    }
  }

  // Shrink to the kept active long tracks
  _long_track_active.resize(write);

  t_end = clock();
  std::cout << " Pruning time: "
            << static_cast<double>(t_end - t_start) / CLOCKS_PER_SEC << " s"
            << std::endl;

  // --------------------------- VSC: Volume Self Calibration
  // --------------------------- //
  // ----- VSC: Only accumulate every N frames and only if not already
  // calibrated -----
  bool has_active_pinhole = false;
  for (const auto &cam : _basic_setting._cam_list) {
    if (cam && cam->is_active && cam->type() == CameraType::Pinhole) {
      has_active_pinhole = true;
      break;
    }
  }

  // bool skip_vsc = !has_active_pinhole ||
  //                 (_obj_config->_vsc_param._camera_calibrated &&
  //                  (!_obj_config->_vsc_param._enable_otf ||
  //                   _obj_config->_vsc_param._otf_calibrated));
  bool skip_vsc = true;

  bool is_accumulate_frame =
      (frame % _obj_config->_vsc_param._accumulate_interval == 0);

  if (!skip_vsc && is_accumulate_frame) {
    clock_t t_vsc_start = clock();
    _vsc.accumulate(frame, _long_track_active, img_orig,
                    _basic_setting._cam_list, *_obj_config);
    clock_t t_vsc_accum = clock();

    std::cout << " VSC Accumulate: " << _vsc.getBufferSize() << " points ("
              << static_cast<double>(t_vsc_accum - t_vsc_start) / CLOCKS_PER_SEC
              << " s)" << std::endl;

    if (_vsc.isReady() && !_obj_config->_vsc_param._camera_calibrated) {
      std::cout << "Running VSC optimization..." << std::endl;
      bool updated = _vsc.runVSC(_basic_setting._cam_list);
      clock_t t_vsc_opt = clock();

      if (updated) {
        std::cout << "Camera parameters updated by VSC!" << std::endl;
        _obj_config->_vsc_param._camera_calibrated = true;

        // // Reset VSC buffer to collect new data with updated cams
        // _vsc.reset();

        // If Tracer, update OTF (only once)
        if (_obj_config->kind() == ObjectKind::Tracer &&
            _obj_config->_vsc_param._enable_otf &&
            !_obj_config->_vsc_param._otf_calibrated) {
          std::cout << "Running OTF update..." << std::endl;
          auto *tracer_cfg = dynamic_cast<TracerConfig *>(_obj_config.get());
          if (tracer_cfg) {
            std::vector<TracerConfig> cfgs;
            cfgs.push_back(*tracer_cfg);

            _vsc.runOTF(cfgs);

            tracer_cfg->_otf = cfgs[0]._otf;
            _obj_config->_vsc_param._otf_calibrated = true;
            std::cout << "OTF parameters updated." << std::endl;
          }
        }
      }

      std::cout << " VSC Optimization time: "
                << static_cast<double>(t_vsc_opt - t_vsc_accum) / CLOCKS_PER_SEC
                << " s" << std::endl;
    }
  }

  // Print outputs
  std::cout << "\tNo. of active short tracks: " << n_sa << " + " << add_sa
            << " - " << rm_sa << " = " << _short_track_active.size()
            << std::endl;
  std::cout << "\tNo. of active long tracks: " << n_la << " + " << add_la
            << " - " << rm_la << " = " << _long_track_active.size()
            << std::endl;
  std::cout << "\tNo. of exited long tracks: " << n_ex << " + " << n_exit
            << " = " << _exit_track.size() << std::endl;
  std::cout << "\tNo. of exited short tracks(deleted): " << n_exit_del
            << std::endl;
  std::cout << "\tNo. of inactive Long tracks: " << n_li << " + " << add_li
            << " = " << _long_track_inactive.size() << std::endl;
  std::cout << "\tNo. of fail shaking intensity/repeated: " << n_fail_shaking
            << std::endl;
  std::cout << "\tNo. of fail linear fit: " << n_fail_lf << std::endl;

  // save all data every 500 frames
  if (frame % 500 == 0) {
    saveTracksAll(_basic_setting._output_path + "ConvergeTrack/", frame);
    _long_track_inactive.clear();
    _exit_track.clear();
  }
}

std::vector<ObjFlag>
STB::checkRepeat(const std::vector<std::unique_ptr<Object3D>> &objs) const {
  const size_t n_obj3d = objs.size();
  std::vector<ObjFlag> flags(n_obj3d, ObjFlag::None);
  if (n_obj3d == 0)
    return flags;

  // get the last point in active long tracks, to avoid frequent reference
  // back()
  std::vector<Pt3D> last_centers;
  last_centers.reserve(_long_track_active.size());
  for (const auto &tr : _long_track_active) {
    const auto &p_last_ptr = tr._obj3d_list.back();
    last_centers.push_back(p_last_ptr->_pt_center);
  }
  if (last_centers.empty())
    return flags;

  const double tol = _obj_config->_sm_param.tol_3d_mm;

  const int ni_obj3d = static_cast<int>(n_obj3d);

#pragma omp parallel for if (!omp_in_parallel())
  for (int i = 0; i < ni_obj3d; ++i) {
    const Pt3D &p = objs[i]->_pt_center;

    // for bubbles, tol needs to be added with radius
    double r_obj = 0.0;
    if (const auto *bubble = dynamic_cast<const Bubble3D *>(objs[i].get())) {
      r_obj = bubble->_r3d;
    }

    for (const Pt3D &q : last_centers) {
      Pt3D d = p - q;
      double dist = d.norm();
      if (dist <= tol + r_obj) {
        flags[i] = ObjFlag::Repeated;
        break;
      }
    }
  }

  return flags;
}

int STB::linkShortTrack(Track const &track, int n_iter,
                        KDTreeObj3d const &tree_obj3d,
                        KDTreeTrack const &tree_track) {
  // --- Early checks --------------------------------------------------------
  if (track._obj3d_list.empty() || !track._obj3d_list.back())
    return UNLINKED;

  const Pt3D &p_last = track._obj3d_list.back()->_pt_center;

  const double base_r = _obj_config->_stb_param._radius_search_track;
  if (base_r <= 0.0)
    return UNLINKED;

  // Iterative expansion factor (grow^step); avoid pow() inside the loop.
  constexpr double grow = 1.1;
  double factor = 1.0; // equals grow^step

  int obj3d_id = UNLINKED;

  for (int step = 0; step < n_iter; ++step) {
    factor = (step == 0) ? 1.0 : factor * grow;

    // Neighbor-track search radius (larger band, e.g., 3x)
    const double r_track = (3.0 * base_r) * factor;
    const double r_track2 = r_track * r_track;

    // Collect neighbor long tracks around p_last within r_track.
    // indices_dists[j].first  -> neighbor track index
    // indices_dists[j].second -> squared L2 distance to p_last
    std::vector<nanoflann::ResultItem<size_t, double>> indices_dists;
    nanoflann::RadiusResultSet<double, size_t> result_set(r_track2,
                                                          indices_dists);

    nanoflann::SearchParameters params;
    params.sorted = false; // no need to sort for a weighted average
    tree_track.findNeighbors(result_set, p_last.data(), params);

    // --- Compute neighbor-based velocity estimate with adaptive L --------
    Pt3D vel{0, 0, 0};

    if (!indices_dists.empty()) {
      // Adaptive scale L from the mean squared distance of neighbors:
      //   L^2 = (alpha^2) * mean(d^2), clamped to [ (0.1R)^2 , (0.8R)^2 ]
      // where R = base_r * factor
      double mean_d2 = 0.0;
      for (const auto &it : indices_dists)
        mean_d2 += it.second;
      mean_d2 /= static_cast<double>(indices_dists.size());

      constexpr double alpha = 0.8; // tweak 0.6~1.2 as needed
      double L2 = (alpha * alpha) * mean_d2;

      const double R = base_r * factor;
      const double Lmin2 = (0.1 * R) * (0.1 * R);
      const double Lmax2 = (0.8 * R) * (0.8 * R);
      L2 = std::clamp(L2, Lmin2, Lmax2);

      const double beta2 = 1.0 / (L2 + 1e-12);

      // Weighted average of last-frame displacements among neighbors:
      //   w = 1 / (1 + beta^2 * dist^2)
      // NOTE: dist^2 comes directly from nanoflann (it.second).
      Pt3D sum_disp{0, 0, 0};
      double sum_w = 0.0;

      for (const auto &it : indices_dists) {
        const size_t idx = it.first;
        const double d2 = it.second;

        if (idx >= _long_track_active.size())
          continue;
        const auto &tr = _long_track_active[idx];
        const size_t len = tr._obj3d_list.size();
        if (len < 2)
          continue;
        if (!tr._obj3d_list[len - 1] || !tr._obj3d_list[len - 2])
          continue;

        const Pt3D disp = tr._obj3d_list[len - 1]->_pt_center -
                          tr._obj3d_list[len - 2]->_pt_center;

        const double w = 1.0 / (1.0 + d2 * beta2);

        sum_w += w;
        sum_disp[0] += w * disp[0];
        sum_disp[1] += w * disp[1];
        sum_disp[2] += w * disp[2];
      }

      if (sum_w > 0.0) {
        const double inv = 1.0 / sum_w;
        vel[0] = sum_disp[0] * inv;
        vel[1] = sum_disp[1] * inv;
        vel[2] = sum_disp[2] * inv;
      } else {
        // No valid neighbor displacements (e.g., all had len<2): fallback to
        // zero velocity. The object-association radius below still expands with
        // 'factor'.
        vel = Pt3D{0, 0, 0};
      }
    } else {
      // No neighbor tracks found within r_track: fallback to zero velocity.
      vel = Pt3D{0, 0, 0};
    }

    // Predicted position for object association (p_last + vel).
    const Pt3D est{p_last[0] + vel[0], p_last[1] + vel[1], p_last[2] + vel[2]};

    // Expanding object-association radius as well
    const double r_obj = base_r * factor;

    obj3d_id = findNN(tree_obj3d, est, r_obj);
    if (obj3d_id != UNLINKED) {
      break; // success
    }

    // Optional extra fallback:
    // If there were zero neighbors, you may also try the zero-velocity point
    // 'p_last' once more with the same r_obj to be symmetric with legacy
    // behavior:
    //
    if (indices_dists.empty()) {
      obj3d_id = findNN(tree_obj3d, p_last, r_obj);
      if (obj3d_id != UNLINKED)
        break;
    }
  }

  return obj3d_id;
}

bool STB::checkLinearFit(Track const &track) {
  int len = track._t_list.size();
  int n_initPhase = _obj_config->_stb_param._n_initial_frames;
  int n_pts = n_initPhase > 4
                  ? 4
                  : n_initPhase; // only select tha last four points at maximum.

  if (len < n_pts)
    return false;

  Matrix<double> coeff(3, 2, 0);
  Matrix<double> x_mat(n_pts, 2, 0);
  Matrix<double> kernel(2, n_pts, 0);
  Matrix<double> y_mat(n_pts, 1, 0);
  Matrix<double> temp(2, 1, 0);

  // prepare x_mat
  for (int i = 0; i < n_pts; i++) {
    x_mat(i, 0) = 1;
    x_mat(i, 1) = i;
  }

  // calculate the kernel matrix
  kernel = myMATH::inverse(x_mat.transpose() * x_mat) * x_mat.transpose();

  // calculate coeff at each direction and the estimate point
  Pt3D est;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < n_pts; j++) {
      y_mat(j, 0) = track._obj3d_list[len - n_pts + j]->_pt_center[i];
    }

    temp = kernel * y_mat;
    coeff(i, 0) = temp(0, 0);
    coeff(i, 1) = temp(1, 0);
    est[i] = coeff(i, 0) + (n_pts - 1) * coeff(i, 1);
  }

  // calculate the residue
  double res = myMATH::dist(track._obj3d_list[len - 1]->_pt_center, est);
  double err_max = MAX_ERR_LINEARFIT;
  double err_min = 0.6 * err_max;

  // std::cout << "residue: " << res2 << std::endl;

  if (res > err_max) {
    return false;
  }

  return true;
}

void STB::saveTracksAll(const std::string &folder, int frame) {
  namespace fs = std::filesystem;
  const fs::path dir(folder);
  const std::string s = std::to_string(frame);

  // Helper: remove all files in 'dir' that start with the given prefix and end
  // with ".csv"
  auto remove_with_prefix = [&](std::string_view prefix) {
    if (!fs::exists(dir))
      return;
    for (const auto &e : fs::directory_iterator(dir)) {
      if (!e.is_regular_file())
        continue;
      const std::string name = e.path().filename().string();
      // Match files like prefix + anything + ".csv"
      if (name.size() >= prefix.size() + 4 &&
          name.compare(0, prefix.size(), prefix) == 0 &&
          name.rfind(".csv") == name.size() - 4) {
        std::error_code ec;
        fs::remove(e.path(), ec); // ignore deletion error
      }
    }
  };

  // Remove old Active track snapshots (with frame numbers in the file name)
  remove_with_prefix("LongTrackActive_");
  remove_with_prefix("ShortTrackActive_");

  // Save current Active tracks (with current frame number in the file name)
  saveTracks((dir / ("LongTrackActive_" + s + ".csv")).string(),
             _long_track_active);
  saveTracks((dir / ("ShortTrackActive_" + s + ".csv")).string(),
             _short_track_active);

  // Save Exit and Inactive tracks; keep historical snapshots for each frame
  saveTracks((dir / ("ExitTrack_" + s + ".csv")).string(), _exit_track);
  saveTracks((dir / ("LongTrackInactive_" + s + ".csv")).string(),
             _long_track_inactive);
  std::cout << "  Saved all tracks to folder: " << folder << std::endl;
}

void STB::saveTracks(std::string const &file, std::deque<Track> &tracks) {
  std::ofstream output(file, std::ios::out);
  REQUIRE_CTX(output.is_open(), ErrorCode::IOfailure,
              "Cannot open file:", file);

  output.setf(std::ios::fixed);
  output.precision(6);

  const int n_cam = _basic_setting._n_cam;
  const ObjectKind kind = _obj_config->kind(); // Tracer or Bubble

  // Header (no more Error,Ncam; FrameID is integer frame index)
  output << "TrackID,FrameID,WorldX,WorldY,WorldZ";
  switch (kind) {
  case ObjectKind::Tracer:
    for (int i = 0; i < n_cam; ++i) {
      output << ",cam" << i << "_x(col),cam" << i << "_y(row)";
    }
    break;
  case ObjectKind::Bubble:
    output << ",R3D";
    for (int i = 0; i < n_cam; ++i) {
      output << ",cam" << i << "_x(col),cam" << i << "_y(row),cam" << i
             << "_rpx";
    }
    break;
  default:
    // If you may add more types in future
    break;
  }
  output << "\n";

  for (size_t i = 0; i < tracks.size(); ++i) {
    tracks[i].saveTrack(output, static_cast<int>(i));
  }

  output.close();
}

void STB::loadTracks(const std::string &file, std::deque<Track> &tracks) {
  std::ifstream fin(file);
  REQUIRE_CTX(fin.is_open(), ErrorCode::IOfailure,
              "LoadTracks: cannot open file:", file);

  std::string header;
  std::getline(fin, header); // skip header

  tracks.clear();

  while (true) {
    std::streampos pos = fin.tellg();
    std::string line;
    if (!std::getline(fin, line))
      break; // EOF
    if (line.empty())
      continue;

    std::istringstream row(line);
    std::string s_tid;
    if (!std::getline(row, s_tid, ','))
      continue;

    int tid = -1;
    try {
      tid = std::stoi(s_tid);
    } catch (...) {
      continue;
    } // try to get track ID.

    // Rewind this line so Track::loadTrack can process it as the first row of
    // that track.
    fin.clear();
    fin.seekg(pos);

    // Ensure we have a Track slot for this tid
    while (static_cast<int>(tracks.size()) <= tid)
      tracks.emplace_back();

    // Let the track consume all its consecutive rows.
    tracks[tid].loadTrack(fin, *_obj_config, _basic_setting._cam_list);
  }

  fin.close();
}

void STB::loadTracksAll(std::string const &folder, int frame) {
  const std::string s = std::to_string(frame);

  struct Item {
    const char *prefix;
    std::deque<Track> &dst;
  } items[] = {
      {"LongTrackActive_", _long_track_active},
      {"ShortTrackActive_", _short_track_active},
  };

  for (auto &it : items) {
    const std::string path = folder + it.prefix + s + ".csv";

    // Just a probe to check existence/openability; rest of logic unchanged.
    std::ifstream probe(path);
    REQUIRE_CTX(probe.is_open(), ErrorCode::IOfailure,
                "LoadTracks: missing file:", path);

    probe.close();

    loadTracks(path,
               it.dst); // dst has _long_track_active and _short_track_active
  }
}
