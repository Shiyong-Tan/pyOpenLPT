
// IPR.cpp (place these before IPR::runIPR)
#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <type_traits>

#include "BubbleRefImg.h"
#include "Camera.h"
#include "Config.h"
#include "IPR.h"
#include "myMATH.h"

// Limit the number of 2D objects PER ACTIVE CAMERA to `cap`.
// - Indices stay aligned with GLOBAL cam_id.
// - Inactive cameras are skipped.
// - Deterministic via fixed RNG seed for reproducibility.
static void limit2DObjectNumber(
    std::vector<std::vector<std::unique_ptr<Object2D>>> &o2d_list_all,
    const std::vector<std::shared_ptr<Camera>> &cams,
    std::size_t cap) {
  if (cap == 0)
    return;
  const std::size_t n_cam = cams.size();
  std::default_random_engine rng(1234);

  for (std::size_t cam_id = 0; cam_id < n_cam; ++cam_id) {
    if (!cams[cam_id]->is_active)
      continue;

    auto &lst = o2d_list_all[cam_id];
    if (lst.size() > cap) {
      // Shuffle then truncate to `cap`
      std::shuffle(lst.begin(), lst.end(), rng);
      lst.resize(cap);
    }
  }
}

// Remove ghosts in-place if a mask is provided (mask[i] == true means "ghost").
static void filterOutInvalid(std::vector<std::unique_ptr<Object3D>> &objs,
                             const std::vector<ObjFlag> &flags) {
  if (flags.size() != objs.size())
    return;

  auto has_any = [](ObjFlag v, ObjFlag mask) {
    using U = std::underlying_type_t<ObjFlag>;
    return (static_cast<U>(v) & static_cast<U>(mask)) != 0;
  };

  const ObjFlag drop_mask = ObjFlag::Ghost | ObjFlag::Repeated;

  std::vector<std::unique_ptr<Object3D>> kept;
  kept.reserve(objs.size());

  for (std::size_t i = 0; i < objs.size(); ++i) {
    if (objs[i] && !has_any(flags[i], drop_mask)) {
      kept.emplace_back(std::move(objs[i]));
    }
  }
  objs.swap(kept);
}

static void setActiveAll(std::vector<std::shared_ptr<Camera>> &camera_models) {
  for (size_t i = 0; i < camera_models.size(); ++i) {
    if (camera_models[i]) {
      camera_models[i]->is_active = true;
    }
  }
}

static void setActiveSubset(std::vector<std::shared_ptr<Camera>> &camera_models,
                            const std::vector<int> &ids) {
  std::vector<char> is_active(camera_models.size(), 0);
  for (int id : ids) {
    if (id >= 0 && static_cast<size_t>(id) < camera_models.size()) {
      is_active[static_cast<size_t>(id)] = 1;
    }
  }

  for (size_t i = 0; i < camera_models.size(); ++i) {
    if (camera_models[i]) {
      const bool active = is_active[i] != 0;
      camera_models[i]->is_active = active;
    }
  }
}

// Run one full IPR iteration on the CURRENT active camera set.
// - Returns newly reconstructed 3D objects (derived from Object3D).
// - Mutates `images` in-place (residual/mask updates) for active cameras only.
// - `cfg` carries both the object type and all IPR parameters.
// - All indices are GLOBAL cam_id aligned (no compacting).
static std::vector<std::unique_ptr<Object3D>>
runSingleIPRIteration(const std::vector<std::shared_ptr<Camera>> &camera_models,
                      std::vector<Image> &images, ObjectConfig &cfg) {
  std::vector<std::unique_ptr<Object3D>> objs_out;

  const auto &ipr = cfg._ipr_param;
  const std::size_t n_cam = camera_models.size();

  // Count active cameras (runIPR ensures >=2; we just need the count for
  // SMParam).
  int n_active = 0;
  for (const auto &c : camera_models)
    if (c->is_active)
      ++n_active;

  // 1) 2D detection (GLOBAL cam_id alignment; inactive slots remain empty)
  ObjectFinder2D finder;
  std::vector<std::vector<std::unique_ptr<Object2D>>>
      o2d_list_all; // smart pointer: automatic memory management, allow
                    // different subclass object
  o2d_list_all.resize(n_cam);

  std::cout << "\t\t2D detections per active camera: ";
  const clock_t t_find2D_start = clock();
#pragma omp parallel for schedule(static)
  for (int cam_id = 0; cam_id < static_cast<int>(n_cam); ++cam_id) {
    if (!camera_models[cam_id]->is_active)
      continue;

    // Implementation should branch internally by reading `cfg` (Tracer/Bubble).
    std::vector<std::unique_ptr<Object2D>> o2d_list =
        finder.findObject2D(images[cam_id], cfg);

    std::cout << o2d_list.size() << "  ";
    o2d_list_all[cam_id] = std::move(o2d_list);
  }
  const clock_t t_find2D_end = clock();
  std::cout << " (" << double(t_find2D_end - t_find2D_start) / CLOCKS_PER_SEC
            << " s)" << "\n";

  // 1.1) Global limiting (drop randomly across active cameras)
  limit2DObjectNumber(o2d_list_all, camera_models,
                      static_cast<std::size_t>(ipr.n_obj2d_process_max));

  // 2) Stereo matching (may parallelize internally; active set is fixed this
  // pass)
  StereoMatch stereo_match(camera_models, o2d_list_all, cfg);

  const clock_t t_match_start = clock();

  // note: the 2D information of objs_out won't be updated since it may be used
  // to calculate bubble reference image
  //       or calibrate camera or OTF parameter
  //       2D information will be updated in shaking

  objs_out = stereo_match.match();

  const clock_t t_match_end = clock();
  const double t_sec = double(t_match_end - t_match_start) / CLOCKS_PER_SEC;

  std::cout << "\t\tMatched " << objs_out.size()
            << " objects. ( 2D tol: " << cfg._sm_param.tol_2d_px
            << ", t:" << t_sec << " s)";

  // Objs_out and o2d_list_all are used for obtaining bubble reference image, or
  // camera calibration, or OTF parameter
  switch (cfg.kind()) {
  case ObjectKind::Tracer:
    // TODO: calibrate OTF and camera can be put here.
    // Note: only match_cam_count cameras have 2D information.
    break;

  case ObjectKind::Bubble: {
    // obatain bubble reference image
    // must pass this before shaking
    auto &bb_cfg = static_cast<BubbleConfig &>(cfg);

      if (!bb_cfg._bb_ref_img._is_valid) {
        const bool ok = bb_cfg._bb_ref_img.calBubbleRefImg(
            objs_out,       // std::vector<std::unique_ptr<Object3D>>
            o2d_list_all,   // std::vector<std::vector<std::unique_ptr<Object2D>>>
            camera_models,  // std::vector<std::shared_ptr<Camera>>
            images,         // const std::vector<Image>&
            bb_cfg._output_path // output folder
        );

      if (!ok) {
        THROW_FATAL_CTX(ErrorCode::NoEnoughData,
                        "Cannot obtain bubble reference image.",
                        "IPR::runSingleIPRIteration");
      }
      std::cout << "\tObtained bubble reference image!";
    }
    break;
  }

  default:
    break;
  }

  if (objs_out.empty()) {
    std::cout << "\n";
    return objs_out;
  }

  // 3) Shake refinement (updates `images` in-place for active cameras)
  Shake shaker(camera_models, cfg);
  const clock_t t_shake_start = clock();

  std::vector<ObjFlag> flags =
      shaker.runShake(objs_out, images); // must honor cams[cam_id]._is_active

  // 4) remove ghost and repeated objects
  filterOutInvalid(objs_out, flags);

  images = shaker.calResidualImage(
      objs_out, images); // get updated images with contructed objects removed.

  const clock_t t_shake_end = clock();
  std::cout << "\tAfter shaking: " << objs_out.size() << " objects remain. ("
            << double(t_shake_end - t_shake_start) / CLOCKS_PER_SEC << " s)\n";

  return objs_out;
}

// ---------- IPR main entry (full + reduced unified) ----------
std::vector<std::unique_ptr<Object3D>> IPR::runIPR(ObjectConfig &cfg,
                                                   std::vector<Image> images) {
  std::cout << "START IPR...\n";
  std::vector<std::unique_ptr<Object3D>> all_objs;

  const size_t n_cam = _cam_list.size();
  if (n_cam < 2) {
    // Need at least 2 cameras for triangulation
    return all_objs;
  }

  const IPRParam &ipr_param = cfg._ipr_param;

  // Always start from a well-defined state: all cameras active
  setActiveAll(_cam_list);

  // drop = 0 .. max_drop; drop=0 means "full set" case.
  // The maximum we can drop is limited by cfg.n_reduced (from ObjectConfig) and
  // by keeping at least 2 cameras.
  const int max_drop = std::min(ipr_param.n_cam_reduced,
                                std::max(0, static_cast<int>(n_cam) - 2));
  const int orig_match_cam_count = cfg._sm_param.match_cam_count;

  for (int drop = 0; drop <= max_drop; ++drop) {
    const size_t K = n_cam - static_cast<size_t>(drop); // subset size (>=2)
    std::vector<std::vector<int>> subsets;
    myMATH::generateCombinations(n_cam, K, subsets);

    // Choose loop count: full set vs reduced set
    const int loops =
        (drop == 0) ? ipr_param.n_loop_ipr : ipr_param.n_loop_ipr_reduced;

    std::cout << ((drop == 0) ? "Full cameras" : "Reduced cameras") << "\n";
    // match_cam_count must not be larger than active cameras
    int match_cam_count = std::min(orig_match_cam_count, static_cast<int>(K));
    cfg._sm_param.match_cam_count = match_cam_count;

    for (size_t si = 0; si < subsets.size(); ++si) {
      const auto &ids = subsets[si];

      // Activate exactly this subset (others inactive)
      setActiveSubset(_cam_list, ids);

      // for obtaining bubble reference image or OTF
      // we need to use all cameras
      // so temporarily change match_cam_count to all active cameras
      bool calibration_loop = false;
      if (drop == 0) {
        switch (cfg.kind()) {
        case ObjectKind::Bubble: {
          auto &bb_cfg = static_cast<BubbleConfig &>(cfg);

          if (!bb_cfg._bb_ref_img._is_valid) {
            // need to locate 2D points on all cameras to get bubble reference
            // images
            cfg._sm_param.match_cam_count = static_cast<int>(_cam_list.size());
            calibration_loop = true;
          }
          break;
        }
        default:
          break;
        }
      }

      std::cout << "Combination " << si << " cams=[";
      for (size_t k = 0; k < ids.size(); ++k) {
        if (k)
          std::cout << ", ";
        std::cout << ids[k];
      }
      std::cout << "]\n";

      double tol_2d_px_orig = cfg._sm_param.tol_2d_px;

      for (int loop = 0; loop < loops; ++loop) {
        // gradually increase 2D tolerance, final loop = tol_2d_px_orig + 1 px
        // cfg._sm_param.tol_2d_px = tol_2d_px_orig + 1.0 / loops * loop; // 1.0
        // is used for "double" calculation

        auto objs = runSingleIPRIteration(_cam_list, images,
                                          cfg); // images will be updated for every loop.

        if (calibration_loop) {
          cfg._sm_param.match_cam_count =
              match_cam_count; // reset back after calibration
          calibration_loop = false;
        }

        if (!objs.empty()) {
          all_objs.insert(all_objs.end(), std::make_move_iterator(objs.begin()),
                          std::make_move_iterator(objs.end()));
        }
        std::cout << "\tLOOP=" << loop
                  << ": TOTAL OBJECTS = " << all_objs.size() << "\n";

        cfg._sm_param.tol_2d_px =
            cfg._sm_param.tol_2d_px * 1.5; // increase by 50% each loop
      }
      cfg._sm_param.tol_2d_px = tol_2d_px_orig; // reset back
    }
  }

  cfg._sm_param.match_cam_count = orig_match_cam_count; // reset back

  // Leave system in a well-defined state: all cameras active on exit
  setActiveAll(_cam_list);

  std::cout << "IPR FINISH! FOUND " << all_objs.size() << " OBJECTS.\n";
  return all_objs;
}

void IPR::saveObjInfo(const std::string &filename,
                      const std::vector<std::unique_ptr<Object3D>> &obj3d_list,
                      const ObjectConfig &cfg) const {
  std::ofstream file(filename);
  REQUIRE_CTX(file.is_open(), ErrorCode::IOfailure, "Cannot open file ",
              filename);

  const int n_cam = static_cast<int>(_cam_list.size());
  const ObjectKind kind = cfg.kind();

  // Header depends on object kind
  file << "WorldX,WorldY,WorldZ,Error,Ncam";
  switch (kind) {
  case ObjectKind::Tracer:
    for (int i = 0; i < n_cam; ++i) {
      file << ",cam" << i << "_x(col)"
           << ",cam" << i << "_y(row)";
    }
    break;
  case ObjectKind::Bubble:
    file << ",R3D";
    for (int i = 0; i < n_cam; ++i) {
      file << ",cam" << i << "_x(col)"
           << ",cam" << i << "_y(row)"
           << ",cam" << i << "_rpx";
    }
    break;
  }
  file << "\n";

  // Rows
  for (const auto &obj : obj3d_list) {
    if (obj)
      obj->saveObject3D(file);
  }
  file.close();
}
