// pyStereoMatch.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#define OPENLPT_EXPOSE_PRIVATE

#include "Camera.h"
#include "Config.h"
#include "ObjectInfo.h"
#include "StereoMatch.h"
#include "py_camera_handle.h"
#include "pybind_utils.h" // make_unique_obj2d_grid


namespace py = pybind11;

// ---------- Debug friend access: requires FRIEND_DEBUG(StereoMatch) in header
// ----------
struct DebugAccess_StereoMatch {
  // main pipeline
  static void buildMatch(const StereoMatch &m,
                         std::vector<int> &match_candidate_id,
                         std::vector<std::vector<int>> &build_candidates) {
    m.buildMatch(match_candidate_id, build_candidates);
  }
  static bool checkMatch(const StereoMatch &m,
                         const std::vector<int> &candidate_ids,
                         double &out_e_check) {
    return m.checkMatch(candidate_ids, out_e_check);
  }

  // NEW SIG: pruneMatch(match_candidates, e_checks)
  static std::vector<std::vector<int>>
  pruneMatch(const StereoMatch &m,
             const std::vector<std::vector<int>> &match_candidates,
             const std::vector<double> &e_checks) {
    return m.pruneMatch(match_candidates, e_checks);
  }
  static std::vector<std::unique_ptr<Object3D>>
  triangulateMatch(const StereoMatch &m,
                   const std::vector<std::vector<int>> &selected_matches) {
    return m.triangulateMatch(selected_matches);
  }

  // helpers
  static void enumerateCandidatesOnCam(const StereoMatch &m,
                                       const std::vector<Line3D> &los3d,
                                       int target_cam,
                                       const std::vector<int> &chosen_cams,
                                       const std::vector<Pt2D> &chosen_pts,
                                       std::vector<int> &out) {
    m.enumerateCandidatesOnCam(los3d, target_cam, chosen_cams, chosen_pts, out);
  }
  static bool makeLine2DFromLOS3D(const StereoMatch &m, int cam_id,
                                  const Line3D &los, Line2D &out_line) {
    return m.makeLine2DFromLOS3D(cam_id, los, out_line);
  }
  static bool buildLinesOnCam(const StereoMatch &m,
                              const std::vector<Line3D> &los3d, int cam_id,
                              std::vector<Line2D> &out) {
    return m.buildLinesOnCam(los3d, cam_id, out);
  }
  static int
  selectSecondCameraByLineLength(const StereoMatch &m, const Line3D &los_ref,
                                 const std::vector<int> &remaining_cams) {
    return m.selectSecondCameraByLineLength(los_ref, remaining_cams);
  }
  static int
  selectNextCameraByMaxPairAngle(const StereoMatch &m,
                                 const std::vector<Line3D> &los3d,
                                 const std::vector<int> &remaining_cams) {
    return m.selectNextCameraByMaxPairAngle(los3d, remaining_cams);
  }
  static double computeMinParallaxDeg(const StereoMatch &m,
                                      const std::vector<Line3D> &los3d) {
    return m.computeMinParallaxDeg(los3d);
  }
  static double calTriangulateTol(const StereoMatch &m, double final_tol_3d_mm,
                                  int k_selected, int k_target,
                                  double min_parallax_deg) {
    return m.calTriangulateTol(final_tol_3d_mm, k_selected, k_target,
                               min_parallax_deg);
  }
  static bool TriangulationCheckWithTol(const StereoMatch &m,
                                        const std::vector<Line3D> &los3d,
                                        double tol_3d_mm) {
    return m.TriangulationCheckWithTol(los3d, tol_3d_mm);
  }
  static bool checkBackProjection(const StereoMatch &m, int target_cam,
                                  const Pt2D &q_t,
                                  const std::vector<int> &chosen_cams,
                                  const std::vector<Pt2D> &chosen_pts) {
    return m.checkBackProjection(target_cam, q_t, chosen_cams, chosen_pts);
  }

  // early checks
  static bool objectEarlyCheck(const StereoMatch &m,
                               const std::vector<int> &cams_used,
                               const std::vector<int> &ids_on_used) {
    return m.objectEarlyCheck(cams_used, ids_on_used);
  }
  static bool tracerEarlyCheck(const StereoMatch &m,
                               const std::vector<int> &cams_in_path,
                               const std::vector<int> &ids_in_path) {
    return m.tracerEarlyCheck(cams_in_path, ids_in_path);
  }
  static bool bubbleEarlyCheck(const StereoMatch &m,
                               const std::vector<int> &cams_in_path,
                               const std::vector<int> &ids_in_path) {
    return m.bubbleEarlyCheck(cams_in_path, ids_in_path);
  }

  // idmaps accessor
  static const std::vector<std::unique_ptr<IDMap>> &
  idmaps(const StereoMatch &m) {
    return m._idmaps;
  }
};

void bind_StereoMatch(py::module_ &m) {
  // ===== StereoMatch =====
  py::class_<StereoMatch>(m, "StereoMatch", py::dynamic_attr())
      .def(
          "__init__",
          [](py::handle self, const std::vector<PyCameraHandle> &cams_in,
             const std::vector<std::vector<Object2D *>> &obj2d_by_cam,
             py::object obj_cfg) {
            auto obj2d_owned_keep = std::make_shared<
                std::vector<std::vector<std::unique_ptr<Object2D>>>>(
                make_unique_obj2d_grid(obj2d_by_cam));
            ObjectConfig &cfg_ref = py::cast<ObjectConfig &>(obj_cfg);

            auto camera_models_keep = std::make_shared<std::vector<std::shared_ptr<Camera>>>(
                make_cam_list_from_handles(cams_in,
                                               "StereoMatch pybind ctor"));

            new (self.cast<StereoMatch *>())
                StereoMatch(*camera_models_keep, *obj2d_owned_keep, cfg_ref);
            py::setattr(
                self, "_keep_obj2d",
                py::capsule(
                    new std::shared_ptr<
                        std::vector<std::vector<std::unique_ptr<Object2D>>>>(
                        std::move(obj2d_owned_keep)),
                    [](void *p) {
                      delete static_cast<std::shared_ptr<std::vector<
                          std::vector<std::unique_ptr<Object2D>>>> *>(p);
                    }));
            py::setattr(
                self, "_keep_cam_list",
                py::capsule(new std::shared_ptr<std::vector<std::shared_ptr<Camera>>>(
                                std::move(camera_models_keep)),
                            [](void *p) {
                              delete static_cast<std::shared_ptr<std::vector<
                                  std::shared_ptr<Camera>>> *>(p);
                            }));
            py::setattr(self, "_keep_cfg", obj_cfg);
          },
          py::arg("cams"), py::arg("obj2d_by_cam"), py::arg("obj_cfg"))
      .def(
          "__init__",
          [](py::handle self,
             const std::vector<std::shared_ptr<Camera>> &camera_models_in,
             const std::vector<std::vector<Object2D *>> &obj2d_by_cam,
             py::object obj_cfg) {
            auto obj2d_owned_keep = std::make_shared<
                std::vector<std::vector<std::unique_ptr<Object2D>>>>(
                make_unique_obj2d_grid(obj2d_by_cam));
            ObjectConfig &cfg_ref = py::cast<ObjectConfig &>(obj_cfg);
            auto camera_models_keep =
                std::make_shared<std::vector<std::shared_ptr<Camera>>>(
                    camera_models_in);

            new (self.cast<StereoMatch *>())
                StereoMatch(*camera_models_keep, *obj2d_owned_keep, cfg_ref);
            py::setattr(
                self, "_keep_obj2d",
                py::capsule(
                    new std::shared_ptr<
                        std::vector<std::vector<std::unique_ptr<Object2D>>>>(
                        std::move(obj2d_owned_keep)),
                    [](void *p) {
                      delete static_cast<std::shared_ptr<std::vector<
                          std::vector<std::unique_ptr<Object2D>>>> *>(p);
                    }));
            py::setattr(
                self, "_keep_cam_list",
                py::capsule(new std::shared_ptr<std::vector<std::shared_ptr<Camera>>>(
                                std::move(camera_models_keep)),
                            [](void *p) {
                              delete static_cast<std::shared_ptr<std::vector<
                                  std::shared_ptr<Camera>>> *>(p);
                            }));
            py::setattr(self, "_keep_cfg", obj_cfg);
          },
          py::arg("camera_models"), py::arg("obj2d_by_cam"),
          py::arg("obj_cfg"))

      // public
      .def("match", &StereoMatch::match)

      // private (debug) — names exactly as in header
      .def(
          "buildMatch",
          [](const StereoMatch &self, std::vector<int> match_candidate_id) {
            std::vector<std::vector<int>> out;
            DebugAccess_StereoMatch::buildMatch(self, match_candidate_id, out);
            return out;
          },
          py::arg("match_candidate_id"))
      .def(
          "checkMatch",
          [](const StereoMatch &self, const std::vector<int> &candidate_ids) {
            double e_check = std::numeric_limits<double>::infinity();
            bool ok = DebugAccess_StereoMatch::checkMatch(self, candidate_ids,
                                                          e_check);
            return py::make_tuple(ok, e_check);
          },
          py::arg("candidate_ids"))

      // NEW: pruneMatch(match_candidates, e_checks)
      .def(
          "pruneMatch",
          [](const StereoMatch &self,
             const std::vector<std::vector<int>> &match_candidates,
             const std::vector<double> &e_checks) {
            return DebugAccess_StereoMatch::pruneMatch(self, match_candidates,
                                                       e_checks);
          },
          py::arg("match_candidates"), py::arg("e_checks"))
      .def(
          "triangulateMatch",
          [](const StereoMatch &self,
             const std::vector<std::vector<int>> &selected_matches) {
            return DebugAccess_StereoMatch::triangulateMatch(self,
                                                             selected_matches);
          },
          py::arg("selected_matches"))

      .def(
          "enumerateCandidatesOnCam",
          [](const StereoMatch &self, const std::vector<Line3D> &los3d,
             int target_cam, const std::vector<int> &chosen_cams,
             const std::vector<Pt2D> &chosen_pts) {
            std::vector<int> out;
            DebugAccess_StereoMatch::enumerateCandidatesOnCam(
                self, los3d, target_cam, chosen_cams, chosen_pts, out);
            return out;
          },
          py::arg("los3d"), py::arg("target_cam"),
          py::arg("chosen_cams") = std::vector<int>{},
          py::arg("chosen_pts") = std::vector<Pt2D>{})
      .def(
          "buildLinesOnCam",
          [](const StereoMatch &self, const std::vector<Line3D> &los3d,
             int cam_id) {
            std::vector<Line2D> lines;
            bool ok = DebugAccess_StereoMatch::buildLinesOnCam(self, los3d,
                                                               cam_id, lines);
            return py::make_tuple(ok, lines);
          },
          py::arg("los3d"), py::arg("cam_id"))
      .def(
          "makeLine2DFromLOS3D",
          [](const StereoMatch &self, int cam_id, const Line3D &los) {
            Line2D line;
            bool ok = DebugAccess_StereoMatch::makeLine2DFromLOS3D(
                self, cam_id, los, line);
            return py::make_tuple(ok, line);
          },
          py::arg("cam_id"), py::arg("los"))
      .def(
          "selectSecondCameraByLineLength",
          [](const StereoMatch &self, const Line3D &los_ref,
             const std::vector<int> &remaining_cams) {
            return DebugAccess_StereoMatch::selectSecondCameraByLineLength(
                self, los_ref, remaining_cams);
          },
          py::arg("los_ref"), py::arg("remaining_cams"))
      .def(
          "selectNextCameraByMaxPairAngle",
          [](const StereoMatch &self, const std::vector<Line3D> &los3d,
             const std::vector<int> &remaining_cams) {
            return DebugAccess_StereoMatch::selectNextCameraByMaxPairAngle(
                self, los3d, remaining_cams);
          },
          py::arg("los3d"), py::arg("remaining_cams"))
      .def(
          "computeMinParallaxDeg",
          [](const StereoMatch &self, const std::vector<Line3D> &los3d) {
            return DebugAccess_StereoMatch::computeMinParallaxDeg(self, los3d);
          },
          py::arg("los3d"))
      .def(
          "calTriangulateTol",
          [](const StereoMatch &self, double final_tol_3d_mm, int k_selected,
             int k_target, double min_parallax_deg) {
            return DebugAccess_StereoMatch::calTriangulateTol(
                self, final_tol_3d_mm, k_selected, k_target, min_parallax_deg);
          },
          py::arg("final_tol_3d_mm"), py::arg("k_selected"),
          py::arg("k_target"), py::arg("min_parallax_deg"))
      .def(
          "TriangulationCheckWithTol",
          [](const StereoMatch &self, const std::vector<Line3D> &los3d,
             double tol_3d_mm) {
            return DebugAccess_StereoMatch::TriangulationCheckWithTol(
                self, los3d, tol_3d_mm);
          },
          py::arg("los3d"), py::arg("tol_3d_mm"))
      .def(
          "checkBackProjection",
          [](const StereoMatch &self, int target_cam, const Pt2D &q_t,
             const std::vector<int> &chosen_cams,
             const std::vector<Pt2D> &chosen_pts) {
            return DebugAccess_StereoMatch::checkBackProjection(
                self, target_cam, q_t, chosen_cams, chosen_pts);
          },
          py::arg("target_cam"), py::arg("q_t"), py::arg("chosen_cams"),
          py::arg("chosen_pts"))

      // early checks
      .def(
          "objectEarlyCheck",
          [](const StereoMatch &self, const std::vector<int> &cams_used,
             const std::vector<int> &ids_on_used) {
            return DebugAccess_StereoMatch::objectEarlyCheck(self, cams_used,
                                                             ids_on_used);
          },
          py::arg("cams_used"), py::arg("ids_on_used"))
      .def(
          "tracerEarlyCheck",
          [](const StereoMatch &self, const std::vector<int> &cams_in_path,
             const std::vector<int> &ids_in_path) {
            return DebugAccess_StereoMatch::tracerEarlyCheck(self, cams_in_path,
                                                             ids_in_path);
          },
          py::arg("cams_in_path"), py::arg("ids_in_path"))
      .def(
          "bubbleEarlyCheck",
          [](const StereoMatch &self, const std::vector<int> &cams_in_path,
             const std::vector<int> &ids_in_path) {
            return DebugAccess_StereoMatch::bubbleEarlyCheck(self, cams_in_path,
                                                             ids_in_path);
          },
          py::arg("cams_in_path"), py::arg("ids_in_path"))

      // 获取内部 IDMap（按需调试）；返回指针，寿命绑定到 StereoMatch
      .def(
          "getIDMap",
          [](StereoMatch &self, int cam_id) -> IDMap * {
            const auto &vec = DebugAccess_StereoMatch::idmaps(self);
            if (cam_id < 0 || cam_id >= (int)vec.size())
              return nullptr;
            return vec[cam_id].get();
          },
          py::arg("cam_id"), py::return_value_policy::reference_internal);

  // ===== IDMap =====
  py::class_<IDMap::RowSpan>(m, "RowSpan")
      .def(py::init<>())
      .def_readwrite("x_min", &IDMap::RowSpan::x_min)
      .def_readwrite("x_max", &IDMap::RowSpan::x_max);

  py::class_<IDMap>(m, "IDMap")
      .def(py::init<int, int, int>(), py::arg("img_rows_px"),
           py::arg("img_cols_px"), py::arg("cell_px"))
      .def(
          "rebuild",
          [](IDMap &self, const std::vector<Object2D *> &objs_in) {
            std::vector<const Object2D *> view;
            view.reserve(objs_in.size());
            for (auto *p : objs_in)
              view.push_back(p);
            self.rebuild(view);
          },
          py::arg("objs"))
      .def(
          "computeStripIntersection",
          [](const IDMap &self, const std::vector<Line2D> &lines_px,
             double tol_px) {
            std::vector<IDMap::RowSpan> spans;
            self.computeStripIntersection(lines_px, tol_px, spans);
            return spans;
          },
          py::arg("lines_px"), py::arg("tol_px"))
      .def(
          "visitPointsInRowSpans",
          [](const IDMap &self, const std::vector<IDMap::RowSpan> &spans,
             const std::vector<Line2D> &lines_px, double tol_px) {
            std::vector<int> out;
            self.visitPointsInRowSpans(spans, lines_px, tol_px, out);
            return out;
          },
          py::arg("spans"), py::arg("lines_px"), py::arg("tol_px"))
      .def("rowsCell", &IDMap::rowsCell)
      .def("colsCell", &IDMap::colsCell)
      .def("cellSizePx", &IDMap::cellSizePx);
}
