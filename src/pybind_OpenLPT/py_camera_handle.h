#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "Camera.h"
#include "myMATH.h"

class PyCameraHandle {
public:
  PyCameraHandle() = default;
  explicit PyCameraHandle(const std::string& file_name) { loadParameters(file_name); }
  explicit PyCameraHandle(std::shared_ptr<Camera> model)
      : _model(std::move(model)) {
    syncCacheFromModel();
  }

  bool hasModel() const { return static_cast<bool>(_model); }
  const std::shared_ptr<Camera>& model() const { return _model; }

  void loadParameters(const std::string& file_name) {
    auto st = CameraFactory::loadFromFile(file_name);
    if (!st) {
      throw std::runtime_error(st.status().err.toString());
    }
    _model = st.value();
    syncCacheFromModel();
  }

  void saveParameters(const std::string& file_name) {
    auto& m = requireModelMutable("Camera.saveParameters");
    m.is_active = _is_active_cache;
    m.max_intensity = _max_intensity_cache;
    auto st = m.saveParameters(file_name);
    if (!st) {
      throw std::runtime_error(st.err.toString());
    }
  }

  int getNRow() const { return requireModel("Camera.getNRow").getNRow(); }
  int getNCol() const { return requireModel("Camera.getNCol").getNCol(); }

  Pt2D project(const Pt3D& pt_world, bool is_print_detail = false) const {
    auto st = requireModel("Camera.project").project(pt_world, is_print_detail);
    if (!st) {
      throw std::runtime_error(st.status().err.toString());
    }
    return st.value();
  }

  Line3D lineOfSight(const Pt2D& pt_img_dist) const {
    auto st = requireModel("Camera.lineOfSight").lineOfSight(pt_img_dist);
    if (!st) {
      throw std::runtime_error(st.status().err.toString());
    }
    return st.value();
  }

  std::vector<std::tuple<bool, Pt2D, std::string>>
  projectBatchStatus(const std::vector<Pt3D>& pts_world,
                     bool is_print_detail = false) const {
    std::vector<std::tuple<bool, Pt2D, std::string>> out;
    out.reserve(pts_world.size());
    for (const auto& pt_world : pts_world) {
      try {
        out.emplace_back(true, project(pt_world, is_print_detail), std::string(""));
      } catch (const std::exception& e) {
        out.emplace_back(false, Pt2D(), std::string(e.what()));
      }
    }
    return out;
  }

  std::vector<std::tuple<bool, Line3D, std::string>>
  lineOfSightBatchStatus(const std::vector<Pt2D>& pts_img_dist) const {
    std::vector<std::tuple<bool, Line3D, std::string>> out;
    out.reserve(pts_img_dist.size());
    for (const auto& pt_img_dist : pts_img_dist) {
      try {
        out.emplace_back(true, lineOfSight(pt_img_dist), std::string(""));
      } catch (const std::exception& e) {
        out.emplace_back(false, Line3D(), std::string(e.what()));
      }
    }
    return out;
  }

  CameraType legacyType() const {
    if (!_model) {
      return _type_cache;
    }
    return _model->type();
  }

  void setLegacyType(CameraType) {
    throw std::runtime_error(
        "Camera._type is read-only in hard migration mode; use setPinplate* methods + commitPinplateUpdate instead.");
  }

  bool isActive() const { return _model ? _model->is_active : _is_active_cache; }
  void setIsActive(bool v) {
    _is_active_cache = v;
    if (_model) {
      _model->is_active = v;
    }
  }

  double maxIntensity() const {
    return _model ? _model->max_intensity : _max_intensity_cache;
  }
  void setMaxIntensity(double v) {
    _max_intensity_cache = v;
    if (_model) {
      _model->max_intensity = v;
    }
  }

  PinholeParam pinholeParam() const {
    const auto* cam = dynamic_cast<const PinholeCamera*>(_model.get());
    if (!cam) {
      throw std::runtime_error("Camera._pinhole_param is only available for pinhole camera model");
    }
    return cam->param();
  }

  PolyParam polyParam() const {
    const auto* cam = dynamic_cast<const PolynomialCamera*>(_model.get());
    if (!cam) {
      throw std::runtime_error("Camera._poly_param is only available for polynomial camera model");
    }
    return cam->param();
  }

  PinPlateParam pinplateParam() const {
    const auto* cam = dynamic_cast<const RefractionPinholeCamera*>(_model.get());
    if (!cam) {
      throw std::runtime_error("Camera._pinplate_param is only available for pinplate camera model");
    }
    return cam->param();
  }

  void setPinplateImageSize(int n_row, int n_col) {
    auto& cam = ensureRefractionModel();
    requireStatus(cam.setImageSize(n_row, n_col), "Camera.setPinplateImageSize");
  }

  void setPinplateIntrinsics(double fx, double fy, double cx, double cy,
                             const std::vector<double>& dist_coeff) {
    auto& cam = ensureRefractionModel();
    requireStatus(cam.setIntrinsics(fx, fy, cx, cy, dist_coeff),
                  "Camera.setPinplateIntrinsics");
  }

  void setPinplateExtrinsics(const Pt3D& rvec, const Pt3D& tvec) {
    auto& cam = ensureRefractionModel();
    requireStatus(cam.setExtrinsics(rvec, tvec), "Camera.setPinplateExtrinsics");
  }

  void setPinplateRefraction(const Pt3D& plane_pt, const Pt3D& plane_n,
                             const std::vector<double>& refract_array,
                             const std::vector<double>& w_array, double proj_tol,
                             int proj_nmax, double lr) {
    auto& cam = ensureRefractionModel();
    requireStatus(cam.setRefraction(plane_pt, plane_n, refract_array, w_array),
                  "Camera.setPinplateRefraction");
    requireStatus(cam.setSolverOptions(proj_tol, proj_nmax, lr),
                  "Camera.setPinplateRefraction");
  }

  void commitPinplateUpdate(bool is_active, double max_intensity) {
    auto& cam = ensureRefractionModel();
    cam.is_active = is_active;
    cam.max_intensity = max_intensity;
    requireStatus(cam.commitUpdate(), "Camera.commitPinplateUpdate");
    syncCacheFromModel();
  }

  Pt3D rmtxTorvec(const Matrix<double>& r_mtx) const {
    Pt3D r_vec;
    double tr = (myMATH::trace<double>(r_mtx) - 1.0) / 2.0;
    tr = tr > 1.0 ? 1.0 : tr < -1.0 ? -1.0 : tr;
    const double theta = std::acos(tr);
    const double s = std::sin(theta);
    if (std::abs(s) > 1e-12) {
      const double ratio = theta / (2.0 * s);
      r_vec[0] = (r_mtx(2, 1) - r_mtx(1, 2)) * ratio;
      r_vec[1] = (r_mtx(0, 2) - r_mtx(2, 0)) * ratio;
      r_vec[2] = (r_mtx(1, 0) - r_mtx(0, 1)) * ratio;
    } else if (tr > 0.0) {
      r_vec[0] = 0.0;
      r_vec[1] = 0.0;
      r_vec[2] = 0.0;
    } else {
      r_vec[0] = theta * std::sqrt((r_mtx(0, 0) + 1.0) / 2.0);
      r_vec[1] = theta * std::sqrt((r_mtx(1, 1) + 1.0) / 2.0) *
                 (r_mtx(0, 1) > 0 ? 1 : -1);
      r_vec[2] = theta * std::sqrt((r_mtx(2, 2) + 1.0) / 2.0) *
                 (r_mtx(0, 2) > 0 ? 1 : -1);
    }
    return r_vec;
  }

private:
  static void requireStatus(const Status& st, const char* where) {
    if (!st) {
      throw std::runtime_error(std::string(where) + ": " + st.err.toString());
    }
  }

  const Camera& requireModel(const char* where) const {
    if (!_model) {
      throw std::runtime_error(std::string(where) +
                               ": Camera is not initialized; call loadParameters(...) first or use setPinplate* to initialize a pinplate model.");
    }
    return *_model;
  }

  Camera& requireModelMutable(const char* where) {
    if (!_model) {
      throw std::runtime_error(std::string(where) +
                               ": Camera is not initialized; call loadParameters(...) first or use setPinplate* to initialize a pinplate model.");
    }
    return *_model;
  }

  RefractionPinholeCamera& ensureRefractionModel() {
    if (!_model) {
      auto cam = std::make_shared<RefractionPinholeCamera>();
      cam->is_active = _is_active_cache;
      cam->max_intensity = _max_intensity_cache;
      _model = std::move(cam);
      _type_cache = CameraType::RefractionPinhole;
      return *static_cast<RefractionPinholeCamera*>(_model.get());
    }
    auto* pinplate = dynamic_cast<RefractionPinholeCamera*>(_model.get());
    if (!pinplate) {
      throw std::runtime_error("Camera is initialized with non-pinplate model; pinplate setter is unavailable");
    }
    pinplate->is_active = _is_active_cache;
    pinplate->max_intensity = _max_intensity_cache;
    return *pinplate;
  }

  void syncCacheFromModel() {
    if (!_model) {
      return;
    }
    _is_active_cache = _model->is_active;
    _max_intensity_cache = _model->max_intensity;
    _type_cache = legacyType();
  }

  std::shared_ptr<Camera> _model;
  bool _is_active_cache = true;
  double _max_intensity_cache = 255.0;
  CameraType _type_cache = CameraType::RefractionPinhole;
};

inline std::vector<std::shared_ptr<Camera>> make_cam_list_from_handles(
    const std::vector<PyCameraHandle>& cams, const char* context) {
  std::vector<std::shared_ptr<Camera>> camera_models;
  camera_models.reserve(cams.size());
  for (const auto& cam : cams) {
    if (!cam.hasModel()) {
      throw std::runtime_error(std::string(context) +
                               ": camera is not initialized; call loadParameters(...) first");
    }
    camera_models.push_back(cam.model());
  }
  return camera_models;
}
