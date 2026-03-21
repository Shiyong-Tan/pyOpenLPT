#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "Camera.h"
#include "STBCommons.h"
#include "py_camera_handle.h"

namespace py = pybind11;

namespace {

[[noreturn]] void throwLegacyFieldWrite(const char* field_name,
                                        const char* replacement) {
  throw std::runtime_error(std::string("Camera.") + field_name +
                           " is read-only in hard migration mode; use " +
                           replacement + " instead.");
}

} // namespace

void bind_Camera(py::module_& m) {
  py::class_<Camera, std::shared_ptr<Camera>>(m, "CameraModel")
      .def_property(
          "is_active", [](const Camera& c) { return c.is_active; },
          [](Camera& c, bool v) { c.is_active = v; })
      .def_property(
          "max_intensity", [](const Camera& c) { return c.max_intensity; },
          [](Camera& c, double v) { c.max_intensity = v; })
      .def("getNRow", &Camera::getNRow)
      .def("getNCol", &Camera::getNCol)
      .def("saveParameters", &Camera::saveParameters, py::arg("file_name"))
      .def(
          "project",
          [](const Camera& c, const Pt3D& pt_world, bool is_print_detail) {
            auto st = c.project(pt_world, is_print_detail);
            if (!st)
              throw std::runtime_error(st.status().err.toString());
            return st.value();
          },
          py::arg("pt_world"), py::arg("is_print_detail") = false)
      .def(
          "lineOfSight",
          [](const Camera& c, const Pt2D& pt_img_dist) {
            auto st = c.lineOfSight(pt_img_dist);
            if (!st)
              throw std::runtime_error(st.status().err.toString());
            return st.value();
          },
          py::arg("pt_img_dist"))
      .def(
          "projectBatchStatus",
          [](const Camera& c, const std::vector<Pt3D>& pts_world,
             bool is_print_detail) {
            std::vector<std::tuple<bool, Pt2D, std::string>> out;
            out.reserve(pts_world.size());
            for (const auto& pt_world : pts_world) {
              auto st = c.project(pt_world, is_print_detail);
              if (st) {
                out.emplace_back(true, st.value(), std::string(""));
              } else {
                out.emplace_back(false, Pt2D(), st.status().err.toString());
              }
            }
            return out;
          },
          py::arg("pts_world"), py::arg("is_print_detail") = false)
      .def(
          "lineOfSightBatchStatus",
          [](const Camera& c, const std::vector<Pt2D>& pts_img_dist) {
            std::vector<std::tuple<bool, Line3D, std::string>> out;
            out.reserve(pts_img_dist.size());
            for (const auto& pt_img_dist : pts_img_dist) {
              auto st = c.lineOfSight(pt_img_dist);
              if (st) {
                out.emplace_back(true, st.value(), std::string(""));
              } else {
                out.emplace_back(false, Line3D(), st.status().err.toString());
              }
            }
            return out;
          },
          py::arg("pts_img_dist"));

  py::class_<PinholeCamera, Camera, std::shared_ptr<PinholeCamera>>(
      m, "PinholeCameraModel")
      .def(py::init<>());
  py::class_<PolynomialCamera, Camera,
             std::shared_ptr<PolynomialCamera>>(m, "PolynomialCameraModel")
      .def(py::init<>());
  py::class_<RefractionPinholeCamera, Camera,
             std::shared_ptr<RefractionPinholeCamera>>(
      m, "RefractionPinholeCameraModel")
      .def(py::init<>());

  m.def(
      "loadCameraModel",
      [](const std::string& file_name) {
        auto st = CameraFactory::loadFromFile(file_name);
        if (!st) {
          throw std::runtime_error(st.status().err.toString());
        }
        return st.value();
      },
      py::arg("file_name"));

  py::enum_<CameraType>(m, "CameraType")
      .value("PINHOLE", CameraType::Pinhole)
      .value("POLYNOMIAL", CameraType::Polynomial)
      .value("PINPLATE", CameraType::RefractionPinhole)
      .export_values();

  py::enum_<RefPlane>(m, "RefPlane")
      .value("REF_X", RefPlane::REF_X)
      .value("REF_Y", RefPlane::REF_Y)
      .value("REF_Z", RefPlane::REF_Z)
      .export_values();

  py::class_<PinholeParam>(m, "PinholeParam")
      .def(py::init<>())
      .def_readwrite("n_row", &PinholeParam::n_row)
      .def_readwrite("n_col", &PinholeParam::n_col)
      .def_readwrite("cam_mtx", &PinholeParam::cam_mtx)
      .def_readwrite("is_distorted", &PinholeParam::is_distorted)
      .def_readwrite("n_dist_coeff", &PinholeParam::n_dist_coeff)
      .def_readwrite("dist_coeff", &PinholeParam::dist_coeff)
      .def_readwrite("r_mtx", &PinholeParam::r_mtx)
      .def_readwrite("t_vec", &PinholeParam::t_vec)
      .def_readwrite("r_mtx_inv", &PinholeParam::r_mtx_inv)
      .def_readwrite("t_vec_inv", &PinholeParam::t_vec_inv);

  py::class_<PolyParam>(m, "PolyParam")
      .def(py::init<>())
      .def_readwrite("n_row", &PolyParam::n_row)
      .def_readwrite("n_col", &PolyParam::n_col)
      .def_readwrite("ref_plane", &PolyParam::ref_plane)
      .def_readwrite("plane", &PolyParam::plane)
      .def_readwrite("n_coeff", &PolyParam::n_coeff)
      .def_readwrite("u_coeffs", &PolyParam::u_coeffs)
      .def_readwrite("du_coeffs", &PolyParam::du_coeffs)
      .def_readwrite("v_coeffs", &PolyParam::v_coeffs)
      .def_readwrite("dv_coeffs", &PolyParam::dv_coeffs);

  py::class_<PinPlateParam, PinholeParam>(m, "PinPlateParam")
      .def(py::init<>())
      .def_readwrite("plane", &PinPlateParam::plane)
      .def_readwrite("plane_array", &PinPlateParam::plane_array)
      .def_readwrite("u_axis", &PinPlateParam::u_axis)
      .def_readwrite("v_axis", &PinPlateParam::v_axis)
      .def_readwrite("refract_array", &PinPlateParam::refract_array)
      .def_readwrite("w_array", &PinPlateParam::w_array)
      .def_readwrite("n_plate", &PinPlateParam::n_plate)
      .def_readwrite("proj_tol", &PinPlateParam::proj_tol)
      .def_readwrite("proj_nmax", &PinPlateParam::proj_nmax)
      .def_readwrite("lr", &PinPlateParam::lr)
      .def_readwrite("refract_ratio_max", &PinPlateParam::refract_ratio_max);

  py::class_<PyCameraHandle>(m, "Camera")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("file_name"))
      .def(py::init<const PyCameraHandle&>(), py::arg("other"))
      .def("loadParameters", &PyCameraHandle::loadParameters,
           py::arg("file_name"))
      .def("saveParameters", &PyCameraHandle::saveParameters,
           py::arg("file_name"))
      .def_property("_type", &PyCameraHandle::legacyType,
                    &PyCameraHandle::setLegacyType)
      .def_property("_pinhole_param", &PyCameraHandle::pinholeParam,
                    [](PyCameraHandle&, const PinholeParam&) {
                      throwLegacyFieldWrite(
                          "_pinhole_param",
                          "setPinplateIntrinsics/setPinplateExtrinsics");
                    })
      .def_property("_poly_param", &PyCameraHandle::polyParam,
                    [](PyCameraHandle&, const PolyParam&) {
                      throwLegacyFieldWrite("_poly_param",
                                            "PolynomialCameraModel setters");
                    })
      .def_property("_pinplate_param", &PyCameraHandle::pinplateParam,
                    [](PyCameraHandle&, const PinPlateParam&) {
                      throwLegacyFieldWrite(
                          "_pinplate_param",
                          "setPinplate* methods + commitPinplateUpdate");
                    })
      .def_property("_max_intensity", &PyCameraHandle::maxIntensity,
                    &PyCameraHandle::setMaxIntensity)
      .def_property("_is_active", &PyCameraHandle::isActive,
                    &PyCameraHandle::setIsActive)
      .def("rmtxTorvec", &PyCameraHandle::rmtxTorvec, py::arg("r_mtx"))
      .def("setPinplateImageSize", &PyCameraHandle::setPinplateImageSize,
           py::arg("n_row"), py::arg("n_col"))
      .def("setPinplateIntrinsics", &PyCameraHandle::setPinplateIntrinsics,
           py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
           py::arg("dist_coeff"))
      .def("setPinplateExtrinsics", &PyCameraHandle::setPinplateExtrinsics,
           py::arg("rvec"), py::arg("tvec"))
      .def("setPinplateRefraction", &PyCameraHandle::setPinplateRefraction,
           py::arg("plane_pt"), py::arg("plane_n"), py::arg("refract_array"),
           py::arg("w_array"), py::arg("proj_tol"), py::arg("proj_nmax"),
           py::arg("lr"))
      .def("commitPinplateUpdate", &PyCameraHandle::commitPinplateUpdate,
           py::arg("is_active"), py::arg("max_intensity"))
      .def("getNRow", &PyCameraHandle::getNRow)
      .def("getNCol", &PyCameraHandle::getNCol)
      .def("project", &PyCameraHandle::project, py::arg("pt_world"),
           py::arg("is_print_detail") = false)
      .def(
          "projectStatus",
          [](const PyCameraHandle& c, const Pt3D& pt_world,
             bool is_print_detail) {
            try {
              return py::make_tuple(true, c.project(pt_world, is_print_detail),
                                    std::string(""));
            } catch (const std::exception& e) {
              return py::make_tuple(false, Pt2D(), std::string(e.what()));
            }
          },
          py::arg("pt_world"), py::arg("is_print_detail") = false)
      .def("projectBatchStatus", &PyCameraHandle::projectBatchStatus,
           py::arg("pts_world"), py::arg("is_print_detail") = false)
      .def("lineOfSight", &PyCameraHandle::lineOfSight,
           py::arg("pt_img_dist"))
      .def(
          "lineOfSightStatus",
          [](const PyCameraHandle& c, const Pt2D& pt_img_dist) {
            try {
              return py::make_tuple(true, c.lineOfSight(pt_img_dist),
                                    std::string(""));
            } catch (const std::exception& e) {
              return py::make_tuple(false, Line3D(), std::string(e.what()));
            }
          },
          py::arg("pt_img_dist"))
      .def("lineOfSightBatchStatus", &PyCameraHandle::lineOfSightBatchStatus,
           py::arg("pts_img_dist"));
}
