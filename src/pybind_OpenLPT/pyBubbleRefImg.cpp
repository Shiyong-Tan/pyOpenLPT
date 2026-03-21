#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BubbleRefImg.h"
#include "Camera.h"
#include "ImageIO.h"
#include "ObjectInfo.h" // Object3D/Object2D, Bubble3D/Bubble2D, Tracer3D/Tracer2D
#include "py_camera_handle.h"
#include "pybind_utils.h"


namespace py = pybind11;

void bind_BubbleRefImg(py::module_ &m) {
  py::class_<BubbleRefImg>(m, "BubbleRefImg")
      .def(py::init<>())

      // 如果这两个返回内部缓存，建议加 reference_internal；若按值返回可去掉策略
      .def("getIntRef", &BubbleRefImg::getIntRef, py::arg("camID"),
           py::return_value_policy::reference_internal)

      .def("__getitem__",
           (const Image &(BubbleRefImg::*)(size_t) const) &
               BubbleRefImg::operator[],
           py::return_value_policy::reference_internal)

      .def(
          "calBubbleRefImg",
          [](BubbleRefImg &self,
             const std::vector<Object3D *> &objs3d, // list[Bubble3D]
              const std::vector<std::vector<Object2D *>>
                  &objs2d_by_cam, // per-cam list[Bubble2D]
              const std::vector<PyCameraHandle> &cams,
              const std::vector<Image> &images,
              std::string output_folder, double r_thres, int n_bb_thres) {
            auto objs3d_uq = make_unique_obj3d_list(objs3d);
            auto objs2d_uq = make_unique_obj2d_grid(objs2d_by_cam);
            auto camera_models =
                make_cam_list_from_handles(cams, "calBubbleRefImg");
            py::gil_scoped_release nogil;
            bool ok = self.calBubbleRefImg(objs3d_uq, objs2d_uq, camera_models,
                                           images, output_folder, r_thres,
                                           n_bb_thres);
            if (!ok)
              throw std::runtime_error("calBubbleRefImg failed");
            return ok;
          },
          py::arg("objs3d"), py::arg("objs2d_by_cam"), py::arg("cams"),
          py::arg("images"), py::arg("output_folder") = "",
          py::arg("r_thres") = 6.0, py::arg("n_bb_thres") = 5)
      .def(
          "calBubbleRefImg",
          [](BubbleRefImg &self,
             const std::vector<Object3D *> &objs3d,
             const std::vector<std::vector<Object2D *>> &objs2d_by_cam,
             const std::vector<std::shared_ptr<Camera>> &camera_models,
             const std::vector<Image> &images, std::string output_folder,
             double r_thres, int n_bb_thres) {
            auto objs3d_uq = make_unique_obj3d_list(objs3d);
            auto objs2d_uq = make_unique_obj2d_grid(objs2d_by_cam);
            py::gil_scoped_release nogil;
            bool ok = self.calBubbleRefImg(objs3d_uq, objs2d_uq, camera_models,
                                           images, output_folder, r_thres,
                                           n_bb_thres);
            if (!ok)
              throw std::runtime_error("calBubbleRefImg failed");
            return ok;
          },
          py::arg("objs3d"), py::arg("objs2d_by_cam"),
          py::arg("camera_models"), py::arg("images"),
          py::arg("output_folder") = "", py::arg("r_thres") = 6.0,
          py::arg("n_bb_thres") = 5)

      .def("saveRefImg", &BubbleRefImg::saveRefImg, py::arg("folder"),
           py::arg("n_cam"))
      .def("loadRefImg", &BubbleRefImg::loadRefImg, py::arg("folder"),
           py::arg("n_cam"));
}
