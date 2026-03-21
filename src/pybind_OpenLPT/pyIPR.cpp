// pyIPR.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "IPR.h"
#include "Config.h"
#include "Camera.h"
#include "ImageIO.h"
#include "ObjectInfo.h"     // Object3D / Bubble3D / ...
#include "py_camera_handle.h"
#include "pybind_utils.h"   // make_unique_obj3d_list (我们刚建的工具头)

namespace py = pybind11;

void bind_IPR(py::module_& m) {
    py::class_<IPR>(m, "IPR", py::dynamic_attr())
        .def(
            "__init__",
            [](py::handle self,
               const std::vector<PyCameraHandle>&                           cams_in)
            {
                auto camera_models_keep = std::make_shared<std::vector<std::shared_ptr<Camera>>>(
                    make_cam_list_from_handles(cams_in, "IPR pybind ctor"));

                new (self.cast<IPR*>()) IPR(*camera_models_keep);
                py::setattr(
                    self, "_keep_cam_list",
                    py::capsule(
                        new std::shared_ptr<std::vector<std::shared_ptr<Camera>>>(std::move(camera_models_keep)),
                        [](void* p){ delete static_cast<std::shared_ptr<std::vector<std::shared_ptr<Camera>>>*>(p); }
                    )
                );
            },
            py::arg("cams")
        )
        .def(
            "__init__",
            [](py::handle self,
               const std::vector<std::shared_ptr<Camera>>& camera_models_in)
            {
                auto camera_models_keep = std::make_shared<std::vector<std::shared_ptr<Camera>>>(camera_models_in);
                new (self.cast<IPR*>()) IPR(*camera_models_keep);
                py::setattr(
                    self, "_keep_cam_list",
                    py::capsule(
                        new std::shared_ptr<std::vector<std::shared_ptr<Camera>>>(std::move(camera_models_keep)),
                        [](void* p){ delete static_cast<std::shared_ptr<std::vector<std::shared_ptr<Camera>>>*>(p); }
                    )
                );
            },
            py::arg("camera_models")
        )
        
        .def("runIPR",
             &IPR::runIPR,
             py::arg("cfg"), py::arg("images"))

        
        .def("saveObjInfo",
             [](IPR& self,
                const std::string& filename,
                const std::vector<Object3D*>& objs,   // Python: list[Bubble3D/...]
                const ObjectConfig& cfg) {
                 auto out = make_unique_obj3d_list(objs);  // 重建 unique_ptr 容器
                 py::gil_scoped_release nogil;
                 self.saveObjInfo(filename, out, cfg);
             },
             py::arg("filename"), py::arg("objs"), py::arg("cfg"));
}
