#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "OTF.h"
#include "Config.h"   // TracerConfig
#include "Camera.h"
#include "STBCommons.h"

namespace py = pybind11;

void bind_OTF(py::module_& m)
{
    py::class_<OTFParam>(m, "OTFParam")
        .def(py::init<>())
        .def_readwrite("dx", &OTFParam::dx)
        .def_readwrite("dy", &OTFParam::dy)
        .def_readwrite("dz", &OTFParam::dz)
        .def_readwrite("n_cam", &OTFParam::n_cam)
        .def_readwrite("nx", &OTFParam::nx)
        .def_readwrite("ny", &OTFParam::ny)
        .def_readwrite("nz", &OTFParam::nz)
        .def_readwrite("n_grid", &OTFParam::n_grid)
        .def_readwrite("boundary", &OTFParam::boundary)
        .def_readwrite("grid_x", &OTFParam::grid_x)
        .def_readwrite("grid_y", &OTFParam::grid_y)
        .def_readwrite("grid_z", &OTFParam::grid_z)
        // 注意：a,b,c,alpha 是 Matrix<double>，如果你之后把 Matrix<double> 映射成 Python 类型，再把下面四个也开放：
        // .def_readwrite("a", &OTFParam::a) ...
        ;

    py::class_<OTF>(m, "OTF")
        .def(py::init<>())
        .def(py::init<int,int,int,int, const AxisLimit&, const std::vector<std::shared_ptr<Camera>>&>(),
             py::arg("n_cam"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("boundary"), py::arg("camera_models"))
        .def(py::init<const std::string&>(), py::arg("otf_file"))
        .def("loadParam",
             py::overload_cast<int,int,int,int, const AxisLimit&, const std::vector<std::shared_ptr<Camera>>&>(&OTF::loadParam),
             py::arg("n_cam"), py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("boundary"), py::arg("camera_models"))
        .def("loadParam",
             py::overload_cast<std::string>(&OTF::loadParam),
             py::arg("otf_file"))
        .def("saveParam", &OTF::saveParam, py::arg("otf_file"))
        .def("estimateUniformOTFFromImage",
             [](OTF& self, int cam_id, TracerConfig& cfg, const std::vector<Image>& imgs){
                 self.estimateUniformOTFFromImage(cam_id, cfg, imgs);
             }, py::arg("cam_id"), py::arg("tracer_config"), py::arg("img_list"))
        .def("getOTFParam",
             [](const OTF& self, int cam_id, const Pt3D& pt_world){
                 auto v = self.getOTFParam(cam_id, pt_world);
                 return py::make_tuple(v[0], v[1], v[2], v[3]); // a,b,c,alpha
             }, py::arg("cam_id"), py::arg("pt_world"))
        .def_readwrite("_param", &OTF::_param)
        .def_readwrite("_output_path", &OTF::_output_path);
}
