#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Track.h"
#include "ObjectInfo.h"     // Object3D / Bubble3D / Tracer3D
#include "pybind_utils.h"   // make_unique_obj3d_list

namespace py = pybind11;

void bind_Track(py::module_& m) {
    py::class_<Track>(m, "Track")
        .def(py::init<>())

        // 只读访问 _obj3d_list：返回 Python list[Object3D]，元素是 C++ 内部 const Object3D* 引用
        .def_property_readonly("_obj3d_list",
            [](Track &self) {
                py::list out;
                auto parent = py::cast(&self);  // 作为 reference_internal 的父对象
                for (auto &uptr : self._obj3d_list) {
                    const Object3D *p = uptr.get();
                    if (p) {
                        out.append(py::cast(p,
                            py::return_value_policy::reference_internal,
                            parent));
                    } else {
                        out.append(py::none());
                    }
                }
                return out;
            })

        // 这两个是普通 STL / 基本类型，可以直接 def_readwrite 暴露
        .def_readwrite("_t_list", &Track::_t_list)
        .def_readwrite("_active", &Track::_active)

        // 仅绑定公开方法；不要暴露/触碰私有成员
        .def("save_track", &Track::saveTrack, py::arg("ostream"), py::arg("track_id"))
        .def("load_track",
             static_cast<void (Track::*)(std::ifstream&, const ObjectConfig&,
                                         const std::vector<std::shared_ptr<Camera>>&)>(&Track::loadTrack),
             py::arg("ifstream"), py::arg("cfg"), py::arg("camera_models"))

        // 关键：add_next 适配器（Python 传 Object3D，绑定层重建 unique_ptr 后调库）
        .def("add_next",
             [](Track& self, Object3D& obj3d, int t) {
                 std::vector<Object3D*> tmp{ &obj3d };
                 auto vec = make_unique_obj3d_list(tmp);   // vector<unique_ptr<Object3D>>，size=1
                 self.addNext(std::move(vec[0]), t);       // 假设签名为 addNext(std::unique_ptr<Object3D>, int)
             },
             py::arg("obj3d"), py::arg("t"));
}
