#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <stdexcept>

#include "ObjectInfo.h"
#include "Camera.h"
#include "pybind_utils.h"

namespace py = pybind11;

static size_t len_obj2d_list(const Object3D& obj) {
    return obj._obj2d_list.size();
}
static Object2D* get_obj2d_at(Object3D& obj, size_t i) {
    if (i >= obj._obj2d_list.size()) throw py::index_error();
    return obj._obj2d_list[i].get();
}

// Binding function
void bind_ObjectInfo(py::module_& m) {
    // ============================== Object2D base & derived ==============================
    py::class_<Object2D, std::unique_ptr<Object2D>>(m, "Object2D")
        .def_readwrite("_pt_center", &Object2D::_pt_center)
        .def("clone", &Object2D::clone);

    py::class_<Tracer2D, Object2D, std::unique_ptr<Tracer2D>>(m, "Tracer2D")
        .def(py::init<>())
        .def(py::init<const Tracer2D&>())
        .def(py::init<const Pt2D&>(), py::arg("pt_center"))
        .def_readwrite("_r_px", &Tracer2D::_r_px);

    py::class_<Bubble2D, Object2D, std::unique_ptr<Bubble2D>>(m, "Bubble2D")
        .def(py::init<>())
        .def(py::init<const Bubble2D&>())
        .def(py::init<const Pt2D&, double>(), py::arg("pt_center"), py::arg("r_px"))
        .def_readwrite("_r_px", &Bubble2D::_r_px);

    // ============================== Object3D base & derived ==============================
    py::class_<Object3D, std::unique_ptr<Object3D>>(m, "Object3D")
        .def_readwrite("_pt_center",  &Object3D::_pt_center)
        .def_readwrite("_is_tracked", &Object3D::_is_tracked)
        // Python property: _obj2d_list as list[Object2D*] (read-only view)
        .def_property_readonly("_obj2d_list",
            [](Object3D& self) {
                std::vector<Object2D*> out;
                out.reserve(self._obj2d_list.size());
                for (auto& u : self._obj2d_list) out.push_back(u.get());
                return out;
            },
            py::return_value_policy::reference_internal,
            "Readonly list[Object2D*] view; ownership stays with Object3D")
        // Convenience: __len__ and __getitem__ for obj3d itself
        .def("__len__", &len_obj2d_list)
        .def("__getitem__",
            [](Object3D& self, size_t i) -> Object2D* { return get_obj2d_at(self, i); },
            py::return_value_policy::reference_internal)
        // Methods
        .def("projectObject2D",
            static_cast<void (Object3D::*)(const std::vector<std::shared_ptr<Camera>>&)>(
                &Object3D::projectObject2D),
            py::arg("camera_models"))
        .def("isReconstructable",
            static_cast<bool (Object3D::*)(const std::vector<std::shared_ptr<Camera>>&)>(
                &Object3D::isReconstructable),
            py::arg("camera_models"))
        // I/O: overloads with file path
        .def("saveObject3D",
            [](const Object3D& self, const std::string& file) {
                std::ofstream out(file);
                if (!out) throw std::runtime_error("Failed to open file: " + file);
                self.saveObject3D(out);
            },
            py::arg("file"))
        .def("loadObject3D",
            [](Object3D& self, const std::string& file) {
                std::ifstream in(file);
                if (!in) throw std::runtime_error("Failed to open file: " + file);
                self.loadObject3D(in);
            },
            py::arg("file"));

    py::class_<Tracer3D, Object3D, std::unique_ptr<Tracer3D>>(m, "Tracer3D")
        .def(py::init<>())
        .def(py::init<const Pt3D&>(), py::arg("pt_center"))
        .def(py::init<const Pt3D&, const double>(), py::arg("pt_center"), py::arg("r2d_px"))
        .def_readwrite("_r2d_px", &Tracer3D::_r2d_px);

    py::class_<Bubble3D, Object3D, std::unique_ptr<Bubble3D>>(m, "Bubble3D")
        .def(py::init<>())
        .def(py::init<const Pt3D&, double>(), py::arg("pt_center"), py::arg("r3d"))
        .def_readwrite("_r3d", &Bubble3D::_r3d);

    // ============================== Bubble namespace functions ==============================
    auto bb = m.def_submodule("Bubble");

    bb.def("calRadiusFromOneCam",
           [](const std::shared_ptr<Camera>& cam, const Pt3D& X, double r_px) {
               if (!cam) {
                   throw std::runtime_error("Bubble.calRadiusFromOneCam: null camera model");
               }
               return Bubble::calRadiusFromOneCam(*cam, X, r_px);
           },
           py::arg("camera_model"), py::arg("X"), py::arg("r_px"));

    bb.def("cal2DRadius",
           [](const std::shared_ptr<Camera>& cam, const Pt3D& X, double R) {
               if (!cam) {
                   throw std::runtime_error("Bubble.cal2DRadius: null camera model");
               }
               return Bubble::cal2DRadius(*cam, X, R);
           },
           py::arg("camera_model"), py::arg("X"), py::arg("R"));

    // Wrapper: accepts list[Object2D*], deep-copies to vector<unique_ptr<Object2D>>
    bb.def("calRadiusFromCams",
           [](const std::vector<std::shared_ptr<Camera>>& camera_models,
              const Pt3D& X,
              const std::vector<Object2D*>& obj2d_list_py) {
               auto tmp = make_unique_obj2d_list(obj2d_list_py);
               return Bubble::calRadiusFromCams(camera_models, X, tmp);
           },
           py::arg("camera_models"), py::arg("X"), py::arg("obj2d_list"));

    // Wrapper: accepts list[CameraModel] and list[Object2D*], converts to const pointer views
    bb.def("checkRadiusConsistency",
           [](const Pt3D& X,
              const std::vector<std::shared_ptr<Camera>>& camera_models,
              const std::vector<Object2D*>& objs_vec,
              double tol_2d_px,
              double tol_3d) {
               std::vector<const Camera*> cams;
               cams.reserve(camera_models.size());
               for (auto const& c : camera_models) {
                   cams.push_back(c.get());
               }

               std::vector<const Object2D*> views;
               views.reserve(objs_vec.size());
               for (auto* p : objs_vec) views.push_back(p);

               return Bubble::checkRadiusConsistency(X, cams, views, tol_2d_px, tol_3d);
           },
           py::arg("X"), py::arg("camera_models"), py::arg("obj2d_by_cam"),
           py::arg("tol_2d_px"), py::arg("tol_3d"));

}
