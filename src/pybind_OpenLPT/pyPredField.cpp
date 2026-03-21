#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PredField.h"
#include "Config.h"
#include "ObjectInfo.h"  // Pt3D
#include "STBCommons.h"

namespace py = pybind11;

// 一个只为绑定而生的“最小 3D 对象”，只用到了 _pt_center；PredField 的邻域/插值逻辑只读取坐标。
struct _PyMinimal3D : public Object3D {
    explicit _PyMinimal3D(const Pt3D& p) { _pt_center = p; }

    // 满足 Object3D 的纯虚接口：都做空实现/默认返回
    void saveObject3D(std::ostream&) const override {}
    void loadObject3D(std::istream&) override {}

protected:
    // projectObject2D() 内会用到这个来生成 2D 对象类型；随便给个轻量实现即可
    std::unique_ptr<Object2D> create2DObject() const override {
        return std::make_unique<Tracer2D>();  // 只需要有个占位的 2D 对象
    }

    void additional2DProjection(
        const std::vector<std::shared_ptr<Camera>>&) override {
        // 对 PredField 用途来说不需要额外投影
    }
};


static std::vector<std::unique_ptr<Object3D>>
_make_uplist_from_pts(const std::vector<Pt3D>& pts)
{
    std::vector<std::unique_ptr<Object3D>> out;
    out.reserve(pts.size());
    for (const auto& p: pts) out.emplace_back(std::make_unique<_PyMinimal3D>(p));
    return out;
}

void bind_PredField(py::module_& m)
{
    py::class_<PredField>(m, "PredField")
        .def(py::init<ObjectConfig&>(), py::arg("obj_cfg"))
        // 直接把 3D 点坐标传进来，内部临时包成 unique_ptr<Object3D> 以调用原生 API
        .def("calPredField_from_points",
             [](PredField& self,
                const std::vector<Pt3D>& prev_pts,
                const std::vector<Pt3D>& curr_pts)
             {
                 auto prev_u = _make_uplist_from_pts(prev_pts);
                 auto curr_u = _make_uplist_from_pts(curr_pts);
                 self.calPredField(prev_u, curr_u);  // 原型见头/源文件
             },
             py::arg("prev_pts"), py::arg("curr_pts"))
        .def("getDisp",
             [](const PredField& self, const Pt3D& p){
                 auto v = self.getDisp(p);
                 return py::make_tuple(v[0], v[1], v[2]);
             }, py::arg("pt_world"))
        .def("saveDispField", &PredField::saveDispField, py::arg("file"));
}
