// pyShake.cpp
#include <memory>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>


#include "Camera.h"
#include "Config.h"
#include "ObjectInfo.h"
#include "STBCommons.h"
#include "Shake.h"
#include "py_camera_handle.h"
#include "pybind_utils.h"


namespace py = pybind11;
using namespace pybind11::literals;

// ========== Debug 访问器（方案 1 的 friend）==========
#ifdef OPENLPT_EXPOSE_PRIVATE
struct DebugAccess_Shake {
  static const ObjectConfig &obj_cfg(Shake &s) { return s._obj_cfg; }
  static std::vector<Image> &img_res_list(Shake &s) { return s._img_res_list; }
  static std::vector<double> &score_list(Shake &s) { return s._score_list; }
  static ShakeStrategy *strategy(Shake &s) { return s._strategy.get(); }

  static void
  calResidueImage(Shake &s, const std::vector<std::unique_ptr<Object3D>> &objs,
                  const std::vector<Image> &imgs, bool nonneg,
                  const std::vector<ObjFlag> *flags) {
    s.calResidueImage(objs, imgs, nonneg, flags);
  }

  static PixelRange calROIBound(const Shake &s, int id_cam, double xc,
                                double yc, double dx, double dy) {
    return s.calROIBound(id_cam, xc, yc, dx, dy);
  }

  static std::vector<ROIInfo> buildROIInfo(const Shake &s, const Object3D &obj,
                                           const std::vector<Image> &img) {
    return s.buildROIInfo(obj, img);
  }

  static double shakeOneObject(const Shake &s, Object3D &obj,
                               std::vector<ROIInfo> &roi, double delta,
                               const std::vector<bool> &shake_cam) {
    return s.shakeOneObject(obj, roi, delta, shake_cam);
  }

  static double calObjectScore(Shake &s, Object3D &obj,
                               std::vector<ROIInfo> &roi,
                               const std::vector<bool> &shake_cam) {
    return s.calObjectScore(obj, roi, shake_cam);
  }

  static std::vector<bool>
  markRepeatedObj(Shake &s,
                  const std::vector<std::unique_ptr<Object3D>> &objs) {
    return s.markRepeatedObj(objs);
  }
};
#endif

void bind_Shake(py::module_ &m) {
  // -------- ObjFlag（位掩码）--------
  py::enum_<ObjFlag>(m, "ObjFlag", py::arithmetic())
      .value("None", ObjFlag::None)
      .value("Ghost", ObjFlag::Ghost)
      .value("Repeated", ObjFlag::Repeated)
      .export_values();

  // -------- ROIInfo --------
  py::class_<ROIInfo>(m, "ROIInfo")
      .def(py::init<>())
      .def("allocAugImg", &ROIInfo::allocAugImg)
      .def("allocCorrMap", &ROIInfo::allocCorrMap)
      .def("getAugImg", &ROIInfo::getAugImg,
           py::return_value_policy::reference_internal)
      .def("inRange", &ROIInfo::inRange, py::arg("row"), py::arg("col"))
      .def(
          "mapToLocal",
          [](const ROIInfo &r, int row, int col) {
            int i = 0, j = 0;
            bool ok = r.mapToLocal(row, col, i, j);
            return py::make_tuple(ok, i, j);
          },
          py::arg("row"), py::arg("col"))
      // 单像素访问（通常不建议在 Python 逐像素改写；需要的话可以包块操作）
      .def("aug_img", (double &(ROIInfo::*)(int, int)) & ROIInfo::aug_img,
           py::return_value_policy::reference_internal)
      .def("aug_img", (double (ROIInfo::*)(int, int) const) & ROIInfo::aug_img,
           py::return_value_policy::reference_internal)
      .def("corr_map", (double &(ROIInfo::*)(int, int)) & ROIInfo::corr_map,
           py::return_value_policy::reference_internal)
      .def("corr_map",
           (double (ROIInfo::*)(int, int) const) & ROIInfo::corr_map,
           py::return_value_policy::reference_internal)
      .def_readwrite("ROI_range", &ROIInfo::_ROI_range);

  // -------- 策略基类与派生（如果不在 Python 里继承/覆盖，trampoline
  // 不是必须）--------
  py::class_<ShakeStrategy>(m, "ShakeStrategy")
      .def("project2DInt", &ShakeStrategy::project2DInt)
      .def("calROISize", &ShakeStrategy::calROISize)
      .def("calShakeResidue", &ShakeStrategy::calShakeResidue)
      .def("selectShakeCam", &ShakeStrategy::selectShakeCam);

  py::class_<TracerShakeStrategy, ShakeStrategy>(m, "TracerShakeStrategy")
      .def("gaussIntensity", &TracerShakeStrategy::gaussIntensity)
      .def("project2DInt", &TracerShakeStrategy::project2DInt)
      .def("calROISize", &TracerShakeStrategy::calROISize)
      .def("calShakeResidue", &TracerShakeStrategy::calShakeResidue);

  py::class_<BubbleShakeStrategy, ShakeStrategy>(m, "BubbleShakeStrategy")
      .def("project2DInt", &BubbleShakeStrategy::project2DInt)
      .def("calROISize", &BubbleShakeStrategy::calROISize)
      .def("selectShakeCam", &BubbleShakeStrategy::selectShakeCam)
      .def("getImgCorr", &BubbleShakeStrategy::getImgCorr)
      .def("calShakeResidue", &BubbleShakeStrategy::calShakeResidue);

  // -------- Shake 主类 --------
  py::class_<Shake>(m, "Shake", py::dynamic_attr())
      .def(
          "__init__",
          [](py::handle self,
             const std::vector<PyCameraHandle> &cams_in, // 支持直接传 Python list
             py::object cfg_obj) // 接收 Python 的具体 ObjectConfig 派生对象
          {
            auto camera_models_keep = std::make_shared<std::vector<std::shared_ptr<Camera>>>(
                make_cam_list_from_handles(cams_in, "Shake pybind ctor"));

            // 2) 从 Python 对象中取出“底层 C++”的 ObjectConfig&（派生类实例）
            ObjectConfig &cfg_ref = py::cast<ObjectConfig &>(cfg_obj);

            // 3) placement-new：在 self 的存储上就地构造 C++ Shake
            new (self.cast<Shake *>()) Shake(*camera_models_keep, cfg_ref);

            // 4) 保活：
            //    - 把相机副本的 shared_ptr 挂到 Python 实例上（随实例一起析构）
            //    - 把原始的 cfg Python 对象也挂上，确保其底层 C++ 实例不被销毁
            py::setattr(
                self, "_keep_cam_list",
                py::capsule(new std::shared_ptr<std::vector<std::shared_ptr<Camera>>>(
                                std::move(camera_models_keep)),
                            [](void *p) {
                              delete static_cast<std::shared_ptr<std::vector<
                                  std::shared_ptr<Camera>>> *>(p);
                            }));
            py::setattr(self, "_keep_cfg", cfg_obj);
          },
          py::arg("cams"), py::arg("obj_cfg"))
      .def(
          "__init__",
          [](py::handle self,
             const std::vector<std::shared_ptr<Camera>> &camera_models_in,
             py::object cfg_obj)
          {
            auto camera_models_keep = std::make_shared<std::vector<std::shared_ptr<Camera>>>(camera_models_in);
            ObjectConfig &cfg_ref = py::cast<ObjectConfig &>(cfg_obj);
            new (self.cast<Shake *>()) Shake(*camera_models_keep, cfg_ref);

            py::setattr(
                self, "_keep_cam_list",
                py::capsule(new std::shared_ptr<std::vector<std::shared_ptr<Camera>>>(
                                std::move(camera_models_keep)),
                            [](void *p) {
                              delete static_cast<std::shared_ptr<std::vector<
                                  std::shared_ptr<Camera>>> *>(p);
                            }));
            py::setattr(self, "_keep_cfg", cfg_obj);
          },
          py::arg("camera_models"), py::arg("obj_cfg"))

      // ⚠️ runShake：避免直接绑定 vector<unique_ptr<Object3D>>&
      // 改为 Python 传 list[Object3D]（或其派生），C++ 里深拷贝成 unique_ptr
      // 容器再调用
      .def(
          "runShake",
          [](Shake &s, const std::vector<Object3D *> &objs_py,
             const std::vector<Image> &img_orig) {
            auto objs = make_unique_obj3d_list(
                objs_py); // deep copy, update of objs won't refect in objs_py
            auto flags = s.runShake(objs, img_orig);
            // return the objs to interface
            py::list out(objs.size());
            size_t i = 0;
            for (auto &up : objs) {
              if (up)
                out[i++] = py::cast(std::move(up));
              else
                out[i++] = py::none();
            }
            return py::make_tuple(flags, out);
          },
          py::arg("objs").noconvert(), py::arg("img_orig").noconvert())

      .def(
          "run_Shake",
          [](Shake &s, const std::vector<Object3D *> &objs_py,
             const std::vector<Image> &img_orig) {
            auto objs = make_unique_obj3d_list(objs_py);
            return s.runShake(objs, img_orig);
          },
          py::arg("objs").noconvert(), py::arg("img_orig").noconvert())

      // ⚠️ calResidualImage（公开）：同理封装，避免绑定 vector<unique_ptr<...>>
      .def(
          "calResidualImage",
          [](Shake &s, const std::vector<Object3D *> &objs_py,
             const std::vector<Image> &img_orig,
             std::optional<std::vector<ObjFlag>> flags) {
            auto objs = make_unique_obj3d_list(objs_py);
            // 原函数签名：const vector<unique_ptr<Object3D>>&, imgs, flags
            const std::vector<ObjFlag> *flags_ptr = flags ? &*flags : nullptr;
            return s.calResidualImage(
                reinterpret_cast<
                    const std::vector<std::unique_ptr<Object3D>> &>(objs),
                img_orig, flags_ptr);
          },
          py::arg("objs"), py::arg("img_orig"), py::arg("flags") = py::none())

#ifdef OPENLPT_EXPOSE_PRIVATE
      // ===== Debug-only：私有成员与私有函数桥接 =====
      .def_property_readonly(
          "obj_cfg",
          [](Shake &s) -> const ObjectConfig & {
            return DebugAccess_Shake::obj_cfg(s);
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "img_res_list",
          [](Shake &s) -> std::vector<Image> & {
            return DebugAccess_Shake::img_res_list(s);
          },
          //[](Shake& s, const std::vector<Image>& v){
          //DebugAccess_Shake::img_res_list(s) = v; },
          py::return_value_policy::reference_internal)
      .def_property(
          "score_list",
          [](Shake &s) -> std::vector<double> & {
            return DebugAccess_Shake::score_list(s);
          },
          [](Shake &s, const std::vector<double> &v) {
            DebugAccess_Shake::score_list(s) = v;
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "strategy",
          [](Shake &s) -> ShakeStrategy * {
            return DebugAccess_Shake::strategy(s);
          },
          py::return_value_policy::reference // 非拥有引用，随 Shake 生命周期
          )

      // 私有 calResidueImage：依然通过 utils 先转 unique_ptr 容器
      .def(
          "calResidueImage",
          [](Shake &s, const std::vector<Object3D *> &objs_py,
             const std::vector<Image> &imgs, bool non_negative = false,
             std::optional<std::vector<ObjFlag>> flags_opt = std::nullopt) {
            auto objs = make_unique_obj3d_list(objs_py);
            const std::vector<ObjFlag> *flags_ptr =
                flags_opt ? &*flags_opt : nullptr;

            DebugAccess_Shake::calResidueImage(
                s,
                reinterpret_cast<
                    const std::vector<std::unique_ptr<Object3D>> &>(objs),
                imgs, non_negative, flags_ptr);
          },
          py::arg("objs"), py::arg("img_orig"),
          py::kw_only(), // <- keyword-only from here on
          py::arg("non_negative") = false, py::arg("flags") = py::none())

      .def(
          "calROIBound",
          [](const Shake &s, int id_cam, double x_center, double y_center,
             double dx, double dy) {
            return DebugAccess_Shake::calROIBound(s, id_cam, x_center, y_center,
                                                  dx, dy);
          },
          py::arg("id_cam"), py::arg("x_center"), py::arg("y_center"),
          py::arg("dx"), py::arg("dy"))

      .def(
          "buildROIInfo",
          [](const Shake &s, const Object3D &obj,
             const std::vector<Image> &img) {
            return DebugAccess_Shake::buildROIInfo(s, obj, img);
          },
          py::arg("obj"), py::arg("img_orig"))

      .def(
          "shakeOneObject",
          [](const Shake &s, Object3D &obj, std::vector<ROIInfo> &roi,
             double delta, const std::vector<bool> &shake_cam) {
            CreateArgs args;
            args._proto = &obj;
            std::unique_ptr<Object3D> obj_shaked =
                DebugAccess_Shake::obj_cfg(const_cast<Shake &>(s))
                    .creatObject3D(std::move(args));
            double residue = DebugAccess_Shake::shakeOneObject(
                s, *obj_shaked, roi, delta, shake_cam);
            return py::make_tuple(residue, std::move(obj_shaked));
          },
          py::arg("obj"), py::arg("roi_info"), py::arg("delta"),
          py::arg("shake_cam"))

      .def(
          "calObjectScore",
          [](Shake &s, Object3D &obj, std::vector<ROIInfo> &roi,
             const std::vector<bool> &shake_cam) {
            return DebugAccess_Shake::calObjectScore(s, obj, roi, shake_cam);
          },
          py::arg("obj"), py::arg("roi_info"), py::arg("shake_cam"))

      // 私有 markRepeatedObj：同理用 utils 转换
      .def(
          "markRepeatedObj",
          [](Shake &s, const std::vector<Object3D *> &objs_py) {
            auto objs = make_unique_obj3d_list(objs_py);
            return DebugAccess_Shake::markRepeatedObj(
                s, reinterpret_cast<
                       const std::vector<std::unique_ptr<Object3D>> &>(objs));
          },
          py::arg("objs"))
#endif
      ;
}
