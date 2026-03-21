#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Config.h"
#include "ObjectInfo.h"
#include "Camera.h"
#include "pybind_utils.h"

namespace py = pybind11;

// ========== CreateArgs 安全绑定辅助 ==========
// 说明：
// 1) _obj2d_ready 使用 add_obj2d(std::unique_ptr<Object2D>) 来接管所有权；
// 2) _proto / _cams 仅保存指针（不接管所有权），请保证在 creatObject3D 调用期间它们有效；
// 3) 对 std::optional 字段，提供 set_/clear_ 方法，避免直接暴露模板类型绑定带来的不确定性。

void bind_Config(py::module_& m) {
    // -------- BasicSetting --------
    py::class_<BasicSetting>(m, "BasicSetting")
        .def(py::init<>())
        .def("readConfig", &BasicSetting::readConfig, py::arg("config_file"))
        .def_readwrite("_n_cam", &BasicSetting::_n_cam)
        .def_readwrite("_frame_start", &BasicSetting::_frame_start)
        .def_readwrite("_frame_end", &BasicSetting::_frame_end)
        .def_readwrite("_fps", &BasicSetting::_fps)
        .def_readwrite("_n_thread", &BasicSetting::_n_thread)
        .def_readwrite("_cam_list", &BasicSetting::_cam_list)
        .def_readwrite("_axis_limit", &BasicSetting::_axis_limit)
        .def_readwrite("_voxel_to_mm", &BasicSetting::_voxel_to_mm)
        .def_readwrite("_output_path", &BasicSetting::_output_path)
        .def_readwrite("_image_file_paths", &BasicSetting::_image_file_paths)
        .def_readwrite("_object_types", &BasicSetting::_object_types)
        .def_readwrite("_object_config_paths", &BasicSetting::_object_config_paths)
        .def_readwrite("_load_track", &BasicSetting::_load_track)
        .def_readwrite("_load_track_frame", &BasicSetting::_load_track_frame)
        .def_readwrite("_load_track_path", &BasicSetting::_load_track_path);

    // -------- ObjectKind --------
    py::enum_<ObjectKind>(m, "ObjectKind")
        .value("Tracer", ObjectKind::Tracer)
        .value("Bubble", ObjectKind::Bubble)
        .export_values();

    // -------- IPR / PF / SM / Shake / STB --------
    py::class_<IPRParam>(m, "IPRParam")
        .def(py::init<>())
        .def_readwrite("n_cam_reduced", &IPRParam::n_cam_reduced)
        .def_readwrite("n_loop_ipr", &IPRParam::n_loop_ipr)
        .def_readwrite("n_loop_ipr_reduced", &IPRParam::n_loop_ipr_reduced)
        .def_readwrite("n_obj2d_process_max", &IPRParam::n_obj2d_process_max);

    py::class_<PFParam>(m, "PFParam")
        .def(py::init<>())
        .def_readwrite("limit", &PFParam::limit)
        .def_readwrite("nx", &PFParam::nx)
        .def_readwrite("ny", &PFParam::ny)
        .def_readwrite("nz", &PFParam::nz)
        .def_readwrite("r",  &PFParam::r)
        .def_readwrite("nBin_x", &PFParam::nBin_x)
        .def_readwrite("nBin_y", &PFParam::nBin_y)
        .def_readwrite("nBin_z", &PFParam::nBin_z)
        .def_readwrite("is_smooth", &PFParam::is_smooth)
        .def_readwrite("sigma_x", &PFParam::sigma_x)
        .def_readwrite("sigma_y", &PFParam::sigma_y)
        .def_readwrite("sigma_z", &PFParam::sigma_z);

    py::class_<SMParam>(m, "SMParam")
        .def(py::init<>())
        .def_readwrite("match_cam_count", &SMParam::match_cam_count)
        .def_readwrite("idmap_cell_px", &SMParam::idmap_cell_px)
        .def_readwrite("tol_2d_px", &SMParam::tol_2d_px)
        .def_readwrite("tol_3d_mm", &SMParam::tol_3d_mm);

    py::class_<ShakeParam>(m, "ShakeParam")
        .def(py::init<>())
        .def_readwrite("_shake_width", &ShakeParam::_shake_width)
        .def_readwrite("_n_shake_loop", &ShakeParam::_n_shake_loop)
        .def_readwrite("_thred_ghost", &ShakeParam::_thred_ghost)
        .def_readwrite("_shakewidth_min", &ShakeParam::_shakewidth_min)
        .def_readwrite("_ratio_augimg", &ShakeParam::_ratio_augimg);

    py::class_<STBParam>(m, "STBParam")
        .def(py::init<>())
        .def_readwrite("_radius_search_obj", &STBParam::_radius_search_obj)
        .def_readwrite("_n_initial_frames", &STBParam::_n_initial_frames)
        .def_readwrite("_radius_search_track", &STBParam::_radius_search_track);

    // -------- CreateArgs（提供安全 setter/clear 与接管 Object2D 所有权的方法）--------
    py::class_<CreateArgs>(m, "CreateArgs")
        .def(py::init<>())

        // _proto : 只保存指针，不接管所有权
        .def("set_proto", [](CreateArgs& a, const Object3D& o){ a._proto = &o; })
        .def("clear_proto", [](CreateArgs& a){ a._proto = nullptr; })

        // _pt_center : 提供显式设置/清空，避免直接暴露 optional 绑定不确定性
        .def("set_pt_center", [](CreateArgs& a, double x, double y, double z){
            a._pt_center = Pt3D{x, y, z};
        })
        .def("clear_pt_center", [](CreateArgs& a){ a._pt_center.reset(); })

        // _obj2d_ready : 通过接管 std::unique_ptr<Object2D> 的方式安全添加
        // 用 helper 把 Python 传来的 list[Object2D] 克隆成 unique_ptr 容器
        .def("set_obj2d_list", [](CreateArgs& a, const std::vector<Object2D*>& objs){
            a._obj2d_ready = make_unique_obj2d_list(objs);
        })

        // _r2d_px_tracer / _r3d_bubble : 显式 set/clear
        .def("set_r2d_px_tracer", [](CreateArgs& a, double v){ a._r2d_px_tracer = v; })
        .def("clear_r2d_px_tracer", [](CreateArgs& a){ a._r2d_px_tracer.reset(); })
        .def("set_r3d_bubble", [](CreateArgs& a, double v){ a._r3d_bubble = v; })
        .def("clear_r3d_bubble", [](CreateArgs& a){ a._r3d_bubble.reset(); })

        .def("set_compute_bubble_radius", [](CreateArgs& a, bool on){
            a._compute_bubble_radius = on;
        });

    // -------- ObjectConfig（名称与 C++ 保持一致；新增 creatObject3D）--------
    py::class_<ObjectConfig>(m, "ObjectConfig")
        .def("readCommonConfig", &ObjectConfig::readCommonConfig,
             py::arg("filepath"), py::arg("settings"))
        .def("kind", &ObjectConfig::kind)
        .def("creatObject3D",
             [](const ObjectConfig& cfg, CreateArgs& args) -> std::unique_ptr<Object3D> {
                 // 直接转发，返回 std::unique_ptr<Object3D> 由 pybind11 接管所有权
                 return cfg.creatObject3D(std::move(args));
             },
             py::arg("args"))
        .def_readwrite("_stb_param", &ObjectConfig::_stb_param)
        .def_readwrite("_shake_param", &ObjectConfig::_shake_param)
        .def_readwrite("_pf_param", &ObjectConfig::_pf_param)
        .def_readwrite("_ipr_param", &ObjectConfig::_ipr_param)
        .def_readwrite("_sm_param", &ObjectConfig::_sm_param);

    // -------- TracerConfig --------
    py::class_<TracerConfig, ObjectConfig>(m, "TracerConfig")
        .def(py::init<>())
        .def("readConfig", &TracerConfig::readConfig, py::arg("filepath"), py::arg("settings"))
        .def_property_readonly("kind", &TracerConfig::kind)
        .def_readwrite("_min_obj_int", &TracerConfig::_min_obj_int)
        .def_readwrite("_radius_obj", &TracerConfig::_radius_obj)
        .def_readwrite("_otf", &TracerConfig::_otf);

    // -------- BubbleConfig --------
    py::class_<BubbleConfig, ObjectConfig>(m, "BubbleConfig")
        .def(py::init<>())
        .def("readConfig", &BubbleConfig::readConfig, py::arg("filepath"), py::arg("settings"))
        .def_property_readonly("kind", &BubbleConfig::kind)
        .def_readwrite("_radius_min", &BubbleConfig::_radius_min)
        .def_readwrite("_radius_max", &BubbleConfig::_radius_max)
        .def_readwrite("_sense", &BubbleConfig::_sense)
        .def_readwrite("_bb_ref_img", &BubbleConfig::_bb_ref_img);
}
