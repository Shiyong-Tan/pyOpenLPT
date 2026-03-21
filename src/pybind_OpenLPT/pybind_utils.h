#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include "Camera.h"
#include "ObjectInfo.h"   // Object3D/Object2D, Bubble3D/Bubble2D, Tracer3D/Tracer2D
#include "py_camera_handle.h"

// ---------- 1D: list[Object2D] -> vector<unique_ptr<Object2D>> ----------
// ---------- 1D: list[Object2D] -> vector<unique_ptr<Object2D>> (ALLOW nullptr) ----------
inline std::vector<std::unique_ptr<Object2D>>
make_unique_obj2d_list(const std::vector<Object2D*>& objs) {
    std::vector<std::unique_ptr<Object2D>> out;
    out.resize(objs.size());

    for (size_t i = 0; i < objs.size(); ++i) {
        const Object2D* p = objs[i];
        if (!p) {
            out[i] = nullptr;                  // preserve nullptr
            continue;
        }
        if (auto* b = dynamic_cast<const Bubble2D*>(p)) {
            out[i] = std::make_unique<Bubble2D>(b->_pt_center, b->_r_px);
        } else if (auto* t = dynamic_cast<const Tracer2D*>(p)) {
            out[i] = std::make_unique<Tracer2D>(t->_pt_center);
        } else {
            throw std::runtime_error("Unsupported Object2D dynamic type");
        }

        // NOTE: If Object2D has more fields needed by projection/ROI, copy them here as well.
        // e.g., out[i]->_bbox = b->_bbox; out[i]->_intensity = b->_intensity; ...
    }
    return out;
}



// ---------- 3D: list[Object3D] -> vector<unique_ptr<Object3D>> ----------
inline std::vector<std::unique_ptr<Object3D>>
make_unique_obj3d_list(const std::vector<Object3D*>& objs) {
    std::vector<std::unique_ptr<Object3D>> out;
    out.reserve(objs.size());

    for (auto* p : objs) {
        if (!p) throw std::runtime_error("nullptr in objs");

        // 1) Construct 3D object body first
        std::unique_ptr<Object3D> q;
        if (auto* b = dynamic_cast<Bubble3D*>(p)) {
            q = std::make_unique<Bubble3D>(b->_pt_center, b->_r3d);
        } else if (auto* t = dynamic_cast<Tracer3D*>(p)) {
            q = std::make_unique<Tracer3D>(t->_pt_center);
        } else {
            throw std::runtime_error("Unsupported Object3D dynamic type");
        }

        // 2) Deep-copy _obj2d_list by reusing make_unique_obj2d_list (preserve nullptr)
        std::vector<Object2D*> src2d;
        src2d.reserve(p->_obj2d_list.size());
        for (auto const& up : p->_obj2d_list) src2d.push_back(up.get());

        q->_obj2d_list = make_unique_obj2d_list(src2d);

        out.emplace_back(std::move(q));
    }
    return out;
}

// ---------- 2D: per-cam list[Object2D] -> vector<vector<unique_ptr<Object2D>>> ----------
inline std::vector<std::vector<std::unique_ptr<Object2D>>>
make_unique_obj2d_grid(const std::vector<std::vector<Object2D*>>& objs_by_cam) {
    std::vector<std::vector<std::unique_ptr<Object2D>>> out(objs_by_cam.size());
    for (size_t cam = 0; cam < objs_by_cam.size(); ++cam) {
        const auto& row = objs_by_cam[cam];
        auto& dst = out[cam];
        dst.reserve(row.size());
        for (auto* p : row) {
            if (!p) throw std::runtime_error("nullptr in objs_by_cam");
            if (auto* b = dynamic_cast<Bubble2D*>(p)) {
                const Pt2D& uv = b->_pt_center;
                dst.emplace_back(std::make_unique<Bubble2D>(uv, b->_r_px));   
            } else if (auto* t = dynamic_cast<Tracer2D*>(p)) {
                const Pt2D& uv = t->_pt_center;                               
                dst.emplace_back(std::make_unique<Tracer2D>(uv));            
            } else {
                throw std::runtime_error("Unsupported Object2D dynamic type");
            }
        }
    }
    return out;
}
