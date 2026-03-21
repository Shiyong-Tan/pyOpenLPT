#ifndef CAMERA_UTIL_H
#define CAMERA_UTIL_H

#include <memory>
#include <vector>
#include "Camera.h"

namespace CameraUtil
{
    // set all camera active
    inline void setActiveAll(std::vector<std::shared_ptr<Camera>>& cams)
    {
        for (auto& cam : cams)
            cam->is_active = true;
    }

    // set subset of camera active，inactive
    inline void setActiveSubset(std::vector<std::shared_ptr<Camera>>& cams, const std::vector<int>& active_ids)
    {
        for (auto& cam : cams)
            cam->is_active = false;
        for (int id : active_ids)
        {
            if (id >= 0 && id < static_cast<int>(cams.size()))
                cams[id]->is_active = true;
        }
    }
}

#endif // CAMERA_UTIL_H
