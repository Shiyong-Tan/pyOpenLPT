#ifndef IPR_H
#define IPR_H

#include <vector>
#include <memory>
#include <fstream>

#include "Matrix.h"
#include "Camera.h"
#include "ObjectInfo.h"   // Object3D + derived types
#include "ObjectFinder.h"
#include "StereoMatch.h"
#include "Shake.h"
//#include "Config.h"       // ObjectConfig (contains IPR parameters)
#include "STBCommons.h"   // NONCOPYABLE_MOVABLE

class ObjectConfig; // avoid circular dependency config.h <-> IPR.h

// -----------------------------------------------------------------------------
// IPR: Iterative Particle Reconstruction
//
// High-level flow (subset level is strictly sequential; DO NOT parallelize):
//   1) Reconstruct once using ALL cameras.
//   2) If cfg.ipr.n_reduced > 0, for r = 1..min(n_reduced, N-2):
//        - Enumerate all camera subsets of size K = N - r (require K >= 2).
//        - For each subset: set Camera::is_active accordingly, run reconstruction,
//          then update the working images to consume already-detected particles.
//   3) If a dedup safety step is desired, call it INSIDE the implementation.
// -----------------------------------------------------------------------------
class IPR
{
public:
    // Cameras are referenced (non-owning). Their is_active flags will be toggled in runIPR().
    explicit IPR(std::vector<std::shared_ptr<Camera>>& camera_models)
        : _cam_list(camera_models) {}

    ~IPR() = default;
    NONCOPYABLE_MOVABLE(IPR)

    // Main entry:
    // - 'images' is passed BY VALUE intentionally: runIPR modifies this local copy
    //   (residual/mask updates). Callers may pass std::move(images) to avoid copying.
    // - Returns reconstructed objects; the concrete derived type depends on ObjectConfig.
    // cfg is not const, it will be updated (Tracer: update OTF; Bubble: update)
    std::vector<std::unique_ptr<Object3D>>
    runIPR(ObjectConfig& cfg,
           std::vector<Image> images);

    // Save object info to CSV (header + per-object lines).
    // Uses the virtual save function of derived Object3D classes.
    void saveObjInfo(const std::string& filename,
                     const std::vector<std::unique_ptr<Object3D>>& obj3d_list,
                     const ObjectConfig& cfg) const;

private:
    std::vector<std::shared_ptr<Camera>>& _cam_list;
};

#endif // IPR_H

