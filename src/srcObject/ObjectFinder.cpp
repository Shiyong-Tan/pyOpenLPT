#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>

#include <omp.h>
#include "ObjectFinder.h"


std::vector<std::unique_ptr<Object2D>>
ObjectFinder2D::findObject2D(Image const& img, ObjectConfig const& obj_cfg)
{
    switch (obj_cfg.kind())
    {
    case ObjectKind::Tracer:
        return findTracer2D(img, static_cast<TracerConfig const&>(obj_cfg));

    case ObjectKind::Bubble:
        return findBubble2D(img, static_cast<BubbleConfig const&>(obj_cfg));

    default:
        // for unsupported object types, return an empty vector
        return {};
    }
}

/**
 * @brief Detect tracer particles in a 2D image and return them as Object2D pointers.
 *
 * This function scans the input image for local intensity maxima above a minimum threshold
 * and refines their positions to sub-pixel accuracy using a 3-point logarithmic parabola fit
 * in both x and y directions. Each detected tracer is returned as a Tracer2D object stored
 * in a unique_ptr<Object2D>.
 *
 * @param img   The input image.
 * @param cfg   TracerConfig containing detection parameters:
 *              - _radius_obj: expected particle radius in pixels
 *              - _min_obj_int: minimum intensity threshold
 * @return      A vector of unique_ptr<Object2D> pointing to detected Tracer2D objects.
 */
std::vector<std::unique_ptr<Object2D>>
ObjectFinder2D::findTracer2D(Image const& img, TracerConfig const& cfg)
{
    const int rows = img.getDimRow();
    const int cols = img.getDimCol();
    const double r_px = cfg._radius_obj;
    const double min_intensity = cfg._min_obj_int;

    // Estimate max possible number of particles based on density and image size
    constexpr double particle_density = 0.125; // estimated particles per (2*r)^2 area
    size_t estimated_count = static_cast<size_t>(
        (rows * cols) * particle_density / ((2.0 * r_px) * (2.0 * r_px))
    );

    std::vector<std::unique_ptr<Object2D>> out;
    out.reserve(estimated_count);

    auto safe_ln = [](double v) {
        const double vv = (v < LOGSMALLNUMBER) ? LOGSMALLNUMBER : v;
        return std::log(vv);
    };

    for (int row = 1; row < rows - 1; ++row)
    {
        for (int col = 1; col < cols - 1; ++col)
        {
            const double centerI = static_cast<double>(img(row, col));
            if (centerI < min_intensity) continue;
            if (!myMATH::isLocalMax(img, row, col)) continue;

            const int x1 = col - 1, x2 = col, x3 = col + 1;
            const int y1 = row - 1, y2 = row, y3 = row + 1;

            // --- X direction fit ---
            const double ln_z1x = safe_ln(static_cast<double>(img(y2, x1)));
            const double ln_z2  = safe_ln(centerI); // center pixel
            const double ln_z3x = safe_ln(static_cast<double>(img(y2, x3)));

            const double num_x =   ln_z1x * ((x2 * x2) - (x3 * x3))
                                 - ln_z2  * ((x1 * x1) - (x3 * x3))
                                 + ln_z3x * ((x1 * x1) - (x2 * x2));
            const double den_x =   ln_z1x * (x3 - x2)
                                 - ln_z3x * (x1 - x2)
                                 + ln_z2  * (x1 - x3);
            if (den_x == 0.0) continue;
            const double xc = -0.5 * (num_x / den_x);
            if (!std::isfinite(xc)) continue;

            // --- Y direction fit ---
            const double ln_z1y = safe_ln(static_cast<double>(img(y1, x2)));
            const double ln_z3y = safe_ln(static_cast<double>(img(y3, x2)));

            const double num_y =   ln_z1y * ((y2 * y2) - (y3 * y3))
                                 - ln_z2  * ((y1 * y1) - (y3 * y3))
                                 + ln_z3y * ((y1 * y1) - (y2 * y2));
            const double den_y =   ln_z1y * (y3 - y2)
                                 - ln_z3y * (y1 - y2)
                                 + ln_z2  * (y1 - y3);
            if (den_y == 0.0) continue;
            const double yc = -0.5 * (num_y / den_y);
            if (!std::isfinite(yc)) continue;

            auto tracer = std::make_unique<Tracer2D>();
            tracer->_r_px = r_px;
            tracer->_pt_center[0] = xc;
            tracer->_pt_center[1] = yc;

            out.emplace_back(std::move(tracer));
        }
    }

    // Free unused reserved space
    out.shrink_to_fit();
    return out;
}


/**
 * @brief Detect bubbles via circular fitting and return as Object2D pointers.
 *
 * This function uses CircleIdentifier to locate bubble centers and radii within
 * the specified radius range. Each detected bubble is wrapped as a Bubble2D and
 * returned through unique_ptr<Object2D> to preserve polymorphism.
 *
 * @param img  Input image.
 * @param cfg  BubbleConfig containing detection parameters:
 *             - _radius_min, _radius_max: allowed radius range in pixels
 *             - _sense: detector sensitivity (higher -> more detections)
 * @return     Vector of unique_ptr<Object2D> pointing to Bubble2D objects.
 */
std::vector<std::unique_ptr<Object2D>>
ObjectFinder2D::findBubble2D(Image const& img, BubbleConfig const& cfg)
{
    const int W = img.getDimCol();
    const int H = img.getDimRow();
    const double metric_thres = 0.3;

    // ---------- Step 0. Sanity checks ----------
    if (cfg._radius_min > cfg._radius_max || W <= 0 || H <= 0) {
        return {};
    }

    const double rmin  = cfg._radius_min;
    const double rmax  = cfg._radius_max;
    const double sense = cfg._sense;

    // Tiling parameters
    bool enable_tiling = true;
    const double tiles_per_thread = 0.8;
    const int halo = int(std::ceil(rmax) + 3);

    // Dedup thresholds (min version)
    const double d_th = std::min(2.0, 0.35 * rmax);
    const double r_th = std::min(2.0, 0.25 * rmax);

    const int num_threads = omp_get_max_threads();

    // ---------- Step 1. Decide tiling or whole image ----------
    const bool use_tiling = enable_tiling && (W * H > 65536) && (num_threads > 1);
    if (!use_tiling) {
        CircleIdentifier circle_id(img);
        std::vector<Pt2D> center;
        std::vector<double> radius;
        std::vector<double> metric;

        metric = circle_id.BubbleCenterAndSizeByCircle(center, radius, rmin, rmax, sense);

        std::vector<std::unique_ptr<Object2D>> out;
        out.reserve(center.size());
        for (size_t i = 0; i < center.size(); ++i) {
            if (metric[i] < metric_thres) continue;

            out.emplace_back(std::make_unique<Bubble2D>(center[i], radius[i]));
        }
        out.shrink_to_fit();
        return out;
    }

    // ---------- Step 2. Plan tiling grid ----------
    // devide the image into pieces for parallelization to increase the speed.
    // 1) Detect context
    const bool in_outer_parallel = (omp_in_parallel() != 0);
    const int  team_size_outer   = in_outer_parallel ? omp_get_num_threads() : 1;
    const int  max_for_new_team  = std::max(1, omp_get_max_threads()); // default for a new parallel

    // 2) Decide T_inner (how many threads we *should* target inside)
    int T_inner = 1;
    bool nested_enabled = (omp_get_nested() != 0);
    #if defined(_OPENMP) && (_OPENMP >= 200805)
        nested_enabled = nested_enabled || (omp_get_max_active_levels() > 1);
    #endif

    if (!in_outer_parallel) {
        T_inner = max_for_new_team;              // top-level call: use all default threads
    } else {
        if (nested_enabled) {
            // Roughly divide the global budget among outer threads
            T_inner = std::max(1, max_for_new_team / std::max(1, team_size_outer));
        } else {
            T_inner = 1;                          // inner parallel would serialize, so plan as single-thread
        }
    }

    // 3) Map T_inner -> N_target
    int N_target = int(std::llround(tiles_per_thread * T_inner));
    const double core_ideal = std::sqrt((double)W * H / N_target);
    const int core = std::clamp(int(std::ceil(core_ideal)), int(2 * rmax + 8), 768);

    const int nx = (W + core - 1) / core;
    const int ny = (H + core - 1) / core;

    // ---------- Step 3. Parallel detection (per-thread buckets, lock-free merge) ----------
    struct Detection { Pt2D c; double r; double metric; };

    // One bucket per thread; we'll concatenate them after the parallel region.
    // Pre-size to the (upper bound) number of threads to avoid reallocation inside the region.
    std::vector<std::vector<Detection>> thread_buckets(std::max(1, omp_get_max_threads()));

    // Parallel region
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        // Thread-local bucket to avoid contention and locks
        auto& bucket = thread_buckets[tid];
        bucket.clear();
        bucket.reserve(1024); // heuristic; adjust per workload to reduce reallocations

        // Iterate tiles in parallel; static scheduling is usually faster than guided(1)
        #pragma omp for collapse(2) schedule(static)
        for (int ty = 0; ty < ny; ++ty) {
            for (int tx = 0; tx < nx; ++tx) {

                // ---- Core region in global coordinates (half-open) ----
                const int cx0 = tx * core;
                const int cy0 = ty * core;
                const int cx1 = std::min(cx0 + core, W);
                const int cy1 = std::min(cy0 + core, H);

                // ---- Input ROI = core expanded by halo, clamped to image ----
                const int ix0 = std::max(0, cx0 - halo);
                const int iy0 = std::max(0, cy0 - halo);
                const int ix1 = std::min(cx1 + halo, W);
                const int iy1 = std::min(cy1 + halo, H);

                // Skip empty ROI (can happen at degenerate edges if core==0 or bad params)
                if (ix1 <= ix0 || iy1 <= iy0) continue;

                // Extract subimage (NOTE: crop is (y0,y1,x0,x1) not (x,y))
                Image subimg = img.crop(iy0, iy1, ix0, ix1);

                // Run original detector on this tile
                CircleIdentifier circle_id(subimg);
                std::vector<Pt2D> center_local;
                std::vector<double> radius_local;
                std::vector<double> metric_local;

                metric_local = circle_id.BubbleCenterAndSizeByCircle(
                    center_local, radius_local, rmin, rmax, sense
                );

                // Accept only results whose centers fall inside the core region (in global coords)
                for (size_t i = 0; i < center_local.size(); ++i) {
                    const double gx = center_local[i][0] + ix0; // local -> global
                    const double gy = center_local[i][1] + iy0;
                    if (gx >= cx0 && gx < cx1 && gy >= cy0 && gy < cy1) {
                        bucket.push_back(Detection{ Pt2D{gx, gy}, radius_local[i], metric_local[i] });
                    }
                }
            }
        }
    } // end parallel region

    // ---- Lock-free concatenation of all thread buckets into one vector ----
    std::vector<Detection> global_detections;
    size_t total = 0;
    for (const auto& b : thread_buckets) total += b.size();
    global_detections.resize(total);

    size_t offset = 0;
    for (auto& b : thread_buckets) {
        std::move(b.begin(), b.end(), global_detections.begin() + offset);
        offset += b.size();
        // optional: free memory held by bucket
        std::vector<Detection>().swap(b);
    }

    // ---------- Step 4. Sort by metric descending ----------
    std::sort(global_detections.begin(), global_detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.metric > b.metric;
        });

    // ---------- Step 5. Deduplicate ----------
    std::vector<Detection> deduped;
    deduped.reserve(global_detections.size());

    for (auto& d : global_detections) {
        bool duplicate = false;
        for (auto& kept : deduped) {
            double dx = d.c[0] - kept.c[0];
            double dy = d.c[1] - kept.c[1];
            double dist = std::sqrt(dx*dx + dy*dy);
            double dr   = std::fabs(d.r - kept.r);
            if (dist <= d_th && dr <= r_th) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) deduped.push_back(d);
    }

    // ---------- Step 6. Assemble output ----------
    std::vector<std::unique_ptr<Object2D>> out;
    out.reserve(deduped.size());
    for (auto& d : deduped) {
        // ---- Metric filter: remove bubbles with low confidence ----
        if (d.metric < metric_thres) continue;

        out.emplace_back(std::make_unique<Bubble2D>(d.c, d.r));
    }
    out.shrink_to_fit();
    return out;
}

std::vector<std::vector<std::unique_ptr<Object2D>>>
ObjectFinder2D::findBubble2DFixedBatch(const std::vector<Image>& images,
                                       const std::vector<char>& active,
                                       const BubbleConfig& cfg)
{
    struct Plan {
        int W = 0;
        int H = 0;
        int core = 0;
        int halo = 0;
        int nx = 0;
        int ny = 0;
    };
    struct Job { std::size_t cam; int tx; int ty; };

    std::vector<std::vector<std::unique_ptr<Object2D>>> output(images.size());
    if (active.size() != images.size() ||
        cfg._radius_min > cfg._radius_max) {
        return output;
    }

    const double rmin = cfg._radius_min;
    const double rmax = cfg._radius_max;
    const double sense = cfg._sense;
    const double metric_thres = 0.3;
    const double d_th = std::min(2.0, 0.35 * rmax);
    const double r_th = std::min(2.0, 0.25 * rmax);

    std::vector<Plan> plans(images.size());
    std::vector<std::vector<std::vector<BubbleTileDetection>>>
        tile_results(images.size());
    std::vector<Job> jobs;

    for (std::size_t cam = 0; cam < images.size(); ++cam) {
        if (!active[cam]) continue;

        Plan& plan = plans[cam];
        plan.W = images[cam].getDimCol();
        plan.H = images[cam].getDimRow();
        if (plan.W <= 0 || plan.H <= 0) continue;

        // This is the legacy nested-path geometry used by IPR. The camera-level
        // OpenMP team makes T_inner equal to one, and the 768 cap produces the
        // existing 2x2 grid for the validated 1280x800 inputs.
        constexpr int n_target = 1;
        const double core_ideal =
            std::sqrt(static_cast<double>(plan.W) * plan.H / n_target);
        plan.core = std::clamp(static_cast<int>(std::ceil(core_ideal)),
                               static_cast<int>(2 * rmax + 8), 768);
        plan.halo = static_cast<int>(std::ceil(rmax) + 3);
        plan.nx = (plan.W + plan.core - 1) / plan.core;
        plan.ny = (plan.H + plan.core - 1) / plan.core;

        tile_results[cam].resize(static_cast<std::size_t>(plan.nx * plan.ny));
        for (int ty = 0; ty < plan.ny; ++ty) {
            for (int tx = 0; tx < plan.nx; ++tx) {
                jobs.push_back(Job{cam, tx, ty});
            }
        }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (std::ptrdiff_t job_id = 0;
         job_id < static_cast<std::ptrdiff_t>(jobs.size()); ++job_id) {
        const Job& job = jobs[static_cast<std::size_t>(job_id)];
        const Plan& plan = plans[job.cam];
        const Image& img = images[job.cam];

        const int cx0 = job.tx * plan.core;
        const int cy0 = job.ty * plan.core;
        const int cx1 = std::min(cx0 + plan.core, plan.W);
        const int cy1 = std::min(cy0 + plan.core, plan.H);
        const int ix0 = std::max(0, cx0 - plan.halo);
        const int iy0 = std::max(0, cy0 - plan.halo);
        const int ix1 = std::min(cx1 + plan.halo, plan.W);
        const int iy1 = std::min(cy1 + plan.halo, plan.H);

        if (ix1 <= ix0 || iy1 <= iy0) continue;

        const std::size_t tile_id =
            static_cast<std::size_t>(job.ty * plan.nx + job.tx);
        auto& detections = tile_results[job.cam][tile_id];

        Image subimg = img.crop(iy0, iy1, ix0, ix1);
        CircleIdentifier circle_id(subimg);
        std::vector<Pt2D> centers;
        std::vector<double> radii;
        std::vector<double> metrics = circle_id.BubbleCenterAndSizeByCircle(
            centers, radii, rmin, rmax, sense);

        detections.reserve(centers.size());
        for (std::size_t i = 0; i < centers.size(); ++i) {
            const double gx = centers[i][0] + ix0;
            const double gy = centers[i][1] + iy0;
            if (gx >= cx0 && gx < cx1 && gy >= cy0 && gy < cy1) {
                detections.push_back(
                    BubbleTileDetection{Pt2D{gx, gy}, radii[i], metrics[i]});
            }
        }
    }

    for (std::size_t cam = 0; cam < images.size(); ++cam) {
        if (!active[cam]) continue;

        std::vector<BubbleTileDetection> global_detections;
        std::size_t total = 0;
        for (const auto& tile : tile_results[cam]) total += tile.size();
        global_detections.reserve(total);
        for (auto& tile : tile_results[cam]) {
            global_detections.insert(global_detections.end(),
                                     std::make_move_iterator(tile.begin()),
                                     std::make_move_iterator(tile.end()));
        }

        std::sort(global_detections.begin(), global_detections.end(),
            [](const BubbleTileDetection& a, const BubbleTileDetection& b) {
                return a.metric > b.metric;
            });

        std::vector<BubbleTileDetection> deduped;
        deduped.reserve(global_detections.size());
        for (auto& detection : global_detections) {
            bool duplicate = false;
            for (const auto& kept : deduped) {
                const double dx = detection.center[0] - kept.center[0];
                const double dy = detection.center[1] - kept.center[1];
                const double dist = std::sqrt(dx * dx + dy * dy);
                const double dr = std::fabs(detection.radius - kept.radius);
                if (dist <= d_th && dr <= r_th) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) deduped.push_back(detection);
        }

        auto& camera_output = output[cam];
        camera_output.reserve(deduped.size());
        for (const auto& detection : deduped) {
            if (detection.metric < metric_thres) continue;
            camera_output.emplace_back(
                std::make_unique<Bubble2D>(detection.center, detection.radius));
        }
        camera_output.shrink_to_fit();
    }

    return output;
}

// std::vector<std::unique_ptr<Object2D>>
// ObjectFinder2D::findBubble2D(Image const& img, BubbleConfig const& cfg)
// {
//     // Basic parameter sanity checks (optional but helpful)
//     if (cfg._radius_min > cfg._radius_max) {
//         // Swap or early return; here we choose early return
//         return {};
//     }

//     CircleIdentifier circle_id(img);

//     std::vector<Pt2D> center;
//     std::vector<double> radius;

//     const double sense = cfg._sense;  // use config, do not hardcode

//     circle_id.BubbleCenterAndSizeByCircle(
//         center, radius, cfg._radius_min, cfg._radius_max, sense
//     );

//     std::vector<std::unique_ptr<Object2D>> out;
//     out.reserve(center.size());

//     for (size_t i = 0; i < center.size(); ++i)
//     {
//         // Preserve subclass type by constructing Bubble2D and storing as base pointer
//         out.emplace_back(std::make_unique<Bubble2D>(center[i], radius[i]));
//     }

//     // Optional: release extra capacity (usually small)
//     out.shrink_to_fit();
//     return out;
// }
