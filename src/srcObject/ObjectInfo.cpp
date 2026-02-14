#include "ObjectInfo.h"

#include "Camera.h"

bool Object3D::isReconstructable(
    const std::vector<std::shared_ptr<Camera>>& camera_models)
{
    int n_outrange = 0;
    projectObject2D(camera_models);

    double x, y;
    const int n_cam = static_cast<int>(camera_models.size());
    for (int i = 0; i < n_cam; i++)
    {
        if (!camera_models[i]) {
            n_outrange++;
            if (n_cam - n_outrange < 2)
            {
                return false;
            }
            continue;
        }

        x = _obj2d_list[i]->_pt_center[0];
        y = _obj2d_list[i]->_pt_center[1];

        if (x < 0 || x >= camera_models[i]->getNCol() ||
            y < 0 || y >= camera_models[i]->getNRow())
        {
            n_outrange ++;
            if (n_cam - n_outrange < 2)
            {
                return false;
            }
        }
    }

    return true;
}

void Object3D::projectObject2D(
    const std::vector<std::shared_ptr<Camera>>& camera_models)
{
    const int n_cam = static_cast<int>(camera_models.size());
    _obj2d_list.clear();
    _obj2d_list.reserve(n_cam);

    for (int i = 0; i < n_cam; i ++)
    {
        auto obj2d = create2DObject();
        if (!camera_models[i]) {
            obj2d->_pt_center = Pt2D{-1e10, -1e10};
            _obj2d_list.push_back(std::move(obj2d));
            continue;
        }

        auto proj_status = camera_models[i]->project(_pt_center);
        if (proj_status) {
            obj2d->_pt_center = proj_status.value();
        } else {
            obj2d->_pt_center = Pt2D{-1e10, -1e10};
        }
        _obj2d_list.push_back(std::move(obj2d));
    }

    additional2DProjection(camera_models);
}

void Tracer3D::additional2DProjection(
    const std::vector<std::shared_ptr<Camera>>& camera_models)
{
    (void)camera_models;
    for (int i = 0; i < static_cast<int>(_obj2d_list.size()); i ++)
    {
        auto* tr2D = static_cast<Tracer2D*>(_obj2d_list[i].get());
        if (tr2D)
        {
            tr2D->_r_px = _r2d_px;
        }

    }
}

void Bubble3D::additional2DProjection(
    const std::vector<std::shared_ptr<Camera>>& camera_models)
{
    for (int i = 0; i < static_cast<int>(_obj2d_list.size()); i ++)
    {
        auto* bb2D = static_cast<Bubble2D*>(_obj2d_list[i].get());
        if (bb2D)
        {
            if (i < static_cast<int>(camera_models.size()) && camera_models[i]) {
                bb2D->_r_px = Bubble::cal2DRadius(*camera_models[i], _pt_center, _r3d);
            } else {
                bb2D->_r_px = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
}

void Tracer3D::saveObject3D(std::ostream& out) const
{
    // 3D center
    out << _pt_center[0] << "," << _pt_center[1] << "," << _pt_center[2];

    // 2D projections: fixed-width (2 cols per camera)
    const size_t n_cam = _obj2d_list.size();
    for (size_t i = 0; i < n_cam; ++i)
    {
        if (auto* tr2d = dynamic_cast<Tracer2D*>(_obj2d_list[i].get())) {
            out << "," << tr2d->_pt_center[0] << "," << tr2d->_pt_center[1];
        } else {
            // keep column count constant even if dynamic type is unexpected
            out << ",,";
        }
    }

    out << "\n";
}

void Bubble3D::saveObject3D(std::ostream& output) const
{
    // 3D center and radius
    output << _pt_center[0] << "," << _pt_center[1] << "," << _pt_center[2]
           << "," << _r3d;

    // 2D projections: fixed-width (3 cols per camera)
    const size_t n_cam = _obj2d_list.size();
    for (size_t i = 0; i < n_cam; ++i)
    {
        if (auto* bb2d = dynamic_cast<Bubble2D*>(_obj2d_list[i].get())) {
            output << "," << bb2d->_pt_center[0] << "," << bb2d->_pt_center[1]
                   << "," << bb2d->_r_px;
        } else {
            // placeholders to keep the line width constant
            output << ",,,";
        }
    }

    output << "\n";
}

// Tracer3D.cpp
void Tracer3D::loadObject3D(std::istream& in)
{
    std::string sx, sy, sz;
    REQUIRE(read_csv_field(in, sx) && read_csv_field(in, sy) && read_csv_field(in, sz),
            ErrorCode::IOfailure, "Tracer3D::loadObject3D: insufficient 3D columns");

    _pt_center[0] = std::stod(sx);
    _pt_center[1] = std::stod(sy);
    _pt_center[2] = std::stod(sz);
    // 2D will be generated later via projectObject2D().
}

// Bubble3D.cpp
void Bubble3D::loadObject3D(std::istream& in)
{
    std::string sx, sy, sz, sr;
    REQUIRE(read_csv_field(in, sx) && read_csv_field(in, sy) && read_csv_field(in, sz) && read_csv_field(in, sr),
            ErrorCode::IOfailure, "Bubble3D::loadObject3D: insufficient 3D/R3D columns");
    
    _pt_center[0] = std::stod(sx);
    _pt_center[1] = std::stod(sy);
    _pt_center[2] = std::stod(sz);
    _r3d          = std::stod(sr);
    // 2D will be generated later via projectObject2D().
}

namespace Bubble {

namespace {

// Finite-difference step in world units [mm] for local projection Jacobian.
constexpr double kJacStepMm = 0.05;
// Smallest valid local projection gain [px/mm].
constexpr double kSigmaEps = 1e-12;
bool isValidProjection(const Pt2D& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]);
}

bool extractPinholeData(const Camera& camera_model,
                        double& fx,
                        double& fy,
                        Pt3D& cam_center)
{
    switch (camera_model.type()) {
    case CameraType::Pinhole: {
        const auto* pinhole = dynamic_cast<const PinholeCamera*>(&camera_model);
        if (!pinhole) {
            return false;
        }
        const auto& p = pinhole->param();
        fx = std::abs(p.cam_mtx(0, 0));
        fy = std::abs(p.cam_mtx(1, 1));
        cam_center = p.t_vec_inv;
        return true;
    }
    case CameraType::RefractionPinhole: {
        const auto* pinplate = dynamic_cast<const RefractionPinholeCamera*>(&camera_model);
        if (!pinplate) {
            return false;
        }
        const auto& p = pinplate->param();
        fx = std::abs(p.cam_mtx(0, 0));
        fy = std::abs(p.cam_mtx(1, 1));
        cam_center = p.t_vec_inv;
        return true;
    }
    default:
        return false;
    }
}

bool calProjRadiusScale(double& radius_scale, const Camera& cam, const Pt3D& X,
                        double h = kJacStepMm)
{
    // Estimate local image-space scaling for a 3D perturbation around X:
    //   J = d(u,v)/d(X,Y,Z).
    // Radius scale uses isotropic RMS gain in the tangent plane:
    //   radius_scale = sqrt((sigma1^2 + sigma2^2)/2),
    // where sigma1, sigma2 are singular values of J.
    // Input:
    //   radius_scale [out]: estimated isotropic gain [px/mm]
    //   cam       [in] : camera model used by cam.project(...)
    //   X         [in] : 3D center point in world coordinates [mm]
    //   h         [in] : finite-difference step [mm]
    // Return:
    //   true  -> radius_scale is valid and > kSigmaEps
    //   false -> invalid projection/Jacobian; caller should treat as unsupported
    radius_scale = std::numeric_limits<double>::quiet_NaN();
    if (!(h > 0.0) || !std::isfinite(h)) {
        return false;
    }

    // 1) Baseline projection at X.
    auto p0_status = cam.project(X);
    if (!p0_status) {
        return false;
    }
    const Pt2D p0 = p0_status.value();
    if (!isValidProjection(p0)) {
        return false;
    }

    // 2) Finite differences per axis to build a 2x3 Jacobian.
    //    Prefer central differences; if one side is invalid, fall back to one-sided.
    double J[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    for (int ax = 0; ax < 3; ++ax) {
        Pt3D Xp = X;
        Pt3D Xm = X;
        Xp[ax] += h;
        Xm[ax] -= h;

        auto pp_status = cam.project(Xp);
        auto pm_status = cam.project(Xm);
        const Pt2D pp = pp_status ? pp_status.value() : Pt2D();
        const Pt2D pm = pm_status ? pm_status.value() : Pt2D();
        const bool vp = isValidProjection(pp);
        const bool vm = isValidProjection(pm);

        if (vp && vm) {
            const double inv = 1.0 / (2.0 * h);
            J[0][ax] = (pp[0] - pm[0]) * inv;
            J[1][ax] = (pp[1] - pm[1]) * inv;
        } else if (vp) {
            const double inv = 1.0 / h;
            J[0][ax] = (pp[0] - p0[0]) * inv;
            J[1][ax] = (pp[1] - p0[1]) * inv;
        } else if (vm) {
            const double inv = 1.0 / h;
            J[0][ax] = (p0[0] - pm[0]) * inv;
            J[1][ax] = (p0[1] - pm[1]) * inv;
        } else {
            return false;
        }

        if (!std::isfinite(J[0][ax]) || !std::isfinite(J[1][ax])) {
            return false;
        }
    }

    // 3) Build A = J*J^T (2x2). Eigenvalues of A are squared singular values of J.
    const double a00 = J[0][0] * J[0][0] + J[0][1] * J[0][1] +
                       J[0][2] * J[0][2];
    const double a01 = J[0][0] * J[1][0] + J[0][1] * J[1][1] +
                       J[0][2] * J[1][2];
    const double a11 = J[1][0] * J[1][0] + J[1][1] * J[1][1] +
                       J[1][2] * J[1][2];

    const double tr = a00 + a11;
    const double det = a00 * a11 - a01 * a01;

    // 4) Closed-form largest eigenvalue of 2x2 symmetric matrix A.
    double disc = tr * tr - 4.0 * det;
    if (disc < 0.0) {
        if (disc > -1e-12) {
            disc = 0.0;
        } else {
            return false;
        }
    }

    const double lambda_max = 0.5 * (tr + std::sqrt(disc));
    if (!std::isfinite(lambda_max) || lambda_max <= 0.0) {
        return false;
    }

    // 5) Radius scale uses isotropic RMS singular-value gain.
    //    trace(A) = sigma1^2 + sigma2^2 for A = J*J^T.
    const double sigma_rms_sq = 0.5 * tr;
    if (!std::isfinite(sigma_rms_sq) || sigma_rms_sq <= 0.0) {
        return false;
    }

    radius_scale = std::sqrt(sigma_rms_sq);
    return std::isfinite(radius_scale) && (radius_scale > kSigmaEps);
}

} // namespace

// Exact 3D radius from one view (pinhole), pixel-domain.
// R = d * k / sqrt(1 + k^2), where k = r_px / f_px, d = ||X - C||.
// Input:
//   cam  : camera used for inversion
//   X    : 3D bubble center in world coordinates [mm]
//   r_px : measured 2D bubble radius [px]
// Output:
//   estimated 3D radius R [mm], or NaN if invalid/unsupported.
double calRadiusFromOneCam(const Camera& cam,
                           const Pt3D&        X,
                           double             r_px)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (!(r_px > 0.0)) return NaN;

    switch (cam.type()) {
    case CameraType::Pinhole:
    {
        double fx = 0.0, fy = 0.0;
        Pt3D C;
        if (!extractPinholeData(cam, fx, fy, C)) {
            return NaN;
        }
        if (!(fx > 0.0) || !(fy > 0.0)) return NaN;

        // Use geometric mean for generality (handles non-square pixels robustly).
        const double f_px = std::sqrt(fx * fy);

        // Distance d = ||X - C|| (same units as desired R)
        Pt3D dX = X - C;
        const double d  = dX.norm();
        if (!(d > 0.0)) return NaN;

        // Exact inversion from pinhole geometry
        const double k     = r_px / f_px;
        const double denom = std::sqrt(1.0 + k*k);
        if (!(denom > 0.0)) return NaN;

        return d * k / denom;
    }

    case CameraType::Polynomial:
        // TODO: polynomial — use Jacobian at X: R ≈ r_px / radius_scale(J(X))
        return NaN;

    case CameraType::RefractionPinhole:
    {
        // Refraction model: use local projection linearization at X.
        // For small bubbles, R ≈ r_px / radius_scale.
        double radius_scale = 0.0;
        if (!calProjRadiusScale(radius_scale, cam, X)) {
            return NaN;
        }

        const double R = r_px / radius_scale;
        return (std::isfinite(R) && (R > 0.0)) ? R : NaN;
    }

    default:
        return NaN;
    }
}

// Forward projection: predict image radius r_px from 3D radius R.
// Pinhole (exact): r_px = f_px * R / sqrt(d^2 - R^2),
//   where f_px is pixel focal length, d = ||X - C||.
// Input:
//   cam : camera used for projection
//   X   : 3D bubble center in world coordinates [mm]
//   R   : 3D bubble radius [mm]
// Output:
//   predicted image radius r_px [px], or NaN if invalid/unsupported.
double cal2DRadius(const Camera& cam,
                   const Pt3D&        X,
                   double             R)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (!(R > 0.0) || !std::isfinite(R)) return NaN;

    switch (cam.type()) {
    case CameraType::Pinhole:
    {
        double fx = 0.0, fy = 0.0;
        Pt3D C;
        if (!extractPinholeData(cam, fx, fy, C)) {
            return NaN;
        }
        if (!(fx > 0.0) || !(fy > 0.0)) return NaN;

        // Use geometric mean for generality (handles non-square pixels robustly).
        const double f_px = std::sqrt(fx * fy);

        // Distance from camera to bubble center
        const double d = (X - C).norm();
        if (!std::isfinite(d) || !(d > R)) {
            // invalid or degenerate: object at/inside the camera center sphere of radius R
            return NaN;
        }
        // Exact forward formula
        const double denom_sq = d*d - R*R;
        if (!(denom_sq > 0.0)) return NaN; // extra guard
        const double r_px = f_px * R / std::sqrt(denom_sq);

        return std::isfinite(r_px) ? r_px : NaN;
    }

    case CameraType::Polynomial:
        // TODO: polynomial — use local Jacobian at X: r_px ≈ radius_scale(J(X)) * R
        return NaN;

    case CameraType::RefractionPinhole:
    {
        // Refraction model: use local projection linearization at X.
        // For small bubbles, r_px ≈ radius_scale * R.
        double radius_scale = 0.0;
        if (!calProjRadiusScale(radius_scale, cam, X)) {
            return NaN;
        }

        const double r_px = radius_scale * R;
        return std::isfinite(r_px) ? r_px : NaN;
    }

    default:
        return NaN;
    }
}

// Multi-view 3D radius estimate for a final accepted match.
// Inputs:
//   cams          : per-view cameras (aligned with obj2d_by_cam by index)
//   X             : 3D bubble center (world coords)
//   obj2d_by_cam  : per-view Object2D pointers; nullptr if that camera has no 2D
// Behavior:
//   - For each view that has a 2D bubble, read r_px from Bubble2D,
//     call calRadiusFromOneCam(cam_i, X, r_px), and collect valid R_i.
//   - Aggregate R_i with the median (robust).
// Returns NaN if no usable views.
double calRadiusFromCams(
    const std::vector<std::shared_ptr<Camera>>& camera_models,
    const Pt3D& X,
    const std::vector<std::unique_ptr<Object2D>>& obj2d_list)
{
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    if (camera_models.empty() || camera_models.size() != obj2d_list.size())
        return NaN;

    std::vector<double> Rs;
    Rs.reserve(camera_models.size());

    for (size_t i = 0; i < camera_models.size(); ++i) {
        const auto& cam = camera_models[i];
        if (!cam) {
            continue;
        }
        const Object2D* base = obj2d_list[i].get();
        if (!base) continue; // no camera or no 2D for this view

        // Container is homogeneous (Bubble2D) per your guarantee → static_cast is fine.
        const Bubble2D* bb = static_cast<const Bubble2D*>(base);
        const double r_px = bb->_r_px;    
        if (!(r_px > 0.0) || !std::isfinite(r_px)) continue;

        const double Ri = calRadiusFromOneCam(*cam, X, r_px);
        if (std::isfinite(Ri) && (Ri > 0.0)) {
            Rs.push_back(Ri);
        }
    }

    if (Rs.empty()) return NaN;

    // Robust aggregation: median of R_i
    const size_t n   = Rs.size();
    const size_t mid = n / 2;
    std::nth_element(Rs.begin(), Rs.begin() + mid, Rs.end());
    double median = Rs[mid];
    if ((n % 2) == 0) {
        const auto max_lhs = std::max_element(Rs.begin(), Rs.begin() + mid);
        median = 0.5 * (median + *max_lhs);
    }

    return median;
}

// Cross-view consistency gate using per-view radius inversion from Object2D list.
// Policy: if fewer than 2 usable views → return true (permissive).
// Dynamic threshold is derived from tol_2d/tol_3d via RMS combination.
bool checkRadiusConsistency(const Pt3D&                          X,
                            const std::vector<const Camera*>& cams,
                            const std::vector<const Object2D*>&  obj2d_by_cam,
                            double                               tol_2d_px,
                            double                               tol_3d)
{
    if (cams.empty() || cams.size() != obj2d_by_cam.size())
        return true; // permissive on size mismatch

    double Rmin =  std::numeric_limits<double>::infinity();
    double Rmax = -std::numeric_limits<double>::infinity();
    size_t n_ok = 0;

    // Track worst-case relative error bound across views to derive a dynamic spread gate.
    double eps_max = 0.0;
    bool   eps_has_data = false;

    for (size_t i = 0; i < cams.size(); ++i) {
        const Camera* cam = cams[i];
        const Object2D* base = obj2d_by_cam[i];
        if (!cam || !base) continue;

        // Container is homogeneous (Bubble2D) per your guarantee → static_cast is fine.
        const Bubble2D* bb = static_cast<const Bubble2D*>(base);
        const double r_px = bb->_r_px;                    // TODO: replace with accessor if available
        if (!(r_px > 0.0) || !std::isfinite(r_px)) continue;

        // Per-view 3D radius via model-dispatched inversion (PINHOLE implemented).
        const double Ri = calRadiusFromOneCam(*cam, X, r_px);
        if (std::isfinite(Ri) && (Ri > 0.0)) {
            Rmin = std::min(Rmin, Ri);
            Rmax = std::max(Rmax, Ri);
            ++n_ok;

            // --- Build dynamic threshold contributions ---
            if (cam->type() == CameraType::Pinhole) {
                double fx = 0.0, fy = 0.0;
                Pt3D C;
                if (!extractPinholeData(*cam, fx, fy, C)) {
                    continue;
                }
                if (fx > 0.0 && fy > 0.0) {
                    const double f_px = std::sqrt(fx * fy); // or use fx if that's your convention
                    const double d_i  = (X - C).norm();
                    if (std::isfinite(d_i) && d_i > 0.0) {
                        const double k = r_px / f_px;

                        // Relative error components:
                        //   δR/R ≈ (δr/r)/(1+k^2)  (from exact pinhole derivative wrt r)
                        //   δR/R ≈ δd/d            (sensitivity wrt distance)
                        double eps_r = 0.0, eps_d = 0.0;
                        if (tol_2d_px > 0.0 && r_px > 0.0) {
                            eps_r = (tol_2d_px / r_px) / (1.0 + k*k);
                        }
                        if (tol_3d > 0.0) {
                            eps_d = tol_3d / d_i;
                        }

                        // Combine contributions:
                        // Recommended default: RMS (assume independence)
                        const double eps_i = std::hypot(eps_r, eps_d); // sqrt(eps_r^2 + eps_d^2)
                        if (std::isfinite(eps_i)) {
                            eps_max = std::max(eps_max, eps_i);
                            eps_has_data = true;
                        }
                    }
                }
            } else if (cam->type() == CameraType::RefractionPinhole) {
                // Refraction model uses local linearization in calRadiusFromOneCam.
                // Estimate dR/dr numerically with pixel perturbation step tol_2d_px.
                double eps_r = 0.0;
                if (tol_2d_px > 0.0 && r_px > 0.0) {
                    const double dr = tol_2d_px;
                    const double r_plus = r_px + dr;
                    const double r_minus = std::max(1e-9, r_px - dr);

                    const double R_plus = calRadiusFromOneCam(*cam, X, r_plus);
                    const double R_minus = calRadiusFromOneCam(*cam, X, r_minus);

                    if (std::isfinite(R_plus) && std::isfinite(R_minus) && r_plus > r_minus) {
                        const double dRdr = (R_plus - R_minus) / (r_plus - r_minus);
                        if (std::isfinite(dRdr)) {
                            eps_r = std::abs(dRdr) * tol_2d_px / Ri;
                        }
                    } else {
                        // Fallback to local proportional model: R ~ r.
                        eps_r = tol_2d_px / r_px;
                    }
                }

                double eps_d = 0.0;
                if (tol_3d > 0.0) {
                    double fx = 0.0, fy = 0.0;
                    Pt3D C;
                    if (extractPinholeData(*cam, fx, fy, C)) {
                        const double d_i = (X - C).norm();
                        if (std::isfinite(d_i) && d_i > 0.0) {
                            eps_d = tol_3d / d_i;
                        }
                    }
                }

                const double eps_i = std::hypot(eps_r, eps_d);
                if (std::isfinite(eps_i)) {
                    eps_max = std::max(eps_max, eps_i);
                    eps_has_data = true;
                }
            }
        }
    }

    if (n_ok < 2) return true;                 // not enough evidence → permissive
    if (!(Rmin > 0.0) || !(Rmax > 0.0)) return true;

    const double spread = Rmax / Rmin;

    // Dynamic gate from tolerances if we have data; otherwise use a conservative default.
    double max_spread = 1.35;                  // fallback default
    if (eps_has_data && (tol_2d_px > 0.0 || tol_3d > 0.0)) {
        // Keep ε within a sane range for the (1+ε)/(1-ε) mapping
        double eps = std::min(std::max(eps_max, 0.0), 0.49); // clamp to avoid blow-up
        max_spread = (1.0 + eps) / (1.0 - eps);
        // avoid being unrealistically tight for tiny eps
        if (max_spread < 1.05) max_spread = 1.05;
    }

    return (spread <= max_spread);
}

} // namespace Bubble
