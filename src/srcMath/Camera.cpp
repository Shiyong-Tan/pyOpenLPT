#include "Camera.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "myMATH.h"

// Convert Rodrigues rotation vector to rotation matrix.
// Input: rotation vector (axis * angle).
// Output: 3x3 rotation matrix.
Matrix<double> Rodrigues(const Pt3D& r_vec) {
    const double theta = std::sqrt(r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] +
                                                                  r_vec[2] * r_vec[2]);
    Matrix<double> R(3, 3, 0.0);
    if (theta < 1e-8) {
        R(0, 0) = 1.0;
        R(1, 1) = 1.0;
        R(2, 2) = 1.0;
        return R;
    }

    const double x = r_vec[0] / theta;
    const double y = r_vec[1] / theta;
    const double z = r_vec[2] / theta;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double t = 1.0 - c;

    R(0, 0) = t * x * x + c;
    R(0, 1) = t * x * y - s * z;
    R(0, 2) = t * x * z + s * y;
    R(1, 0) = t * x * y + s * z;
    R(1, 1) = t * y * y + c;
    R(1, 2) = t * y * z - s * x;
    R(2, 0) = t * x * z - s * y;
    R(2, 1) = t * y * z + s * x;
    R(2, 2) = t * z * z + c;
    return R;
}

// Convert rotation matrix to Rodrigues vector.
// Input: 3x3 rotation matrix.
// Output: Rodrigues rotation vector.
Pt3D rmtxTorvec(Matrix<double> const& r_mtx) {
    Pt3D r_vec;
    double tr = (myMATH::trace<double>(r_mtx) - 1) / 2;
    tr = tr > 1 ? 1 : tr < -1 ? -1 : tr;
    double theta = std::acos(tr);
    double s = std::sin(theta);

    if (s > SMALLNUMBER) {
        double ratio = theta / (2 * s);
        r_vec[0] = (r_mtx(2, 1) - r_mtx(1, 2)) * ratio;
        r_vec[1] = (r_mtx(0, 2) - r_mtx(2, 0)) * ratio;
        r_vec[2] = (r_mtx(1, 0) - r_mtx(0, 1)) * ratio;
    } else if (tr > 0) {
        r_vec[0] = 0;
        r_vec[1] = 0;
        r_vec[2] = 0;
    } else {
        r_vec[0] = theta * std::sqrt((r_mtx(0, 0) + 1) / 2);
        r_vec[1] =
                theta * std::sqrt((r_mtx(1, 1) + 1) / 2) * (r_mtx(0, 1) > 0 ? 1 : -1);
        r_vec[2] =
                theta * std::sqrt((r_mtx(2, 2) + 1) / 2) * (r_mtx(0, 2) > 0 ? 1 : -1);
    }
    return r_vec;
}

StatusOr<Pt2D> normalizeProjectResult(const Pt2D& p) {
    const double cut = -1e300;
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || p[0] <= cut ||
            p[1] <= cut) {
        return STATUS_OR_ERR(Pt2D, ErrorCode::GeometryFailure,
                                                  "project failed: invalid projection result");
    }
    return p;
}

Status mapExceptionToStatus(const std::exception& e, const char* where) {
    const std::string msg = e.what();
    if (msg.find("total internal reflection") != std::string::npos) {
        return STATUS_ERR_CTX(ErrorCode::TotalInternalReflection, where, msg);
    }
    if (msg.find("parallel") != std::string::npos) {
        return STATUS_ERR_CTX(ErrorCode::ParallelGeometry, where, msg);
    }
    return STATUS_ERR_CTX(ErrorCode::GeometryFailure, where, msg);
}

// Shared pinhole projection helper.
// Input: world point + pinhole parameters.
// Output: normalized undistorted image coordinates.
Pt2D Camera::worldToUndistImg(Pt3D const& pt_world,
                              PinholeParam const& param) {
    Pt3D temp = param.r_mtx * pt_world + param.t_vec;
    Pt2D pt_img_mm;
    pt_img_mm[0] = temp[2] ? (temp[0] / temp[2]) : temp[0];
    pt_img_mm[1] = temp[2] ? (temp[1] / temp[2]) : temp[1];
    return pt_img_mm;
}

// Shared distortion helper.
// Input: normalized undistorted image coordinate.
// Output: distorted image coordinate in pixels.
Pt2D Camera::distort(Pt2D const& pt_img_undist,
                     PinholeParam const& param) {
    double x = pt_img_undist[0];
    double y = pt_img_undist[1];
    double xd, yd;

    if (param.is_distorted) {
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;
        double a1 = 2 * x * y;
        double a2 = r2 + 2 * x * x;
        double a3 = r2 + 2 * y * y;

        double cdist = 1 + param.dist_coeff[0] * r2 + param.dist_coeff[1] * r4;
        if (param.n_dist_coeff > 4) {
            cdist += param.dist_coeff[4] * r6;
        }

        double icdist2 = 1.0;
        if (param.n_dist_coeff > 5) {
            icdist2 = 1.0 / (1 + param.dist_coeff[5] * r2 + param.dist_coeff[6] * r4 +
                                              param.dist_coeff[7] * r6);
        }

        xd = x * cdist * icdist2 + param.dist_coeff[2] * a1 +
                  param.dist_coeff[3] * a2;
        yd = y * cdist * icdist2 + param.dist_coeff[2] * a3 +
                  param.dist_coeff[3] * a1;
        if (param.n_dist_coeff > 8) {
            xd += param.dist_coeff[8] * r2 + param.dist_coeff[9] * r4;
            yd += param.dist_coeff[10] * r2 + param.dist_coeff[11] * r4;
        }
    } else {
        xd = x;
        yd = y;
    }

    return Pt2D(xd * param.cam_mtx(0, 0) + param.cam_mtx(0, 2),
                            yd * param.cam_mtx(1, 1) + param.cam_mtx(1, 2));
}

// Shared undistortion helper.
// Input: distorted image coordinate in pixels.
// Output: normalized undistorted image coordinate.
Pt2D Camera::undistort(Pt2D const& pt_img_dist,
                       PinholeParam const& param) {
    double fx = param.cam_mtx(0, 0);
    double fy = param.cam_mtx(1, 1);
    double cx = param.cam_mtx(0, 2);
    double cy = param.cam_mtx(1, 2);

    double u = pt_img_dist[0];
    double v = pt_img_dist[1];
    double x = (u - cx) / fx;
    double y = (v - cy) / fy;
    Pt2D img_undist(x, y);
    Pt2D img_dist(0, 0);

    if (param.is_distorted) {
        double x0 = x;
        double y0 = y;
        double error = std::numeric_limits<double>::max();
        int iter = 0;

        while (error > UNDISTORT_EPS && iter < UNDISTORT_MAX_ITER) {
            double r2 = x * x + y * y;
            double a1 = 2 * x * y;
            double a2 = r2 + 2 * x * x;
            double a3 = r2 + 2 * y * y;

            double icdist;
            if (param.n_dist_coeff == 4) {
                icdist = 1 / (1 + (param.dist_coeff[1] * r2 + param.dist_coeff[0]) * r2);
            } else if (param.n_dist_coeff == 5) {
                icdist =
                        1 / (1 + ((param.dist_coeff[4] * r2 + param.dist_coeff[1]) * r2 +
                                            param.dist_coeff[0]) *
                                                  r2);
            } else {
                icdist = (1 + ((param.dist_coeff[7] * r2 + param.dist_coeff[6]) * r2 +
                                              param.dist_coeff[5]) *
                                                    r2) /
                                  (1 + ((param.dist_coeff[4] * r2 + param.dist_coeff[1]) * r2 +
                                              param.dist_coeff[0]) *
                                                    r2);
            }

            if (icdist < 0) {
                img_undist[0] = (u - cx) / fx;
                img_undist[1] = (v - cy) / fy;
                break;
            }

            double deltaX = param.dist_coeff[2] * a1 + param.dist_coeff[3] * a2;
            double deltaY = param.dist_coeff[2] * a3 + param.dist_coeff[3] * a1;
            if (param.n_dist_coeff > 8) {
                deltaX += param.dist_coeff[8] * r2 + param.dist_coeff[9] * r2 * r2;
                deltaY += param.dist_coeff[10] * r2 + param.dist_coeff[11] * r2 * r2;
            }
            x = (x0 - deltaX) * icdist;
            y = (y0 - deltaY) * icdist;

            img_undist[0] = x;
            img_undist[1] = y;
            img_dist = distort(img_undist, param);
            error = std::sqrt(std::pow(img_dist[0] - u, 2) +
                                                std::pow(img_dist[1] - v, 2));
            iter++;
        }
    }

    return img_undist;
}

// Read text file and strip inline comments, preserving legacy parsing behavior.
void loadFileToStream(const std::string& file_name, std::stringstream& out) {
    std::ifstream infile(file_name.c_str(), std::ios::in);
    if (!infile) {
        throw std::runtime_error("cannot open file: " + file_name);
    }
    std::string line;
    while (std::getline(infile, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos > 0) {
            if (comment_pos < std::string::npos) {
                line.erase(comment_pos);
            }
        } else if (comment_pos == 0) {
            continue;
        }
        out << line << '\t';
    }
}

// Parse distortion coefficients from one comma-separated field.
void loadDistCoeff(const std::string& dist_coeff_str, PinholeParam& p) {
    p.dist_coeff.clear();
    p.is_distorted = false;
    p.n_dist_coeff = 0;

    std::stringstream dist_coeff_stream(dist_coeff_str);
    double dist_coeff;
    int id = 0;
    while (dist_coeff_stream >> dist_coeff) {
        p.dist_coeff.push_back(dist_coeff);
        id++;
        if (dist_coeff > SMALLNUMBER) {
            p.is_distorted = true;
        }
        if (dist_coeff_stream.peek() == ',') {
            dist_coeff_stream.ignore();
        }
    }

    if (p.is_distorted) {
        p.n_dist_coeff = id;
        if (id != 4 && id != 5 && id != 8 && id != 12) {
            throw std::runtime_error("number of distortion coefficients is wrong");
        }
    }
}

template <typename Param>
void saveDistCoeffLine(std::ofstream& outfile, const Param& p) {
    int size = static_cast<int>(p.dist_coeff.size());
    if (size <= 0) {
        outfile << std::endl;
        return;
    }
    for (int i = 0; i < size - 1; i++) {
        outfile << p.dist_coeff[i] << ",";
    }
    outfile << p.dist_coeff[size - 1] << std::endl;
}

// Project world point using pinhole model.
// Input: world point in physical coordinates.
// Output: distorted image coordinate in pixels.
StatusOr<Pt2D> PinholeCamera::project(const Pt3D& pt_world, bool) const {
    try {
        // Main flow:
        // 1) Transform world point to normalized pinhole image coordinates.
        // 2) Apply lens distortion and camera intrinsics.
        Pt3D pt_cam = _param.r_mtx * pt_world + _param.t_vec;
        Pt2D pt_img_undist;
        pt_img_undist[0] = pt_cam[2] ? (pt_cam[0] / pt_cam[2]) : pt_cam[0];
        pt_img_undist[1] = pt_cam[2] ? (pt_cam[1] / pt_cam[2]) : pt_cam[1];
        Pt2D pt_img_dist = PinholeCamera::distort(pt_img_undist, _param);
        return normalizeProjectResult(pt_img_dist);
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "PinholeCamera::project");
        return STATUS_OR_ERR_CTX(Pt2D, st.err.code, st.err.message, st.err.context);
    }
}

// Build line-of-sight from one image point using pinhole model.
// Input: distorted image coordinate in pixels.
// Output: line in world coordinates.
StatusOr<Line3D> PinholeCamera::lineOfSight(const Pt2D& pt_img_dist) const {
    try {
        // Main flow:
        // 1) Remove image distortion to get normalized image coordinates.
        // 2) Back-project through inverse extrinsics to construct world-space ray.
        Pt2D pt_img_undist = PinholeCamera::undistort(pt_img_dist, _param);
        Pt3D pt_world(pt_img_undist[0], pt_img_undist[1], 1.0);
        pt_world = _param.r_mtx_inv * pt_world + _param.t_vec_inv;

        Line3D line;
        line.pt = _param.t_vec_inv;
        line.unit_vector = myMATH::createUnitVector(_param.t_vec_inv, pt_world);
        return line;
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "PinholeCamera::lineOfSight");
        return STATUS_OR_ERR_CTX(Line3D, st.err.code, st.err.message, st.err.context);
    }
}

int PinholeCamera::getNRow() const { return _param.n_row; }
int PinholeCamera::getNCol() const { return _param.n_col; }

// Save pinhole parameters to text file.
// Input: output file path.
// Output: file with exactly the legacy pinhole format.
Status PinholeCamera::saveParameters(const std::string& file_name) const {
    try {
        std::ofstream outfile(file_name.c_str(), std::ios::out);
        outfile << "# Camera Model: (PINHOLE/POLYNOMIAL)" << std::endl;
        outfile << "PINHOLE" << std::endl;
        outfile << "# Camera Calibration Error: \nNone\n# Pose Calibration Error: \nNone"
                        << std::endl;
        outfile << "# Image Size: (n_row,n_col)" << std::endl;
        outfile << _param.n_row << "," << _param.n_col << std::endl;
        outfile << "# Camera Matrix: " << std::endl;
        auto cam_mtx = _param.cam_mtx;
        cam_mtx.write(outfile);
        outfile << "# Distortion Coefficients: " << std::endl;
        saveDistCoeffLine(outfile, _param);
        outfile << "# Rotation Vector: " << std::endl;
        auto r_vec = rmtxTorvec(_param.r_mtx);
        auto r_vec_t = r_vec.transpose();
        r_vec_t.write(outfile);
        outfile << "# Rotation Matrix: " << std::endl;
        auto r_mtx = _param.r_mtx;
        r_mtx.write(outfile);
        outfile << "# Inverse of Rotation Matrix: " << std::endl;
        auto r_mtx_inv = _param.r_mtx_inv;
        r_mtx_inv.write(outfile);
        outfile << "# Translation Vector: " << std::endl;
        auto t_vec = _param.t_vec;
        auto t_vec_t = t_vec.transpose();
        t_vec_t.write(outfile);
        outfile << "# Inverse of Translation Vector: " << std::endl;
        auto t_vec_inv = _param.t_vec_inv;
        auto t_vec_inv_t = t_vec_inv.transpose();
        t_vec_inv_t.write(outfile);
        return Status::OK();
    } catch (const std::exception& e) {
        return STATUS_ERR_CTX(ErrorCode::IOfailure, "PinholeCamera::saveParameters", e.what());
    }
}

std::shared_ptr<Camera> PinholeCamera::clone() const {
    return std::make_shared<PinholeCamera>(*this);
}

Status PinholeCamera::setImageSize(int n_row, int n_col) {
    REQUIRE(n_row > 0 && n_col > 0, ErrorCode::InvalidArgument,
                    "PinholeCamera::setImageSize invalid image size");
    _param.n_row = n_row;
    _param.n_col = n_col;
    return Status::OK();
}

Status PinholeCamera::setIntrinsics(double fx, double fy, double cx, double cy,
                                                                        const std::vector<double>& dist_coeff) {
    // Main flow:
    // 1) Write intrinsic matrix.
    // 2) Store distortion coefficients.
    // 3) Infer distortion enable flag.
    REQUIRE(fx > 0 && fy > 0, ErrorCode::InvalidArgument,
                    "PinholeCamera::setIntrinsics invalid focal length");
    _param.cam_mtx = Matrix<double>(3, 3, 0.0);
    _param.cam_mtx(0, 0) = fx;
    _param.cam_mtx(1, 1) = fy;
    _param.cam_mtx(0, 2) = cx;
    _param.cam_mtx(1, 2) = cy;
    _param.cam_mtx(2, 2) = 1.0;
    _param.dist_coeff = dist_coeff;
    _param.n_dist_coeff = static_cast<int>(dist_coeff.size());
    _param.is_distorted = false;
    for (double v : dist_coeff) {
        if (std::abs(v) > SMALLNUMBER) {
            _param.is_distorted = true;
            break;
        }
    }
    return Status::OK();
}

Status PinholeCamera::setExtrinsics(const Pt3D& rvec, const Pt3D& tvec) {
    // Main flow:
    // 1) Convert Rodrigues vector to rotation matrix.
    // 2) Cache inverse rotation and camera center.
    _param.r_mtx = Rodrigues(rvec);
    _param.t_vec = tvec;
    _param.r_mtx_inv = myMATH::inverse(_param.r_mtx);
    _param.t_vec_inv = (_param.r_mtx_inv * _param.t_vec) * -1.0;
    return Status::OK();
}

Status PinholeCamera::commitUpdate() {
    return Status::OK();
}

// Build orthonormal in-plane basis from plane normal.
// Input: plane normal and point.
// Output: u_axis and v_axis lying on plane.
void RefractionPinholeCamera::buildPlaneOrthonormalBasis(Pt3D& u_axis,
                                                         Pt3D& v_axis,
                                                         const Plane3D& plane) {
    if (std::abs(plane.norm_vector[2]) < 0.9) {
        u_axis = myMATH::cross(plane.norm_vector, {0, 0, 1});
    } else {
        u_axis = myMATH::cross(plane.norm_vector, {1, 0, 0});
    }
    u_axis /= u_axis.norm();
    v_axis = myMATH::cross(plane.norm_vector, u_axis);
    v_axis /= v_axis.norm();
}

// Build all refractive interface planes and in-plane basis.
// Input: farthest refractive plane, widths and plate count.
// Output: plane_array/u_axis/v_axis ready for tracing.
void RefractionPinholeCamera::buildRefractionPlaneStackAndBasis(
    PinPlateParam& pin) {
    // Main flow:
    // 1) Build refractive interface planes from farthest to nearest.
    // 2) Build a 2D coordinate basis on the farthest reference plane.
    pin.plane_array = std::vector<Plane3D>(pin.n_plate + 1);
    pin.plane_array[0] = pin.plane;

    for (int i = 1; i <= pin.n_plate; i++) {
        Plane3D& prev_plane = pin.plane_array[i - 1];
        Plane3D& curr_plane = pin.plane_array[i];
        curr_plane.pt = prev_plane.pt - prev_plane.norm_vector * pin.w_array[i - 1];
        curr_plane.norm_vector = prev_plane.norm_vector;
    }

    buildPlaneOrthonormalBasis(pin.u_axis, pin.v_axis, pin.plane);
}

// Project a 3D point to 2D coordinates on a plane basis.
void RefractionPinholeCamera::projectToPlaneBasis(std::vector<double>& coords,
                                                  const Pt3D& pt,
                                                  const Plane3D& plane,
                                                  const Pt3D& u_axis,
                                                  const Pt3D& v_axis) {
    Pt3D diff = {pt[0] - plane.pt[0], pt[1] - plane.pt[1], pt[2] - plane.pt[2]};
    coords[0] = myMATH::dot(diff, u_axis);
    coords[1] = myMATH::dot(diff, v_axis);
}

// Reconstruct a 3D point from 2D plane basis coordinates.
void RefractionPinholeCamera::reconstructFromPlaneBasis(
    Pt3D& result, const Pt3D& origin, const Pt3D& u_axis, const Pt3D& v_axis,
    const std::vector<double>& coords) {
    for (int i = 0; i < 3; i++) {
        result[i] = origin[i] + coords[0] * u_axis[i] + coords[1] * v_axis[i];
    }
}

// Refract a direction vector across one interface.
// Input: current direction, interface normal, n_in/n_out ratio.
// Output: updated refracted direction, false if total internal reflection.
bool RefractionPinholeCamera::refractDirection(Pt3D& dir_refract,
                                               const Pt3D& normal,
                                               double refract_ratio,
                                               bool is_forward) {
    double cos_1 = std::clamp(myMATH::dot(dir_refract, normal), -1.0, 1.0);
    double cos_2 = 1.0 - refract_ratio * refract_ratio * (1.0 - cos_1 * cos_1);
    if (cos_2 <= 0) {
        return false;
    }

    if (std::abs(cos_1) < SMALLNUMBER) {
        cos_1 = 0.0;
        double tan_2 = std::sqrt((1.0 - cos_2) / cos_2);
        double factor = -dir_refract.norm() / tan_2 * (is_forward ? 1.0 : -1.0);
        for (int i = 0; i < 3; i++) {
            dir_refract[i] = dir_refract[i] + normal[i] * factor;
        }
    } else {
        double factor = -refract_ratio * cos_1 -
                                        std::sqrt(cos_2) * (is_forward ? 1.0 : -1.0);
        for (int i = 0; i < 3; i++) {
            dir_refract[i] = dir_refract[i] * refract_ratio + normal[i] * factor;
        }
    }

    dir_refract /= dir_refract.norm();
    return true;
}

// Forward trace from world point to camera side through plate stack.
// Input: world point + entry point at farthest interface.
// Output: exit point/direction at nearest interface and final medium.
bool RefractionPinholeCamera::traceRayToCam(Pt3D& pt_exit,
                                            Pt3D& exit_direction,
                                            const Pt3D& pt_world,
                                            const Pt3D& pt_entry,
                                            const PinPlateParam& pin) {
    Line3D line;
    line.pt = pt_world;
    line.unit_vector = myMATH::createUnitVector(pt_world, pt_entry);
    Pt3D pt_cross = pt_entry;

    for (int plane_id = 1; plane_id <= pin.n_plate; plane_id++) {
        const Plane3D& plane_curr = pin.plane_array[plane_id - 1];
        const Plane3D& plane_next = pin.plane_array[plane_id];
        double refract_ratio = pin.refract_array[plane_id - 1] / pin.refract_array[plane_id];

        if (!refractDirection(line.unit_vector, plane_curr.norm_vector, refract_ratio, true)) {
            return false;
        }
        line.pt = pt_cross;
        if (myMATH::crossPoint(pt_cross, line, plane_next)) {
            return false;
        }
    }

    pt_exit = pt_cross;
    exit_direction = line.unit_vector;
    double refract_ratio = pin.refract_array[pin.n_plate] / pin.refract_array[pin.n_plate + 1];
    return refractDirection(exit_direction, pin.plane_array[pin.n_plate].norm_vector,
                                        refract_ratio, true);
}

// Refractive projection solver (LM + line search).
// Input: world point and full pinplate parameters.
// Output: (failure_flag, refracted_point_on_farthest_plane, residual, iterations).
std::tuple<bool, Pt3D, double, int>
RefractionPinholeCamera::solveProjectionByRefractionLM(
    Pt3D const& pt_world, const PinPlateParam& pin) {
    std::tuple<bool, Pt3D, double, int> result = {false, {0, 0, 0}, -1, 0};
    const int proj_nmax = pin.proj_nmax;
    const double proj_tol = pin.proj_tol;
    const double eps = 1e-5;

    // Radius guard is enabled when total internal reflection can occur.
    bool is_check_radius = false;
    double radius_max2 = 0;
    if (pin.refract_ratio_max > 1.0) {
        is_check_radius = true;
        double sin_angle = 1.0 / pin.refract_ratio_max;
        radius_max2 = myMATH::dist2(pt_world, pin.plane) / (1.0 - sin_angle * sin_angle);
    }

    // Build initial ray from object point to camera center.
    Line3D line;
    line.pt = pt_world;
    line.unit_vector = myMATH::createUnitVector(pt_world, pin.t_vec_inv);

    Pt3D pt_init;
    if (myMATH::crossPoint(pt_init, line, pin.plane)) {
        std::get<0>(result) = true;
        return result;
    }

    if (is_check_radius) {
        // Clamp the initial entry point inside valid radius before LM starts.
        double radius2 = myMATH::dist2(pt_init, pt_world);
        if (radius2 >= radius_max2) {
            double scale = std::sqrt(radius_max2 * 0.95 / radius2);
            for (int i = 0; i < 3; i++) {
                pt_init[i] = pt_world[i] + (pt_init[i] - pt_world[i]) * scale;
            }
            Pt3D pt_exit_test, dir_test;
            if (!traceRayToCam(pt_exit_test, dir_test, pt_world, pt_init, pin)) {
                std::get<0>(result) = true;
                return result;
            }
        }
    }

    // Convert the initial 3D entry point to 2D plane-basis coordinates.
    std::vector<double> x0(2);
    projectToPlaneBasis(x0, pt_init, pin.plane, pin.u_axis, pin.v_axis);

    // Estimate scaling between 2D basis updates and world-space displacement.
    const double L_min = 0.1;
    const double mu_min = 1e-4;
    const double mu_scale = 0.1;
    double L = L_min;
    {
        Pt3D pt_exit_scale, exit_dir_scale;
        if (traceRayToCam(pt_exit_scale, exit_dir_scale, pt_world, pt_init, pin)) {
            Pt3D diff = pt_exit_scale - pin.plane.pt;
            double dist = myMATH::dot(diff, pin.plane.norm_vector);
            Pt3D pt_exit_proj = pt_exit_scale - pin.plane.norm_vector * dist;
            double line_dist = myMATH::dist(pt_exit_proj, line);
            L = std::max(line_dist, L_min);
        }
    }

    Pt3D pt_current, pt_exit, exit_direction;
    Pt3D vec_to_pinhole, residual_vec;
    std::vector<double> delta_y(2, 0.0);
    std::vector<double> delta_x(2);
    std::vector<double> x(2);
    std::vector<double> delta_y_perturb(2);
    std::vector<double> x_perturb(2);
    double J[2][2];
    double F[2];
    double JTJ[2][2];

    double lambda = 0.01;
    const double lambda_min = 1e-10;
    const double lambda_max = 1e10;
    const double mu_max = 0.1 * L;

    for (int iter = 0; iter < proj_nmax; iter++) {
        // Reconstruct entry point on farthest interface from current 2D parameters.
        delta_x[0] = L * delta_y[0];
        delta_x[1] = L * delta_y[1];
        x[0] = x0[0] + delta_x[0];
        x[1] = x0[1] + delta_x[1];
        reconstructFromPlaneBasis(pt_current, pin.plane.pt, pin.u_axis, pin.v_axis, x);

        double radius2_current = 0.0;
        if (is_check_radius) {
            radius2_current = myMATH::dist2(pt_current, pt_world);
            if (radius2_current >= radius_max2) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                return result;
            }
        }

        // Forward trace through interfaces ordered from farthest to nearest.
        if (!traceRayToCam(pt_exit, exit_direction, pt_world, pt_current, pin)) {
            std::get<0>(result) = true;
            std::get<3>(result) = iter + 1;
            return result;
        }

        vec_to_pinhole = pin.t_vec_inv - pt_exit;
        double proj_along_ray = myMATH::dot(vec_to_pinhole, exit_direction);
        residual_vec = vec_to_pinhole - exit_direction * proj_along_ray;

        double residual = residual_vec.norm();
        double barrier_current = 0.0;
        if (is_check_radius) {
            double mu = std::max(mu_min, std::min(mu_scale * residual, mu_max));
            double d = radius_max2 - radius2_current;
            barrier_current = -mu * std::log(d);
        }

        // Stop when ray-to-camera-center distance is under tolerance.
        if (residual < proj_tol) {
            std::get<1>(result) = pt_exit;
            std::get<2>(result) = residual;
            std::get<3>(result) = iter + 1;
            return result;
        }

        F[0] = myMATH::dot(residual_vec, pin.u_axis);
        F[1] = myMATH::dot(residual_vec, pin.v_axis);

        // Build numerical Jacobian in the 2D basis parameter space.
        for (int j = 0; j < 2; j++) {
            delta_y_perturb = delta_y;
            delta_y_perturb[j] += eps;
            delta_x[0] = L * delta_y_perturb[0];
            delta_x[1] = L * delta_y_perturb[1];
            x_perturb[0] = x0[0] + delta_x[0];
            x_perturb[1] = x0[1] + delta_x[1];
            reconstructFromPlaneBasis(pt_current, pin.plane.pt, pin.u_axis, pin.v_axis, x_perturb);

            if (is_check_radius) {
                double radius2 = myMATH::dist2(pt_current, pt_world);
                if (radius2 >= radius_max2) {
                    std::get<0>(result) = true;
                    std::get<3>(result) = iter + 1;
                    return result;
                }
            }

            if (!traceRayToCam(pt_exit, exit_direction, pt_world, pt_current, pin)) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                return result;
            }

            vec_to_pinhole = pin.t_vec_inv - pt_exit;
            proj_along_ray = myMATH::dot(vec_to_pinhole, exit_direction);
            residual_vec = vec_to_pinhole - exit_direction * proj_along_ray;
            double F_perturb[2];
            F_perturb[0] = myMATH::dot(residual_vec, pin.u_axis);
            F_perturb[1] = myMATH::dot(residual_vec, pin.v_axis);
            J[0][j] = (F_perturb[0] - F[0]) / (delta_y_perturb[j] - delta_y[j]);
            J[1][j] = (F_perturb[1] - F[1]) / (delta_y_perturb[j] - delta_y[j]);
        }

        JTJ[0][0] = J[0][0] * J[0][0] + J[1][0] * J[1][0] + lambda;
        JTJ[0][1] = J[0][0] * J[0][1] + J[1][0] * J[1][1];
        JTJ[1][0] = JTJ[0][1];
        JTJ[1][1] = J[0][1] * J[0][1] + J[1][1] * J[1][1] + lambda;

        double JTF[2];
        JTF[0] = -(J[0][0] * F[0] + J[1][0] * F[1]);
        JTF[1] = -(J[0][1] * F[0] + J[1][1] * F[1]);

        // Solve damped normal equation and reject ill-conditioned updates.
        double det = JTJ[0][0] * JTJ[1][1] - JTJ[0][1] * JTJ[1][0];
        double trace = JTJ[0][0] + JTJ[1][1];
        double condition_estimate = trace / std::abs(det);
        if (std::abs(det) < 1e-20 || condition_estimate > 1e12) {
            lambda = std::min(lambda * 10.0, lambda_max);
            if (lambda >= lambda_max) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                return result;
            }
            continue;
        }

        double dx[2];
        dx[0] = (JTF[0] * JTJ[1][1] - JTF[1] * JTJ[0][1]) / det;
        dx[1] = (JTF[1] * JTJ[0][0] - JTF[0] * JTJ[1][0]) / det;

        // Line-search the LM step in case of radius or residual barrier violations.
        double alpha = 1.0;
        bool step_accepted = false;
        const int max_line_search = 10;
        for (int ls = 0; ls < max_line_search; ls++) {
            delta_y_perturb[0] = delta_y[0] + alpha * dx[0];
            delta_y_perturb[1] = delta_y[1] + alpha * dx[1];
            delta_x[0] = L * delta_y_perturb[0];
            delta_x[1] = L * delta_y_perturb[1];
            x_perturb[0] = x0[0] + delta_x[0];
            x_perturb[1] = x0[1] + delta_x[1];
            reconstructFromPlaneBasis(pt_current, pin.plane.pt, pin.u_axis, pin.v_axis, x_perturb);

            if (is_check_radius) {
                double radius2 = myMATH::dist2(pt_current, pt_world);
                if (radius2 >= radius_max2) {
                    alpha *= 0.5;
                    continue;
                }
            }

            if (traceRayToCam(pt_exit, exit_direction, pt_world, pt_current, pin)) {
                vec_to_pinhole = pin.t_vec_inv - pt_exit;
                double proj_along_ray_ls = myMATH::dot(vec_to_pinhole, exit_direction);
                residual_vec = vec_to_pinhole - exit_direction * proj_along_ray_ls;
                double residual_new = residual_vec.norm();

                double barrier_new = 0.0;
                if (is_check_radius) {
                    double mu = std::max(mu_min, std::min(mu_scale * residual_new, mu_max));
                    double d = radius_max2 - myMATH::dist2(pt_current, pt_world);
                    barrier_new = -mu * std::log(d);
                }

                if (residual_new + barrier_new < residual + barrier_current) {
                    projectToPlaneBasis(x, pt_current, pin.plane, pin.u_axis, pin.v_axis);
                    delta_y[0] = (x[0] - x0[0]) / L;
                    delta_y[1] = (x[1] - x0[1]) / L;
                    step_accepted = true;
                    lambda = std::max(lambda * 0.1, lambda_min);
                    break;
                }
            }
            alpha *= 0.5;
        }

        // Increase damping when no acceptable step is found.
        if (!step_accepted) {
            lambda = std::min(lambda * 10.0, lambda_max);
            if (lambda >= lambda_max) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                return result;
            }
        }
    }

    std::get<1>(result) = pt_exit;
    std::get<2>(result) = residual_vec.norm();
    std::get<3>(result) = proj_nmax;
    return result;
}


// Project world point using pinplate refraction model.
// Input: world point in physical coordinates.
// Output: distorted image coordinate in pixels after refraction solve.
StatusOr<Pt2D> RefractionPinholeCamera::project(const Pt3D& pt_world,
                                                bool is_print_detail) const {
    try {
        // Step 1: Solve for the effective refracted point on the farthest interface.
        std::tuple<bool, Pt3D, double, int> result = solveProjectionByRefractionLM(pt_world, _param);
        if (is_print_detail) {
            std::cout << "RefractionPinholeCamera::project result (is_parallel,error^2): "
                                << std::get<0>(result) << "," << std::get<2>(result) << std::endl;
        }
        // Step 2: Return sentinel for invalid geometry branch.
        if (std::get<0>(result)) {
          // Refraction solve failed.
          // Return a sentinel image point (very large negative values) so
          // normalizeProjectResult() reports a projection failure.
            double value = std::numeric_limits<double>::lowest();
            return normalizeProjectResult(Pt2D(value, value));
        }
        // Step 3: Apply pinhole projection and distortion from refracted point.
        return normalizeProjectResult(RefractionPinholeCamera::distort(
            RefractionPinholeCamera::worldToUndistImg(std::get<1>(result),
                                                      _param),
            _param));
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "RefractionPinholeCamera::project");
        return STATUS_OR_ERR_CTX(Pt2D, st.err.code, st.err.message, st.err.context);
    }
}

// Build line-of-sight from one image point through refractive plates.
// Input: distorted image coordinate in pixels.
// Output: line in world coordinates after backward refraction.
StatusOr<Line3D> RefractionPinholeCamera::lineOfSight(const Pt2D& pt_img_dist) const {
    try {
        // Step 1: Undistort image point and build initial pinhole ray.
        Pt2D pt_img_undist = RefractionPinholeCamera::undistort(pt_img_dist, _param);
        Pt3D pt_world(pt_img_undist[0], pt_img_undist[1], 1.0);
        pt_world = _param.r_mtx_inv * pt_world + _param.t_vec_inv;

        Line3D line;
        line.pt = _param.t_vec_inv;
        line.unit_vector = myMATH::createUnitVector(_param.t_vec_inv, pt_world);

        // Step 2: Backward-trace through refractive interfaces.
        // Ordering note: backward tracing always visits nearest -> farthest.
        for (int plane_id = _param.n_plate; plane_id >= 0; plane_id--) {
            const Plane3D& plane = _param.plane_array[plane_id];
            Pt3D pt_cross;
            bool is_parallel = myMATH::crossPoint(pt_cross, line, plane);
            if (is_parallel) {
                throw std::runtime_error("line is parallel to the plane");
            }

            double refract_ratio =
                    _param.refract_array[plane_id + 1] / _param.refract_array[plane_id];
            bool success = refractDirection(line.unit_vector, plane.norm_vector, refract_ratio, false);
            if (!success) {
                throw std::runtime_error("total internal reflection");
            }
            line.pt = pt_cross;
        }

        return line;
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "RefractionPinholeCamera::lineOfSight");
        return STATUS_OR_ERR_CTX(Line3D, st.err.code, st.err.message, st.err.context);
    }
}

int RefractionPinholeCamera::getNRow() const { return _param.n_row; }
int RefractionPinholeCamera::getNCol() const { return _param.n_col; }

// Save pinplate parameters to text file.
// Input: output file path.
// Output: file with exactly the legacy pinplate format.
Status RefractionPinholeCamera::saveParameters(const std::string& file_name) const {
    try {
        std::ofstream outfile(file_name.c_str(), std::ios::out);
        outfile << "# Camera Model: (PINHOLE/POLYNOMIAL/PINPLATE)" << std::endl;
        outfile << "PINPLATE" << std::endl;
        outfile << "# Camera Calibration Error: \nNone\n# Pose Calibration Error: \nNone"
                        << std::endl;
        outfile << "# Image Size: (n_row,n_col)" << std::endl;
        outfile << _param.n_row << "," << _param.n_col << std::endl;
        outfile << "# Camera Matrix: " << std::endl;
        auto cam_mtx = _param.cam_mtx;
        cam_mtx.write(outfile);
        outfile << "# Distortion Coefficients: " << std::endl;
        saveDistCoeffLine(outfile, _param);
        outfile << "# Rotation Vector: " << std::endl;
        auto r_vec = rmtxTorvec(_param.r_mtx);
        auto r_vec_t = r_vec.transpose();
        r_vec_t.write(outfile);
        outfile << "# Rotation Matrix: " << std::endl;
        auto r_mtx = _param.r_mtx;
        r_mtx.write(outfile);
        outfile << "# Inverse of Rotation Matrix: " << std::endl;
        auto r_mtx_inv = _param.r_mtx_inv;
        r_mtx_inv.write(outfile);
        outfile << "# Translation Vector: " << std::endl;
        auto t_vec = _param.t_vec;
        auto t_vec_t = t_vec.transpose();
        t_vec_t.write(outfile);
        outfile << "# Inverse of Translation Vector: " << std::endl;
        auto t_vec_inv = _param.t_vec_inv;
        auto t_vec_inv_t = t_vec_inv.transpose();
        t_vec_inv_t.write(outfile);
        outfile << "# Reference Point of Refractive Plate: " << std::endl;
        auto plane_pt = _param.plane.pt;
        auto plane_pt_t = plane_pt.transpose();
        plane_pt_t.write(outfile);
        outfile << "# Normal Vector of Refractive Plate: " << std::endl;
        auto plane_norm = _param.plane.norm_vector;
        auto plane_norm_t = plane_norm.transpose();
        plane_norm_t.write(outfile);
        outfile << "# Refractive Index Array: " << std::endl;
        for (int i = 0; i < _param.n_plate + 1; i++) {
            outfile << _param.refract_array[i] << ",";
        }
        outfile << _param.refract_array[_param.n_plate + 1] << std::endl;
        outfile << "# Width of the Refractive Plate: (same as other physical units)"
                        << std::endl;
        for (int i = 0; i < _param.n_plate - 1; i++) {
            outfile << _param.w_array[i] << ",";
        }
        outfile << _param.w_array[_param.n_plate - 1] << std::endl;
        outfile << "# Projection Tolerance: " << std::endl;
        outfile << _param.proj_tol << std::endl;
        outfile << "# Maximum Number of Iterations for Projection: " << std::endl;
        outfile << _param.proj_nmax << std::endl;
        outfile << "# Optimization Rate: " << std::endl;
        outfile << _param.lr << std::endl;
        return Status::OK();
    } catch (const std::exception& e) {
        return STATUS_ERR_CTX(ErrorCode::IOfailure,
                                                    "RefractionPinholeCamera::saveParameters", e.what());
    }
}

std::shared_ptr<Camera> RefractionPinholeCamera::clone() const {
    return std::make_shared<RefractionPinholeCamera>(*this);
}

Status RefractionPinholeCamera::setImageSize(int n_row, int n_col) {
    REQUIRE(n_row > 0 && n_col > 0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setImageSize invalid image size");
    _param.n_row = n_row;
    _param.n_col = n_col;
    return Status::OK();
}

Status RefractionPinholeCamera::setIntrinsics(double fx, double fy, double cx,
                                                                                            double cy,
                                                                                            const std::vector<double>& dist_coeff) {
    // Main flow:
    // 1) Write intrinsic matrix.
    // 2) Store distortion coefficients.
    // 3) Infer distortion enable flag.
    REQUIRE(fx > 0 && fy > 0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setIntrinsics invalid focal length");
    _param.cam_mtx = Matrix<double>(3, 3, 0.0);
    _param.cam_mtx(0, 0) = fx;
    _param.cam_mtx(1, 1) = fy;
    _param.cam_mtx(0, 2) = cx;
    _param.cam_mtx(1, 2) = cy;
    _param.cam_mtx(2, 2) = 1.0;
    _param.dist_coeff = dist_coeff;
    _param.n_dist_coeff = static_cast<int>(dist_coeff.size());
    _param.is_distorted = false;
    for (double v : dist_coeff) {
        if (std::abs(v) > SMALLNUMBER) {
            _param.is_distorted = true;
            break;
        }
    }
    return Status::OK();
}

Status RefractionPinholeCamera::setExtrinsics(const Pt3D& rvec, const Pt3D& tvec) {
    // Main flow:
    // 1) Convert Rodrigues vector to rotation matrix.
    // 2) Cache inverse rotation and camera center.
    _param.r_mtx = Rodrigues(rvec);
    _param.t_vec = tvec;
    _param.r_mtx_inv = myMATH::inverse(_param.r_mtx);
    _param.t_vec_inv = (_param.r_mtx_inv * _param.t_vec) * -1.0;
    return Status::OK();
}

// Set refraction model parameters.
// Input:
// - plane_pt / plane_n: farthest refractive interface.
// - refract_array: refractive indices from farthest medium to nearest medium.
// - w_array: plate thicknesses from farthest plate to nearest plate.
// Output: updates refraction buffers and derived ratio bound.
Status RefractionPinholeCamera::setRefraction(const Pt3D& plane_pt,
                                                                                            const Pt3D& plane_n,
                                                                                            const std::vector<double>& refract_array,
                                                                                            const std::vector<double>& w_array) {
    // Main flow:
    // 1) Validate array lengths: refract_array size must be n_plate + 2.
    // 2) Store farthest interface plane and normalize its normal vector.
    // 3) Store refractive stack ordered from farthest to nearest.
    // 4) Precompute refract_ratio_max for radius safety checks.
    REQUIRE(!refract_array.empty(), ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setRefraction empty refract array");
    REQUIRE(refract_array.size() == w_array.size() + 2, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setRefraction size mismatch");

    _param.plane.pt = plane_pt;
    Pt3D plane_n_local = plane_n;
    const double nrm = plane_n_local.norm();
    REQUIRE(nrm > SMALLNUMBER, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setRefraction invalid plane normal");
    _param.plane.norm_vector = plane_n_local / nrm;
    _param.refract_array = refract_array;
    _param.w_array = w_array;
    _param.n_plate = static_cast<int>(w_array.size());
    _param.refract_ratio_max = 0.0;
    for (double n_i : _param.refract_array) {
        _param.refract_ratio_max = std::max(_param.refract_ratio_max,
                                                                                _param.refract_array[0] / n_i);
    }
    return Status::OK();
}

Status RefractionPinholeCamera::setSolverOptions(double proj_tol, int proj_nmax,
                                                                                                  double lr) {
    // Main flow: validate and store projection solver parameters.
    REQUIRE(proj_tol > 0.0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setSolverOptions invalid proj_tol");
    REQUIRE(proj_nmax > 0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setSolverOptions invalid proj_nmax");
    REQUIRE(lr > 0.0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setSolverOptions invalid lr");
    _param.proj_tol = proj_tol;
    _param.proj_nmax = proj_nmax;
    _param.lr = lr;
    return Status::OK();
}

Status RefractionPinholeCamera::commitUpdate() {
    buildRefractionPlaneStackAndBasis(_param);
    return Status::OK();
}

// Build polynomial derivative coefficient buffers.
// Input: polynomial projection coefficients.
// Output: du/dv coefficient tables for inverse solver Jacobian.
void PolynomialCamera::buildPolyDerivatives(PolyParam& poly) {
    int derivative_id[2] = {1, 2};
    if (poly.ref_plane == REF_X) {
        derivative_id[0] = 2;
        derivative_id[1] = 3;
    } else if (poly.ref_plane == REF_Y) {
        derivative_id[0] = 1;
        derivative_id[1] = 3;
    } else if (poly.ref_plane == REF_Z) {
        derivative_id[0] = 1;
        derivative_id[1] = 2;
    } else {
        throw std::runtime_error("reference plane is wrong");
    }

    poly.du_coeffs = Matrix<double>(poly.n_coeff * 2, 4, 0);
    poly.dv_coeffs = Matrix<double>(poly.n_coeff * 2, 4, 0);
    for (int i = 0; i < poly.n_coeff; i++) {
        poly.du_coeffs(i, 0) = poly.u_coeffs(i, 0) * poly.u_coeffs(i, derivative_id[0]);
        poly.du_coeffs(i, poly.ref_plane) = poly.u_coeffs(i, poly.ref_plane);
        poly.du_coeffs(i, derivative_id[0]) =
                std::max(poly.u_coeffs(i, derivative_id[0]) - 1, 0.0);
        poly.du_coeffs(i, derivative_id[1]) = poly.u_coeffs(i, derivative_id[1]);

        int j = i + poly.n_coeff;
        poly.du_coeffs(j, 0) = poly.u_coeffs(i, 0) * poly.u_coeffs(i, derivative_id[1]);
        poly.du_coeffs(j, poly.ref_plane) = poly.u_coeffs(i, poly.ref_plane);
        poly.du_coeffs(j, derivative_id[0]) = poly.u_coeffs(i, derivative_id[0]);
        poly.du_coeffs(j, derivative_id[1]) =
                std::max(poly.u_coeffs(i, derivative_id[1]) - 1, 0.0);

        poly.dv_coeffs(i, 0) = poly.v_coeffs(i, 0) * poly.v_coeffs(i, derivative_id[0]);
        poly.dv_coeffs(i, poly.ref_plane) = poly.v_coeffs(i, poly.ref_plane);
        poly.dv_coeffs(i, derivative_id[0]) =
                std::max(poly.v_coeffs(i, derivative_id[0]) - 1, 0.0);
        poly.dv_coeffs(i, derivative_id[1]) = poly.v_coeffs(i, derivative_id[1]);

        poly.dv_coeffs(j, 0) = poly.v_coeffs(i, 0) * poly.v_coeffs(i, derivative_id[1]);
        poly.dv_coeffs(j, poly.ref_plane) = poly.v_coeffs(i, poly.ref_plane);
        poly.dv_coeffs(j, derivative_id[0]) = poly.v_coeffs(i, derivative_id[0]);
        poly.dv_coeffs(j, derivative_id[1]) =
                std::max(poly.v_coeffs(i, derivative_id[1]) - 1, 0.0);
    }
}

// Solve inverse polynomial mapping on one fixed reference plane.
// Input: image coordinate + one plane coordinate + polynomial parameters.
// Output: world point on the requested reference plane.
Pt3D PolynomialCamera::solveWorldOnRefPlane(Pt2D const& pt_img_dist,
                                            double plane_world,
                                            PolyParam const& poly) {
    // Initialize one seed point and clamp the reference-axis coordinate.
    Pt3D pt_world((double)rand() / RAND_MAX, (double)rand() / RAND_MAX,
                                (double)rand() / RAND_MAX);

    switch (poly.ref_plane) {
    case REF_X:
        pt_world[0] = plane_world;
        break;
    case REF_Y:
        pt_world[1] = plane_world;
        break;
    case REF_Z:
        pt_world[2] = plane_world;
        break;
    default:
        throw std::runtime_error("unknown reference plane");
    }

    Matrix<double> jacobian(2, 2, 0);
    Matrix<double> jacobian_inv(2, 2, 0);
    Pt2D pt_img_temp;
    double du, dv, dx, dy;
    double err = std::numeric_limits<double>::max();
    int iter = 0;

    while (err > UNDISTORT_EPS && iter < UNDISTORT_MAX_ITER) {
        // Build local Jacobian from derivative coefficient buffers.
        jacobian *= 0;
        jacobian_inv *= 0;

        for (int i = 0; i < poly.n_coeff; i++) {
            double dudx = poly.du_coeffs(i, 0);
            double dudy = poly.du_coeffs(i + poly.n_coeff, 0);
            double dvdx = poly.dv_coeffs(i, 0);
            double dvdy = poly.dv_coeffs(i + poly.n_coeff, 0);

            for (int j = 1; j < 4; j++) {
                dudx *= std::pow(pt_world[j - 1], poly.du_coeffs(i, j));
                dudy *= std::pow(pt_world[j - 1], poly.du_coeffs(i + poly.n_coeff, j));
                dvdx *= std::pow(pt_world[j - 1], poly.dv_coeffs(i, j));
                dvdy *= std::pow(pt_world[j - 1], poly.dv_coeffs(i + poly.n_coeff, j));
            }

            jacobian(0, 0) += dudx;
            jacobian(0, 1) += dudy;
            jacobian(1, 0) += dvdx;
            jacobian(1, 1) += dvdy;
        }

        // Evaluate forward polynomial projection at current estimate.
        jacobian_inv = myMATH::inverse(jacobian, "det");
        pt_img_temp = Pt2D(0.0, 0.0);
        for (int i = 0; i < poly.u_coeffs.getDimRow(); i++) {
            double u_val = poly.u_coeffs(i, 0);
            double v_val = poly.v_coeffs(i, 0);
            for (int j = 1; j < 4; j++) {
                u_val *= std::pow(pt_world[j - 1], poly.u_coeffs(i, j));
                v_val *= std::pow(pt_world[j - 1], poly.v_coeffs(i, j));
            }
            pt_img_temp[0] += u_val;
            pt_img_temp[1] += v_val;
        }
        du = pt_img_dist[0] - pt_img_temp[0];
        dv = pt_img_dist[1] - pt_img_temp[1];
        dx = jacobian_inv(0, 0) * du + jacobian_inv(0, 1) * dv;
        dy = jacobian_inv(1, 0) * du + jacobian_inv(1, 1) * dv;

        // Update the two solved axes while keeping the fixed reference axis.
        switch (poly.ref_plane) {
        case REF_X:
            pt_world[1] += dx;
            pt_world[2] += dy;
            break;
        case REF_Y:
            pt_world[0] += dx;
            pt_world[2] += dy;
            break;
        case REF_Z:
            pt_world[0] += dx;
            pt_world[1] += dy;
            break;
        default:
            throw std::runtime_error("unknown reference plane");
        }

        err = std::sqrt(du * du + dv * dv);
        iter++;
    }

    return pt_world;
}


// Project world point using polynomial model.
// Input: world point in physical coordinates.
// Output: image coordinate in pixels.
StatusOr<Pt2D> PolynomialCamera::project(const Pt3D& pt_world, bool) const {
    try {
        // Main flow:
        // 1) Evaluate polynomial projection for u and v.
        // 2) Return projected image coordinate.
        Pt2D pt_img(0.0, 0.0);
        for (int i = 0; i < _param.u_coeffs.getDimRow(); i++) {
            double u_val = _param.u_coeffs(i, 0);
            double v_val = _param.v_coeffs(i, 0);
            for (int j = 1; j < 4; j++) {
                u_val *= std::pow(pt_world[j - 1], _param.u_coeffs(i, j));
                v_val *= std::pow(pt_world[j - 1], _param.v_coeffs(i, j));
            }
            pt_img[0] += u_val;
            pt_img[1] += v_val;
        }
        return normalizeProjectResult(pt_img);
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "PolynomialCamera::project");
        return STATUS_OR_ERR_CTX(Pt2D, st.err.code, st.err.message, st.err.context);
    }
}

// Build line-of-sight from one image point using polynomial model.
// Input: image coordinate in pixels.
// Output: line in world coordinates.
StatusOr<Line3D> PolynomialCamera::lineOfSight(const Pt2D& pt_img_dist) const {
    try {
        // Step 1: Solve inverse projection on two reference planes.
        Pt3D pt_world_1 = solveWorldOnRefPlane(pt_img_dist, _param.plane[0], _param);
        Pt3D pt_world_2 = solveWorldOnRefPlane(pt_img_dist, _param.plane[1], _param);
        // Step 2: Build line passing through the two recovered world points.
        Line3D line = {pt_world_1, myMATH::createUnitVector(pt_world_1, pt_world_2)};
        return line;
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "PolynomialCamera::lineOfSight");
        return STATUS_OR_ERR_CTX(Line3D, st.err.code, st.err.message, st.err.context);
    }
}

int PolynomialCamera::getNRow() const { return _param.n_row; }
int PolynomialCamera::getNCol() const { return _param.n_col; }

// Save polynomial parameters to text file.
// Input: output file path.
// Output: file with exactly the legacy polynomial format.
Status PolynomialCamera::saveParameters(const std::string& file_name) const {
    try {
        std::ofstream outfile(file_name.c_str(), std::ios::out);
        outfile << "# Camera Model: (PINHOLE/POLYNOMIAL)" << std::endl;
        outfile << "POLYNOMIAL" << std::endl;
        outfile << "# Camera Calibration Error: \nNone" << std::endl;
        outfile << "# Image Size: (n_row,n_col)" << std::endl;
        outfile << _param.n_row << "," << _param.n_col << std::endl;
        outfile << "# Reference Plane: (REF_X/REF_Y/REF_Z,coordinate,coordinate)"
                        << std::endl;
        if (_param.ref_plane == REF_X) {
            outfile << "REF_X,";
        } else if (_param.ref_plane == REF_Y) {
            outfile << "REF_Y,";
        } else {
            outfile << "REF_Z,";
        }
        outfile << _param.plane[0] << "," << _param.plane[1] << std::endl;
        outfile << "# Number of Coefficients: " << std::endl;
        outfile << _param.n_coeff << std::endl;
        outfile << "# U_Coeff,X_Power,Y_Power,Z_Power" << std::endl;
        auto u_coeffs = _param.u_coeffs;
        u_coeffs.write(outfile);
        outfile << "# V_Coeff,X_Power,Y_Power,Z_Power" << std::endl;
        auto v_coeffs = _param.v_coeffs;
        v_coeffs.write(outfile);
        return Status::OK();
    } catch (const std::exception& e) {
        return STATUS_ERR_CTX(ErrorCode::IOfailure,
                                                    "PolynomialCamera::saveParameters", e.what());
    }
}

std::shared_ptr<Camera> PolynomialCamera::clone() const {
    return std::make_shared<PolynomialCamera>(*this);
}

Status PolynomialCamera::setImageSize(int n_row, int n_col) {
    REQUIRE(n_row > 0 && n_col > 0, ErrorCode::InvalidArgument,
                    "PolynomialCamera::setImageSize invalid image size");
    _param.n_row = n_row;
    _param.n_col = n_col;
    return Status::OK();
}

Status PolynomialCamera::setReferencePlane(RefPlane ref_plane, double p0, double p1) {
    _param.ref_plane = ref_plane;
    _param.plane[0] = p0;
    _param.plane[1] = p1;
    return Status::OK();
}

Status PolynomialCamera::setCoefficients(const Matrix<double>& u_coeffs,
                                                                                  const Matrix<double>& v_coeffs) {
    // Main flow:
    // 1) Validate coefficient matrix dimensions.
    // 2) Store u/v coefficients and coefficient count.
    REQUIRE(u_coeffs.getDimRow() == v_coeffs.getDimRow(), ErrorCode::InvalidArgument,
                    "PolynomialCamera::setCoefficients row mismatch");
    REQUIRE(u_coeffs.getDimCol() == 4 && v_coeffs.getDimCol() == 4,
                    ErrorCode::InvalidArgument,
                    "PolynomialCamera::setCoefficients coeff matrix must have 4 cols");
    _param.n_coeff = u_coeffs.getDimRow();
    _param.u_coeffs = u_coeffs;
    _param.v_coeffs = v_coeffs;
    return Status::OK();
}

Status PolynomialCamera::commitUpdate() {
    buildPolyDerivatives(_param);
    return Status::OK();
}

// Load camera model from legacy text format.
// Input: camera parameter file path.
// Output: typed camera model with parameters populated.
StatusOr<std::shared_ptr<Camera>> CameraFactory::loadFromFile(
        const std::string& file_name) {
    try {
        std::stringstream file_content;
        loadFileToStream(file_name, file_content);

        std::string type_name;
        file_content >> type_name;

        if (type_name == "PINHOLE") {
            auto cam = std::make_shared<PinholeCamera>();
            std::string useless;
            file_content >> useless;
            file_content >> useless;

            std::string img_size_str;
            file_content >> img_size_str;
            std::stringstream img_size_stream(img_size_str);
            std::string temp;
            std::getline(img_size_stream, temp, ',');
            cam->param().n_row = std::stoi(temp);
            std::getline(img_size_stream, temp, ',');
            cam->param().n_col = std::stoi(temp);

            cam->param().cam_mtx = Matrix<double>(3, 3, file_content);

            std::string dist_coeff_str;
            file_content >> dist_coeff_str;
            loadDistCoeff(dist_coeff_str, cam->param());

            file_content >> useless; // rvec, ignored for compatibility
            cam->param().r_mtx = Matrix<double>(3, 3, file_content);
            cam->param().r_mtx_inv = Matrix<double>(3, 3, file_content);
            cam->param().t_vec = Pt3D(file_content);
            cam->param().t_vec_inv = Pt3D(file_content);
            return std::static_pointer_cast<Camera>(cam);
        }

        if (type_name == "POLYNOMIAL") {
            auto cam = std::make_shared<PolynomialCamera>();
            std::string useless;
            file_content >> useless;

            std::string img_size_str;
            file_content >> img_size_str;
            std::stringstream img_size_stream(img_size_str);
            std::string temp;
            std::getline(img_size_stream, temp, ',');
            cam->param().n_row = std::stoi(temp);
            std::getline(img_size_stream, temp, ',');
            cam->param().n_col = std::stoi(temp);

            std::string ref_plane_str;
            file_content >> ref_plane_str;
            std::stringstream plane_stream(ref_plane_str);
            std::getline(plane_stream, temp, ',');
            if (temp == "REF_X") {
                cam->param().ref_plane = REF_X;
            } else if (temp == "REF_Y") {
                cam->param().ref_plane = REF_Y;
            } else if (temp == "REF_Z") {
                cam->param().ref_plane = REF_Z;
            } else {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::UnsupportedType,
                                                          "reference plane is wrong");
            }
            std::getline(plane_stream, temp, ',');
            cam->param().plane[0] = std::stod(temp);
            std::getline(plane_stream, temp, ',');
            cam->param().plane[1] = std::stod(temp);

            file_content >> cam->param().n_coeff;
            cam->param().u_coeffs = Matrix<double>(cam->param().n_coeff, 4, file_content);
            cam->param().v_coeffs = Matrix<double>(cam->param().n_coeff, 4, file_content);
            auto st = cam->commitUpdate();
            if (!st) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, st.err.code,
                                     st.err.message);
            }
            return std::static_pointer_cast<Camera>(cam);
        }

        if (type_name == "PINPLATE") {
            auto cam = std::make_shared<RefractionPinholeCamera>();
            // Read and ignore calibration-error tokens to keep parsing aligned with
            // PINHOLE-style header:
            //   # Camera Calibration Error:
            //   <mean,std>
            //   # Pose Calibration Error:
            //   <mean,std>
            // Legacy pinplate files that used placeholder numeric tokens remain
            // compatible because they still provide two tokens here.
            std::string calib_err_token;
            std::string pose_err_token;
            file_content >> calib_err_token;
            file_content >> pose_err_token;

            std::string img_size_str;
            file_content >> img_size_str;
            std::stringstream img_size_stream(img_size_str);
            std::string temp;
            std::getline(img_size_stream, temp, ',');
            cam->param().n_row = std::stoi(temp);
            std::getline(img_size_stream, temp, ',');
            cam->param().n_col = std::stoi(temp);

            cam->param().cam_mtx = Matrix<double>(3, 3, file_content);
            std::string dist_coeff_str;
            file_content >> dist_coeff_str;
            loadDistCoeff(dist_coeff_str, cam->param());

            std::string rvec_token;
            file_content >> rvec_token; // rvec, ignored for compatibility
            cam->param().r_mtx = Matrix<double>(3, 3, file_content);
            cam->param().r_mtx_inv = Matrix<double>(3, 3, file_content);
            cam->param().t_vec = Pt3D(file_content);
            cam->param().t_vec_inv = Pt3D(file_content);

            cam->param().plane.pt = Pt3D(file_content);
            cam->param().plane.norm_vector = Pt3D(file_content);
            {
                double nrm = cam->param().plane.norm_vector.norm();
                if (nrm < 1e-12) {
                    return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::InvalidArgument,
                                                              "plane normal is near zero");
                }
                cam->param().plane.norm_vector /= nrm;
            }

            std::string refract_str;
            file_content >> refract_str;
            std::stringstream refract_stream(refract_str);
            double refract;
            while (refract_stream >> refract) {
                cam->param().refract_array.push_back(refract);
                if (refract_stream.peek() == ',') {
                    refract_stream.ignore();
                }
            }

            cam->param().refract_ratio_max = 0.0;
            for (int i = 0; i < static_cast<int>(cam->param().refract_array.size()); i++) {
                cam->param().refract_ratio_max =
                        std::max(cam->param().refract_ratio_max,
                                          cam->param().refract_array[0] / cam->param().refract_array[i]);
            }

            std::string w_str;
            file_content >> w_str;
            std::stringstream w_stream(w_str);
            double w;
            while (w_stream >> w) {
                cam->param().w_array.push_back(w);
                if (w_stream.peek() == ',') {
                    w_stream.ignore();
                }
            }

            if (cam->param().refract_array.size() != cam->param().w_array.size() + 2 ||
                    cam->param().refract_array.size() < 3) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::InvalidArgument,
                                                          "number of refractive index and width is not consistent");
            }
            file_content >> cam->param().proj_tol;
            file_content >> cam->param().proj_nmax;
            if (!(file_content >> cam->param().lr)) {
                cam->param().lr = 0.1;
            }

            cam->param().n_plate = static_cast<int>(cam->param().w_array.size());
            auto st = cam->commitUpdate();
            if (!st) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, st.err.code,
                                     st.err.message);
            }

            return std::static_pointer_cast<Camera>(cam);
        }

        return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::UnsupportedType,
                                                  "CameraFactory::loadFromFile unknown camera type");
    } catch (const std::exception& e) {
        return STATUS_OR_ERR_CTX(std::shared_ptr<Camera>, ErrorCode::IOfailure,
                                                          "CameraFactory::loadFromFile failed", e.what());
    }
}
