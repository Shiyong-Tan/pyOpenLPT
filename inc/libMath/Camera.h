#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "../error.hpp"
#include "Matrix.h"
#include "STBCommons.h"

/**
  * @brief Parameters for a standard pinhole camera model.
  */
struct PinholeParam {
    int n_row = 0; // Image height in pixels.
    int n_col = 0; // Image width in pixels.
    Matrix<double> cam_mtx; // Intrinsic matrix K.
    bool is_distorted = false; // True if distortion is enabled.
    int n_dist_coeff = 0; // Number of distortion coefficients (4/5/8/12).
    std::vector<double> dist_coeff; // OpenCV-compatible distortion coefficients.
    Matrix<double> r_mtx; // Rotation matrix: world -> camera.
    Pt3D t_vec; // Translation vector: world -> camera.
    Matrix<double> r_mtx_inv; // Inverse rotation: camera -> world.
    Pt3D t_vec_inv; // Camera center in world coordinates.
};

enum RefPlane {
    REF_X = 1,
    REF_Y,
    REF_Z
};

/**
  * @brief Parameters for the polynomial camera model.
  */
struct PolyParam {
    int n_row = 0; // Image height in pixels.
    int n_col = 0; // Image width in pixels.
    RefPlane ref_plane = REF_Z; // Two depth planes are defined along this axis.
    std::vector<double> plane = {0, 0}; // Two reference plane coordinates.
    int n_coeff = 0; // Number of rows in polynomial coefficient matrices.
    Matrix<double> u_coeffs; // u projection polynomial coefficients.
    Matrix<double> du_coeffs; // Partial derivatives of u wrt the two solved axes.
    Matrix<double> v_coeffs; // v projection polynomial coefficients.
    Matrix<double> dv_coeffs; // Partial derivatives of v wrt the two solved axes.
};

struct PinPlateParam : public PinholeParam {
    Plane3D plane; // Farthest refractive interface plane.

    // Refraction ordering is always FROM FARTHEST to NEAREST to the camera.
    // - refract_array size = n_plate + 2 (media interfaces)
    // - w_array       size = n_plate     (plate thicknesses)
    std::vector<double> refract_array;
    std::vector<double> w_array;
    int n_plate = 0; // Number of physical plates.
    double proj_tol = 0.0; // Projection solver tolerance.
    int proj_nmax = 0; // Projection solver max iterations.
    double lr = 0.0; // Projection solver update scale.
    double refract_ratio_max = 0.0; // Max refractive ratio n0/ni for radius checks.

    // Refractive interfaces ordered FROM FARTHEST to NEAREST.
    std::vector<Plane3D> plane_array;

    // 2D basis on the farthest reference plane.
    Pt3D u_axis, v_axis;
};

enum class CameraType {
    Pinhole,
    Polynomial,
    RefractionPinhole,
};

class Camera {
public:
    virtual ~Camera() = default;

    /** @brief Return camera model type. */
    virtual CameraType type() const = 0;
    /**
      * @brief Project a world point to image space.
      * @param pt_world World point in physical units.
      * @param is_print_detail Enable debug print for iterative models.
      * @return Distorted image point on success.
      */
    virtual StatusOr<Pt2D> project(const Pt3D& pt_world,
                                                                  bool is_print_detail = false) const = 0;
    /**
      * @brief Compute line-of-sight in world coordinates from a distorted image point.
      * @param pt_img_dist Distorted image point in pixels.
      * @return World-space ray as a line.
      */
    virtual StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const = 0;

    /** @brief Image height in pixels. */
    virtual int getNRow() const = 0;
    /** @brief Image width in pixels. */
    virtual int getNCol() const = 0;

    /**
      * @brief Save camera parameters in OpenLPT text format.
      * @param file_name Output file path.
      */
    virtual Status saveParameters(const std::string& file_name) const = 0;
    /** @brief Deep clone camera model object. */
    virtual std::shared_ptr<Camera> clone() const = 0;

    bool is_active = true;
    double max_intensity = 255.0;

protected:
    /**
      * @brief Project world point to normalized pinhole image coordinates.
      * @param pt_world World point.
      * @param param Pinhole-compatible camera parameters.
      * @return Undistorted normalized image point.
      */
    static Pt2D worldToUndistImg(const Pt3D& pt_world, const PinholeParam& param);
    /**
      * @brief Apply distortion and intrinsics to normalized image point.
      * @param pt_img_undist Undistorted normalized image point.
      * @param param Pinhole-compatible camera parameters.
      * @return Distorted image point in pixel coordinates.
      */
    static Pt2D distort(const Pt2D& pt_img_undist, const PinholeParam& param);
    /**
      * @brief Remove distortion from pixel coordinate to normalized image point.
      * @param pt_img_dist Distorted image point in pixel coordinates.
      * @param param Pinhole-compatible camera parameters.
      * @return Undistorted normalized image point.
      */
    static Pt2D undistort(const Pt2D& pt_img_dist, const PinholeParam& param);
};

class CameraFactory {
public:
    /**
      * @brief Load one camera model from a parameter text file.
      * @param file_name Input file path.
      * @return Concrete camera model on success.
      */
    static StatusOr<std::shared_ptr<Camera>> loadFromFile(const std::string& file_name);
};


class PinholeCamera final : public Camera {
public:
    PinholeCamera() = default;

    /** @brief Return pinhole camera model type. */
    CameraType type() const override { return CameraType::Pinhole; }
    /** @brief Project one world point to distorted image coordinate. */
    StatusOr<Pt2D> project(const Pt3D& pt_world,
                           bool is_print_detail = false) const override;
    /** @brief Back-project one distorted image point to world-space line. */
    StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const override;
    /** @brief Image height in pixels. */
    int getNRow() const override;
    /** @brief Image width in pixels. */
    int getNCol() const override;
    /** @brief Save pinhole parameters to text file. */
    Status saveParameters(const std::string& file_name) const override;
    /** @brief Deep clone pinhole camera object. */
    std::shared_ptr<Camera> clone() const override;

    /**
      * @brief Set image size in pixels.
      * @param n_row Image height.
      * @param n_col Image width.
      * @return OK on success.
      */
    Status setImageSize(int n_row, int n_col);
    /**
      * @brief Set pinhole intrinsics and distortion coefficients.
      * @param fx,fy Focal lengths in pixels.
      * @param cx,cy Principal point in pixels.
      * @param dist_coeff Distortion coefficient array.
      * @return OK on success.
      */
    Status setIntrinsics(double fx, double fy, double cx, double cy,
                                              const std::vector<double>& dist_coeff);
    /**
      * @brief Set extrinsics from Rodrigues rvec and translation tvec.
      * @param rvec Rodrigues rotation vector.
      * @param tvec Translation vector.
      * @return OK on success.
      */
    Status setExtrinsics(const Pt3D& rvec, const Pt3D& tvec);
    /** @brief Finalize derived fields after setters. @return OK on success. */
    Status commitUpdate();

    const PinholeParam& param() const { return _param; }
    PinholeParam& param() { return _param; }

private:
    PinholeParam _param;
};

class RefractionPinholeCamera final : public Camera {
public:
    RefractionPinholeCamera() = default;

    /** @brief Return refraction-pinhole camera model type. */
    CameraType type() const override { return CameraType::RefractionPinhole; }
    /** @brief Project one world point with refractive correction to image coordinate. */
    StatusOr<Pt2D> project(const Pt3D& pt_world,
                           bool is_print_detail = false) const override;
    /** @brief Back-project one distorted image point through refractive stack. */
    StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const override;
    /** @brief Image height in pixels. */
    int getNRow() const override;
    /** @brief Image width in pixels. */
    int getNCol() const override;
    /** @brief Save pinplate parameters to text file. */
    Status saveParameters(const std::string& file_name) const override;
    /** @brief Deep clone refraction-pinhole camera object. */
    std::shared_ptr<Camera> clone() const override;

    /**
      * @brief Set image size in pixels.
      * @param n_row Image height.
      * @param n_col Image width.
      * @return OK on success.
      */
    Status setImageSize(int n_row, int n_col);
    /**
      * @brief Set pinhole intrinsics and distortion coefficients.
      * @param fx,fy Focal lengths in pixels.
      * @param cx,cy Principal point in pixels.
      * @param dist_coeff Distortion coefficient array.
      * @return OK on success.
      */
    Status setIntrinsics(double fx, double fy, double cx, double cy,
                                              const std::vector<double>& dist_coeff);
    /**
      * @brief Set extrinsics from Rodrigues rvec and translation tvec.
      * @param rvec Rodrigues rotation vector.
      * @param tvec Translation vector.
      * @return OK on success.
      */
    Status setExtrinsics(const Pt3D& rvec, const Pt3D& tvec);
    /**
      * @brief Configure refractive stack.
      * @param plane_pt Point on the farthest refractive interface.
      * @param plane_n Interface normal (pointing away from camera).
      * @param refract_array Refractive indices from farthest medium to nearest.
      * @param w_array Plate widths from farthest plate to nearest.
      */
    Status setRefraction(const Pt3D& plane_pt, const Pt3D& plane_n,
                                              const std::vector<double>& refract_array,
                                              const std::vector<double>& w_array);
    /**
      * @brief Configure projection solver options.
      * @param proj_tol Residual tolerance.
      * @param proj_nmax Maximum solver iterations.
      * @param lr Solver update scale.
      * @return OK on success.
      */
    Status setSolverOptions(double proj_tol, int proj_nmax, double lr);
    /** @brief Finalize derived refractive geometry buffers. @return OK on success. */
    Status commitUpdate();

    const PinPlateParam& param() const { return _param; }
    PinPlateParam& param() { return _param; }

private:
    /**
      * @brief Build orthonormal in-plane basis from refractive plane normal.
      * @param u_axis Output first basis axis.
      * @param v_axis Output second basis axis.
      * @param plane Input plane.
      */
    static void buildPlaneOrthonormalBasis(Pt3D& u_axis, Pt3D& v_axis,
                                           const Plane3D& plane);
    /**
      * @brief Build refractive plane stack and basis buffers.
      * @param pin Refractive parameter container to update.
      */
    static void buildRefractionPlaneStackAndBasis(PinPlateParam& pin);
    /**
      * @brief Project 3D point to 2D coordinates under plane basis.
      * @param coords Output 2D coordinates.
      * @param pt Input 3D point.
      * @param plane Basis anchor plane.
      * @param u_axis Basis axis u.
      * @param v_axis Basis axis v.
      */
    static void projectToPlaneBasis(std::vector<double>& coords, const Pt3D& pt,
                                    const Plane3D& plane, const Pt3D& u_axis,
                                    const Pt3D& v_axis);
    /**
      * @brief Reconstruct 3D point from plane-basis coordinates.
      * @param result Output 3D point.
      * @param origin Plane origin.
      * @param u_axis Basis axis u.
      * @param v_axis Basis axis v.
      * @param coords Input 2D basis coordinates.
      */
    static void reconstructFromPlaneBasis(Pt3D& result, const Pt3D& origin,
                                          const Pt3D& u_axis, const Pt3D& v_axis,
                                          const std::vector<double>& coords);
    /**
      * @brief Refract direction across one interface.
      * @param dir_refract Input/output direction vector.
      * @param normal Interface normal.
      * @param refract_ratio Refractive index ratio n_in/n_out.
      * @param is_forward True for farthest->nearest propagation.
      * @return False if total internal reflection occurs.
      */
    static bool refractDirection(Pt3D& dir_refract, const Pt3D& normal,
                                 double refract_ratio, bool is_forward);
    /**
      * @brief Trace one ray through refractive interfaces toward camera side.
      * @param pt_exit Output exit point at nearest interface.
      * @param exit_direction Output direction after exiting stack.
      * @param pt_world Input world point.
      * @param pt_entry Input entry point on farthest interface.
      * @param pin Refractive parameter container.
      * @return False if tracing fails or total internal reflection occurs.
      */
    static bool traceRayToCam(Pt3D& pt_exit, Pt3D& exit_direction,
                              const Pt3D& pt_world, const Pt3D& pt_entry,
                              const PinPlateParam& pin);
    /**
      * @brief Solve refractive projection with LM + line-search.
      * @param pt_world Input world point.
      * @param pin Refractive parameter container.
      * @return (failed, refracted_point, residual, iterations).
      */
    static std::tuple<bool, Pt3D, double, int>
    solveProjectionByRefractionLM(const Pt3D& pt_world, const PinPlateParam& pin);

    PinPlateParam _param;
};

class PolynomialCamera final : public Camera {
public:
    PolynomialCamera() = default;

    /** @brief Return polynomial camera model type. */
    CameraType type() const override { return CameraType::Polynomial; }
    /** @brief Project one world point to image coordinate using polynomial model. */
    StatusOr<Pt2D> project(const Pt3D& pt_world,
                                                  bool is_print_detail = false) const override;
    /** @brief Back-project one image point to world-space line using two reference planes. */
    StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const override;
    /** @brief Image height in pixels. */
    int getNRow() const override;
    /** @brief Image width in pixels. */
    int getNCol() const override;
    /** @brief Save polynomial parameters to text file. */
    Status saveParameters(const std::string& file_name) const override;
    /** @brief Deep clone polynomial camera object. */
    std::shared_ptr<Camera> clone() const override;

    /**
      * @brief Set image size in pixels.
      * @param n_row Image height.
      * @param n_col Image width.
      * @return OK on success.
      */
    Status setImageSize(int n_row, int n_col);
    /**
      * @brief Set two reference planes used for polynomial inversion.
      * @param ref_plane Axis along which two planes are specified.
      * @param p0,p1 Coordinates of the two reference planes.
      * @return OK on success.
      */
    Status setReferencePlane(RefPlane ref_plane, double p0, double p1);
    /**
      * @brief Set polynomial coefficients for u and v projections.
      * @param u_coeffs U polynomial table.
      * @param v_coeffs V polynomial table.
      * @return OK on success.
      */
    Status setCoefficients(const Matrix<double>& u_coeffs,
                                                  const Matrix<double>& v_coeffs);
    /** @brief Finalize derivative coefficient buffers. @return OK on success. */
    Status commitUpdate();

    const PolyParam& param() const { return _param; }
    PolyParam& param() { return _param; }

private:
    /**
      * @brief Build derivative coefficient buffers for polynomial inversion.
      * @param poly Polynomial parameter container.
      */
    static void buildPolyDerivatives(PolyParam& poly);
    /**
      * @brief Solve one world point on selected reference plane from image point.
      * @param pt_img_dist Input image coordinate.
      * @param plane_world Target plane coordinate.
      * @param poly Polynomial parameter container.
      * @return Reconstructed world point on requested plane.
      */
    static Pt3D solveWorldOnRefPlane(const Pt2D& pt_img_dist, double plane_world,
                                     const PolyParam& poly);

    PolyParam _param;
};
