#include "Camera.h"
#include <cmath>
#include <vector>
#include <cstdlib>
#include <string>

Camera::Camera() {};

Camera::Camera(const Camera &c)
    : _type(c._type), _pinhole_param(c._pinhole_param),
      _poly_param(c._poly_param), _pinplate_param(c._pinplate_param),
      _is_active(c._is_active), _max_intensity(c._max_intensity) {}

Camera::Camera(std::istream &is) { loadParameters(is); }

Camera::Camera(std::string file_name) { loadParameters(file_name); }

void Camera::loadParameters(std::istream &is) {
  std::string type_name;
  is >> type_name;
  if (type_name == "PINHOLE") {
    _type = PINHOLE;

    // do not read errors
    std::string useless;
    is >> useless;
    is >> useless;

    // read image size (n_row,n_col)
    std::string img_size_str;
    is >> img_size_str;
    std::stringstream img_size_stream(img_size_str);
    std::string temp;
    std::getline(img_size_stream, temp, ',');
    _pinhole_param.n_row = std::stoi(temp);
    std::getline(img_size_stream, temp, ',');
    _pinhole_param.n_col = std::stoi(temp);

    // read camera matrix
    _pinhole_param.cam_mtx = Matrix<double>(3, 3, is);

    // initialize distortion parameters
    _pinhole_param.dist_coeff.clear();
    _pinhole_param.is_distorted = false;
    _pinhole_param.n_dist_coeff = 0;

    // read distortion coefficients
    std::string dist_coeff_str;
    is >> dist_coeff_str;
    std::stringstream dist_coeff_stream(dist_coeff_str);
    double dist_coeff;
    int id = 0;
    while (dist_coeff_stream >> dist_coeff) {
      _pinhole_param.dist_coeff.push_back(dist_coeff);
      id++;

      if (dist_coeff > SMALLNUMBER) {
        _pinhole_param.is_distorted = true;
      }

      if (dist_coeff_stream.peek() == ',') {
        dist_coeff_stream.ignore();
      }
    }
    if (_pinhole_param.is_distorted) {
      _pinhole_param.n_dist_coeff = id;
      if (id != 4 && id != 5 && id != 8 && id != 12) {
        // Note: id == 14
        // additional distortion by projecting onto a tilt plane
        // not supported yet

        std::cerr << "Camera::LoadParameters line " << __LINE__
                  << " : Error: number of distortion coefficients is wrong: "
                  << id << std::endl;
        throw error_type;
      }
    }

    // do not read rotation vector
    is >> useless;

    // read rotation matrix
    _pinhole_param.r_mtx = Matrix<double>(3, 3, is);

    // read inverse rotation matrix
    _pinhole_param.r_mtx_inv = Matrix<double>(3, 3, is);

    // read translation vector
    _pinhole_param.t_vec = Pt3D(is);

    // read inverse translation vector
    _pinhole_param.t_vec_inv = Pt3D(is);
  } else if (type_name == "POLYNOMIAL") {
    _type = POLYNOMIAL;

    // do not read errors
    std::string useless;
    is >> useless;

    // read image size (n_row,n_col)
    std::string img_size_str;
    is >> img_size_str;
    std::stringstream img_size_stream(img_size_str);
    std::string temp;
    std::getline(img_size_stream, temp, ',');
    _poly_param.n_row = std::stoi(temp);
    std::getline(img_size_stream, temp, ',');
    _poly_param.n_col = std::stoi(temp);

    // read reference plane
    std::string ref_plane_str;
    is >> ref_plane_str;
    std::stringstream plane_stream(ref_plane_str);
    // std::string temp;
    std::getline(plane_stream, temp, ',');
    if (temp == "REF_X") {
      _poly_param.ref_plane = REF_X;
    } else if (temp == "REF_Y") {
      _poly_param.ref_plane = REF_Y;
    } else if (temp == "REF_Z") {
      _poly_param.ref_plane = REF_Z;
    } else {
      std::cerr << "Camera::LoadParameters line " << __LINE__
                << " : Error: reference plane is wrong: " << temp << std::endl;
      throw error_type;
    }

    std::getline(plane_stream, temp, ',');
    _poly_param.plane[0] = std::stod(temp);
    std::getline(plane_stream, temp, ',');
    _poly_param.plane[1] = std::stod(temp);

    // read number of coefficients
    is >> _poly_param.n_coeff;

    // read u coefficients
    _poly_param.u_coeffs = Matrix<double>(_poly_param.n_coeff, 4, is);

    // read v coefficients
    _poly_param.v_coeffs = Matrix<double>(_poly_param.n_coeff, 4, is);

    // calculate du, dv coefficients
    updatePolyDuDv();
  } else if (type_name == "PINPLATE") {
    _type = PINPLATE;

    // do not read errors
    std::string useless;
    is >> useless;
    is >> useless;

    // read image size (n_row,n_col)
    std::string img_size_str;
    is >> img_size_str;
    std::stringstream img_size_stream(img_size_str);
    std::string temp;
    std::getline(img_size_stream, temp, ',');
    _pinplate_param.n_row = std::stoi(temp);
    std::getline(img_size_stream, temp, ',');
    _pinplate_param.n_col = std::stoi(temp);

    // read camera matrix
    _pinplate_param.cam_mtx = Matrix<double>(3, 3, is);

    // initialize distortion parameters
    _pinplate_param.dist_coeff.clear();
    _pinplate_param.is_distorted = false;
    _pinplate_param.n_dist_coeff = 0;

    // read distortion coefficients
    std::string dist_coeff_str;
    is >> dist_coeff_str;
    std::stringstream dist_coeff_stream(dist_coeff_str);
    double dist_coeff;
    int id = 0;
    while (dist_coeff_stream >> dist_coeff) {
      _pinplate_param.dist_coeff.push_back(dist_coeff);
      id++;

      if (dist_coeff > SMALLNUMBER) {
        _pinplate_param.is_distorted = true;
      }

      if (dist_coeff_stream.peek() == ',') {
        dist_coeff_stream.ignore();
      }
    }
    if (_pinplate_param.is_distorted) {
      _pinplate_param.n_dist_coeff = id;
      if (id != 4 && id != 5 && id != 8 && id != 12) {
        // Note: id == 14
        // additional distortion by projecting onto a tilt plane
        // not supported yet

        std::cerr << "Camera::LoadParameters line " << __LINE__
                  << " : Error: number of distortion coefficients is wrong: "
                  << id << std::endl;
        throw error_type;
      }
    }

    // do not read rotation vector
    is >> useless;

    // read rotation matrix
    _pinplate_param.r_mtx = Matrix<double>(3, 3, is);

    // read inverse rotation matrix
    _pinplate_param.r_mtx_inv = Matrix<double>(3, 3, is);

    // read translation vector
    _pinplate_param.t_vec = Pt3D(is);

    // read inverse translation vector
    _pinplate_param.t_vec_inv = Pt3D(is);

    // read reference point of refractive plate
    _pinplate_param.plane.pt = Pt3D(is);

    // read normal vector of refractive plate
    _pinplate_param.plane.norm_vector = Pt3D(is);
    // Task B1: Normalize immediately
    {
        double nrm = _pinplate_param.plane.norm_vector.norm();
        if (nrm < 1e-12) {
            std::cerr << "Camera::LoadParameters: Error: plane normal is near zero" << std::endl;
            throw error_type;
        }
        _pinplate_param.plane.norm_vector /= nrm;
    }

    // read refractive index array
    std::string refract_str;
    is >> refract_str;
    std::stringstream refract_stream(refract_str);
    double refract;
    while (refract_stream >> refract) {
      _pinplate_param.refract_array.push_back(refract);
      if (refract_stream.peek() == ',') {
        refract_stream.ignore();
      }
    }
    double ratio = 0.0;
    for (int i = 0; i < _pinplate_param.refract_array.size(); i++) {
      ratio = std::max(ratio, _pinplate_param.refract_array[0] /
                                  _pinplate_param.refract_array[i]);
    }
    _pinplate_param.refract_ratio_max = ratio;

    // read width of the refractive plate
    std::string w_str;
    is >> w_str;
    std::stringstream w_stream(w_str);
    double w;
    while (w_stream >> w) {
      _pinplate_param.w_array.push_back(w);
      if (w_stream.peek() == ',') {
        w_stream.ignore();
      }
    }

    // check number of plate is consistent
    if (_pinplate_param.refract_array.size() !=
            _pinplate_param.w_array.size() + 2 ||
        _pinplate_param.refract_array.size() < 3) {
      std::cerr << "Camera::LoadParameters line " << __LINE__
                << " : Error: number of refractive index and width is not "
                   "consistent (n_plate_min=1): "
                << _pinplate_param.refract_array.size() << " vs "
                << _pinplate_param.w_array.size() << std::endl;
      throw error_type;
    }
    _pinplate_param.n_plate = _pinplate_param.w_array.size();

    updatePinPlateParam();

    // read projection tolerance
    is >> _pinplate_param.proj_tol;

    // read maximum number of iterations for projection
    is >> _pinplate_param.proj_nmax;

    // read learning rate if exists
    if (is >> _pinplate_param.lr) {
      // do nothing
    } else {
      _pinplate_param.lr = 0.1;
    }
  } else {
    std::cerr << "Camera::LoadParameters line " << __LINE__
              << " : Error: unknown camera type: " << type_name << std::endl;
    throw error_type;
  }
}

void Camera::loadParameters(std::string file_name) {
  std::ifstream infile(file_name.c_str(), std::ios::in);

  if (!infile) {
    std::cerr << "Camera::LoadParameters line " << __LINE__
              << " : Error: cannot open file: " << file_name << std::endl;
    throw error_io;
  }

  std::string line;
  std::stringstream file_content;
  while (std::getline(infile, line)) {
    size_t comment_pos = line.find('#');
    if (comment_pos > 0) {
      if (comment_pos < std::string::npos) {
        line.erase(comment_pos);
      }
    } else if (comment_pos == 0) {
      continue;
    }

    file_content << line << '\t';
  }
  infile.close();

  loadParameters(file_content);
}

void Camera::updatePolyDuDv() {
  if (_type == POLYNOMIAL) {
    int derivative_id[2] = {1, 2};
    if (_poly_param.ref_plane == REF_X) {
      derivative_id[0] = 2;
      derivative_id[1] = 3;
    } else if (_poly_param.ref_plane == REF_Y) {
      derivative_id[0] = 1;
      derivative_id[1] = 3;
    } else if (_poly_param.ref_plane == REF_Z) {
      derivative_id[0] = 1;
      derivative_id[1] = 2;
    } else {
      std::cerr << "Camera::updatePolyDuDv line " << __LINE__
                << " : Error: reference plane is wrong: "
                << _poly_param.ref_plane << std::endl;
      throw error_type;
    }

    _poly_param.du_coeffs = Matrix<double>(_poly_param.n_coeff * 2, 4, 0);
    _poly_param.dv_coeffs = Matrix<double>(_poly_param.n_coeff * 2, 4, 0);
    for (int i = 0; i < _poly_param.n_coeff; i++) {
      // calculate du coeff
      _poly_param.du_coeffs(i, 0) = _poly_param.u_coeffs(i, 0) *
                                    _poly_param.u_coeffs(i, derivative_id[0]);
      _poly_param.du_coeffs(i, _poly_param.ref_plane) =
          _poly_param.u_coeffs(i, _poly_param.ref_plane);
      _poly_param.du_coeffs(i, derivative_id[0]) =
          std::max(_poly_param.u_coeffs(i, derivative_id[0]) - 1, 0.0);
      _poly_param.du_coeffs(i, derivative_id[1]) =
          _poly_param.u_coeffs(i, derivative_id[1]);

      int j = i + _poly_param.n_coeff;
      _poly_param.du_coeffs(j, 0) = _poly_param.u_coeffs(i, 0) *
                                    _poly_param.u_coeffs(i, derivative_id[1]);
      _poly_param.du_coeffs(j, _poly_param.ref_plane) =
          _poly_param.u_coeffs(i, _poly_param.ref_plane);
      _poly_param.du_coeffs(j, derivative_id[0]) =
          _poly_param.u_coeffs(i, derivative_id[0]);
      _poly_param.du_coeffs(j, derivative_id[1]) =
          std::max(_poly_param.u_coeffs(i, derivative_id[1]) - 1, 0.0);

      // calculate dv coeff
      _poly_param.dv_coeffs(i, 0) = _poly_param.v_coeffs(i, 0) *
                                    _poly_param.v_coeffs(i, derivative_id[0]);
      _poly_param.dv_coeffs(i, _poly_param.ref_plane) =
          _poly_param.v_coeffs(i, _poly_param.ref_plane);
      _poly_param.dv_coeffs(i, derivative_id[0]) =
          std::max(_poly_param.v_coeffs(i, derivative_id[0]) - 1, 0.0);
      _poly_param.dv_coeffs(i, derivative_id[1]) =
          _poly_param.v_coeffs(i, derivative_id[1]);

      _poly_param.dv_coeffs(j, 0) = _poly_param.v_coeffs(i, 0) *
                                    _poly_param.v_coeffs(i, derivative_id[1]);
      _poly_param.dv_coeffs(j, _poly_param.ref_plane) =
          _poly_param.v_coeffs(i, _poly_param.ref_plane);
      _poly_param.dv_coeffs(j, derivative_id[0]) =
          _poly_param.v_coeffs(i, derivative_id[0]);
      _poly_param.dv_coeffs(j, derivative_id[1]) =
          std::max(_poly_param.v_coeffs(i, derivative_id[1]) - 1, 0.0);
    }
  }
}

Pt3D Camera::rmtxTorvec(Matrix<double> const &r_mtx) {
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

void Camera::saveParameters(std::string file_name) {
  std::ofstream outfile(file_name.c_str(), std::ios::out);
  if (_type == PINHOLE) {
    outfile << "# Camera Model: (PINHOLE/POLYNOMIAL)" << std::endl;
    outfile << "PINHOLE" << std::endl;
    outfile << "# Camera Calibration Error: \nNone\n# Pose Calibration Error: "
               "\nNone"
            << std::endl;

    outfile << "# Image Size: (n_row,n_col)" << std::endl;
    outfile << _pinhole_param.n_row << "," << _pinhole_param.n_col << std::endl;

    outfile << "# Camera Matrix: " << std::endl;
    _pinhole_param.cam_mtx.write(outfile);

    outfile << "# Distortion Coefficients: " << std::endl;
    int size = _pinhole_param.dist_coeff.size();
    for (int i = 0; i < size - 1; i++) {
      outfile << _pinhole_param.dist_coeff[i] << ",";
    }
    outfile << _pinhole_param.dist_coeff[size - 1] << std::endl;

    outfile << "# Rotation Vector: " << std::endl;
    Pt3D r_vec = rmtxTorvec(_pinhole_param.r_mtx);
    r_vec.transpose().write(outfile);

    outfile << "# Rotation Matrix: " << std::endl;
    _pinhole_param.r_mtx.write(outfile);

    outfile << "# Inverse of Rotation Matrix: " << std::endl;
    _pinhole_param.r_mtx_inv.write(outfile);

    outfile << "# Translation Vector: " << std::endl;
    _pinhole_param.t_vec.transpose().write(outfile);

    outfile << "# Inverse of Translation Vector: " << std::endl;
    _pinhole_param.t_vec_inv.transpose().write(outfile);

    outfile.close();
  } else if (_type == POLYNOMIAL) {
    outfile << "# Camera Model: (PINHOLE/POLYNOMIAL)" << std::endl;
    outfile << "POLYNOMIAL" << std::endl;
    outfile << "# Camera Calibration Error: \nNone" << std::endl;

    outfile << "# Image Size: (n_row,n_col)" << std::endl;
    outfile << _poly_param.n_row << "," << _poly_param.n_col << std::endl;

    outfile << "# Reference Plane: (REF_X/REF_Y/REF_Z,coordinate,coordinate)"
            << std::endl;
    if (_poly_param.ref_plane == REF_X) {
      outfile << "REF_X,";
    } else if (_poly_param.ref_plane == REF_Y) {
      outfile << "REF_Y,";
    } else if (_poly_param.ref_plane == REF_Z) {
      outfile << "REF_Z,";
    }
    outfile << _poly_param.plane[0] << "," << _poly_param.plane[1] << std::endl;

    outfile << "# Number of Coefficients: " << std::endl;
    outfile << _poly_param.n_coeff << std::endl;

    outfile << "# U_Coeff,X_Power,Y_Power,Z_Power" << std::endl;
    _poly_param.u_coeffs.write(outfile);

    outfile << "# V_Coeff,X_Power,Y_Power,Z_Power" << std::endl;
    _poly_param.v_coeffs.write(outfile);

    outfile.close();
  } else if (_type == PINPLATE) {
    outfile << "# Camera Model: (PINHOLE/POLYNOMIAL)" << std::endl;
    outfile << "PINPLATE" << std::endl;
    outfile << "# Camera Calibration Error: \nNone\n# Pose Calibration Error: "
               "\nNone"
            << std::endl;

    outfile << "# Image Size: (n_row,n_col)" << std::endl;
    outfile << _pinplate_param.n_row << "," << _pinplate_param.n_col
            << std::endl;

    outfile << "# Camera Matrix: " << std::endl;
    _pinplate_param.cam_mtx.write(outfile);

    outfile << "# Distortion Coefficients: " << std::endl;
    int size = _pinplate_param.dist_coeff.size();
    for (int i = 0; i < size - 1; i++) {
      outfile << _pinplate_param.dist_coeff[i] << ",";
    }
    outfile << _pinplate_param.dist_coeff[size - 1] << std::endl;

    outfile << "# Rotation Vector: " << std::endl;
    Pt3D r_vec = rmtxTorvec(_pinplate_param.r_mtx);
    r_vec.transpose().write(outfile);

    outfile << "# Rotation Matrix: " << std::endl;
    _pinplate_param.r_mtx.write(outfile);

    outfile << "# Inverse of Rotation Matrix: " << std::endl;
    _pinplate_param.r_mtx_inv.write(outfile);

    outfile << "# Translation Vector: " << std::endl;
    _pinplate_param.t_vec.transpose().write(outfile);

    outfile << "# Inverse of Translation Vector: " << std::endl;
    _pinplate_param.t_vec_inv.transpose().write(outfile);

    outfile << "# Reference Point of Refractive Plate: " << std::endl;
    _pinplate_param.plane.pt.transpose().write(outfile);

    outfile << "# Normal Vector of Refractive Plate: " << std::endl;
    _pinplate_param.plane.norm_vector.transpose().write(outfile);

    outfile << "# Refractive Index Array: " << std::endl;
    for (int i = 0; i < _pinplate_param.n_plate + 1; i++) {
      outfile << _pinplate_param.refract_array[i] << ",";
    }
    outfile << _pinplate_param.refract_array[_pinplate_param.n_plate + 1]
            << std::endl;

    outfile << "# Width of the Refractive Plate: (same as other physical units)"
            << std::endl;
    for (int i = 0; i < _pinplate_param.n_plate - 1; i++) {
      outfile << _pinplate_param.w_array[i] << ",";
    }
    outfile << _pinplate_param.w_array[_pinplate_param.n_plate - 1]
            << std::endl;

    outfile << "# Projection Tolerance: " << std::endl;
    outfile << _pinplate_param.proj_tol << std::endl;

    outfile << "# Maximum Number of Iterations for Projection: " << std::endl;
    outfile << _pinplate_param.proj_nmax << std::endl;

    outfile << "# Optimization Rate: " << std::endl;
    outfile << _pinplate_param.lr << std::endl;

    outfile.close();
  } else {
    std::cerr << "Camera::SaveParameters line " << __LINE__
              << " : Error: unknown camera type: " << _type << std::endl;
    throw error_type;
  }
}

//
// Get image size
//
int Camera::getNRow() const {
  if (_type == PINHOLE) {
    return _pinhole_param.n_row;
  } else if (_type == POLYNOMIAL) {
    return _poly_param.n_row;
  } else if (_type == PINPLATE) {
    return _pinplate_param.n_row;
  } else {
    std::cerr << "Camera::GetNRow line " << __LINE__
              << " : Error: unknown camera type: " << _type << std::endl;
    throw error_type;
  }
}

int Camera::getNCol() const {
  if (_type == PINHOLE) {
    return _pinhole_param.n_col;
  } else if (_type == POLYNOMIAL) {
    return _poly_param.n_col;
  } else if (_type == PINPLATE) {
    return _pinplate_param.n_col;
  } else {
    std::cerr << "Camera::GetNCol line " << __LINE__
              << " : Error: unknown camera type: " << _type << std::endl;
    throw error_type;
  }
}

//
// Project world coordinate [mm] to image points [px]
//
Pt2D Camera::project(Pt3D const &pt_world, bool is_print_detail) const {
  if (_type == PINHOLE) {
    return distort(worldToUndistImg(pt_world, _pinhole_param), _pinhole_param);
  } else if (_type == POLYNOMIAL) {
    return polyProject(pt_world);
  } else if (_type == PINPLATE) {
    std::tuple<bool, Pt3D, double, int> result = refractPlate(pt_world);
    if (is_print_detail) {
      std::cout << "Camera::Project line " << __LINE__
                << " : result (is_parallel,error^2): " << std::get<0>(result)
                << "," << std::get<2>(result) << std::endl;
    }
    if (std::get<0>(result)) {
      double value = std::numeric_limits<double>::lowest();
      return Pt2D(value, value);
    }
    // Pt3D pt_refract = std::get<1>(result);
    return distort(worldToUndistImg(std::get<1>(result), _pinplate_param),
                   _pinplate_param);
  } else {
    THROW_FATAL(ErrorCode::InvalidArgument,
                "Unknown camera type: " + std::to_string(_type));
  }
}

// Pinhole  model
Pt2D Camera::worldToUndistImg(Pt3D const &pt_world,
                              PinholeParam const &param) const {
  Pt3D temp = param.r_mtx * pt_world + param.t_vec;

  Pt2D pt_img_mm;
  pt_img_mm[0] = temp[2] ? (temp[0] / temp[2]) : temp[0];
  pt_img_mm[1] = temp[2] ? (temp[1] / temp[2]) : temp[1];

  return pt_img_mm;
}


Pt2D Camera::distort(Pt2D const &pt_img_undist,
                     PinholeParam const &param) const {
  // opencv distortion model
  double x = pt_img_undist[0];
  double y = pt_img_undist[1];
  double xd, yd;

  // judge if distortion is needed
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
    // double icdist2 = 1.0 / (1 + param.dist_coeff[5]*r2 +
    // param.dist_coeff[6]*r4 + param.dist_coeff[7]*r6);

    xd = x * cdist * icdist2 + param.dist_coeff[2] * a1 +
         param.dist_coeff[3] * a2;
    yd = y * cdist * icdist2 + param.dist_coeff[2] * a3 +
         param.dist_coeff[3] * a1;
    if (param.n_dist_coeff > 8) {
      // additional distortion by projecting onto a tilt plane (14)
      // not supported yet
      xd += param.dist_coeff[8] * r2 + param.dist_coeff[9] * r4;
      yd += param.dist_coeff[10] * r2 + param.dist_coeff[11] * r4;
    }
  } else {
    xd = x;
    yd = y;
  }

  Pt2D pt_img_dist(xd * param.cam_mtx(0, 0) + param.cam_mtx(0, 2),
                   yd * param.cam_mtx(1, 1) + param.cam_mtx(1, 2));

  return pt_img_dist;
}

// Polynomial model
Pt2D Camera::polyProject(Pt3D const &pt_world) const {
  double u = 0;
  double v = 0;

  for (int i = 0; i < _poly_param.u_coeffs.getDimRow(); i++) {
    double u_val = _poly_param.u_coeffs(i, 0);
    double v_val = _poly_param.v_coeffs(i, 0);

    for (int j = 1; j < 4; j++) {
      u_val *= std::pow(pt_world[j - 1], _poly_param.u_coeffs(i, j));
      v_val *= std::pow(pt_world[j - 1], _poly_param.v_coeffs(i, j));
    }

    u += u_val;
    v += v_val;
  }

  return Pt2D(u, v);
}

//
// Calculate line of sight
//
Line3D Camera::lineOfSight(Pt2D const &pt_img_dist) const {
  if (_type == PINHOLE) {
    return pinholeLine(undistort(pt_img_dist, _pinhole_param));
  } else if (_type == POLYNOMIAL) {
    return polyLineOfSight(pt_img_dist);
  } else if (_type == PINPLATE) {
    return pinplateLine(undistort(pt_img_dist, _pinplate_param));
  } else {
    std::cerr << "Camera::LineOfSight line " << __LINE__
              << " : Error: unknown camera type: " << _type << std::endl;
    throw error_type;
  }
}

// Pinhole  model
Pt2D Camera::undistort(Pt2D const &pt_img_dist,
                       PinholeParam const &param) const {
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

    // undistort iteratively
    int iter = 0;
    while (error > UNDISTORT_EPS && iter < UNDISTORT_MAX_ITER) {
      double r2 = x * x + y * y;
      double a1 = 2 * x * y;
      double a2 = r2 + 2 * x * x;
      double a3 = r2 + 2 * y * y;

      // double icdist = (1 + ( (k[7]*r2+k[6])*r2 + k[5] )*r2)
      //                /(1 + ( (k[4]*r2+k[1])*r2 + k[0] )*r2);
      double icdist;
      if (param.n_dist_coeff == 4) {
        icdist =
            1 / (1 + (param.dist_coeff[1] * r2 + param.dist_coeff[0]) * r2);
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

      // update x, y
      double deltaX = param.dist_coeff[2] * a1 + param.dist_coeff[3] * a2;
      double deltaY = param.dist_coeff[2] * a3 + param.dist_coeff[3] * a1;
      if (param.n_dist_coeff > 8) {
        deltaX += param.dist_coeff[8] * r2 + param.dist_coeff[9] * r2 * r2;
        deltaY += param.dist_coeff[10] * r2 + param.dist_coeff[11] * r2 * r2;
      }
      x = (x0 - deltaX) * icdist;
      y = (y0 - deltaY) * icdist;

      // calculate distorted xd, yd
      img_undist[0] = x;
      img_undist[1] = y;
      img_dist = distort(img_undist, param);

      // update error
      error = std::sqrt(std::pow(img_dist[0] - u, 2) +
                        std::pow(img_dist[1] - v, 2));

      iter++;
    }
  }

  return img_undist;
}

Line3D Camera::pinholeLine(Pt2D const &pt_img_undist) const {
  Pt3D pt_world(pt_img_undist[0], pt_img_undist[1], 1.0);

  // calculate line of sight
  pt_world = _pinhole_param.r_mtx_inv * pt_world + _pinhole_param.t_vec_inv;

  Line3D line;
  line.pt = _pinhole_param.t_vec_inv;
  line.unit_vector =
      myMATH::createUnitVector(_pinhole_param.t_vec_inv, pt_world);

  return line;
}

void Camera::updatePinPlateParam ()
{
    if (_type == PINPLATE)
    {
        // from farthest to nearest to camera
        _pinplate_param.plane_array = std::vector<Plane3D>(_pinplate_param.n_plate + 1);
        _pinplate_param.plane_array[0] = _pinplate_param.plane;

        for (int i = 1; i <= _pinplate_param.n_plate; i ++)
        {
            Plane3D& prev_plane = _pinplate_param.plane_array[i-1];
            Plane3D& curr_plane = _pinplate_param.plane_array[i];

            // point on the current plane
            curr_plane.pt = prev_plane.pt - prev_plane.norm_vector * _pinplate_param.w_array[i-1];

            // normal vector of the current plane
            curr_plane.norm_vector = prev_plane.norm_vector;
        }

        // 2D coordinate system on refractive planes
        getPlaneCoordinateSystem(_pinplate_param.u_axis, _pinplate_param.v_axis, _pinplate_param.plane);
    }
}

Line3D Camera::pinplateLine (Pt2D const& pt_img_undist) const
{
    Pt3D pt_world(pt_img_undist[0], pt_img_undist[1], 1.0);

    // calculate line of sight
    pt_world = _pinplate_param.r_mtx_inv * pt_world + _pinplate_param.t_vec_inv;
    
    Line3D line;
    line.pt = _pinplate_param.t_vec_inv;
    line.unit_vector = myMATH::createUnitVector(_pinplate_param.t_vec_inv, pt_world);

    // backward projection
    Pt3D pt_cross;
    bool is_parallel = false;
    double refract_ratio;
    double cos_1;
    double cos_2;
    double factor;

    for (int plane_id = _pinplate_param.n_plate; plane_id >= 0; plane_id --)
    {
        const Plane3D& plane = _pinplate_param.plane_array[plane_id];
        is_parallel = myMATH::crossPoint(pt_cross, line, plane);
        if (is_parallel)
        {
            std::cerr << "Camera::PinplateLine line " << __LINE__ << " : Error: line is parallel to the plane" << std::endl;
            throw error_type;
        }

        double refract_ratio = _pinplate_param.refract_array[plane_id+1] / _pinplate_param.refract_array[plane_id];
        bool success = refractDir(line.unit_vector, plane.norm_vector, refract_ratio, false);
        if (!success)
        {
            std::cerr << "Camera::PinplateLine line " << __LINE__ << " : Error: total internal reflection" << std::endl;
            throw error_type;
        }
        line.pt = pt_cross;
    }

    return line;
}

// Plate refraction model
std::tuple<bool, Pt3D, double, int> Camera::refractPlate(Pt3D const& pt_world) const {
    return refractPlateNewton(pt_world);
}

std::tuple<bool, Pt3D, double, int> Camera::refractPlateNewton(Pt3D const& pt_world) const
{
    std::tuple<bool, Pt3D, double, int> result = {false, {0,0,0}, -1, 0};
    
    const int proj_nmax = _pinplate_param.proj_nmax;
    const double proj_tol = _pinplate_param.proj_tol;
    const double eps = 1e-5;
    
    // Check if we need to enforce radius constraint (total internal reflection zone)
    bool is_check_radius = false;
    double radius_max2 = 0;
    if (_pinplate_param.refract_ratio_max > 1.0) {
        is_check_radius = true;
        double sin_angle = 1.0 / _pinplate_param.refract_ratio_max;
        radius_max2 = myMATH::dist2(pt_world, _pinplate_param.plane) / (1.0 - sin_angle*sin_angle);
    }
    
    // Initialize: find initial point on first plane
    Line3D line;
    line.pt = pt_world;
    line.unit_vector = myMATH::createUnitVector(pt_world, _pinplate_param.t_vec_inv);
    
    Pt3D pt_init;
    if (myMATH::crossPoint(pt_init, line, _pinplate_param.plane)) {
        std::get<0>(result) = true;
        return result;
    }
    
    // Check and adjust initial point if it's in forbidden region
    if (is_check_radius) {
        double radius2 = myMATH::dist2(pt_init, pt_world);
        if (radius2 >= radius_max2) {
            // Project onto boundary of valid region (95% of max radius for safety)
            double scale = std::sqrt(radius_max2 * 0.95 / radius2);
            for (int i = 0; i < 3; i++) {
                pt_init[i] = pt_world[i] + (pt_init[i] - pt_world[i]) * scale;
            }
            
            // Verify the adjusted point can trace without total internal reflection
            Pt3D pt_exit_test, dir_test;
            if (!forwardTrace(pt_exit_test, dir_test, pt_world, pt_init)) {
                std::get<0>(result) = true;
                return result;
            }
        }
    }
    
    // 2D parametrization on first plane
    std::vector<double> x0(2);
    projectPointToPlane2D(x0, pt_init, _pinplate_param.plane,
                         _pinplate_param.u_axis, _pinplate_param.v_axis);

    const double L_min = 0.1;
    const double mu_min = 1e-4;
    const double mu_scale = 0.1;
    double L = L_min;
    {
        Pt3D pt_exit_scale, exit_dir_scale;
        if (forwardTrace(pt_exit_scale, exit_dir_scale, pt_world, pt_init)) {
            Pt3D diff = pt_exit_scale - _pinplate_param.plane.pt;
            double dist = myMATH::dot(diff, _pinplate_param.plane.norm_vector);
            Pt3D pt_exit_proj = pt_exit_scale - _pinplate_param.plane.norm_vector * dist;
            double line_dist = myMATH::dist(pt_exit_proj, line);
            L = std::max(line_dist, L_min);
        }
    }
    
    // Reusable variables
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
    
    // Levenberg-Marquardt damping parameter
    double lambda = 0.01;
    const double lambda_min = 1e-10;
    const double lambda_max = 1e10;

    const double mu_max = 0.1 * L;

    for (int iter = 0; iter < proj_nmax; iter++) {
        // Reconstruct 3D point from 2D parameters
        delta_x[0] = L * delta_y[0];
        delta_x[1] = L * delta_y[1];
        x[0] = x0[0] + delta_x[0];
        x[1] = x0[1] + delta_x[1];
        reconstructFrom2D(pt_current, _pinplate_param.plane.pt, 
                         _pinplate_param.u_axis, _pinplate_param.v_axis, x);

        double radius2_current = 0.0;
        if (is_check_radius) {
            radius2_current = myMATH::dist2(pt_current, pt_world);
            if (radius2_current >= radius_max2) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                return result;
            }
        }
        
        // Forward trace to get exit ray
        if (!forwardTrace(pt_exit, exit_direction, pt_world, pt_current)) {
            std::get<0>(result) = true;
            std::get<3>(result) = iter + 1;
            return result;
        }
        
        // RESIDUAL: Distance from exit ray to pinhole
        vec_to_pinhole = _pinplate_param.t_vec_inv - pt_exit;
        double proj_along_ray = myMATH::dot(vec_to_pinhole, exit_direction);
        residual_vec = vec_to_pinhole - exit_direction * proj_along_ray;
        
        double residual = residual_vec.norm();
        double barrier_current = 0.0;
        if (is_check_radius) {
            double mu = std::max(mu_min, std::min(mu_scale * residual, mu_max));
            double d = radius_max2 - radius2_current;
            barrier_current = -mu * std::log(d);
        }
        
        if (residual < proj_tol) {
            std::get<1>(result) = pt_exit;
            std::get<2>(result) = residual;
            std::get<3>(result) = iter + 1;
            return result;
        }
        
        // Project residual onto plane basis to get 2D residual
        F[0] = myMATH::dot(residual_vec, _pinplate_param.u_axis);
        F[1] = myMATH::dot(residual_vec, _pinplate_param.v_axis);
        
        // Compute Jacobian using finite differences with radius constraint
        for (int j = 0; j < 2; j++) {
            delta_y_perturb = delta_y;
            delta_y_perturb[j] += eps;
            delta_x[0] = L * delta_y_perturb[0];
            delta_x[1] = L * delta_y_perturb[1];
            x_perturb[0] = x0[0] + delta_x[0];
            x_perturb[1] = x0[1] + delta_x[1];

            // Reconstruct perturbed 3D point
            reconstructFrom2D(pt_current, _pinplate_param.plane.pt, 
                              _pinplate_param.u_axis, _pinplate_param.v_axis, x_perturb);
            
            // Enforce radius constraint by clamping to valid region
            if (is_check_radius) {
                double radius2 = myMATH::dist2(pt_current, pt_world);
                if (radius2 >= radius_max2) {
                    std::get<0>(result) = true;
                    std::get<3>(result) = iter + 1;
                    return result;
                }
            }
            
            if (!forwardTrace(pt_exit, exit_direction, pt_world, pt_current)) {
                // Should not happen after clamping, but handle it
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                return result;
            }
            
            // Compute perturbed residual
            vec_to_pinhole = _pinplate_param.t_vec_inv - pt_exit;
            proj_along_ray = myMATH::dot(vec_to_pinhole, exit_direction);
            residual_vec = vec_to_pinhole - exit_direction * proj_along_ray;
            
            double F_perturb[2];
            F_perturb[0] = myMATH::dot(residual_vec, _pinplate_param.u_axis);
            F_perturb[1] = myMATH::dot(residual_vec, _pinplate_param.v_axis);
            
            // Compute finite difference with actual (possibly clamped) perturbation
            J[0][j] = (F_perturb[0] - F[0]) / (delta_y_perturb[j] - delta_y[j]);
            J[1][j] = (F_perturb[1] - F[1]) / (delta_y_perturb[j] - delta_y[j]);
        }
        
        // Levenberg-Marquardt: Solve (J^T*J + lambda*I) * dx = -J^T*F
        JTJ[0][0] = J[0][0]*J[0][0] + J[1][0]*J[1][0] + lambda;
        JTJ[0][1] = J[0][0]*J[0][1] + J[1][0]*J[1][1];
        JTJ[1][0] = JTJ[0][1];
        JTJ[1][1] = J[0][1]*J[0][1] + J[1][1]*J[1][1] + lambda;
        
        double JTF[2];
        JTF[0] = -(J[0][0]*F[0] + J[1][0]*F[1]);
        JTF[1] = -(J[0][1]*F[0] + J[1][1]*F[1]);
        
        // Solve 2x2 system
        double det = JTJ[0][0]*JTJ[1][1] - JTJ[0][1]*JTJ[1][0];
        
        // Check condition number
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
        dx[0] = (JTF[0]*JTJ[1][1] - JTF[1]*JTJ[0][1]) / det;
        dx[1] = (JTF[1]*JTJ[0][0] - JTF[0]*JTJ[1][0]) / det;
        
        // Line search with radius constraint
        double alpha = 1.0;
        bool step_accepted = false;
        const int max_line_search = 10;
        
        for (int ls = 0; ls < max_line_search; ls++) {
            delta_y_perturb[0] = delta_y[0] + alpha*dx[0];
            delta_y_perturb[1] = delta_y[1] + alpha*dx[1];
            delta_x[0] = L * delta_y_perturb[0];
            delta_x[1] = L * delta_y_perturb[1];
            x_perturb[0] = x0[0] + delta_x[0];
            x_perturb[1] = x0[1] + delta_x[1];
            
            reconstructFrom2D(pt_current, _pinplate_param.plane.pt, 
                            _pinplate_param.u_axis, _pinplate_param.v_axis, x_perturb);

            // Enforce radius constraint during line search too
            if (is_check_radius) {
                double radius2 = myMATH::dist2(pt_current, pt_world);
                if (radius2 >= radius_max2) {
                    alpha *= 0.5;
                    continue;
                }
            }
            
            if (forwardTrace(pt_exit, exit_direction, pt_world, pt_current)) {
                vec_to_pinhole = _pinplate_param.t_vec_inv - pt_exit;
                proj_along_ray = myMATH::dot(vec_to_pinhole, exit_direction);
                residual_vec = vec_to_pinhole - exit_direction * proj_along_ray;
                double residual_new = residual_vec.norm();
                
                double barrier_new = 0.0;
                if (is_check_radius) {
                    double mu = std::max(mu_min, std::min(mu_scale * residual_new, mu_max));
                    double d = radius_max2 - myMATH::dist2(pt_current, pt_world);
                    barrier_new = -mu * std::log(d);
                }

                if (residual_new + barrier_new < residual + barrier_current) {
                    // Update x to reflect actual (possibly clamped) position
                    projectPointToPlane2D(x, pt_current, _pinplate_param.plane,
                                         _pinplate_param.u_axis, _pinplate_param.v_axis);
                    delta_y[0] = (x[0] - x0[0]) / L;
                    delta_y[1] = (x[1] - x0[1]) / L;
                    step_accepted = true;
                    lambda = std::max(lambda * 0.1, lambda_min);
                    break;
                }
            }
            alpha *= 0.5;
        }
        
        if (!step_accepted) {
            lambda = std::min(lambda * 10.0, lambda_max);
            if (lambda >= lambda_max) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                return result;
            }
        }
    }
    
    // If reached here, maximum iterations exceeded
    std::get<1>(result) = pt_exit;
    std::get<2>(result) = residual_vec.norm();
    std::get<3>(result) = proj_nmax;
    return result;
}

void Camera::getPlaneCoordinateSystem(Pt3D& u_axis, Pt3D& v_axis, const Plane3D& plane) const {
    // Create orthonormal basis on the plane
    if (std::abs(plane.norm_vector[2]) < 0.9) {
        u_axis = myMATH::cross(plane.norm_vector, {0, 0, 1});
    } else {
        u_axis = myMATH::cross(plane.norm_vector, {1, 0, 0});
    }
    u_axis /= u_axis.norm();
    v_axis = myMATH::cross(plane.norm_vector, u_axis);
    v_axis /= v_axis.norm();
}

void Camera::projectPointToPlane2D(std::vector<double>& coords, 
                                   const Pt3D& pt, const Plane3D& plane,
                                   const Pt3D& u_axis, const Pt3D& v_axis) const
{
    Pt3D diff = {pt[0] - plane.pt[0], pt[1] - plane.pt[1], pt[2] - plane.pt[2]};
    coords[0] = myMATH::dot(diff, u_axis);
    coords[1] = myMATH::dot(diff, v_axis);
}

void Camera::reconstructFrom2D(Pt3D& result, const Pt3D& origin, const Pt3D& u_axis,
                               const Pt3D& v_axis, const std::vector<double>& coords) const
{
    for (int i = 0; i < 3; i++) {
        result[i] = origin[i] + coords[0]*u_axis[i] + coords[1]*v_axis[i];
    }
}

bool Camera::forwardTrace(Pt3D& pt_exit, Pt3D& exit_direction,
                          const Pt3D& pt_world, const Pt3D& pt_entry) const
{
    Line3D line;
    line.pt = pt_world;
    line.unit_vector = myMATH::createUnitVector(pt_world, pt_entry);
    
    Pt3D pt_cross = pt_entry;
    
    for (int plane_id = 1; plane_id <= _pinplate_param.n_plate; plane_id++) {
        const Plane3D& plane_curr = _pinplate_param.plane_array[plane_id-1];
        const Plane3D& plane_next = _pinplate_param.plane_array[plane_id];

        double refract_ratio = _pinplate_param.refract_array[plane_id-1] / 
                               _pinplate_param.refract_array[plane_id];
        
        if (!refractDir(line.unit_vector, plane_curr.norm_vector, refract_ratio, true)) {
            return false;
        }
        line.pt = pt_cross;
        
        if (myMATH::crossPoint(pt_cross, line, plane_next)) {
            return false;
        }
    }
    
    pt_exit = pt_cross;
    exit_direction = line.unit_vector;
    double refract_ratio = _pinplate_param.refract_array[_pinplate_param.n_plate] / 
                           _pinplate_param.refract_array[_pinplate_param.n_plate + 1];
    bool success = refractDir(exit_direction, _pinplate_param.plane_array[_pinplate_param.n_plate].norm_vector, refract_ratio, true);
    if (!success) {
        return false;
    }
    
    return true;
}

bool Camera::refractDir(Pt3D& dir_refract, const Pt3D& normal, double refract_ratio, bool is_forward) const {
    double cos_1 = std::clamp(myMATH::dot(dir_refract, normal), -1.0, 1.0);
    double cos_2 = 1.0 - refract_ratio*refract_ratio*(1.0 - cos_1*cos_1);
    if (cos_2 <= 0) {
        return false;
    }

    if (std::abs(cos_1) < SMALLNUMBER) {
        cos_1 = 0.0;
        double tan_2 = std::sqrt((1.0 - cos_2) / cos_2);
        double factor = - dir_refract.norm() / tan_2 * (is_forward ? 1.0 : -1.0); 
        for (int i = 0; i < 3; i++) {
            dir_refract[i] = dir_refract[i] + normal[i] * factor;
        }
    } else {
        double factor = - refract_ratio * cos_1 - std::sqrt(cos_2) * (is_forward ? 1.0 : -1.0);
        for (int i = 0; i < 3; i++) {
            dir_refract[i] = dir_refract[i] * refract_ratio + normal[i] * factor;
        }
    }

    dir_refract /= dir_refract.norm();
    return true;
}


// Polynomial model
Pt3D Camera::polyImgToWorld(Pt2D const &pt_img_dist, double plane_world) const {
  Pt3D pt_world((double)rand() / RAND_MAX, (double)rand() / RAND_MAX,
                (double)rand() / RAND_MAX);

  switch (_poly_param.ref_plane) {
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
    std::cerr << "Camera::PolyImgToWorld line " << __LINE__
              << " : Error: unknown reference plane: " << _poly_param.ref_plane
              << std::endl;
    throw error_type;
  }

  Matrix<double> jacobian(2, 2, 0);
  Matrix<double> jacobian_inv(2, 2, 0);
  Pt2D pt_img_temp;
  double du, dv, dx, dy;
  double err = std::numeric_limits<double>::max();
  int iter = 0;
  jacobian = Matrix<double>(2, 2, 0);
  jacobian_inv = Matrix<double>(2, 2, 0);

  while (err > UNDISTORT_EPS && iter < UNDISTORT_MAX_ITER) {
    // calculate jacobian matrix (e.g. REF_Z)
    // du = u_true - u, dv = v_true - v
    // |du| = | du/dx, du/dy | |dx|
    // |dv| = | dv/dx, dv/dy | |dy|
    // x = x0 + dx
    // y = y0 + dy
    jacobian *= 0;
    jacobian_inv *= 0;
    // pt_img_temp = polyProject(pt_world);
    // du = pt_img_dist[0] - pt_img_temp[0];
    // dv = pt_img_dist[1] - pt_img_temp[1];

    // calculate jacobian matrix
    for (int i = 0; i < _poly_param.n_coeff; i++) {
      double dudx = _poly_param.du_coeffs(i, 0);
      double dudy = _poly_param.du_coeffs(i + _poly_param.n_coeff, 0);
      double dvdx = _poly_param.dv_coeffs(i, 0);
      double dvdy = _poly_param.dv_coeffs(i + _poly_param.n_coeff, 0);

      for (int j = 1; j < 4; j++) {
        dudx *= std::pow(pt_world[j - 1], _poly_param.du_coeffs(i, j));
        dudy *= std::pow(pt_world[j - 1],
                         _poly_param.du_coeffs(i + _poly_param.n_coeff, j));

        dvdx *= std::pow(pt_world[j - 1], _poly_param.dv_coeffs(i, j));
        dvdy *= std::pow(pt_world[j - 1],
                         _poly_param.dv_coeffs(i + _poly_param.n_coeff, j));
      }

      jacobian(0, 0) += dudx;
      jacobian(0, 1) += dudy;
      jacobian(1, 0) += dvdx;
      jacobian(1, 1) += dvdy;
    }

    // calculate dx, dy
    jacobian_inv = myMATH::inverse(jacobian, "det");
    pt_img_temp = polyProject(pt_world);
    du = pt_img_dist[0] - pt_img_temp[0];
    dv = pt_img_dist[1] - pt_img_temp[1];

    dx = jacobian_inv(0, 0) * du + jacobian_inv(0, 1) * dv;
    dy = jacobian_inv(1, 0) * du + jacobian_inv(1, 1) * dv;

    // update pt_world
    switch (_poly_param.ref_plane) {
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
      std::cerr << "Camera::PolyImgToWorld line " << __LINE__
                << " : Error: unknown reference plane: "
                << _poly_param.ref_plane << std::endl;
      throw error_type;
    }

    // update error, iter
    err = std::sqrt(du * du + dv * dv);
    iter++;
  }

  // for debug
  // std::cout << "iter: " << iter << std::endl;
  // std::cout << "err: " << err << std::endl;

  return pt_world;
}

Line3D Camera::polyLineOfSight(Pt2D const &pt_img_dist) const {
  Pt3D pt_world_1 = polyImgToWorld(pt_img_dist, _poly_param.plane[0]);
  Pt3D pt_world_2 = polyImgToWorld(pt_img_dist, _poly_param.plane[1]);
  Pt3D unit_vec = myMATH::createUnitVector(pt_world_1, pt_world_2);

  Line3D line = {pt_world_1, unit_vec};

  return line;
}
