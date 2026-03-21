#include <algorithm>
#include <cmath>
#include <random>

#include "Config.h"
#include "OTF.h"
#include "ObjectFinder.h"
#include "ObjectInfo.h"

OTF::OTF(int n_cam, int nx, int ny, int nz, AxisLimit const &boundary,
         const std::vector<std::shared_ptr<Camera>> &camera_models) {
  loadParam(n_cam, nx, ny, nz, boundary, camera_models);
};

OTF::OTF(std::string otf_file) { loadParam(otf_file); };

void OTF::estimateUniformOTFFromImage(int cam_id, TracerConfig &tracer_config,
                                      const std::vector<Image> &img_list) {
  // estimate OTF parameters
  // Gaussian intensity:
  //  projection center: (xc,yc), x=col, y=row
  //  dx =  (x-xc)*cos(alpha) + (y-yc)*sin(alpha)
  //  dy = -(x-xc)*sin(alpha) + (y-yc)*sin(alpha)
  //  I(x,y) = a * exp( - b*dx^2 - c*dx^2 )
  //  - ln(I) + ln(a) = b*dx^2 + c*dy^2
  //  a = mean(I_c) + 2*std(I_c)
  // Assume a,b,c,alpha are the same in the view volume, alpha=0
  //  nx,ny,nz = 2,2,2
  // otf_param: need to be initialized before calling this function

  // this can only be applied to Tracer3D
  std::vector<std::vector<std::unique_ptr<Object2D>>> tr2d_list_all;
  ObjectFinder2D objfinder;
  int n_obj2d_max = 1000; // maximum number of tracers in each camera

  // find objects in each image
  std::cout << "Camera " << cam_id << " ";
  for (int j = 0; j < img_list.size(); j++) {
    std::vector<std::unique_ptr<Object2D>> obj2d_list =
        objfinder.findObject2D(img_list[j], tracer_config); // TODO:

    // if tr2d_list is too large, randomly select some tracers
    int seed = 123;
    if (obj2d_list.size() > n_obj2d_max) {
      std::shuffle(obj2d_list.begin(), obj2d_list.end(),
                   std::default_random_engine(seed));
      obj2d_list.erase(obj2d_list.begin() + n_obj2d_max, obj2d_list.end());
    } else if (obj2d_list.size() == 0) {
      Status st = STATUS_ERR_CTX(ErrorCode::NoEnoughData,
                                 "OTF calibration skipped: no tracer found",
                                 "cam_id=" + std::to_string(cam_id));
      std::cerr << "[WARN] " << st.err.message << " | ctx: " << st.err.context
                << " | at " << SRC_FILE(st.err.where) << ":"
                << SRC_LINE(st.err.where) << std::endl;
      return;
    }
    tr2d_list_all.push_back(std::move(obj2d_list));
  }

  // estimate OTF parameters
  std::vector<double> I_list;
  std::vector<double> a_list;
  std::vector<std::vector<double>> coeff_list;
  std::vector<double> coeff(2, 0);
  double xc, yc;
  int n_row, n_col;
  int row_min, row_max, col_min, col_max;
  double r_pt = tracer_config._radius_obj;
  for (int i = 0; i < img_list.size(); i++) {
    n_row = img_list[i].getDimRow();
    n_col = img_list[i].getDimCol();

    for (int j = 0; j < tr2d_list_all[i].size(); j++) {
      xc = tr2d_list_all[i][j]->_pt_center[0];
      yc = tr2d_list_all[i][j]->_pt_center[1];

      int yc0 = std::clamp<int>(int(std::round(yc)), 0, n_row - 1);
      int xc0 = std::clamp<int>(int(std::round(xc)), 0, n_col - 1);
      a_list.push_back(img_list[i](yc0, xc0));

      row_min = std::max(0, int(std::floor(yc - r_pt)));
      row_max = std::min(n_row, int(std::ceil(yc + r_pt + 1)));
      col_min = std::max(0, int(std::floor(xc - r_pt)));
      col_max = std::min(n_col, int(std::ceil(xc + r_pt + 1)));

      for (int row = row_min; row < row_max; row++) {
        for (int col = col_min; col < col_max; col++) {
          I_list.push_back(img_list[i](row, col));
          coeff[0] = (std::pow(col - xc, 2));
          coeff[1] = (std::pow(row - yc, 2));
          coeff_list.push_back(coeff);
        }
      }
    }
  }

  // a = mean(I_c) + 2.*std(I_c)
  double a;
  double a_mean = 0;
  double a_std = 0;
  double a_max = *std::max_element(a_list.begin(), a_list.end());
  int n_a = a_list.size();
  for (int i = 0; i < n_a; i++) {
    a_mean += a_list[i];
  }
  a_mean /= n_a;
  for (int i = 0; i < n_a; i++) {
    a_std += std::pow(a_list[i] - a_mean, 2);
  }
  a_std = std::sqrt(a_std / n_a);
  a = std::min(a_mean + 2 * a_std, a_max);
  // estimate the coefficients
  int n_coeff = coeff_list.size();
  double logI;
  Matrix<double> coeff_mat(n_coeff, 2, 0);
  Matrix<double> logI_mat(n_coeff, 1, 0);
  for (int i = 0; i < n_coeff; i++) {
    for (int j = 0; j < 2; j++) {
      coeff_mat(i, j) = coeff_list[i][j];
    }

    logI = I_list[i] < LOGSMALLNUMBER ? std::log(LOGSMALLNUMBER)
                                      : std::log(I_list[i]);
    logI_mat(i, 0) = -logI + std::log(a);
  }
  Matrix<double> coeff_est =
      myMATH::inverse(coeff_mat.transpose() * coeff_mat) *
      coeff_mat.transpose() * logI_mat;

  for (int i = 0; i < _param.n_grid; i++) {
    _param.a(cam_id, i) = a;
    _param.b(cam_id, i) = coeff_est(0, 0);
    _param.c(cam_id, i) = coeff_est(1, 0);
    _param.alpha(cam_id, i) = 0;
  }

  std::cout << "(a,b,c,alpha) = " << _param.a(cam_id, 0) << ","
            << _param.b(cam_id, 0) << "," << _param.c(cam_id, 0) << ","
            << _param.alpha(cam_id, 0) << std::endl
            << std::endl;

  // Initial estimation complete. Parameters are held in memory for VSC
  // refinement.
}

void OTF::setGrid() {
  _param.grid_x =
      myMATH::linspace(_param.boundary.x_min, _param.boundary.x_max, _param.nx);
  _param.grid_y =
      myMATH::linspace(_param.boundary.y_min, _param.boundary.y_max, _param.ny);
  _param.grid_z =
      myMATH::linspace(_param.boundary.z_min, _param.boundary.z_max, _param.nz);

  _param.dx = (_param.boundary.x_max - _param.boundary.x_min) / (_param.nx - 1);
  _param.dy = (_param.boundary.y_max - _param.boundary.y_min) / (_param.ny - 1);
  _param.dz = (_param.boundary.z_max - _param.boundary.z_min) / (_param.nz - 1);
}

void OTF::loadParam(int n_cam, int nx, int ny, int nz,
                    AxisLimit const &boundary,
                    const std::vector<std::shared_ptr<Camera>> &camera_models) {
  if (nx < 2 || ny < 2 || nz < 2) {
    Status st = STATUS_ERR(ErrorCode::InvalidArgument,
                           "Grid number should be larger than 1");
    std::cerr << "[WARN] " << st.err.message << " | at "
              << SRC_FILE(st.err.where) << ":" << SRC_LINE(st.err.where)
              << std::endl;
    nx = nx < 2 ? 2 : nx;
    ny = ny < 2 ? 2 : ny;
    nz = nz < 2 ? 2 : nz;
  }

  _param.n_cam = n_cam;
  _param.nx = nx;
  _param.ny = ny;
  _param.nz = nz;
  _param.n_grid = nx * ny * nz;

  _param.boundary = boundary;

  // a depends on each camera's max_intensity / 2
  _param.a = Matrix<double>(n_cam, _param.n_grid, 125);
  for (int cam = 0; cam < n_cam; ++cam) {
    double init_val = camera_models[cam] ? camera_models[cam]->max_intensity / 2.0
                                         : 125.0;
    for (int g = 0; g < _param.n_grid; ++g)
      _param.a(cam, g) = init_val;
  }
  _param.b = Matrix<double>(n_cam, _param.n_grid, 1.5);
  _param.c = Matrix<double>(n_cam, _param.n_grid, 1.5);
  _param.alpha = Matrix<double>(n_cam, _param.n_grid, 0);

  setGrid();
}

void OTF::loadParam(std::string otf_file) {
  std::ifstream infile(otf_file);

  REQUIRE_CTX(infile.is_open(), ErrorCode::IOfailure,
              "Cannot load OTF from file ", otf_file);

  std::string line;
  std::stringstream file_content;
  while (std::getline(infile, line)) {
    size_t comment_pos = line.find("#");
    if (comment_pos > 0 && comment_pos < std::string::npos) {
      line.erase(comment_pos);
    } else if (comment_pos == 0) {
      continue;
    }

    file_content << line << '\t';
  }
  infile.close();

  file_content >> _param.n_cam;
  file_content.ignore();
  file_content >> _param.nx;
  file_content.ignore();
  file_content >> _param.ny;
  file_content.ignore();
  file_content >> _param.nz;
  file_content.ignore();
  file_content >> _param.n_grid;

  file_content >> _param.boundary.x_min;
  file_content.ignore();
  file_content >> _param.boundary.x_max;
  file_content.ignore();
  file_content >> _param.boundary.y_min;
  file_content.ignore();
  file_content >> _param.boundary.y_max;
  file_content.ignore();
  file_content >> _param.boundary.z_min;
  file_content.ignore();
  file_content >> _param.boundary.z_max;

  _param.a = Matrix<double>(_param.n_cam, _param.n_grid, file_content);
  _param.b = Matrix<double>(_param.n_cam, _param.n_grid, file_content);
  _param.c = Matrix<double>(_param.n_cam, _param.n_grid, file_content);
  _param.alpha = Matrix<double>(_param.n_cam, _param.n_grid, file_content);

  setGrid();
}

void OTF::saveParam(std::string otf_file) {
  std::ofstream outfile(otf_file);

  REQUIRE_CTX(outfile.is_open(), ErrorCode::IOfailure,
              "Cannot save OTF to file ", otf_file);

  outfile << "# Size: (n_cam,nx,ny,nz,n_grid)" << std::endl;
  outfile << _param.n_cam << ',' << _param.nx << ',' << _param.ny << ','
          << _param.nz << ',' << _param.n_grid << std::endl;

  outfile << "# Boundary: (xmin,xmax,ymin,ymax,zmin,zmax)" << std::endl;
  outfile << _param.boundary.x_min << ',' << _param.boundary.x_max << ','
          << _param.boundary.y_min << ',' << _param.boundary.y_max << ','
          << _param.boundary.z_min << ',' << _param.boundary.z_max << std::endl;

  outfile << "# a: (n_cam,n_grid)" << std::endl;
  _param.a.write(outfile);

  outfile << "# b: (n_cam,n_grid)" << std::endl;
  _param.b.write(outfile);

  outfile << "# c: (n_cam,n_grid)" << std::endl;
  _param.c.write(outfile);

  outfile << "# alpha: (n_cam,n_grid)" << std::endl;
  _param.alpha.write(outfile);
}

std::vector<double> OTF::getOTFParam(int cam_id, Pt3D const &pt3d) const {
  double pt3d_x =
      std::min(std::max(pt3d[0], _param.boundary.x_min), _param.boundary.x_max);
  double pt3d_y =
      std::min(std::max(pt3d[1], _param.boundary.y_min), _param.boundary.y_max);
  double pt3d_z =
      std::min(std::max(pt3d[2], _param.boundary.z_min), _param.boundary.z_max);

  // find out the limits of interpolation cube
  int x_id = std::max(
      0, (int)std::floor((pt3d_x - _param.boundary.x_min) / _param.dx));
  int y_id = std::max(
      0, (int)std::floor((pt3d_y - _param.boundary.y_min) / _param.dy));
  int z_id = std::max(
      0, (int)std::floor((pt3d_z - _param.boundary.z_min) / _param.dz));
  x_id = std::min(x_id, _param.nx - 2);
  y_id = std::min(y_id, _param.ny - 2);
  z_id = std::min(z_id, _param.nz - 2);

  AxisLimit grid_limit;

  grid_limit.x_min = _param.grid_x[x_id];
  grid_limit.x_max = _param.grid_x[x_id + 1];
  grid_limit.y_min = _param.grid_y[y_id];
  grid_limit.y_max = _param.grid_y[y_id + 1];
  grid_limit.z_min = _param.grid_z[z_id];
  grid_limit.z_max = _param.grid_z[z_id + 1];

  int i_000 = mapGridID(x_id, y_id, z_id);
  int i_100 = mapGridID(x_id + 1, y_id, z_id);
  int i_101 = mapGridID(x_id + 1, y_id, z_id + 1);
  int i_001 = mapGridID(x_id, y_id, z_id + 1);
  int i_010 = mapGridID(x_id, y_id + 1, z_id);
  int i_110 = mapGridID(x_id + 1, y_id + 1, z_id);
  int i_111 = mapGridID(x_id + 1, y_id + 1, z_id + 1);
  int i_011 = mapGridID(x_id, y_id + 1, z_id + 1);

  std::vector<double> a_value = {
      _param.a(cam_id, i_000), _param.a(cam_id, i_100), _param.a(cam_id, i_101),
      _param.a(cam_id, i_001), _param.a(cam_id, i_010), _param.a(cam_id, i_110),
      _param.a(cam_id, i_111), _param.a(cam_id, i_011)};
  std::vector<double> b_value = {
      _param.b(cam_id, i_000), _param.b(cam_id, i_100), _param.b(cam_id, i_101),
      _param.b(cam_id, i_001), _param.b(cam_id, i_010), _param.b(cam_id, i_110),
      _param.b(cam_id, i_111), _param.b(cam_id, i_011)};
  std::vector<double> c_value = {
      _param.c(cam_id, i_000), _param.c(cam_id, i_100), _param.c(cam_id, i_101),
      _param.c(cam_id, i_001), _param.c(cam_id, i_010), _param.c(cam_id, i_110),
      _param.c(cam_id, i_111), _param.c(cam_id, i_011)};
  std::vector<double> alpha_value = {
      _param.alpha(cam_id, i_000), _param.alpha(cam_id, i_100),
      _param.alpha(cam_id, i_101), _param.alpha(cam_id, i_001),
      _param.alpha(cam_id, i_010), _param.alpha(cam_id, i_110),
      _param.alpha(cam_id, i_111), _param.alpha(cam_id, i_011)};

  std::vector<double> res(5, 0); // a,b,c,alpha
  std::vector<double> pt_vec = {pt3d_x, pt3d_y, pt3d_z};
  res[0] = myMATH::triLinearInterp(grid_limit, a_value, pt_vec);
  res[1] = myMATH::triLinearInterp(grid_limit, b_value, pt_vec);
  res[2] = myMATH::triLinearInterp(grid_limit, c_value, pt_vec);
  double alpha = myMATH::triLinearInterp(grid_limit, alpha_value, pt_vec);
  res[3] = std::cos(alpha);
  res[4] = std::sin(alpha);

  return res;
}
