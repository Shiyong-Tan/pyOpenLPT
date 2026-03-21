#include "Config.h"
#include "ImageIO.h" // ensure correct case

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <system_error>

namespace {
namespace fs = std::filesystem;

fs::path absolute_lexical(const fs::path &path) {
  std::error_code ec;
  const fs::path abs_path = fs::absolute(path, ec);
  if (ec) {
    return path.lexically_normal();
  }
  return abs_path.lexically_normal();
}

std::string resolve_path_from_config_root(const fs::path &config_root,
                                          const std::string &raw_path) {
  if (raw_path.empty()) {
    return raw_path;
  }
  const fs::path input(raw_path);
  const fs::path resolved =
      input.is_absolute() ? input : (config_root / input);
  return resolved.lexically_normal().generic_string();
}
} // namespace

// --------------------------- BasicSetting ---------------------------
bool BasicSetting::readConfig(const std::string &config_path) {
  const fs::path config_abs = absolute_lexical(fs::path(config_path));
  const fs::path config_root = config_abs.parent_path();
  _config_root = config_root.generic_string();
  if (!_config_root.empty()) {
    char back = _config_root.back();
    if (back != '/' && back != '\\') {
      _config_root.push_back('/');
    }
  }

  std::ifstream file(config_abs.string());
  if (!file.is_open()) {
    std::cerr << "Failed to open config file: " << config_abs.string()
              << std::endl;
    return false;
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    // trim spaces
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    // remove comments
    size_t comment_pos = line.find('#');
    if (comment_pos == 0)
      continue;
    if (comment_pos != std::string::npos)
      line.erase(comment_pos);

    // trim again
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    lines.push_back(line);
  }
  file.close();

  if (lines.size() < 11) {
    std::cerr << "Invalid config: too few lines." << std::endl;
    return false;
  }

  std::stringstream parser;
  int line_id = 0;

  // frame range
  parser.str(lines[line_id++]);
  std::getline(parser, line, ',');
  _frame_start = std::stoi(line);
  std::getline(parser, line, ',');
  _frame_end = std::stoi(line);
  parser.clear();

  // fps and threads
  _fps = std::stoi(lines[line_id++]);
  _n_thread = std::stoi(lines[line_id++]);

  // number of cameras
  _n_cam = std::stoi(lines[line_id++]);
  _cam_list.clear();
  _cam_list.reserve(static_cast<size_t>(_n_cam));
  for (int i = 0; i < _n_cam; ++i) {
    parser.str(lines[line_id++]);
    std::string cam_path;
    std::getline(parser, cam_path, ',');
    std::getline(parser, line, ',');
    int max_intensity = std::stoi(line);

    cam_path = resolve_path_from_config_root(config_root, cam_path);

    auto cam_model_st = CameraFactory::loadFromFile(cam_path);
    if (!cam_model_st) {
      THROW_FATAL_CTX(ErrorCode::InvalidCameraState,
                      "Failed to load camera model from:", cam_path);
    }
    auto cam_model = cam_model_st.value();
    cam_model->max_intensity = max_intensity;
    cam_model->is_active = true;
    _cam_list.emplace_back(std::move(cam_model));
    parser.clear();
  }

  // image file paths
  _image_file_paths.reserve(static_cast<size_t>(_n_cam));
  for (int i = 0; i < _n_cam; ++i) {
    _image_file_paths.push_back(
        resolve_path_from_config_root(config_root, lines[line_id++]));
  }

  // axis limits
  parser.str(lines[line_id++]);
  std::getline(parser, line, ',');
  _axis_limit.x_min = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.x_max = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.y_min = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.y_max = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.z_min = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.z_max = std::stod(line);
  parser.clear();

  // unit conversion and output path
  _voxel_to_mm = std::stod(lines[line_id++]);
  _output_path = resolve_path_from_config_root(config_root, lines[line_id++]);
  if (!_output_path.empty()) {
    char back = _output_path.back();
    if (back != '/' && back != '\\')
      _output_path.push_back('/');
  }

  // object types
  parser.str(lines[line_id++]);
  while (std::getline(parser, line, ',')) {
    if (!line.empty())
      _object_types.push_back(line);
  }
  parser.clear();

  // object config file paths
  _object_config_paths.reserve(_object_types.size());
  for (size_t i = 0; i < _object_types.size(); ++i) {
    _object_config_paths.push_back(
        resolve_path_from_config_root(config_root, lines[line_id++]));
  }

  // track loading
  parser.str(lines[line_id++]);
  std::getline(parser, line, ',');
  _load_track = std::stoi(line) != 0;
  _load_track_frame = _frame_start;
  if (_load_track) {
    std::getline(parser, line, ',');
    if (!line.empty())
      _load_track_frame = std::stoi(line);
    if (line_id < static_cast<int>(lines.size()))
      _load_track_path =
          resolve_path_from_config_root(config_root, lines[line_id++]);
  }
  parser.clear();

  // sanity checks
  if (_n_cam <= 0) {
    THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                    "Invalid n_cam in config: ", config_path + "\n");
  }
  if (_frame_end < _frame_start) {
    THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                    "Invalid frame range in config: ", config_path + "\n");
  }
  if ((int)_image_file_paths.size() != _n_cam) {
    THROW_FATAL_CTX(
        ErrorCode::InvalidArgument,
        "Image path count != n_cam in config: ", config_path + "\n");
  }
  if (_object_types.size() != _object_config_paths.size()) {
    THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                    "Object type count != config path count in config: ",
                    config_path + "\n");
  }
  if (_n_thread < 0)
    _n_thread = 0;

  return true;
}

// --------------------------- ObjectConfig (Common) ---------------------------
std::pair<std::vector<std::string>, int>
ObjectConfig::readCommonConfig(const std::string &filepath,
                               BasicSetting &settings) {
  std::ifstream file(filepath);
  REQUIRE_CTX(file.is_open(), ErrorCode::IOfailure,
              "Config Error: Cannot open file ", filepath + "\n");

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    // trim
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    // remove comments
    size_t commentpos = line.find('#');
    if (commentpos == 0)
      continue;
    if (commentpos != std::string::npos)
      line.erase(commentpos);

    // trim again
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    lines.push_back(line);
  }
  file.close();

  int id = 0;
  try {
    // STB params
    _stb_param._radius_search_obj =
        std::stod(lines[id++]) * settings._voxel_to_mm;
    _stb_param._n_initial_frames = std::stoi(lines[id++]);
    if (_stb_param._n_initial_frames < 4)
      _stb_param._n_initial_frames = 4; // n_initial_frames must be >= 4.
    _stb_param._radius_search_track =
        std::stod(lines[id++]) * settings._voxel_to_mm;

    // Shake params
    _shake_param._shake_width = std::stod(lines[id++]) * settings._voxel_to_mm;
    _shake_param._shakewidth_min = _shake_param._shake_width / 20;

    // Predictive field params
    _pf_param.limit = settings._axis_limit;
    _pf_param.nx = std::stoi(lines[id++]);
    _pf_param.ny = std::stoi(lines[id++]);
    _pf_param.nz = std::stoi(lines[id++]);
    _pf_param.r = std::stod(lines[id++]) * settings._voxel_to_mm;

    // IPR & shake loop
    _ipr_param.n_loop_ipr = std::stoi(lines[id++]);
    _shake_param._n_shake_loop = std::stoi(lines[id++]);
    _shake_param._thred_ghost = std::stod(lines[id++]);

    // StereoMatch params
    _sm_param.tol_2d_px = std::stod(lines[id++]);
    _sm_param.tol_3d_mm =
        std::stod(lines[id++]) * settings._voxel_to_mm; // in mm
    _sm_param.match_cam_count =
        std::min(_sm_param.match_cam_count,
                 settings._n_cam); // cannot be larger than n_cam
    _sm_param.limit = settings._axis_limit;

    _ipr_param.n_cam_reduced = std::stoi(lines[id++]);
    _ipr_param.n_loop_ipr_reduced = std::stoi(lines[id++]);

    _vsc_param._output_path = settings._output_path;

  } catch (...) {
    THROW_FATAL_CTX(ErrorCode::IOfailure,
                    "Config Error: Failed to parse common config in ",
                    filepath + "\n");
  }

  return {lines, id};
}

// --------------------------- TracerConfig ---------------------------
bool TracerConfig::readConfig(const std::string &filepath,
                              BasicSetting &settings) {
  auto [lines, id] = readCommonConfig(filepath, settings);

  try {
    _min_obj_int = std::stoi(lines[id++]);
    _radius_obj = std::stod(lines[id++]);

    // Initialize OTF
    _otf.loadParam(settings._n_cam, 5, 5, 5, settings._axis_limit,
                   settings._cam_list);
    _otf._output_path = settings._output_path;

    // Estimate OTF using first few frames
    std::vector<ImageIO> imgio_list;
    imgio_list.reserve(settings._n_cam);
    for (const auto &path : settings._image_file_paths) {
      ImageIO io;
      io.loadImgPath(settings._config_root, path);
      imgio_list.push_back(io);
    }

    int n_img_otf = 5;
    std::vector<Image> img_list(n_img_otf);
    std::cout << "Estimating OTF...\n";
    for (int i = 0; i < settings._n_cam; i++) {
      for (int j = 0; j < n_img_otf; j++) {
        img_list[j] = imgio_list[i].loadImg(settings._frame_start + j);
      }

      _otf.estimateUniformOTFFromImage(i, *this, img_list);
    }
  } catch (...) {
    THROW_FATAL_CTX(ErrorCode::IOfailure,
                    "Config Error: Failed to parse common config in ",
                    filepath + "\n");
  }

  return true;
}

// --------------------------- BubbleConfig ---------------------------
bool BubbleConfig::readConfig(const std::string &filepath,
                              BasicSetting &settings) {
  auto [lines, id] = readCommonConfig(filepath, settings);

  try {
    _radius_min = std::stoi(lines[id++]);
    _radius_max = std::stoi(lines[id++]);
    _sense = std::stod(lines[id++]);

  } catch (...) {
    THROW_FATAL_CTX(ErrorCode::IOfailure,
                    "Config Error: Failed to parse common config in ",
                    filepath + "\n");
  }

  _output_path = settings._output_path;

  if (settings._load_track) {
    if (!_bb_ref_img.loadRefImg(_output_path, settings._n_cam)) {
      std::cerr << "Warning: Failed to load bubble reference images from "
                << _output_path << std::endl;
    } else {
      std::cout << "Loaded bubble reference images from " << _output_path
                << std::endl;
    }
  }

  return true;
}

std::unique_ptr<Object3D> TracerConfig::creatObject3D(CreateArgs args) const {
  auto up = std::make_unique<Tracer3D>();

  if (args._proto)
    up->copyCommonFrom(*args._proto);
  if (args._pt_center)
    up->_pt_center = *args._pt_center;
  if (!args._obj2d_ready.empty())
    up->_obj2d_list = std::move(args._obj2d_ready);

  // Tracer radius selection
  if (args._r2d_px_tracer) {
    up->_r2d_px = *args._r2d_px_tracer;
  } else if (args._proto) {
    if (auto *tr = dynamic_cast<const Tracer3D *>(args._proto))
      up->_r2d_px = tr->_r2d_px;
    else
      up->_r2d_px = _radius_obj;
  } else {
    up->_r2d_px = _radius_obj;
  }

  up->_is_tracked = false;
  return up;
}

// --------------------------- BubbleConfig ---------------------------
std::unique_ptr<Object3D> BubbleConfig::creatObject3D(CreateArgs args) const {
  auto up = std::make_unique<Bubble3D>();

  if (args._proto)
    up->copyCommonFrom(*args._proto);
  if (args._pt_center)
    up->_pt_center = *args._pt_center;
  if (!args._obj2d_ready.empty())
    up->_obj2d_list = std::move(args._obj2d_ready);

  // Bubble radius selection
  if (args._r3d_bubble) {
    up->_r3d = *args._r3d_bubble;
  } else if (args._proto) {
    if (auto *bb = dynamic_cast<const Bubble3D *>(args._proto))
      up->_r3d = bb->_r3d;
  }
  if (up->_r3d <= 0.0 && args._compute_bubble_radius &&
      !up->_obj2d_list.empty()) {
    if (args._cam_list) {
      up->_r3d = Bubble::calRadiusFromCams(*args._cam_list, up->_pt_center,
                                           up->_obj2d_list);
    }
  }
  if (up->_r3d <= 0.0)
    up->_r3d = -1.0;

  up->_is_tracked = false;
  return up;
}
