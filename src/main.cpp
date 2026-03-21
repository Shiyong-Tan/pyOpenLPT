// main.cpp
#include "STB.h"
#include <cstdio>
#include <cstdlib> // EXIT_SUCCESS / EXIT_FAILURE
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <system_error>
#include <vector>


#include "Camera.h"
#include "ImageIO.h"
#include "Matrix.h"
#include "ObjectInfo.h"
#include "STBCommons.h"
#include "error.hpp" // 含 FatalError


inline void init_omp_global(int n_threads = 0) {
  omp_set_dynamic(0); // 禁止运行时动态减线程

#if defined(_MSC_VER) && !defined(__clang__)
  omp_set_nested(1); // MSVC vcomp 没有 omp_set_max_active_levels
#else
  omp_set_max_active_levels(3); // 只保留2层并行
#endif

  const int hw = std::max(1, omp_get_num_procs());
  const int n = (n_threads > 0) ? std::min(n_threads, hw)
                                : hw; // 0/negative => use all threads

  if (!omp_in_parallel())
    omp_set_num_threads(n);
}

// -------------------- 核心逻辑 --------------------
int run_openlpt(const std::string &config_path) {
  namespace fs = std::filesystem;

  // 全局设置：所有后续浮点输出两位小数
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(2);

  std::string resolved_config_path = config_path;
  {
    std::error_code ec;
    const fs::path cfg_abs = fs::absolute(fs::path(config_path), ec);
    if (!ec) {
      resolved_config_path = cfg_abs.lexically_normal().string();
    }
  }

  BasicSetting basic_settings;
  if (!basic_settings.readConfig(resolved_config_path)) {
    std::cerr << "Error: Failed to read basic configuration from file: "
              << resolved_config_path << std::endl;
    return EXIT_FAILURE;
  }

  // set global thread
  init_omp_global(basic_settings._n_thread);

  try {
    // Create STB objects
    std::vector<STB> stb_objects;
    stb_objects.reserve(basic_settings._object_types.size());
    for (size_t i = 0; i < basic_settings._object_types.size(); ++i) {
      const std::string &type = basic_settings._object_types[i];
      const std::string &obj_config_path =
          basic_settings._object_config_paths[i];
      stb_objects.emplace_back(STB(basic_settings, type, obj_config_path));
    }

    // load previous tracks if needed
    if (basic_settings._load_track) {
      for (size_t i = 0; i < basic_settings._object_types.size(); ++i) {
        stb_objects[i].loadTracksAll(basic_settings._load_track_path,
                                     basic_settings._load_track_frame);
        std::cout << "Load previous tracks at frame "
                  << basic_settings._load_track_frame << "\n";
        std::cout << "  Active short tracks: "
                  << stb_objects[i]._short_track_active.size() << "\n";
        std::cout << "  Active long tracks: "
                  << stb_objects[i]._long_track_active.size() << "\n\n";
      }
    }

    // --- Prepare image IO ---
    if (basic_settings._cam_list.size() !=
        basic_settings._image_file_paths.size()) {
      std::cerr << "Error: #cam_list ("
                << basic_settings._cam_list.size()
                << ") != #image paths ("
                << basic_settings._image_file_paths.size() << ")\n";
      return EXIT_FAILURE;
    }

    std::vector<ImageIO> imgio_list;
    imgio_list.reserve(basic_settings._image_file_paths.size());
    for (const auto &path : basic_settings._image_file_paths) {
      ImageIO io;
      io.loadImgPath(basic_settings._config_root, path);
      imgio_list.push_back(io);
    }

    std::vector<Image> image_list(imgio_list.size());

    std::cout << "**************" << std::endl;
    std::cout << "OpenLPT start!" << std::endl;
    std::cout << "**************\n" << std::endl;

    int frame_start = basic_settings._frame_start;
    int frame_end = basic_settings._frame_end;
    int num_cams = static_cast<int>(imgio_list.size());
    if (basic_settings._load_track) {
      frame_start = basic_settings._load_track_frame + 1;
    }

    clock_t start = clock();
    for (int frame_id = frame_start; frame_id <= frame_end; ++frame_id) {
      for (int i = 0; i < num_cams; ++i) {
        image_list[i] = imgio_list[i].loadImg(frame_id);
      }
      for (auto &stb : stb_objects) {
        stb.processFrame(frame_id, image_list);
      }
    }
    clock_t end = clock();

    std::cout << "\nTotal time for STB: "
              << double(end - start) / CLOCKS_PER_SEC << "s\n"
              << std::endl;
    std::cout << "***************" << std::endl;
    std::cout << "OpenLPT finish!" << std::endl;
    std::cout << "***************" << std::endl;
  } catch (const FatalError &e) {
    std::cerr << "Program aborted due to error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (const std::exception &e) {
    std::cerr << "Unhandled std exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Unknown error occurred!" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

// -------------------- CLI 入口 --------------------
#ifdef OPENLPT_BUILD_CLI

int main(int argc, char *argv[]) {

  if (argc != 2) {
    std::cerr << "Usage: OpenLPT <config_file_path>" << std::endl;
    return EXIT_FAILURE;
  }

  return run_openlpt(argv[1]);
}

#endif
