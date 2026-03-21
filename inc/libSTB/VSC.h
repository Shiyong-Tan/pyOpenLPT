/**
 * @file VSC.h
 * @brief Volume Self Calibration (VSC) module for OpenLPT.
 *
 * This module implements Volume Self Calibration, a method to refine camera
 * extrinsic parameters using the geometric consistency of tracked particles
 * (Tracers or Bubbles). It collects reliable track points during the processing
 * loop and performs Levenberg-Marquardt optimization to minimize reprojection
 * error.
 *
 * Features:
 * - Online Accumulation: Collects data frame-by-frame from reliable tracks.
 * - Strategy Pattern: Handles different object types (Tracer vs Bubble)
 *   polymorphically without explicit if/else branching.
 * - Spatial Binning: Ensures uniform spatial distribution of calibration
 *   points across the measurement volume.
 * - OTF Extension: Optionally updates Optical Transfer Function (Gaussian
 * shape) parameters for Tracers based on collected intensity patches.
 *
 * Algorithm Overview:
 * 1. accumulate(): Called each frame to collect isolated, high-quality 3D
 * points.
 * 2. isReady(): Returns true when enough points are collected.
 * 3. runVSC(): Optimizes camera extrinsics using Levenberg-Marquardt.
 * 4. runOTF(): (Optional) Fits Gaussian OTF parameters from stored patches.
 * 5. reset(): Clears buffer after successful calibration.
 */

#ifndef LIBSTB_VSC_H
#define LIBSTB_VSC_H

#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

// Project headers - order matters for proper type resolution
#include "Camera.h"
#include "Config.h"     // Uses Camera, ObjectInfo
#include "ImageIO.h"    // Uses Image
#include "Matrix.h"     // Must come first (defines Pt2D, Pt3D, Image)
#include "ObjectInfo.h" // Defines Object2D, Object3D
#include "Track.h"      // Uses Object3D

// ========================================================================
// Strategy Pattern (External Classes)
// ========================================================================

/**
 * @brief Abstract base strategy for object-specific VSC logic.
 */
class VSCStrategy {
public:
  virtual ~VSCStrategy() = default;
  virtual double getObject2DSize(const Object2D &obj) const = 0;
};

/**
 * @brief Strategy implementation for Tracers.
 */
class VSCTracerStrategy : public VSCStrategy {
public:
  double getObject2DSize(const Object2D &obj) const override {
    const auto *tr = static_cast<const Tracer2D *>(&obj);
    return tr ? 2.0 * tr->_r_px : 0;
  }
};

/**
 * @brief Strategy implementation for Bubbles.
 */
class VSCBubbleStrategy : public VSCStrategy {
public:
  double getObject2DSize(const Object2D &obj) const override {
    const auto *bb = static_cast<const Bubble2D *>(&obj);
    return bb ? 2.0 * bb->_r_px : 0;
  }
};

// ========================================================================
// Main VSC Class
// ========================================================================

/**
 * @brief Main Volume Self Calibration class.
 *
 * Manages the accumulation of calibration points from tracked particles
 * and the optimization of camera extrinsic parameters.
 *
 * Usage:
 * 1. Call configure() to set VSC parameters.
 * 2. Call accumulate() each frame during STB processing.
 * 3. When isReady() returns true, call runVSC() to optimize cameras.
 * 4. Optionally call runOTF() to update OTF parameters.
 * 5. Call reset() after successful calibration.
 */
class VSC {
public:
  // ========================================================================
  // Data Structures
  // ========================================================================
  /**
   * @brief OTF parameters identified from a single observation.
   *     //  2D projection point: (x0, y0)
   *     //  xx =  (x-x0) * cos(alpha) + (y-y0) * sin(alpha)
   *     //  yy = -(x-x0) * sin(alpha) + (y-y0) * cos(alpha)
   *     //  img[y,x] = a * exp(-b * xx^2 - c * yy^2)
   *     // default: a = 125, b = 1.5, c = 1.5, alpha = 0
   */
  struct OTFParams {
    double a = 125;
    double b = 1.5;
    double c = 1.5;
    double alpha = 0; // Orientation angle
    bool valid = false;
  };

  /**
   * @brief A single 2D observation of a calibration point in one camera.
   */
  struct Observation {
    int _cam_id;           ///< Camera ID
    Pt2D _meas_2d;         ///< Measured 2D center [px]
    Pt2D _proj_2d;         ///< Projected 2D center (from 3D) [px]
    double _quality_score; ///< Detection quality (0-1)
    double _obj_radius;    ///< Object radius [px]
    OTFParams _otf_params; ///< Identified OTF parameters (Tracer only)

    // For OTF Verification
    Matrix<double> _roi_intensity; ///< Raw intensity crop for validation
  };

  /**
   * @brief A 3D calibration point with its 2D observations.
   */
  struct CalibrationPoint {
    Pt3D _pos_3d;                  ///< Triangulated 3D world position
    int _frame_id;                 ///< Frame where this point was collected
    std::vector<Observation> _obs; ///< 2D observations in all visible cameras
  };

  // ========================================================================
  // Main Methods
  // ========================================================================

  VSC() = default;
  ~VSC() = default;
  VSC(VSC &&) = default;
  VSC &operator=(VSC &&) = default;
  VSC(const VSC &) = delete;
  VSC &operator=(const VSC &) = delete;

  /**
   * @brief Configure the VSC module parameters.
   * @param cfg Configuration struct with all parameters.
   */
  void configure(const VSCParam &cfg);

  /**
   * @brief Initialize strategy based on object type.
   * @param obj_cfg The current object configuration.
   */
  void initStrategy(const ObjectConfig &obj_cfg);

  /**
   * @brief Accumulate reliable calibration points from the current frame.
   *
   * This function:
   * 1. Detects 2D objects in all camera images.
   * 2. Builds KD-trees for fast neighbor queries (isolation check).
   * 3. For each long-enough track, checks visibility and isolation in all
   * cameras.
   * 4. Stores qualified 3D points with their 2D measurements.
   *
   * @param frame_id Current frame index.
   * @param active_tracks List of active tracks to sample from.
   * @param images Current images from all cameras.
   * @param camera_models Current camera parameters.
   * @param obj_cfg Current object configuration for ObjectFinder.
   */
  void accumulate(int frame_id, const std::deque<Track> &active_tracks,
                  const std::vector<Image> &images,
                  const std::vector<std::shared_ptr<Camera>> &camera_models,
                  const ObjectConfig &obj_cfg);

  /**
   * @brief Check if enough data has been collected to run calibration.
   * @return true if buffer size >= min_points_to_trigger.
   */
  bool isReady() const;

  /**
   * @brief Get number of accumulated calibration points.
   */
  size_t getBufferSize() const { return _buffer.size(); }

  /**
   * @brief Run Volume Self Calibration to optimize camera extrinsics.
   *
   * Uses Levenberg-Marquardt optimization to minimize reprojection error.
   * For each camera independently:
   * 1. Collects all 3D-2D correspondences.
   * 2. Computes Jacobian numerically (6 DOF: rotation + translation).
   * 3. Iteratively updates parameters if error reduces.
   * 4. Only commits changes if final error < original error.
   *
   * @param camera_models [in/out] Camera parameters to optimize.
   * @return true if any camera was updated.
   */
  bool runVSC(std::vector<std::shared_ptr<Camera>> &camera_models);

  /**
   * @brief Run OTF Calibration to optimize Tracer shape parameters.
   *
   * Fits 2D Gaussian models (I = a * exp(-b*dx^2 - c*dy^2)) to stored
   * intensity patches. Uses linear least squares in log domain.
   * Results are spatially binned and averaged per OTF grid cell.
   *
   * @param tracer_cfgs [in/out] Tracer configurations to update.
   * @return true if parameters were updated.
   */
  bool runOTF(std::vector<TracerConfig> &tracer_cfgs);

  /**
   * @brief Clear all accumulated calibration points.
   *
   * Should be called after a successful calibration update to collect
   * fresh data with the new camera parameters.
   */
  void reset();

private:
  VSCParam _cfg;                          ///< Current configuration
  std::vector<CalibrationPoint> _buffer;  ///< Storage for calibration points
  std::unique_ptr<VSCStrategy> _strategy; ///< Object-type strategy

  // ----- Grid Balancing State -----
  Pt3D _grid_min;                             ///< Minimum corner of grid bounds
  Pt3D _grid_max;                             ///< Maximum corner of grid bounds
  bool _grid_initialized = false;             ///< Whether grid bounds are set
  std::unordered_map<int, int> _voxel_counts; ///< Count of points in each voxel

  /**
   * @brief Compute voxel index for a 3D point.
   * @param pt 3D point in world coordinates.
   * @return Voxel index, or -1 if grid not initialized.
   */
  int computeVoxelIndex(const Pt3D &pt) const;

  /**
   * @brief Update grid boundaries and re-balance existing points if needed.
   *
   * Checks if the new point falls outside current grid. If so, expands grid,
   * clears voxel counts, and re-computes voxel IDs for all stored points.
   *
   * @param pt New 3D point to accommodate.
   */
  void updateGridAndRebalance(const Pt3D &pt);

  /**
   * @brief Estimate OTF parameters from an image patch using moments.
   *
   * @param img Image containing the object.
   * @param center Measured 2D center of the object.
   * @param half_w Half-width of the patch to process.
   * @return OTFParams struct with a, b, c, alpha.
   */
  OTFParams estimateOTFParams(const Matrix<double> &roi, const Pt2D &center_rel,
                              double obj_radius) const;
};

#endif // LIBSTB_VSC_H
