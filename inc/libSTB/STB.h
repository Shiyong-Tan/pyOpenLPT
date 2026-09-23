#ifndef STB_H
#define STB_H

#include "nanoflann.hpp"
#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <time.h>
#include <typeinfo>
#include <vector>

#include "Camera.h"
#include "Config.h"
#include "IPR.h"
#include "ImageIO.h"
#include "Matrix.h"
#include "OTF.h"
#include "ObjectFinder.h"
#include "ObjectInfo.h"
#include "PredField.h"
#include "Shake.h"
#include "StereoMatch.h"
#include "Track.h"
#include "VSC.h"

class STB {
public:
  std::deque<Track> _short_track_active;
  std::deque<Track> _long_track_active;
  std::deque<Track> _long_track_inactive; // only save very long tracks
  std::deque<Track> _exit_track;

  // FUNCTIONS //
  STB(const BasicSetting &setting, const std::string &type,
      const std::string &obj_cfg_path);

  // Process STB on frame frame_id
  // img_list: images for current frame
  // return img_list: residue images
  void processFrame(int frame_id, std::vector<Image> &img_list);

  // Load tracks
  void loadTracks(const std::string &file, std::deque<Track> &tracks);

  // save tracks of one status
  void saveTracks(std::string const &file, std::deque<Track> &tracks);
  void saveTracksCheckpoint(std::string const &file,
                            std::deque<Track> &tracks);

  // save all tracks at any status
  void saveTracksAll(std::string const &folder, int frame);

  // load tracks, only active long tracks and active short tracks are needed to
  // load
  void loadTracksAll(std::string const &folder, int frame);

  // Frame-boundary checkpoints contain every in-memory track collection and
  // the lossless Bubble reference state. They remain separate from scientific
  // result snapshots so pausing does not change result-file segmentation.
  bool canCheckpoint(int frame) const;
  void saveCheckpoint(std::string const &checkpoint_root, int frame);

private:
  BasicSetting _basic_setting; // basic settings such as frame range, fps, etc.
  std::unique_ptr<ObjectConfig>
      _obj_config; // settings for IPR, Tracking, Shaking for different object
                   // types
  VSC _vsc;        // Volume Self Calibration module

  std::vector<std::vector<std::unique_ptr<Object3D>>>
      _ipr_candidate; // objects from IPR for building tracks

  using KDTreeObj3d = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, Obj3dCloud>, Obj3dCloud,
      3  // dimensionality
      >; // KD-tree is used for fast nearest neighbor search in 3D space

  using KDTreeTrack = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, TrackCloud>, TrackCloud,
      3 // dimensionality
      >;

  struct LinkCandidate {
    int id = UNLINKED;
    double cost = std::numeric_limits<double>::infinity();
  };

  // Track based on velocity field
  void runInitPhase(int frame_id, std::vector<Image> &img_list);

  // Track based on STB
  void runConvPhase(int frame_id, std::vector<Image> &img_list);

  // initialize or connect tracks based on the predicted field
  void buildTrackFromPredField(int frame_id, const PredField *pf);

  // predict the next location
  std::unique_ptr<Object3D> predictNext(const Track &tr) const;

  // find nearest neighbor around a position
  LinkCandidate findNN(KDTreeObj3d const &tree_obj3d,
                       Pt3D const &pt3d_est, double radius) const;

  // check whether objects found by IPR are repeated with the last point in
  // active long tracks
  std::vector<ObjFlag>
  checkRepeat(const std::vector<std::unique_ptr<Object3D>> &objs) const;

  // int linkShortTrack (Track<T3D> const& track, std::vector<T3D> const&
  // obj3d_list, int n_iter);
  LinkCandidate linkShortTrack(Track const &track, int n_iter,
                               KDTreeObj3d const &tree_obj3d,
                               KDTreeTrack const &tree_track);

  bool checkLinearFit(Track const &track);

  FRIEND_DEBUG(STB);
};

#endif
