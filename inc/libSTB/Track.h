#ifndef TRACK_H
#define TRACK_H

#include <deque>
#include <vector>
#include <memory>    // unique_ptr in public API
#include <iostream>
#include <sstream>
#include <fstream>   // std::ifstream in public API
#include <algorithm>
#include <typeinfo>

#include "Config.h"
#include "STBCommons.h"
#include "Matrix.h"
#include "ObjectInfo.h"


class Track
{
public:
    std::vector<std::unique_ptr<const Object3D>> _obj3d_list; // 3D objects, content of objects shouldn't be altered
    std::vector<int> _t_list; // frame ID list

    bool _active = true;

    // Functions //
    Track() {};
    Track(std::unique_ptr<Object3D> obj3d, int t);
    NONCOPYABLE_MOVABLE(Track); //Track cannot be copied, but can be moved
    ~Track() {};

    // Add a new point to the Track
    // only non-fake points should be added to the track
    void addNext(std::unique_ptr<Object3D> obj3d, int t);

    // write the track to a file
    void saveTrack(std::ostream& output, int track_id);

    // Read from the current file position: consume all consecutive rows
    // that belong to the same TrackID. Stop (and rewind one line) when the ID changes.
    void loadTrack(std::ifstream& fin, const ObjectConfig& cfg,
                   const std::vector<std::shared_ptr<Camera>>& camera_models);

};


// Define TrackCloud class for KD-tree

struct TrackCloud 
{
    std::deque<Track> const& _track_list;  // 3D points
    TrackCloud(std::deque<Track> const& track_list) : _track_list(track_list) {}

    // Must define the interface required by nanoflann
    inline size_t kdtree_get_point_count() const { return _track_list.size(); }
    inline float kdtree_get_pt(const size_t idx, int dim) const 
    { return _track_list[idx]._obj3d_list[_track_list[idx]._t_list.size() - 2]->_pt_center[dim]; } 
    // note: use the point before the last point, this is for linking short tracks in STB

    // Bounding box (not needed for standard KD-tree queries)
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

#endif
