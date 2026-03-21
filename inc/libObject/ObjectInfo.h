#ifndef OBJECTINFO_H
#define OBJECTINFO_H

#include "Camera.h"
#include "Matrix.h"
#include "STBCommons.h"
#include <memory> // for std::unique_ptr
#include <sstream>
#include <vector>


// ============================== 2D object classes
// ==============================
class Object2D {
public:
  Pt2D _pt_center;

  Object2D() = default;
  // Copy constructor
  Object2D(const Object2D &object) : _pt_center(object._pt_center) {}
  // Constructor with center point
  Object2D(const Pt2D &pt_center) : _pt_center(pt_center) {}

  virtual ~Object2D() = default;

  // clone method for polymorphic behavior
  virtual std::unique_ptr<Object2D> clone() const = 0;
};

class Tracer2D : public Object2D {
public:
  double _r_px = 2; // [px], for shaking

  Tracer2D() = default;
  Tracer2D(const Tracer2D &tracer) : Object2D(tracer), _r_px(tracer._r_px) {}
  Tracer2D(const Pt2D &pt_center) : Object2D(pt_center) {}

  ~Tracer2D() override = default;

  ENABLE_CLONE(Tracer2D, Object2D);
};

class Bubble2D : public Object2D {
public:
  double _r_px = 2; // [px], for shaking

  Bubble2D() = default;
  Bubble2D(const Bubble2D &bubble) : Object2D(bubble), _r_px(bubble._r_px) {}
  Bubble2D(const Pt2D &pt_center, double r_px)
      : Object2D(pt_center), _r_px(r_px) {}

  ~Bubble2D() override = default;

  ENABLE_CLONE(Bubble2D, Object2D);
};

// ============================== 3D object classes
// ==============================
class Object3D {
public:
  Pt3D _pt_center; // center of the object in world coordinate [mm]
  std::vector<std::unique_ptr<Object2D>>
      _obj2d_list; // 2D projections of the object in different cameras, this
                   // can be specialized to different types of 2D objects
  bool _is_tracked = false;

  Object3D() = default;

  // forbid copy constructor to prevent slicing since _pt2d_list is a vector of
  // unique_ptr enable move semantics
  NONCOPYABLE_MOVABLE(Object3D);

  // Constructor with center point
  Object3D(const Pt3D &pt_center) : _pt_center(pt_center) {}

  // Copy only lightweight/common fields; do NOT deep-copy ownership fields.
  void copyCommonFrom(const Object3D &src) {
    _pt_center = src._pt_center;
    _is_tracked = src._is_tracked;
  }

  // calculate the 2D information (_obj2d_list)
  void projectObject2D(
      const std::vector<std::shared_ptr<Camera>> &camera_models);

  // to check whether the object is seen by more than 2 cameras;
  bool isReconstructable(
      const std::vector<std::shared_ptr<Camera>> &camera_models);

  // function to output the 3D object information
  // output: output stream to save the bubble information
  virtual void
  saveObject3D(std::ostream &out) const = 0; // pure virtual function, must be
                                             // implemented in derived classes

  // function to read the 3D object from input stream
  // Tracer: reads 3 fields (X,Y,Z). Bubble: reads 4 fields (X,Y,Z,R3D).
  // 2D projections are intentionally ignored; they will be projected later.
  virtual void loadObject3D(std::istream &in) = 0;

  virtual ~Object3D() = default;

protected:
  // subclasses must initialize the type of _pt2d_list
  // this function is used in projectObject2D to create a 2D object of the
  // specific type
  virtual std::unique_ptr<Object2D>
  create2DObject() const = 0; // = 0 means pure virtual function, must be
                              // implemented in derived classes

  // additional 2D projection, for example, bubbles need to project the radius
  virtual void additional2DProjection(
      const std::vector<std::shared_ptr<Camera>> &camera_models) = 0;
};

class Tracer3D : public Object3D {
public:
  double _r2d_px = 2; // [px], for shaking

  Tracer3D() = default;
  NONCOPYABLE_MOVABLE(Tracer3D);
  Tracer3D(const Pt3D &pt_center) : Object3D(pt_center) {}
  Tracer3D(const Pt3D &pt_center, const double r2d_px)
      : Object3D(pt_center), _r2d_px(r2d_px) {}

  // save the 3D tracer to a file
  void saveObject3D(std::ostream &out) const override;

  void loadObject3D(std::istream &in) override;

  ~Tracer3D() override = default;

protected:
  // Create a 2D object of type Tracer2D
  std::unique_ptr<Object2D> create2DObject() const override {
    return std::make_unique<Tracer2D>();
  }

  void additional2DProjection(
      const std::vector<std::shared_ptr<Camera>> &camera_models) override;
};

class Bubble3D : public Object3D {
public:
  double _r3d = -1; // [mm], bubble radius in 3D

  Bubble3D() = default;
  NONCOPYABLE_MOVABLE(Bubble3D);
  Bubble3D(const Pt3D &pt_center, double r3d)
      : Object3D(pt_center), _r3d(r3d) {}

  // save the 3D bubble to a file
  void saveObject3D(std::ostream &output) const override;

  void loadObject3D(std::istream &in) override;

  ~Bubble3D() override = default;

protected:
  // Create a 2D object of type Bubble2D
  std::unique_ptr<Object2D> create2DObject() const override {
    return std::make_unique<Bubble2D>();
  }

  void additional2DProjection(
      const std::vector<std::shared_ptr<Camera>> &camera_models) override;
};

// useful functions for spherical bubbles
namespace Bubble {

double calRadiusFromOneCam(const Camera &cam, const Pt3D &X, double r_px);

double cal2DRadius(const Camera &cam, const Pt3D &X, double R);

// Forward projection (predict image radius from a 3D radius)
// Same dispatch as above. Return NaN if unsupported/invalid.
double
calRadiusFromCams(const std::vector<std::shared_ptr<Camera>> &camera_models,
                  const Pt3D &X,
                  const std::vector<std::unique_ptr<Object2D>> &obj2d_list);

// Cross-view consistency gate (early check)
// Uses calRadiusFromOneCam() per view, then gates by max(R_i)/min(R_i) ≤
// max_spread. Policy: if inputs insufficient (e.g., <2 views / size mismatch),
// return true (permissive).
bool checkRadiusConsistency(const Pt3D &X,
                            const std::vector<const Camera *> &cams,
                            const std::vector<const Object2D *> &obj2d_by_cam,
                            double tol_2d_px, double tol_3d);

} // namespace Bubble

// ============================== Obj3dCloud for KD-tree
// ============================== Define Obj3dCloud class for KD-tree NOTE:
// Stores a const reference to the owning container of unique_ptr<Object3D>.
//       Objects must outlive this adapter. Mixed derived types
//       (Tracer3D/Bubble3D) are fine.
struct Obj3dCloud {
  const std::vector<std::unique_ptr<Object3D>>
      &_obj3d_list; // 3D points, view only (no ownership)

  explicit Obj3dCloud(const std::vector<std::unique_ptr<Object3D>> &obj3d_list)
      : _obj3d_list(obj3d_list) {}

  // Must define the interface required by nanoflann
  inline size_t kdtree_get_point_count() const { return _obj3d_list.size(); }

  // Return coordinate 'dim' (0/1/2) of point 'idx'
  inline double kdtree_get_pt(const size_t idx, int dim) const {
    // _pt_center is assumed indexable like _pt_center[dim]
    return _obj3d_list[idx]->_pt_center[dim];
  }

  // Bounding box (not needed for standard KD-tree queries)
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

#endif
