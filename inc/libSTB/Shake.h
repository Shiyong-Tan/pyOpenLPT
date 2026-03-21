#ifndef SHAKE_H
#define SHAKE_H

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits> // ★ for std::underlying_type_t
#include <vector>

#include "Camera.h"
#include "Config.h"
#include "ObjectInfo.h"
#include "STBCommons.h"

class ShakeStrategy;
class TracerShakeStrategy;
class BubbleShakeStrategy;
class ROIInfo; // a pack to the region of interest (ROI) for an object, which
               // contains: ROI image, correlation map
enum class ObjFlag : uint8_t { None = 0, Ghost = 1 << 0, Repeated = 1 << 1 };
inline ObjFlag operator|(ObjFlag a, ObjFlag b) {
  using U = std::underlying_type_t<ObjFlag>;
  return static_cast<ObjFlag>(static_cast<U>(a) | static_cast<U>(b));
}
inline ObjFlag &operator|=(ObjFlag &a, ObjFlag b) {
  a = (a | b);
  return a;
}

class Shake {
public:
  Shake(const std::vector<std::shared_ptr<Camera>> &camera_models,
        const ObjectConfig &obj_cfg);
  ~Shake() = default;

  // shake objects, objs will be updated, return a vector denoting ghost objects
  // or repeated objects
  std::vector<ObjFlag> runShake(std::vector<std::unique_ptr<Object3D>> &objs,
                                const std::vector<Image> &img_orig);

  // get updated residual image, this is for next loop of IPR on residual image
  std::vector<Image>
  calResidualImage(const std::vector<std::unique_ptr<Object3D>> &objs,
                   const std::vector<Image> &img_orig,
                   const std::vector<ObjFlag> *flags = nullptr) {
    calResidueImage(objs, img_orig, true,
                    flags); // output_ipr = true: all negative value should be
                            // set to 0, and calculate for all cameras
    return _img_res_list;
  };

private:
  const std::vector<std::shared_ptr<Camera>> &_cam_list;
  const ObjectConfig &_obj_cfg;
  std::vector<Image> _img_res_list; // residual image
  std::vector<double> _score_list;  // shaking score for objects, which is
                                    // updated for every loop of shake
  std::unique_ptr<ShakeStrategy> _strategy;

  // calculate residual image list: original image - project image
  // if non_negative = true, then replace negative value as 0,
  // flags is the flag marking ghost and repeated objects, which are not
  // consider when calculating residual image
  void calResidueImage(const std::vector<std::unique_ptr<Object3D>> &objs,
                       const std::vector<Image> &img_orig,
                       bool output_ipr = false,
                       const std::vector<ObjFlag> *flags = nullptr);

  // prepare the information for the shaked object
  // calculate the boundary of ROI
  PixelRange calROIBound(int id_cam, double x_center, double y_center,
                         double dx, double dy) const;
  std::vector<ROIInfo> buildROIInfo(const Object3D &obj,
                                    const std::vector<Image> &img_orig) const;

  // shake one object, input delta is the shake width, return the score for
  // updating, obj is also updated
  double shakeOneObject(Object3D &obj, std::vector<ROIInfo> &ROI_info,
                        double delta, const std::vector<bool> &shake_cam) const;

  // calculate the score of object based on intensity
  double calObjectScore(Object3D &obj, std::vector<ROIInfo> &ROI_info,
                        const std::vector<bool> &shake_cam) const;

  // mark repeated objects
  std::vector<bool>
  markRepeatedObj(const std::vector<std::unique_ptr<Object3D>> &objs);

  FRIEND_DEBUG(Shake); // for debugging private members
};

class ROIInfo {
  // each camera has one ROIInfo
public:
  PixelRange _ROI_range;

  ROIInfo() = default; // Initialize augimg, corr_map, range
  ~ROIInfo() = default;

  // allocate augmented image
  void allocAugImg();
  // allocate correlation map
  void allocCorrMap();

  // get the whole augmented image
  const Image &getAugImg() const { return _ROI_augimg; };

  // for indexing ROI:
  bool inRange(int row, int col) const;
  bool mapToLocal(int row, int col, int &i, int &j) const;
  double &aug_img(int row, int col);
  double aug_img(int row, int col) const;
  double &corr_map(int row, int col);
  double corr_map(int row, int col) const;

private:
  Image _ROI_augimg;  // image of ROI
  Image _ROI_corrmap; // correlation map of ROI
};

// class for ShakeStrategy, which is inherited by different object strategy
class ShakeStrategy {
protected:
  const std::vector<std::shared_ptr<Camera>> &_cam_list;
  const ObjectConfig &_obj_cfg;

public:
  explicit ShakeStrategy(
                         const std::vector<std::shared_ptr<Camera>> &camera_models,
                         const ObjectConfig &obj_cfg)
      : _cam_list(camera_models), _obj_cfg(obj_cfg) {}
  virtual ~ShakeStrategy() = default;

  // 2D projection for an object
  virtual double project2DInt(const Object3D &obj, int id_cam, int row,
                              int col) const = 0;

  // calculate the size of ROI, return (dx, dy)
  struct ROISize {
    double dx;
    double dy;
  }; // dx, dy are half size of the object in x and y direction
  virtual ROISize calROISize(const Object3D &obj, int id_cam) const = 0;

  // calculate the shaking residue
  virtual double calShakeResidue(const Object3D &obj_candidate,
                                 std::vector<ROIInfo> &roi_info,
                                 const std::vector<bool> &shake_cam) const = 0;

  // obtain cameras that can be used for shaking(true for use, false for not to
  // use), remove those in which objects are out of range or blocked
  virtual std::vector<bool>
  selectShakeCam(const Object3D &obj, const std::vector<ROIInfo> &roi_info,
                 const std::vector<Image> &imgOrig) const {
    return std::vector<bool>(_cam_list.size(), true);
  };

  // additional check for object after shaking, for example, bubble needs to
  // check the image correlation
  virtual bool additionalObjectCheck(const Object3D &obj,
                                     std::vector<ROIInfo> &roi_info,
                                     const std::vector<bool> &shakecam) const {
    return true;
  };
};

class TracerShakeStrategy : public ShakeStrategy {
public:
  using ShakeStrategy::ShakeStrategy; // <- inherit (cams, obj_cfg) construction
  // Gaussian projection model
  double gaussIntensity(int x, int y, Pt2D const &pt2d,
                        std::vector<double> const &otf_param) const;

  double project2DInt(const Object3D &obj, int id_cam, int row,
                      int col) const override;

  ROISize calROISize(const Object3D &obj, int id_cam) const override;

  double calShakeResidue(const Object3D &obj_candidate,
                         std::vector<ROIInfo> &roi_info,
                         const std::vector<bool> &shake_cam) const override;
};

class BubbleShakeStrategy : public ShakeStrategy {
public:
  using ShakeStrategy::ShakeStrategy; // <- inherit (cams, obj_cfg) construction

  double project2DInt(const Object3D &obj, int id_cam, int row,
                      int col) const override;

  ROISize calROISize(const Object3D &obj, int id_cam) const override;

  std::vector<bool>
  selectShakeCam(const Object3D &obj, const std::vector<ROIInfo> &roi_info,
                 const std::vector<Image> &imgOrig) const override;

  // calcualte the image crosscorrelation at location (x,y)
  double getImgCorr(ROIInfo &roi_info, const int x, const int y,
                    const Image &ref_img) const;

  double calShakeResidue(const Object3D &obj_candidate,
                         std::vector<ROIInfo> &roi_info,
                         const std::vector<bool> &shake_cam) const override;

  bool additionalObjectCheck(const Object3D &obj,
                             std::vector<ROIInfo> &roi_info,
                             const std::vector<bool> &shake_cam) const override;
};

#endif // !SHAKE_H
