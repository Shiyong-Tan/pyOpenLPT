#include "BubbleRefImg.h"
#include "Camera.h"
#include "ImageIO.h"
#include <filesystem>
#include <iostream>
#include <tiffio.h>

// Build bubble reference images (one per active camera) from the current 3D
// reconstruction and all raw 2D detections. Succeeds only if *every* active
// camera builds a template.
//
// Contract
//  - Returns true IFF every active camera produced a valid reference image.
//  - _img_Ref_list / _intRef_list are resized to cams.size(), aligned by camera
//  index.
//  - No ownership is taken; all inputs are read-only.
//  - Pixel windows use half-open ranges [min, max).
//
// Inputs
//  - objs_out      : reconstructed 3D objects (polymorphic). Only Bubble3D are
//  used.
//  - bb2d_list_all : all 2D detections per camera (polymorphic). Only Bubble2D
//  are used.
//  - cams          : camera array; only cameras with _is_active == true are
//  required to succeed.
//  - img_input     : current working images per camera (aligned to cams).
//  - r_thres       : candidate filter — a 3D bubble is selected only if every
//  active camera
//                    sees its 2D radius > r_thres.
//  - n_bb_thres    : minimum number of selected 3D bubbles required to proceed.
//
bool BubbleRefImg::calBubbleRefImg(
    const std::vector<std::unique_ptr<Object3D>> &objs_out,
    const std::vector<std::vector<std::unique_ptr<Object2D>>> &bb2d_list_all,
    const std::vector<std::shared_ptr<::Camera>> &camera_models,
    const std::vector<Image> &img_input,
    std::string output_folder, double r_thres, int n_bb_thres) {
  _img_Ref_list.clear();
  _intRef_list.clear();

  const int n_cam = static_cast<int>(camera_models.size());
  if (n_cam == 0)
    return false;
  if (static_cast<int>(img_input.size()) != n_cam)
    return false;
  if (static_cast<int>(bb2d_list_all.size()) != n_cam)
    return false;

  // NEW: all cameras must be active
  for (int cam = 0; cam < n_cam; ++cam) {
    if (!camera_models[cam] || !camera_models[cam]->is_active)
      return false;
  }

  // ---- Build non-owning pointer views: Bubble3D* and Bubble2D* ----
  std::vector<const Bubble3D *> bb3d_ptrs;
  bb3d_ptrs.reserve(objs_out.size());
  for (const auto &up : objs_out) {
    if (!up)
      continue;
    // no dynamic_cast: you guaranteed they are Bubble3D
    bb3d_ptrs.push_back(static_cast<const Bubble3D *>(up.get()));
  }
  if (bb3d_ptrs.empty())
    return false;

  std::vector<std::vector<const Bubble2D *>> bb2d_ptrs(n_cam);
  for (int cam = 0; cam < n_cam; ++cam) {
    const auto &vec = bb2d_list_all[cam];
    auto &out = bb2d_ptrs[cam];
    out.reserve(vec.size());
    for (const auto &u : vec) {
      if (!u)
        continue;
      // no dynamic_cast: you guaranteed they are Bubble2D
      out.push_back(static_cast<const Bubble2D *>(u.get()));
    }
  }

  // ---- 1) Select 3D bubbles: every camera must see a large enough 2D radius
  // ----
  std::vector<int> id_select;
  id_select.reserve(bb3d_ptrs.size());
  for (int i = 0; i < static_cast<int>(bb3d_ptrs.size()); ++i) {
    bool ok = true;
    for (int cam = 0; cam < n_cam; ++cam) {
      const Object2D *obj2d = bb3d_ptrs[i]->_obj2d_list[cam].get();
      if (!obj2d) {
        ok = false;
        break;
      }
      const auto *b2 = static_cast<const Bubble2D *>(obj2d);
      if (b2->_r_px <= r_thres) {
        ok = false;
        break;
      }
    }
    if (ok)
      id_select.push_back(i);
  }
  const int n_select = static_cast<int>(id_select.size());
  if (n_select <= n_bb_thres)
    return false;

  // ---- 2) Determine per-camera template size (max diameter among selected)
  // ----
  std::vector<double> dia_ref(n_cam, 0.0);
  for (int cam = 0; cam < n_cam; ++cam) {
    double dmax = 0.0;
    for (int j = 0; j < n_select; ++j) {
      const int id = id_select[j];
      const Object2D *obj2d = bb3d_ptrs[id]->_obj2d_list[cam].get();
      if (!obj2d)
        continue;
      const auto *b2 = static_cast<const Bubble2D *>(obj2d);
      dmax = std::max(dmax, 2.0 * b2->_r_px);
    }
    if (dmax <= 0.0)
      return false; // every camera must have usable 2D projections
    dia_ref[cam] = dmax;
  }

  // ---- 3) Allocate outputs aligned with cameras ----
  _img_Ref_list.assign(n_cam, Image{});
  _intRef_list.assign(n_cam, 0.0);

  // ---- 4) Build reference image for EVERY camera; any failure -> false ----
  for (int cam = 0; cam < n_cam; ++cam) {
    const int npix = static_cast<int>(std::round(dia_ref[cam]));
    if (npix <= 0) {
      _img_Ref_list.clear();
      _intRef_list.clear();
      return false;
    }

    _img_Ref_list[cam] = Image(npix, npix, 0.0);

    const int nrow = img_input[cam].getDimRow();
    const int ncol = img_input[cam].getDimCol();
    const double intensity_max = camera_models[cam]->max_intensity;

    std::vector<Image> bb_img_i(n_select);
    std::vector<double> max_peak(n_select, 0.0);
    std::vector<int> ok_crop(n_select, 0);

    // Collect non-overlapping crops and resize to npix × npix
    for (int j = 0; j < n_select; ++j) {
      const int id = id_select[j];
      const Object2D *obj2d = bb3d_ptrs[id]->_obj2d_list[cam].get();
      if (!obj2d)
        continue;
      const auto *b2 = static_cast<const Bubble2D *>(obj2d);

      const double r = b2->_r_px;
      const double xc = b2->_pt_center[0];
      const double yc = b2->_pt_center[1];

      // Strict overlap rejection against all 2D detections on this camera.
      bool overlap = false;
      for (const Bubble2D *q : bb2d_ptrs[cam]) {
        if (!q)
          continue;
        if (q == b2)
          continue; // same pointer (shared storage)
        const double dist = myMATH::dist(q->_pt_center, b2->_pt_center);
        if (dist < 0.25)
          continue; // nearly same location → self-like
        if (dist < 1.2 * (r + q->_r_px)) {
          overlap = true;
          break;
        } // stricter than touching
      }
      if (overlap)
        continue;

      // Half-open crop window [min, max)
      const int x_min = static_cast<int>(std::round(xc - r));
      const int x_max = static_cast<int>(std::round(xc + r)) + 1;
      const int y_min = static_cast<int>(std::round(yc - r));
      const int y_max = static_cast<int>(std::round(yc + r)) + 1;

      // Bounds check against image dimensions
      if (x_min < 0 || y_min < 0 || x_max > ncol || y_max > nrow)
        continue;

      const int dx = x_max - x_min;
      const int dy = y_max - y_min;
      const int sz = std::min(dx, dy);
      if (sz <= 0)
        continue;

      Image crop(sz, sz, 0.0);
      double peak = 0.0;

      for (int yy = 0; yy < sz; ++yy) {
        for (int xx = 0; xx < sz; ++xx) {
          const double dxr = (x_min + xx) - xc;
          const double dyr = (y_min + yy) - yc;
          if (dxr * dxr + dyr * dyr > (r + 1) * (r + 1)) {
            crop(yy, xx) = 0.0;
          } else {
            const double val = img_input[cam](y_min + yy, x_min + xx);
            crop(yy, xx) = val;
            if (val > peak)
              peak = val;
          }
        }
      }

      BubbleResize resizer;
      bb_img_i[j] = resizer.ResizeBubble(crop, npix, intensity_max);

      max_peak[j] = peak;
      ok_crop[j] = 1;
    }

    // Mean-peak check (must have at least one usable crop)
    double mean_peak = 0.0;
    int cnt = 0;
    for (int j = 0; j < n_select; ++j)
      if (ok_crop[j]) {
        mean_peak += max_peak[j];
        ++cnt;
      }
    if (cnt == 0) {
      _img_Ref_list.clear();
      _intRef_list.clear();
      return false;
    }
    mean_peak /= cnt;

    // Global precheck: how many resized crops pass the peak filter?
    int n_eff = 0;
    for (int j = 0; j < n_select; ++j) {
      if (ok_crop[j] && max_peak[j] > 0.8 * mean_peak)
        ++n_eff;
    }
    if (n_eff < n_bb_thres) {
      _img_Ref_list.clear();
      _intRef_list.clear();
      return false; // not enough qualified samples overall
    }

    // Average resized crops whose peak > 0.8 × mean_peak.
    // Enforce per-pixel coverage: navg >= n_bb_thres inside the inscribed
    // circle.
    const double core_margin =
        1.0; // shrink radius by 1px to avoid boundary artifacts
    const double cx = (npix - 1) / 2.0;
    const double cy = (npix - 1) / 2.0;
    const double rr =
        npix / 2.0 - core_margin; // effective radius for per-pixel coverage

    for (int yy = 0; yy < npix; ++yy) {
      for (int xx = 0; xx < npix; ++xx) {
        // const double d = std::hypot(yy - cy, xx - cx);
        // if (d >= rr) {
        //     _img_Ref_list[cam](yy, xx) = 0.0;
        //     continue; // outside the effective disk
        // }

        double acc = 0.0;
        int navg = 0;
        for (int j = 0; j < n_select; ++j) {
          if (ok_crop[j] && max_peak[j] > 0.8 * mean_peak) {
            acc += bb_img_i[j](yy, xx);
            ++navg;
          }
        }

        if (navg < n_bb_thres) {
          _img_Ref_list.clear();
          _intRef_list.clear();
          return false; // not enough contributing samples at this pixel
        }

        _img_Ref_list[cam](yy, xx) = acc / navg;
      }
    }

    // Template mean intensity within the FULL inscribed circle (rename vars to
    // avoid redecl)
    double sum = 0.0;
    int nsum = 0;
    const double cx2 = (npix - 1) / 2.0;
    const double cy2 = (npix - 1) / 2.0;
    const double rr2 = npix / 2.0;
    for (int row = 0; row < npix; ++row) {
      for (int col = 0; col < npix; ++col) {
        const double d = std::hypot(row - cy2, col - cx2);
        if (d < rr2) {
          sum += _img_Ref_list[cam](row, col);
          ++nsum;
        }
      }
    }
    _intRef_list[cam] = (nsum > 0) ? (sum / nsum) : 0.0;
  }

  _is_valid = true;

  // Save reference images if output folder is provided
  if (!output_folder.empty()) {
    if (!saveRefImg(output_folder, n_cam)) {
      std::cerr << "Warning: Failed to save bubble reference image to "
                << output_folder << std::endl;
    }
  }

  return true; // every camera succeeded
}

// -------------------- Persistence --------------------

bool BubbleRefImg::saveRefImg(std::string folder, int n_cam) const {
  if (!_is_valid)
    return false;
  namespace fs = std::filesystem;
  if (!fs::exists(folder))
    return false;

  // ensure folder ends with separator
  if (folder.back() != '/' && folder.back() != '\\')
    folder += "/";

  for (int i = 0; i < n_cam; ++i) {
    if (i >= static_cast<int>(_img_Ref_list.size()))
      break;
    std::string path = folder + "BubbleRefImg_" + std::to_string(i) + ".tif";

    // Check max intensity to decide bit depth
    double max_val = 0.0;
    int rows = _img_Ref_list[i].getDimRow();
    int cols = _img_Ref_list[i].getDimCol();
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        double val = _img_Ref_list[i](r, c);
        if (val > max_val)
          max_val = val;
      }
    }

    int bits = 32;
    if (max_val <= 255.0)
      bits = 8;
    else if (max_val <= 65535.0)
      bits = 16;

    _img_Ref_list[i].save(path, bits);
  }
  return true;
}

bool BubbleRefImg::loadRefImg(std::string folder, int n_cam) {
  namespace fs = std::filesystem;

  // ensure folder ends with separator
  if (folder.back() != '/' && folder.back() != '\\')
    folder += "/";

  _img_Ref_list.clear();
  _intRef_list.clear();
  _img_Ref_list.resize(n_cam);
  _intRef_list.resize(n_cam);

  for (int i = 0; i < n_cam; ++i) {
    std::string path = folder + "BubbleRefImg_" + std::to_string(i) + ".tif";
    if (!fs::exists(path))
      return false;

    TIFF *tif = TIFFOpen(path.c_str(), "r");
    if (!tif)
      return false;

    uint32 w = 0, h = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);

    if (w == 0 || h == 0) {
      TIFFClose(tif);
      return false;
    }

    Image img(h, w, 0.0);

    uint16 bits = 0;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);

    // buffer for one scanline
    tdata_t buf = _TIFFmalloc(TIFFScanlineSize(tif));
    if (!buf) {
      TIFFClose(tif);
      return false;
    }

    for (uint32 row = 0; row < h; row++) {
      TIFFReadScanline(tif, buf, row);

      // Handle different bit depths
      if (bits == 32) {
        uint32 *data = (uint32 *)buf;
        for (uint32 col = 0; col < w; col++) {
          img(row, col) = static_cast<double>(data[col]);
        }
      } else if (bits == 8) {
        uint8 *data = (uint8 *)buf;
        for (uint32 col = 0; col < w; col++) {
          img(row, col) = static_cast<double>(data[col]);
        }
      } else if (bits == 16) {
        uint16 *data = (uint16 *)buf;
        for (uint32 col = 0; col < w; col++) {
          img(row, col) = static_cast<double>(data[col]);
        }
      } else if (bits == 64) {
        uint64 *data = (uint64 *)buf;
        for (uint32 col = 0; col < w; col++) {
          img(row, col) = static_cast<double>(data[col]);
        }
      }
    }
    _TIFFfree(buf);
    TIFFClose(tif);

    _img_Ref_list[i] = std::move(img);

    // Recalculate Intensity (Mean within inscribed circle)
    double sum = 0.0;
    int nsum = 0;
    int npix = w; // assumed square
    const double cx2 = (npix - 1) / 2.0;
    const double cy2 = (npix - 1) / 2.0;
    const double rr2 = npix / 2.0;
    for (int row = 0; row < npix; ++row) {
      for (int col = 0; col < npix; ++col) {
        const double d = std::hypot(row - cy2, col - cx2);
        if (d < rr2) {
          sum += _img_Ref_list[i](row, col);
          ++nsum;
        }
      }
    }
    _intRef_list[i] = (nsum > 0) ? (sum / nsum) : 0.0;
  }

  _is_valid = true;
  return true;
}
