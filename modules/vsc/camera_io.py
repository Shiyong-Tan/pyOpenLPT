"""
Camera Parameter I/O utilities for VSC.
Handles reading and writing camera parameters in OpenLPT format.
"""
import numpy as np
import cv2
import os
import re


def parse_camera_file(file_path: str) -> dict:
    """
    Parse camera parameter file in OpenLPT format.
    
    Returns:
        dict with keys: 'model', 'img_size', 'K', 'dist', 'rvec', 'R', 'R_inv', 'tvec', 'tvec_inv'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Camera file not found: {file_path}")
    
    result = {}
    current_section = None
    section_lines = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    text = "".join(lines)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('#'):
            # Process previous section
            if current_section and section_lines:
                result.update(_parse_section(current_section, section_lines))
            # Start new section
            current_section = line[1:].strip()
            section_lines = []
        else:
            section_lines.append(line)
    
    # Process last section
    if current_section and section_lines:
        result.update(_parse_section(current_section, section_lines))
    
    # Parse optional refraction metadata block (exported by refractive calibrator).
    # This is comment-only metadata and is safe to ignore when absent.
    result.update(_parse_refraction_meta(text))

    return result


def _parse_section(section_name: str, lines: list) -> dict:
    """Parse a single section from camera file."""
    result = {}

    def _to_floats(line: str):
        vals = [v for v in re.split(r'[\s,]+', line.strip()) if v]
        return [float(v) for v in vals]
    
    if 'Camera Model' in section_name:
        result['model'] = lines[0].strip()
    
    elif 'Camera Calibration Error' in section_name:
        val = lines[0].strip()
        if val.lower() != 'none':
            parts = val.split(',')
            if len(parts) >= 2:
                result['calib_error'] = (float(parts[0]), float(parts[1]))
        else:
            result['calib_error'] = None
    
    elif 'Pose Calibration Error' in section_name:
        val = lines[0].strip()
        if val.lower() != 'none':
            parts = val.split(',')
            if len(parts) >= 2:
                result['pose_error'] = (float(parts[0]), float(parts[1]))
        else:
            result['pose_error'] = None
    
    elif 'Image Size' in section_name:
        parts = [v for v in re.split(r'[\s,]+', lines[0].strip()) if v]
        result['img_size'] = (int(parts[0]), int(parts[1]))  # (n_row, n_col)
    
    elif 'Camera Matrix' in section_name and 'not' not in section_name.lower():
        K = np.zeros((3, 3))
        for i, row_line in enumerate(lines[:3]):
            vals = _to_floats(row_line)
            K[i] = vals[:3]
        result['K'] = K
    
    elif 'Distortion Coefficients' in section_name:
        vals = _to_floats(lines[0])
        result['dist'] = np.array(vals)
    
    elif 'Rotation Vector' in section_name:
        vals = _to_floats(lines[0])
        result['rvec'] = np.array(vals)
    
    elif 'Rotation Matrix' in section_name and 'Inverse' not in section_name:
        R = np.zeros((3, 3))
        for i, row_line in enumerate(lines[:3]):
            vals = _to_floats(row_line)
            R[i] = vals[:3]
        result['R'] = R
    
    elif 'Inverse of Rotation Matrix' in section_name:
        R_inv = np.zeros((3, 3))
        for i, row_line in enumerate(lines[:3]):
            vals = _to_floats(row_line)
            R_inv[i] = vals[:3]
        result['R_inv'] = R_inv
    
    elif 'Translation Vector' in section_name and 'Inverse' not in section_name:
        vals = _to_floats(lines[0])
        result['tvec'] = np.array(vals)
    
    elif 'Inverse of Translation Vector' in section_name:
        vals = _to_floats(lines[0])
        result['tvec_inv'] = np.array(vals)

    elif 'Refractive plane reference point' in section_name:
        vals = _to_floats(lines[0])
        if len(vals) >= 3:
            result['plane_pt'] = np.array(vals[:3], dtype=np.float64)

    elif 'Refractive plane normal' in section_name:
        vals = _to_floats(lines[0])
        if len(vals) >= 3:
            result['plane_n'] = np.array(vals[:3], dtype=np.float64)
    
    return result


def _parse_refraction_meta(text: str) -> dict:
    """Parse optional BEGIN/END_REFRACTION_META comment block."""
    result = {}
    m = re.search(
        r"#\s*---\s*BEGIN_REFRACTION_META\s*---(?P<body>.*?)#\s*---\s*END_REFRACTION_META\s*---",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return result

    body = m.group("body")

    def _get_int(key: str):
        mm = re.search(rf"#\s*{re.escape(key)}\s*=\s*([-+]?\d+)", body, flags=re.IGNORECASE)
        return int(mm.group(1)) if mm else None

    def _get_vec3(key: str):
        mm = re.search(rf"#\s*{re.escape(key)}\s*=\s*\[([^\]]+)\]", body, flags=re.IGNORECASE)
        if not mm:
            return None
        vals = [v for v in re.split(r"[\s,]+", mm.group(1).strip()) if v]
        if len(vals) < 3:
            return None
        return np.array([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float64)

    meta = {}
    wid = _get_int("WINDOW_ID")
    cid = _get_int("CAM_ID")
    plane_pt = _get_vec3("PLANE_PT_EXPORT")
    plane_n = _get_vec3("PLANE_N")
    if cid is not None:
        meta["cam_id"] = cid
    if wid is not None:
        meta["window_id"] = wid
    if plane_pt is not None:
        meta["plane_pt_export"] = plane_pt
    if plane_n is not None:
        meta["plane_n"] = plane_n

    if meta:
        result["ref_meta"] = meta
    return result


def save_camera_file(file_path: str, params: dict, proj_error: tuple = None, tri_error: tuple = None):
    """
    Save camera parameters to OpenLPT format file.
    
    Args:
        file_path: Output file path
        params: dict with 'img_size', 'K', 'dist', 'R', 'tvec'
        proj_error: (mean, std) of projection error in pixels
        tri_error: (mean, std) of triangulation error in mm
    """
    K = params['K']
    dist = params['dist']
    R = params['R']
    tvec = params['tvec']
    img_size = params['img_size']
    
    # Compute rotation vector
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.flatten()
    
    # Compute inverse
    R_inv = R.T
    tvec_col = tvec.reshape(3, 1)
    tvec_inv = -R_inv @ tvec_col
    tvec_inv = tvec_inv.flatten()
    
    # Format error strings
    proj_err_str = f"{proj_error[0]},{proj_error[1]}" if proj_error else "None"
    tri_err_str = f"{tri_error[0]},{tri_error[1]}" if tri_error else "None"
    
    with open(file_path, 'w') as f:
        f.write("# Camera Model: (PINHOLE/POLYNOMIAL)\n")
        f.write("PINHOLE\n")
        
        f.write("# Camera Calibration Error: \n")
        f.write(f"{proj_err_str}\n")
        
        f.write("# Pose Calibration Error: \n")
        f.write(f"{tri_err_str}\n")
        
        f.write("# Image Size: (n_row,n_col)\n")
        f.write(f"{img_size[0]},{img_size[1]}\n")
        
        f.write("# Camera Matrix: \n")
        for i in range(3):
            f.write(f"{K[i,0]},{K[i,1]},{K[i,2]}\n")
        
        f.write("# Distortion Coefficients: \n")
        dist_str = ",".join(map(str, dist.flatten()))
        f.write(f"{dist_str}\n")
        
        f.write("# Rotation Vector: \n")
        f.write(f"{rvec[0]},{rvec[1]},{rvec[2]}\n")
        
        f.write("# Rotation Matrix: \n")
        for i in range(3):
            f.write(f"{R[i,0]},{R[i,1]},{R[i,2]}\n")
        
        f.write("# Inverse of Rotation Matrix: \n")
        for i in range(3):
            f.write(f"{R_inv[i,0]},{R_inv[i,1]},{R_inv[i,2]}\n")
        
        f.write("# Translation Vector: \n")
        f.write(f"{tvec[0]},{tvec[1]},{tvec[2]}\n")
        
        f.write("# Inverse of Translation Vector: \n")
        f.write(f"{tvec_inv[0]},{tvec_inv[1]},{tvec_inv[2]}\n")


def project_point(pt3d: np.ndarray, K: np.ndarray, R: np.ndarray, tvec: np.ndarray, dist: np.ndarray = None) -> np.ndarray:
    """
    Project 3D point to 2D using camera parameters.
    
    Args:
        pt3d: (3,) array - 3D point
        K: (3,3) camera matrix
        R: (3,3) rotation matrix
        tvec: (3,) translation vector
        dist: distortion coefficients (optional)
    
    Returns:
        (2,) array - 2D projection
    """
    rvec, _ = cv2.Rodrigues(R)
    if dist is None:
        dist = np.zeros(5)
    
    pt3d = pt3d.reshape(1, 3).astype(np.float64)
    projected, _ = cv2.projectPoints(pt3d, rvec, tvec, K, dist)
    return projected.flatten()


def get_camera_params_vector(cam_params: dict) -> np.ndarray:
    """
    Extract 8-parameter vector from camera dict for optimization.
    
    Returns:
        [rx, ry, rz, tx, ty, tz, f_eff, k1]
    """
    K = cam_params['K']
    R = cam_params['R']
    tvec = cam_params['tvec']
    dist = cam_params.get('dist', np.zeros(5))
    
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.flatten()
    
    f_eff = K[0, 0]  # Assume fx = fy
    k1 = dist[0] if len(dist) > 0 else 0.0
    
    return np.array([rvec[0], rvec[1], rvec[2], 
                     tvec[0], tvec[1], tvec[2], 
                     f_eff, k1])


def set_camera_params_from_vector(cam_params: dict, params_vec: np.ndarray) -> dict:
    """
    Update camera dict from 8-parameter vector.
    
    Args:
        cam_params: Original camera parameters dict
        params_vec: [rx, ry, rz, tx, ty, tz, f_eff, k1]
    
    Returns:
        Updated camera parameters dict
    """
    result = cam_params.copy()
    
    rvec = params_vec[0:3]
    tvec = params_vec[3:6]
    f_eff = params_vec[6]
    k1 = params_vec[7]
    
    R, _ = cv2.Rodrigues(rvec)
    
    # Update K (preserve cx, cy)
    K = result['K'].copy()
    K[0, 0] = f_eff
    K[1, 1] = f_eff
    
    # Update distortion
    dist = result.get('dist', np.zeros(5)).copy()
    dist[0] = k1
    
    result['K'] = K
    result['R'] = R
    result['R_inv'] = R.T
    result['tvec'] = tvec
    result['tvec_inv'] = (-R.T @ tvec.reshape(3, 1)).flatten()
    result['dist'] = dist
    result['rvec'] = rvec
    
    return result
