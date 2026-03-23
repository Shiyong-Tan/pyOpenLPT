"""
Volume Self-Calibration (VSC) Service.
Main service class implementing the complete VSC workflow.
"""
import os
import re
import csv
import shutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

from .camera_io import (
    parse_camera_file, save_camera_file, 
    get_camera_params_vector, set_camera_params_from_vector,
    project_point
)
from .optimizer import VSCOptimizer


class VSCService:
    """
    Volume Self-Calibration Service.
    
    Workflow:
    1. Load tracks from ConvergeTrack CSV files
    2. Filter good tracks (length > min_track_len)
    3. Sample 3D points uniformly from volume
    4. Find 2D correspondences with isolation check
    5. Run model-aware TRF optimization (ray-consistency for refraction)
    6. Save optimized cameras and update config.txt
    """
    
    def __init__(self, proj_dir: str, log_callback: Optional[Callable[[str], None]] = None):
        """
        Args:
            proj_dir: Project directory path
            log_callback: Optional callback for logging messages
        """
        self.proj_dir = proj_dir
        self.log_callback = log_callback
        self.log_file = None
        
        # Parameters
        self.min_track_len = 15
        self.sample_points = 20000
        self.min_valid_points = 2000
        self.n_divisions = 10
        self.isolation_margin = 2.0
        self.margin_factor = 4.0  # ROI margin = margin_factor * obj_radius
        self.search_radius = 4.0  # Max distance from projection to detection
        
        # Data
        self.cameras: Dict[int, dict] = {}
        self.camera_paths: Dict[int, str] = {}
        self.camera_models: Dict[int, str] = {}
        self.cpp_cameras: Dict[int, object] = {}
        self.window_planes: Dict[int, dict] = {}
        self.cam_to_window: Dict[int, int] = {}
        self._overlay_points_optim: Dict[int, np.ndarray] = {}
        self.obj_radius = 5.0
        self.obj_type = "Tracer"
        self.image_size = (1024, 1024)
        self.cam_output_dir = None
        self.tolerance_mode = 'default'
        self.tolerance_value = 5.0
        self._tolerance_thresholds = None
        
    def set_params(self, min_track_len: int = 15, sample_points: int = 20000,
                   min_valid_points: int = 2000, tolerance_mode: str = 'default',
                   tolerance_value: float = 5.0):
        """Set VSC parameters."""
        self.min_track_len = min_track_len
        self.sample_points = sample_points
        self.min_valid_points = min_valid_points
        self.tolerance_mode = tolerance_mode
        self.tolerance_value = tolerance_value
    
    def _log(self, msg: str):
        """Log message to callback and file."""
        if self.log_callback:
            self.log_callback(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()
        print(msg)
    
    def run(self) -> Tuple[bool, str, Dict]:
        """
        Run the complete VSC workflow.
        
        Returns:
            (success, message, data_dict)
            data_dict contains:
            - 'valid_points': List of correspondences
            - 'cameras_init': Initial camera parameters
            - 'cameras_optim': Optimized camera parameters
        """
        # Open log file
        log_path = os.path.join(self.proj_dir, "VSC_log.txt")
        try:
            self.log_file = open(log_path, 'w')
        except Exception as e:
            return False, f"Failed to create log file: {e}", {}
        
        vsc_data = {}
        
        try:
            self._overlay_points_optim = {}
            self._log("=" * 60)
            self._log("Volume Self-Calibration (VSC) Started")
            self._log("=" * 60)
            self._log(f"Project: {self.proj_dir}")
            
            # Step 1: Load cameras
            self._log("\n[Step 1] Loading camera parameters...")
            success, msg = self._load_cameras()
            if not success:
                return False, msg, {}
            
            # Save initial state using deep copy avoids reference issues
            import copy
            vsc_data['cameras_init'] = copy.deepcopy(self.cameras)
            
            # Validate and log tolerance mode
            if self.tolerance_mode == 'reproj':
                missing = []
                for cam_idx, cam in self.cameras.items():
                    ce = cam.get('calib_error')
                    if ce is None or not isinstance(ce, tuple) or len(ce) < 2:
                        missing.append(cam_idx)
                if missing:
                    msg = (f"Reprojection Error tolerance mode selected but camera(s) "
                           f"{missing} lack calibration error stats. "
                           f"Cannot proceed. Please recalibrate or choose a different mode.")
                    self._log(f"  ERROR: {msg}")
                    return False, msg, {}
                reproj_thresholds = {}
                for cam_idx, cam in self.cameras.items():
                    mean, std = cam['calib_error']
                    reproj_thresholds[cam_idx] = mean + 3.0 * std
                self._log(f"  Tolerance mode: Reprojection Error Based")
                for cam_idx in sorted(reproj_thresholds):
                    mean, std = self.cameras[cam_idx]['calib_error']
                    self._log(f"    Cam {cam_idx}: reproj_mean={mean:.3f}, reproj_std={std:.3f}, threshold={reproj_thresholds[cam_idx]:.3f} px")
                self._tolerance_thresholds = reproj_thresholds
            elif self.tolerance_mode == 'custom':
                self._log(f"  Tolerance mode: Custom = {self.tolerance_value:.1f} px")
                self._tolerance_thresholds = None
            else:
                self._log(f"  Tolerance mode: Default (point_r + 1)")
                self._tolerance_thresholds = None
            
            # Step 2: Load object config
            self._log("\n[Step 2] Loading object configuration...")
            success, msg = self._load_object_config()
            if not success:
                return False, msg, {}
            
            # Step 3: Load and filter tracks
            self._log("\n[Step 3] Loading and filtering tracks...")
            tracks = self._load_tracks()
            self.tracks = tracks # Store for GUI return
            
            if not tracks:
                return False, "No tracks found in Results/ConvergeTrack", {}
            
            good_tracks = self._filter_good_tracks(tracks)
            self._log(f"  Total tracks: {len(tracks)}")
            self._log(f"  Good tracks (length >= {self.min_track_len}): {len(good_tracks)}")
            
            if len(good_tracks) == 0:
                return False, f"No tracks with length >= {self.min_track_len}", {}
            
            # Step 4: Sample 3D points uniformly
            self._log("\n[Step 4] Sampling 3D points uniformly...")
            sampled_points = self._sample_uniform_points(good_tracks)
            self._log(f"  Sampled {len(sampled_points)} 3D points")
            
            # Step 5: Find 2D correspondences
            self._log("\n[Step 5] Finding 2D correspondences with isolation check...")
            correspondences = self._find_correspondences(sampled_points)
            for corr_id, corr in enumerate(correspondences):
                corr['corr_id'] = int(corr_id)
            self._log(f"  Valid correspondences: {len(correspondences)}")
            
            vsc_data['valid_points'] = correspondences
            
            if len(correspondences) < self.min_valid_points:
                return False, f"Only {len(correspondences)} valid points found, need {self.min_valid_points}", vsc_data
            
            # Step 6: Run model-aware optimization
            self._log("\n[Step 6] Running TRF optimization (model-aware ray consistency)...")
            success, msg = self._run_optimization(correspondences)
            if not success:
                return False, msg, vsc_data
            
            # Step 7: Save cameras and update config
            self._log("\n[Step 7] Saving optimized cameras...")
            self._save_cameras()
            self._update_config()
            
            vsc_data['cameras_optim'] = copy.deepcopy(self.cameras)
            vsc_data['tracks'] = self.tracks # Add tracks for GUI return
            vsc_data['overlay_points_optim'] = self._overlay_points_optim
            
            self._log("\n" + "=" * 60)
            self._log("VSC Completed Successfully!")
            self._log("=" * 60)
            
            return True, "VSC completed successfully", vsc_data
            
        except Exception as e:
            import traceback
            self._log(f"\nERROR: {e}")
            self._log(traceback.format_exc())
            return False, f"VSC failed: {e}", vsc_data
        finally:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
    
    def _load_cameras(self) -> Tuple[bool, str]:
        """Load camera parameters from paths specified in config.txt."""
        config_path = os.path.join(self.proj_dir, "config.txt")
        
        if not os.path.exists(config_path):
            # Fallback to old behavior if no config.txt
            cam_dir = os.path.join(self.proj_dir, "camFile")
            if not os.path.exists(cam_dir):
                return False, f"config.txt not found and camFile directory not found: {cam_dir}"
            return self._load_cameras_from_dir(cam_dir)
        
        # Parse config.txt for camera file paths
        # Format: "# Camera File Path, Max Intensity" followed by lines like:
        # G:/path/to/cam0.txt,255
        camera_paths = []
        in_camera_section = False
        
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if '# Camera File Path' in line:
                    in_camera_section = True
                    continue
                
                if in_camera_section:
                    # Check if line starts with '#' (new section header)
                    if line.startswith('#'):
                        in_camera_section = False
                        continue
                    
                    # Parse camera path line: "path/to/cam.txt,intensity" or just "path/to/cam.txt"
                    parts = line.split(',')
                    if parts:
                        cam_path = parts[0].strip()
                        if cam_path.endswith('.txt'):
                            # Handle relative vs absolute paths
                            if not os.path.isabs(cam_path):
                                cam_path = os.path.join(self.proj_dir, cam_path)
                            camera_paths.append(cam_path)
        
        if not camera_paths:
            self._log("  No camera paths found in config.txt, trying camFile directory...")
            cam_dir = os.path.join(self.proj_dir, "camFile")
            if not os.path.exists(cam_dir):
                return False, f"Camera directory not found: {cam_dir}"
            return self._load_cameras_from_dir(cam_dir)
        
        # Load cameras from parsed paths
        for cam_path in camera_paths:
            if not os.path.exists(cam_path):
                self._log(f"  Warning: Camera file not found: {cam_path}")
                continue
                
            # Extract camera index from filename (cam0.txt -> 0)
            filename = os.path.basename(cam_path)
            match = re.search(r'cam(\d+)', filename)
            if match:
                cam_idx = int(match.group(1))
                try:
                    params = parse_camera_file(cam_path)
                    params['file_path'] = cam_path
                    self.cameras[cam_idx] = params
                    self.camera_paths[cam_idx] = cam_path
                    self.camera_models[cam_idx] = str(params.get('model', 'PINHOLE')).upper()
                    self._log(f"  Loaded Camera {cam_idx} from {cam_path}")
                    
                    # Track output directory (use first camera's directory)
                    if self.cam_output_dir is None:
                        self.cam_output_dir = os.path.dirname(cam_path)
                    
                    # Get image size from first camera
                    if 'img_size' in params:
                        self.image_size = params['img_size']
                except Exception as e:
                    self._log(f"  Warning: Failed to load {cam_path}: {e}")
        
        if not self.cameras:
            return False, "Failed to load any cameras"

        self._build_refraction_window_map()
        
        self._log(f"  Loaded {len(self.cameras)} cameras")
        return True, ""
    
    def _load_cameras_from_dir(self, cam_dir: str) -> Tuple[bool, str]:
        """Fallback: Load cameras from a directory."""
        cam_files = sorted([f for f in os.listdir(cam_dir) 
                           if f.startswith("cam") and f.endswith(".txt") 
                           and not f.startswith("vsc_cam")])
        
        if not cam_files:
            return False, f"No camera files found in {cam_dir}"
        
        for cam_file in cam_files:
            match = re.search(r'cam(\d+)', cam_file)
            if match:
                cam_idx = int(match.group(1))
                file_path = os.path.join(cam_dir, cam_file)
                
                try:
                    params = parse_camera_file(file_path)
                    params['file_path'] = file_path
                    self.cameras[cam_idx] = params
                    self.camera_paths[cam_idx] = file_path
                    self.camera_models[cam_idx] = str(params.get('model', 'PINHOLE')).upper()
                    self._log(f"  Loaded Camera {cam_idx} from {cam_file}")
                    
                    if 'img_size' in params:
                        self.image_size = params['img_size']
                except Exception as e:
                    self._log(f"  Warning: Failed to load {cam_file}: {e}")
        
        if not self.cameras:
            return False, "Failed to load any cameras"

        self._build_refraction_window_map()
        
        self._log(f"  Loaded {len(self.cameras)} cameras")
        return True, ""

    def _build_refraction_window_map(self):
        """Build per-window plane map from camera refraction metadata."""
        self.cam_to_window = {}
        self.window_planes = {}

        for cam_idx, cam in self.cameras.items():
            model = str(cam.get('model', 'PINHOLE')).upper()
            if model != 'PINPLATE':
                continue

            meta = cam.get('ref_meta', {}) or {}
            wid = meta.get('window_id', None)
            plane_pt = meta.get('plane_pt_export', None)
            plane_n = meta.get('plane_n', None)

            # Fallback when optional metadata block is missing.
            if wid is None:
                wid = int(cam_idx)
                self._log(
                    f"  Warning: cam{cam_idx} missing REFRACTION_META WINDOW_ID; "
                    f"fallback window_id={wid}"
                )
            if plane_pt is None:
                plane_pt = cam.get('plane_pt', None)
            if plane_n is None:
                plane_n = cam.get('plane_n', None)

            self.cam_to_window[int(cam_idx)] = int(wid)
            if plane_pt is not None and plane_n is not None:
                self.window_planes[int(wid)] = {
                    'plane_pt': np.asarray(plane_pt, dtype=np.float64).reshape(3),
                    'plane_n': np.asarray(plane_n, dtype=np.float64).reshape(3),
                }

        if self.window_planes:
            self._log(f"  Refractive windows loaded: {len(self.window_planes)}")

    def _build_refraction_window_map_from_cpp(self):
        """Rebuild refractive window map using C++ PINPLATE state as source of truth."""
        self.cam_to_window = {}
        self.window_planes = {}

        for cam_idx, cam in self.cameras.items():
            model = str(self.camera_models.get(cam_idx, cam.get('model', 'PINHOLE'))).upper()
            if model != 'PINPLATE':
                continue

            meta = cam.get('ref_meta', {}) or {}
            wid = meta.get('window_id', None)
            if wid is None:
                wid = int(cam_idx)
                self._log(
                    f"  Warning: cam{cam_idx} missing REFRACTION_META WINDOW_ID; "
                    f"fallback window_id={wid}"
                )
            wid = int(wid)
            self.cam_to_window[int(cam_idx)] = wid

            cpp_cam = self.cpp_cameras.get(cam_idx)
            if cpp_cam is None:
                self._log(f"  Warning: Missing cpp camera for cam{cam_idx}; cannot read refractive plane")
                continue

            try:
                pin = cpp_cam._pinplate_param
                plane = pin.plane
                pt = plane.pt
                n = plane.norm_vector

                plane_pt = np.array([float(pt[0]), float(pt[1]), float(pt[2])], dtype=np.float64)
                plane_n = np.array([float(n[0]), float(n[1]), float(n[2])], dtype=np.float64)
                n_norm = np.linalg.norm(plane_n)
                if n_norm > 1e-12:
                    plane_n = plane_n / n_norm

                if not (np.all(np.isfinite(plane_pt)) and np.all(np.isfinite(plane_n))):
                    raise ValueError("non-finite plane parameters")

                if wid in self.window_planes:
                    prev = self.window_planes[wid]
                    dp = float(np.linalg.norm(plane_pt - prev['plane_pt']))
                    dn = float(np.linalg.norm(plane_n - prev['plane_n']))
                    if dp > 1e-6 or dn > 1e-6:
                        self._log(
                            f"  Warning: window {wid} plane mismatch across cameras; "
                            f"using cam{cam_idx} values (dpt={dp:.3e}, dn={dn:.3e})"
                        )

                self.window_planes[wid] = {
                    'plane_pt': plane_pt,
                    'plane_n': plane_n,
                }
            except Exception as e:
                self._log(f"  Warning: Failed reading pinplate plane from cam{cam_idx}: {e}")

        if self.window_planes:
            self._log(f"  Refractive windows loaded from CPP: {len(self.window_planes)}")
        else:
            self._log("  Warning: No refractive window plane loaded from CPP")

    def _build_window_refraction_from_cpp(self) -> Dict[int, dict]:
        """Collect per-window refractive media/solver settings from C++ PINPLATE state."""
        window_refraction: Dict[int, dict] = {}

        for cam_idx, cpp_cam in self.cpp_cameras.items():
            wid = self.cam_to_window.get(int(cam_idx), None)
            if wid is None:
                continue
            wid = int(wid)

            try:
                pin = cpp_cam._pinplate_param
                refract_array = [float(v) for v in list(pin.refract_array)]
                w_array = [float(v) for v in list(pin.w_array)]
                proj_tol = float(pin.proj_tol)
                proj_nmax = int(pin.proj_nmax)
                lr = float(pin.lr)
            except Exception as e:
                self._log(f"  Warning: Failed reading refraction settings from cam{cam_idx}: {e}")
                continue

            if len(refract_array) < 3 or len(w_array) < 1:
                self._log(
                    f"  Warning: cam{cam_idx} has incomplete refraction arrays; "
                    f"skip window {wid} settings"
                )
                continue

            curr = {
                'refract_array': refract_array,
                'w_array': w_array,
                'proj_tol': proj_tol,
                'proj_nmax': proj_nmax,
                'lr': lr,
            }

            if wid in window_refraction:
                prev = window_refraction[wid]
                mismatch = (
                    len(prev['refract_array']) != len(curr['refract_array'])
                    or len(prev['w_array']) != len(curr['w_array'])
                    or np.linalg.norm(np.asarray(prev['refract_array']) - np.asarray(curr['refract_array'])) > 1e-9
                    or np.linalg.norm(np.asarray(prev['w_array']) - np.asarray(curr['w_array'])) > 1e-9
                    or abs(float(prev['proj_tol']) - float(curr['proj_tol'])) > 1e-12
                    or int(prev['proj_nmax']) != int(curr['proj_nmax'])
                    or abs(float(prev['lr']) - float(curr['lr'])) > 1e-12
                )
                if mismatch:
                    self._log(
                        f"  Warning: window {wid} refraction settings mismatch across cameras; "
                        f"using cam{cam_idx} values"
                    )

            window_refraction[wid] = curr

        self._log(f"  Refractive window settings loaded: {len(window_refraction)}")
        return window_refraction
    
    def _load_object_config(self) -> Tuple[bool, str]:
        """Load object configuration to get obj_radius."""
        config_path = os.path.join(self.proj_dir, "config.txt")
        if not os.path.exists(config_path):
            self._log("  Warning: config.txt not found, using defaults")
            return True, ""
        
        # Determine object type
        with open(config_path, 'r') as f:
            content = f.read().lower()
            if 'bubble' in content:
                self.obj_type = "Bubble"
            else:
                self.obj_type = "Tracer"
        
        # Load object-specific config
        if self.obj_type == "Tracer":
            tracer_config = os.path.join(self.proj_dir, "tracerConfig.txt")
            if os.path.exists(tracer_config):
                self._load_tracer_config(tracer_config)
        else:
            bubble_config = os.path.join(self.proj_dir, "bubbleConfig.txt")
            if os.path.exists(bubble_config):
                self._load_bubble_config(bubble_config)
        
        self._log(f"  Object type: {self.obj_type}")
        self._log(f"  Object radius: {self.obj_radius} px")
        return True, ""
    
    def _load_tracer_config(self, path: str):
        """Parse tracerConfig.txt for obj_radius.
        Format: 'value # comment' - e.g. '2 # Particle radius [px]...'
        """
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Format: "value # comment"
                # Look for "Particle radius" in the comment part
                if 'Particle radius' in line:
                    # Extract the value before the # comment
                    parts = line.split('#')
                    if parts:
                        try:
                            self.obj_radius = float(parts[0].strip())
                            self._log(f"  Found Particle radius: {self.obj_radius} px")
                            return
                        except ValueError:
                            pass
        
        # Default tracer radius
        self.obj_radius = 5.0
    
    def _load_bubble_config(self, path: str):
        """Parse bubbleConfig.txt for minimum bubble size as obj_radius.
        Format: 'value # comment' - e.g. '2 # minimum bubble size to track'
        """
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Format: "value # comment"
                # Look for "minimum bubble size" in the comment part
                if 'minimum bubble size' in line.lower():
                    parts = line.split('#')
                    if parts:
                        try:
                            self.obj_radius = float(parts[0].strip())
                            self._log(f"  Found minimum bubble size: {self.obj_radius} px")
                            return
                        except ValueError:
                            pass
        
        # Default bubble radius
        self.obj_radius = 10.0
    
    def _load_tracks(self) -> Dict[int, List[Tuple]]:
        """
        Load tracks from ConvergeTrack CSV files.
        
        Returns:
            dict: {track_id: [(frame_id, x, y, z, r3d_mm, cam_2d_dict), ...]}
        """
        # Read Output Folder Path from config.txt
        config_path = os.path.join(self.proj_dir, "config.txt")
        output_dir = os.path.join(self.proj_dir, "Results")  # Default fallback
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "Output Folder Path" in line:
                        if i + 1 < len(lines):
                            path_line = lines[i+1].strip()
                            if path_line and not path_line.startswith('#'):
                                output_dir = path_line
                                break
        
        track_dir = os.path.join(output_dir, "ConvergeTrack")
        if not os.path.exists(track_dir):
            # Fallback to default location
            fallback_dir = os.path.join(self.proj_dir, "Results", "ConvergeTrack")
            if os.path.exists(fallback_dir):
                track_dir = fallback_dir
            else:
                return {}
        
        def natsort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]
        
        patterns = ["LongTrackActive", "LongTrackInactive", "ExitTrack"]
        tracks = defaultdict(list)
        max_id_overall = -1  # Use -1 like tracking_view
        
        for pattern in patterns:
            files = sorted([f for f in os.listdir(track_dir) 
                           if f.startswith(pattern) and f.endswith(".csv")],
                          key=natsort_key)
            
            for filename in files:
                file_path = os.path.join(track_dir, filename)
                local_max_id_in_file = -1
                
                # Use max_id_overall + 1 as offset for each file (like tracking_view)
                offset = max_id_overall + 1
                
                with open(file_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header:
                        continue
                    
                    for row in reader:
                        if not row:
                            continue
                        try:
                            orig_id = int(row[0])
                            frame_id = int(row[1])
                            is_bubble = (self.obj_type == "Bubble")
                            x, y, z = float(row[2]), float(row[3]), float(row[4])
                            r3d_mm = 0.0
                            if is_bubble and len(row) > 5:
                                try:
                                    r3d_mm = float(row[5])
                                except Exception:
                                    r3d_mm = 0.0
                            
                            track_id = orig_id + offset
                            
                            # Parse 2D coords for each camera
                            # Tracer: [ID, F, X, Y, Z, C0_x, C0_y, C1_x, C1_y, ...] -> Start 5, Stride 2
                            # Bubble: [ID, F, X, Y, Z, R3D, C0_x, C0_y, C0_r, C1_x, ...] -> Start 6, Stride 3
                            
                            if is_bubble:
                                start_col = 6
                                stride = 3
                            else:
                                start_col = 5
                                stride = 2
                            
                            cam_2d = {}
                            cam_idx = 0
                            col = start_col
                            while col + 1 < len(row):
                                try:
                                    cam_x = float(row[col])
                                    cam_y = float(row[col + 1])
                                    if cam_x > 0 and cam_y > 0:
                                        # For bubble: store (x, y, r2d)
                                        # For tracer: store (x, y, obj_radius from config)
                                        if is_bubble and col + 2 < len(row):
                                            cam_r = float(row[col + 2])
                                            cam_2d[cam_idx] = (cam_x, cam_y, cam_r)
                                        else:
                                            # Tracer: use fixed radius from config
                                            cam_2d[cam_idx] = (cam_x, cam_y, self.obj_radius)
                                except (ValueError, IndexError):
                                    pass
                                col += stride
                                cam_idx += 1
                            
                            tracks[track_id].append((frame_id, x, y, z, r3d_mm, cam_2d))
                            
                            if orig_id > local_max_id_in_file:
                                local_max_id_in_file = orig_id
                                
                        except (ValueError, IndexError):
                            continue
                
                # Update global max after each file (like tracking_view)
                if local_max_id_in_file != -1:
                    max_id_overall += (local_max_id_in_file + 1)
        
        return dict(tracks)
    
    def _filter_good_tracks(self, tracks: Dict[int, List]) -> Dict[int, List]:
        """Filter tracks with length >= min_track_len."""
        return {tid: pts for tid, pts in tracks.items() 
                if len(pts) >= self.min_track_len}
    
    def _sample_uniform_points(self, tracks: Dict[int, List]) -> List[Tuple]:
        """
        Sample 3D points uniformly from view volume using voxel grid.
        
        Returns:
            List of (x, y, z, frame_id, r3d_mm, cam_2d_dict) tuples
        """
        # Collect all points
        all_points = []
        for track_id, pts in tracks.items():
            for frame_id, x, y, z, r3d_mm, cam_2d in pts:
                all_points.append((x, y, z, frame_id, r3d_mm, cam_2d))
        
        if not all_points:
            return []
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Compute bounding box
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        zs = [p[2] for p in all_points]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)
        
        self._log(f"  Volume bounds: X=[{x_min:.2f}, {x_max:.2f}], "
                  f"Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")
        
        # Voxel sizes
        n_div = self.n_divisions
        dx = (x_max - x_min) / n_div if x_max > x_min else 1.0
        dy = (y_max - y_min) / n_div if y_max > y_min else 1.0
        dz = (z_max - z_min) / n_div if z_max > z_min else 1.0
        
        # Bin points into voxels
        voxels = defaultdict(list)
        for pt in all_points:
            xi = min(int((pt[0] - x_min) / dx), n_div - 1)
            yi = min(int((pt[1] - y_min) / dy), n_div - 1)
            zi = min(int((pt[2] - z_min) / dz), n_div - 1)
            voxel_id = xi * n_div * n_div + yi * n_div + zi
            voxels[voxel_id].append(pt)
        
        # Sample uniformly
        n_voxels = len(voxels)
        pts_per_voxel = max(1, self.sample_points // n_voxels)
        
        sampled = []
        for voxel_id, pts in voxels.items():
            if len(pts) <= pts_per_voxel:
                sampled.extend(pts)
            else:
                indices = np.random.choice(len(pts), pts_per_voxel, replace=False)
                sampled.extend([pts[i] for i in indices])
        
        # Limit to sample_points
        if len(sampled) > self.sample_points:
            indices = np.random.choice(len(sampled), self.sample_points, replace=False)
            sampled = [sampled[i] for i in indices]
        
        return sampled

    def _find_correspondences(self, sampled_points: List[Tuple]) -> List[dict]:
        """
        Find 2D correspondences for sampled 3D points by detecting in actual images.
        
        For each point:
        1. Load the corresponding frame image
        2. Crop ROI around the stored projection
        3. Run ObjectFinder2D on ROI
        4. Validate: single detection, close to projection, isolated
        
        Returns:
            List of dicts with keys: 'pt3d', '2d_per_cam'
        """
        valid_correspondences = []
        n_cams = len(self.cameras)
        
        # Import pyopenlpt for image detection
        try:
            import pyopenlpt as lpt
        except ImportError:
            self._log("  WARNING: pyopenlpt not available, using stored projections as fallback")
            return self._find_correspondences_fallback(sampled_points)
        
        # Load ImageIO for each camera from config.txt paths
        img_loaders = self._load_image_loaders(lpt)
        
        if not img_loaders:
            self._log("  WARNING: No image loaders available, using fallback")
            return self._find_correspondences_fallback(sampled_points)
        
        # Load ObjectConfig for detection
        try:
            obj_cfg = self._load_obj_config_for_detection()
            if obj_cfg is None:
                self._log("  WARNING: Failed to load object config, using fallback")
                return self._find_correspondences_fallback(sampled_points)
        except Exception as e:
            self._log(f"  WARNING: ObjectConfig error: {e}, using fallback")
            return self._find_correspondences_fallback(sampled_points)
        
        # Group points by frame for efficient image loading
        points_by_frame = {}
        for pt in sampled_points:
            x, y, z, frame_id, r3d_mm, cam_2d = pt
            if frame_id not in points_by_frame:
                points_by_frame[frame_id] = []
            points_by_frame[frame_id].append(pt)
        
        self._log(f"  Processing {len(points_by_frame)} unique frames...")
        
        # Helper for thread-safe ObjectFinder
        from concurrent.futures import ThreadPoolExecutor
        import threading
        thread_local = threading.local()
        
        def get_finder():
            if not hasattr(thread_local, "finder"):
                thread_local.finder = lpt.ObjectFinder2D()
            return thread_local.finder
        
        # Process each frame
        processed = 0
        total_points = len(sampled_points)
        
        # Debug counters
        fail_no_detection = 0
        fail_distance = 0
        fail_not_isolated = 0
        fail_exception = 0
        fail_roi = 0
        
        if not points_by_frame:
            return []
            
        self._log(f"  Processing {len(points_by_frame)} unique frames in parallel processes...")
        
        # Prepare parameters for workers
        # Prepare parameters for workers
        params = {
            'obj_type': self.obj_type,
            'obj_radius': self.obj_radius,
            'margin_factor': self.margin_factor,
            'search_radius': self.search_radius,
            'isolation_margin': self.isolation_margin,
            'tolerance_mode': self.tolerance_mode,
            'tolerance_value': self.tolerance_value,
        }
        if self.tolerance_mode == 'reproj':
            params['reproj_thresholds'] = self._tolerance_thresholds
        
        # Use ProcessPoolExecutor to bypass GIL
        # Workers will load their own images using config paths
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        valid_correspondences = []
        
        # Debug counters (accumulated from workers)
        fail_stats = {
            'processed': 0, 'valid': 0, 
            'roi': 0, 'no_detection': 0, 
            'distance': 0, 'not_isolated': 0, 
            'exception': 0, 'error': 0
        }
        
        max_workers = min(os.cpu_count(), len(points_by_frame))
        
        # Pass initializer to setup worker environment once
        # Note: We share cameras keys to let worker know which cameras to init
        with ProcessPoolExecutor(max_workers=max_workers, 
                               initializer=init_worker, 
                               initargs=(self.proj_dir, params, list(self.cameras.keys()))) as executor:
            futures = []
            for frame_id, frame_points in points_by_frame.items():
                futures.append(executor.submit(
                    process_frame_task, 
                    frame_id, frame_points, 
                    self.image_size, params
                ))
            
            processed_count = 0
            total_frames = len(futures)
            log_step = max(1, total_frames // 10)
            
            for i, future in enumerate(as_completed(futures)):
                processed_count += 1
                try:
                    res_list, stats = future.result()
                    valid_correspondences.extend(res_list)
                    
                    # Accumulate stats
                    for key in fail_stats:
                        if key in stats:
                            if key == 'error' and stats[key] != 0:
                                fail_stats['error'] += 1 # Count error tasks
                                self._log(f"  Worker error: {stats['error']}")
                            else:
                                fail_stats[key] += stats[key]
                                
                    if processed_count % log_step == 0 or processed_count == total_frames:
                        self._log(f"  Processed frame {processed_count}/{total_frames}. Total valid: {len(valid_correspondences)}")
                        
                except Exception as e:
                    fail_stats['exception'] += 1
                    self._log(f"  Task exception: {e}")
        
        # Log failure breakdown
        self._log(f"  FINAL Detection Stats:")
        self._log(f"    Total Checked: {fail_stats['processed']}")
        self._log(f"    Valid found: {len(valid_correspondences)}")
        self._log(f"    Failures - NoDet: {fail_stats['no_detection']}, "
                  f"Dist: {fail_stats['distance']}, Iso: {fail_stats['not_isolated']}, "
                  f"ROI: {fail_stats['roi']}, Exc: {fail_stats['exception']}")
        
        return valid_correspondences
    
    def _debug_show_roi(self, roi_img, detections, best_det, 
                        roi_proj_x, roi_proj_y, point_r,
                        cam_idx, frame_id, sep, min_sep):
        """Debug visualization for not_isolated cases using Qt dialog for blocking."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton
        from PySide6.QtCore import Qt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        import numpy as np
        
        # Convert ROI image to numpy array
        h = roi_img.getDimRow()
        w = roi_img.getDimCol()
        img_data = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                img_data[i, j] = roi_img[i, j]
        
        # Create Qt dialog
        dialog = QDialog()
        dialog.setWindowTitle(f"NOT ISOLATED - Cam {cam_idx}, Frame {frame_id}")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
        dialog.resize(800, 850)
        
        layout = QVBoxLayout(dialog)
        
        # Create matplotlib figure with black background
        fig = Figure(figsize=(8, 8), facecolor='black')
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        
        ax.imshow(img_data, cmap='gray')
        ax.set_title(f"NOT ISOLATED - Cam {cam_idx}, Frame {frame_id}\n"
                     f"sep={sep:.2f}, min_sep={min_sep:.2f}", color='white')
        
        # Plot projection position (green cross)
        ax.scatter([roi_proj_x], [roi_proj_y], c='green', marker='x', s=200, 
                   linewidths=3, label=f'Projection ({roi_proj_x:.1f}, {roi_proj_y:.1f})')
        
        # Plot all detected points
        colors = ['red', 'blue', 'orange', 'purple', 'cyan']
        for i, det in enumerate(detections):
            cx, cy = det._pt_center[0], det._pt_center[1]
            r = det._r_px if hasattr(det, '_r_px') else point_r
            
            if det is best_det:
                color = 'lime'
                label = f'BEST ({cx:.1f}, {cy:.1f}), r={r:.1f}'
            else:
                color = colors[i % len(colors)]
                label = f'Det {i} ({cx:.1f}, {cy:.1f}), r={r:.1f}'
            
            ax.scatter([cx], [cy], c=color, marker='o', s=100, label=label)
            circle = fig.gca().add_patch(
                __import__('matplotlib.patches', fromlist=['Circle']).Circle(
                    (cx, cy), r, fill=False, color=color, linewidth=2))
        
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel('X (pixels)', color='white')
        ax.set_ylabel('Y (pixels)', color='white')
        ax.tick_params(colors='white')
        fig.tight_layout()
        
        layout.addWidget(canvas)
        
        # Add status label for hover info
        from PySide6.QtWidgets import QLabel
        status_label = QLabel("Hover over image to see pixel info")
        layout.addWidget(status_label)
        
        # Mouse motion event handler - use round() for correct pixel position
        def on_motion(event):
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                x, y = round(event.xdata), round(event.ydata)
                if 0 <= x < img_data.shape[1] and 0 <= y < img_data.shape[0]:
                    intensity = img_data[y, x]
                    status_label.setText(f"X={x}, Y={y}, Intensity={intensity:.1f}")
        
        canvas.mpl_connect('motion_notify_event', on_motion)
        
        # Button layout
        from PySide6.QtWidgets import QHBoxLayout
        btn_layout = QHBoxLayout()
        
        # Add Continue button
        btn_continue = QPushButton("Continue")
        btn_continue.clicked.connect(dialog.accept)
        btn_layout.addWidget(btn_continue)
        
        # Add Skip All button
        def on_skip_all():
            self._skip_debug = True
            dialog.accept()
        
        btn_skip = QPushButton("Skip All")
        btn_skip.clicked.connect(on_skip_all)
        btn_layout.addWidget(btn_skip)
        
        layout.addLayout(btn_layout)
        
        # exec() blocks until dialog is closed
        dialog.exec()
    
    def _find_correspondences_fallback(self, sampled_points: List[Tuple]) -> List[dict]:
        """
        Fallback: use stored projections when image detection is not available.
        NOTE: This is for testing only - results will not be accurate for VSC.
        """
        valid_correspondences = []
        n_cams = len(self.cameras)
        
        for pt in sampled_points:
            x, y, z, frame_id, r3d_mm, cam_2d = pt
            
            if len(cam_2d) < n_cams:
                continue
            
            pts_2d = {}
            all_valid = True
            
            for cam_idx in self.cameras.keys():
                if cam_idx not in cam_2d:
                    all_valid = False
                    break
                
                pt2d_data = cam_2d[cam_idx]
                pt2d_x = pt2d_data[0]
                pt2d_y = pt2d_data[1]
                pt2d_r = pt2d_data[2] if len(pt2d_data) > 2 else self.obj_radius
                
                if (pt2d_x < pt2d_r or 
                    pt2d_x >= self.image_size[1] - pt2d_r or
                    pt2d_y < pt2d_r or 
                    pt2d_y >= self.image_size[0] - pt2d_r):
                    all_valid = False
                    break
                
                pts_2d[cam_idx] = np.array([pt2d_x, pt2d_y])
            
            if all_valid:
                valid_correspondences.append({
                    'pt3d': np.array([x, y, z]),
                    'r3d_mm': float(r3d_mm),
                    '2d_per_cam': pts_2d
                })
        
        return valid_correspondences
    
    def _load_image_loaders(self, lpt) -> Dict[int, object]:
        """
        Load ImageIO objects for each camera using basic_settings._image_file_paths.
        
        Returns:
            Dict mapping cam_idx to ImageIO loader
        """
        img_loaders = {}
        
        # Load basic settings to get image paths
        config_path = os.path.join(self.proj_dir, "config.txt")
        if not os.path.exists(config_path):
            self._log(f"  Warning: config.txt not found")
            return img_loaders
        
        try:
            settings = lpt.BasicSetting()
            settings.readConfig(config_path)
            
            # Get image paths from settings
            image_paths = settings._image_file_paths
            
            if not image_paths:
                self._log(f"  Warning: No image paths found in settings")
                return img_loaders
            
            # Create ImageIO for each camera
            folder_base = os.path.abspath(self.proj_dir).replace('\\', '/') + '/'
            for cam_idx, path in enumerate(image_paths):
                try:
                    io = lpt.ImageIO()
                    io.loadImgPath(folder_base, path)
                    img_loaders[cam_idx] = io
                    self._log(f"  Loaded ImageIO for camera {cam_idx}: {path}")
                except Exception as e:
                    self._log(f"  Warning: Failed to load ImageIO for camera {cam_idx}: {e}")
            
        except Exception as e:
            self._log(f"  Warning: Error loading image paths: {e}")
        
        return img_loaders
    
    def _load_obj_config_for_detection(self):
        """Load ObjectConfig (TracerConfig or BubbleConfig) for ObjectFinder2D."""
        import pyopenlpt as lpt
        
        # Load basic settings first
        config_path = os.path.join(self.proj_dir, "config.txt")
        settings = lpt.BasicSetting()
        settings.readConfig(config_path)
        
        # Get config file from settings._object_config_paths[0]
        if not settings._object_config_paths or len(settings._object_config_paths) == 0:
            self._log("  Warning: No object config paths in settings")
            return None
        
        config_file = settings._object_config_paths[0]
        
        if self.obj_type == "Tracer":
            obj_cfg = lpt.TracerConfig()
        else:
            obj_cfg = lpt.BubbleConfig()
        
        if os.path.exists(config_file):
            obj_cfg.readConfig(config_file, settings)
            return obj_cfg
        else:
            self._log(f"  Warning: Config file not found: {config_file}")
        
        return None
    
    def _run_optimization(self, correspondences: List[dict]) -> Tuple[bool, str]:
        """Run model-aware camera optimization (PINHOLE or PINPLATE)."""
        self._log("Running joint camera optimization...")

        has_refraction = any(
            str(self.camera_models.get(cid, self.cameras.get(cid, {}).get('model', 'PINHOLE'))).upper() == 'PINPLATE'
            for cid in self.cameras.keys()
        )

        if has_refraction:
            try:
                import pyopenlpt as lpt
            except Exception as e:
                return False, f"Refraction VSC requires pyopenlpt: {e}"

            from .refraction_optimizer import RefractionVSCOptimizer

            self.cpp_cameras = {}
            camera_states = {}
            for cam_idx, cam in self.cameras.items():
                model = str(self.camera_models.get(cam_idx, cam.get('model', 'PINHOLE'))).upper()
                if model != 'PINPLATE':
                    continue
                cam_path = self.camera_paths.get(cam_idx, cam.get('file_path', None))
                if not cam_path:
                    return False, f"Missing camera file path for cam{cam_idx}"
                cpp_cam = lpt.Camera(cam_path)
                self.cpp_cameras[cam_idx] = cpp_cam

                # Use C++ camera state as source of truth (do not trust parsed rvec text).
                try:
                    pin = cpp_cam._pinplate_param
                    rvec_cpp = cpp_cam.rmtxTorvec(pin.r_mtx)
                    tvec_cpp = pin.t_vec
                    rvec = np.asarray([rvec_cpp[0], rvec_cpp[1], rvec_cpp[2]],
                                      dtype=np.float64).reshape(3)
                    tvec = np.asarray([tvec_cpp[0], tvec_cpp[1], tvec_cpp[2]],
                                      dtype=np.float64).reshape(3)
                except Exception:
                    # Fallback for compatibility if direct pinplate state is unavailable.
                    rvec = np.asarray(cam.get('rvec', np.zeros(3)), dtype=np.float64).reshape(3)
                    tvec = np.asarray(cam.get('tvec', np.zeros(3)), dtype=np.float64).reshape(3)
                camera_states[cam_idx] = {
                    'rvec': rvec,
                    'tvec': tvec,
                    'is_active': bool(getattr(cpp_cam, '_is_active', True)),
                    'max_intensity': float(getattr(cpp_cam, '_max_intensity', 255.0)),
                }

            if len(self.cpp_cameras) < 2:
                return False, "Refraction VSC needs at least 2 PINPLATE cameras"

            # Rebuild cam->window and window plane map from C++ camera state.
            # This avoids text-parser brittleness and ensures barrier uses loaded PINPLATE geometry.
            self._build_refraction_window_map_from_cpp()
            self._log(
                f"  Refraction map: cam_to_window={len(self.cam_to_window)}, "
                f"window_planes={len(self.window_planes)}"
            )

            # Keep only correspondences with at least 2 participating refraction cameras.
            corr_ref = []
            ref_cam_set = set(self.cpp_cameras.keys())
            for c in correspondences:
                obs = c.get('2d_per_cam', {})
                n_ref = sum(1 for cid in obs.keys() if cid in ref_cam_set)
                if n_ref >= 2:
                    corr_ref.append(c)

            if len(corr_ref) < 10:
                return False, f"Not enough PINPLATE correspondences ({len(corr_ref)})"

            window_refraction = self._build_window_refraction_from_cpp()

            self._log("  [Refraction VSC] Stage 1/3: optimize camera extrinsics (max_nfev=50)")
            optimizer_stage1 = RefractionVSCOptimizer(
                max_nfev=50,
                ftol=1e-6,
                xtol=1e-6,
                margin_side_mm=0.05,
                alpha_side_gate=10.0,
                beta_side_dir=1e4,
                tau=0.01,
            )
            optimizer_stage1.set_log_callback(self._log)
            optimized_states_stage1, info_stage1 = optimizer_stage1.optimize_all_cameras(
                self.cpp_cameras,
                camera_states,
                corr_ref,
                self.cam_to_window,
                self.window_planes,
            )

            self._log(
                "  [Refraction VSC] Stage 2/3: optimize window planes "
                "(delta_d<5mm, delta_theta<2deg, max_nfev=50)"
            )
            optimizer_stage2 = RefractionVSCOptimizer(
                max_nfev=50,
                ftol=1e-6,
                xtol=1e-6,
                margin_side_mm=0.05,
                alpha_side_gate=10.0,
                beta_side_dir=1e4,
                tau=0.01,
            )
            optimizer_stage2.set_log_callback(self._log)
            window_planes_stage2, info_stage2 = optimizer_stage2.optimize_window_planes(
                self.cpp_cameras,
                optimized_states_stage1,
                corr_ref,
                self.cam_to_window,
                self.window_planes,
                window_refraction,
                max_nfev=50,
                max_delta_d_mm=5.0,
                max_delta_theta_deg=2.0,
            )
            self.window_planes = window_planes_stage2

            self._log("  [Refraction VSC] Stage 3/3: refine camera extrinsics (max_nfev=20)")
            optimizer_stage3 = RefractionVSCOptimizer(
                max_nfev=20,
                ftol=1e-6,
                xtol=1e-6,
                margin_side_mm=0.05,
                alpha_side_gate=10.0,
                beta_side_dir=1e4,
                tau=0.01,
            )
            optimizer_stage3.set_log_callback(self._log)
            optimized_states, info = optimizer_stage3.optimize_all_cameras(
                self.cpp_cameras,
                optimized_states_stage1,
                corr_ref,
                self.cam_to_window,
                self.window_planes,
            )
            self._overlay_points_optim = info.get('overlay_points_optim', {})

            # Reflect optimized extrinsics back to parsed camera dict.
            import cv2
            for cam_idx, st in optimized_states.items():
                self.cameras[cam_idx]['rvec'] = np.asarray(st['rvec'], dtype=np.float64)
                self.cameras[cam_idx]['tvec'] = np.asarray(st['tvec'], dtype=np.float64)
                R, _ = cv2.Rodrigues(self.cameras[cam_idx]['rvec'])
                self.cameras[cam_idx]['R'] = R
                self.cameras[cam_idx]['R_inv'] = R.T
                self.cameras[cam_idx]['tvec_inv'] = (-R.T @ self.cameras[cam_idx]['tvec'].reshape(3, 1)).flatten()

            stats = info.get('full_stats', {})
            proj_mean = stats.get('proj_mean', 0.0)
            proj_std = stats.get('proj_std', 0.0)
            triang_mean = stats.get('triang_mean', 0.0)
            triang_std = stats.get('triang_std', 0.0)
            self._log(
                f"  Final ProjErr: {proj_mean:.4f} ± {proj_std:.4f} px "
                f"(Tol: {stats.get('proj_tol', 0.0):.4f})"
            )
            self._log(
                f"  Final TriangErr: {triang_mean:.4f} ± {triang_std:.4f} mm "
                f"(Tol: {stats.get('triang_tol', 0.0):.4f})"
            )

            for cam_idx in self.cpp_cameras.keys():
                self.cameras[cam_idx]['proj_error'] = (proj_mean, proj_std)
                self.cameras[cam_idx]['triang_error'] = (triang_mean, triang_std)

            # Update tolerance fields using available keys.
            stats_for_cfg = {
                'proj_tol': stats.get('proj_tol', proj_mean + 3.0 * proj_std),
                'triang_tol': stats.get('triang_tol', triang_mean + 3.0 * triang_std),
            }
            self._update_object_config(stats_for_cfg)
            return True, ""

        # PINHOLE path (existing behavior)
        from .optimizer import VSCOptimizer

        optimizer = VSCOptimizer()
        optimizer.set_log_callback(self._log)

        optimized_cameras, info = optimizer.optimize_all_cameras(
            self.cameras, correspondences, self.image_size
        )
        self.cameras = optimized_cameras
        self._overlay_points_optim = {}

        if 'full_stats' in info:
            stats = info['full_stats']
            proj_mean = stats.get('proj_mean', 0.0)
            proj_std = stats.get('proj_std', 0.0)
            triang_mean = stats.get('triang_mean', 0.0)
            triang_std = stats.get('triang_std', 0.0)

            self._log(f"  Final ProjErr: {proj_mean:.4f} ± {proj_std:.4f} px (Tol: {stats.get('proj_tol',0):.4f})")
            self._log(f"  Final TriangErr: {triang_mean:.4f} ± {triang_std:.4f} mm (Tol: {stats.get('triang_tol',0):.4f})")

            for cam_idx in self.cameras:
                self.cameras[cam_idx]['proj_error'] = (proj_mean, proj_std)
                self.cameras[cam_idx]['triang_error'] = (triang_mean, triang_std)

            self._update_object_config(stats)
        else:
            self._log("  Computing final error statistics...")
            proj_errors, triang_errors = self._compute_final_errors(correspondences)

            if len(proj_errors) > 0:
                proj_mean = np.mean(proj_errors)
                proj_std = np.std(proj_errors)
                triang_mean = np.mean(triang_errors)
                triang_std = np.std(triang_errors)

                self._log(f"  Final ProjErr: {proj_mean:.4f} ± {proj_std:.4f} px")
                self._log(f"  Final TriangErr: {triang_mean:.4f} ± {triang_std:.4f} mm")

                for cam_idx in self.cameras:
                    self.cameras[cam_idx]['proj_error'] = (proj_mean, proj_std)
                    self.cameras[cam_idx]['triang_error'] = (triang_mean, triang_std)
            else:
                self._log("  Warning: No valid errors computed")
                for cam_idx in self.cameras:
                    self.cameras[cam_idx]['proj_error'] = (0.0, 0.0)
                    self.cameras[cam_idx]['triang_error'] = (0.0, 0.0)

        return True, ""
    
    def _compute_final_errors(self, correspondences: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute final projection and triangulation errors using optimized cameras.
        
        Returns:
            proj_errors: Array of projection errors (pixels)
            triang_errors: Array of triangulation errors (mm)
        """
        import cv2
        
        # Build projection matrices for triangulation
        P_matrices = {}
        cam_params = {}
        for cam_idx, cam in self.cameras.items():
            K = cam['K']
            R = cam['R']
            tvec = cam['tvec'].reshape(3, 1)
            P = K @ np.hstack([R, tvec])
            P_matrices[cam_idx] = P
            
            R_inv = cam.get('R_inv', R.T)
            C = (-R_inv @ tvec).flatten()
            
            rvec, _ = cv2.Rodrigues(R)
            cam_params[cam_idx] = {
                'K': K, 'R': R, 'R_inv': R_inv, 'tvec': cam['tvec'],
                'rvec': rvec.flatten(), 'dist': cam.get('dist', np.zeros(5)),
                'C': C
            }
        
        proj_errors = []
        triang_errors = []
        
        for corr in correspondences:
            pts_2d = corr['2d_per_cam']
            
            # 1. Triangulate 3D point from observed 2D points
            pt3d = self._triangulate_dlt(P_matrices, pts_2d)
            if pt3d is None:
                continue
            
            # 2. For each camera, compute errors
            for cam_idx, pt_2d_obs in pts_2d.items():
                if cam_idx not in cam_params:
                    continue
                cp = cam_params[cam_idx]
                
                # Projection error: project 3D -> 2D, compare with observed
                projected, _ = cv2.projectPoints(
                    pt3d.reshape(1, 3), cp['rvec'], cp['tvec'], cp['K'], cp['dist']
                )
                projected = projected.reshape(2)
                proj_err = np.sqrt((projected[0] - pt_2d_obs[0])**2 + (projected[1] - pt_2d_obs[1])**2)
                proj_errors.append(proj_err)
                
                # Triangulation error: point-to-ray distance
                pts_2d_undist = cv2.undistortPoints(
                    pt_2d_obs.reshape(1, 1, 2), cp['K'], cp['dist']
                ).reshape(2)
                
                ray_cam = np.array([pts_2d_undist[0], pts_2d_undist[1], 1.0])
                ray_cam = ray_cam / np.linalg.norm(ray_cam)
                ray_world = cp['R_inv'] @ ray_cam
                ray_world = ray_world / np.linalg.norm(ray_world)
                
                v = pt3d - cp['C']
                proj_len = np.dot(v, ray_world)
                perp = v - proj_len * ray_world
                triang_err = np.linalg.norm(perp)
                triang_errors.append(triang_err)
        
        return np.array(proj_errors), np.array(triang_errors)
    
    def _triangulate_dlt(self, P_matrices: dict, pts_2d: dict) -> Optional[np.ndarray]:
        """Triangulate 3D point using DLT."""
        if len(pts_2d) < 2:
            return None
        
        A = []
        for cam_idx, pt_2d in pts_2d.items():
            if cam_idx not in P_matrices:
                continue
            P = P_matrices[cam_idx]
            x, y = pt_2d[0], pt_2d[1]
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        if len(A) < 4:
            return None
        
        A = np.array(A)
        try:
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            X = X[:3] / X[3]
            return X
        except:
            return None
    
    def _save_cameras(self):
        """Save optimized camera parameters to vsc_cam*.txt files."""
        # Set output directory to camFile_VSC
        cam_dir = os.path.join(self.proj_dir, "camFile_VSC")
        
        # Ensure directory exists
        os.makedirs(cam_dir, exist_ok=True)
        
        self._log(f"  Output directory: {cam_dir}")
        
        for cam_idx, cam_params in self.cameras.items():
            output_path = os.path.join(cam_dir, f"vsc_cam{cam_idx}.txt")
            model = str(self.camera_models.get(cam_idx, cam_params.get('model', 'PINHOLE'))).upper()

            if model == 'PINPLATE':
                cpp_cam = self.cpp_cameras.get(cam_idx)
                if cpp_cam is None:
                    self._log(f"  Warning: Missing PINPLATE camera handle for cam{cam_idx}; skipping save")
                    continue
                cpp_cam.saveParameters(output_path)

                # Patch error-stat lines because current C++ saveParameters() writes
                # these two fields as "None" for pinhole/pinplate models.
                proj_error = cam_params.get('proj_error', None)
                tri_error = cam_params.get('triang_error', None)
                self._patch_camfile_error_stats(output_path, proj_error, tri_error)
                self._patch_camfile_refraction_meta(output_path, cam_idx)

                self._log(f"  Saved Camera {cam_idx} to vsc_cam{cam_idx}.txt")
                continue
            
            # Get error stats
            proj_error = cam_params.get('proj_error', None)
            tri_error = cam_params.get('triang_error', None)
            
            save_camera_file(output_path, cam_params, proj_error, tri_error)
            self._log(f"  Saved Camera {cam_idx} to vsc_cam{cam_idx}.txt")

    def _patch_camfile_error_stats(self, file_path: str, proj_error: tuple = None, tri_error: tuple = None):
        """Replace camera/pose error lines in existing cam file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            proj_line = "None\n"
            tri_line = "None\n"
            if proj_error and len(proj_error) >= 2:
                proj_line = f"{float(proj_error[0]):.8g},{float(proj_error[1]):.8g}\n"
            if tri_error and len(tri_error) >= 2:
                tri_line = f"{float(tri_error[0]):.8g},{float(tri_error[1]):.8g}\n"

            def patch_after_header(header: str, value_line: str):
                for i, line in enumerate(lines):
                    if header in line and i + 1 < len(lines):
                        lines[i + 1] = value_line
                        return True
                return False

            ok_proj = patch_after_header("# Camera Calibration Error", proj_line)
            ok_tri = patch_after_header("# Pose Calibration Error", tri_line)

            if ok_proj or ok_tri:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            else:
                self._log(f"  Warning: Could not patch error stats in {os.path.basename(file_path)}")
        except Exception as e:
            self._log(f"  Warning: Failed to patch error stats for {os.path.basename(file_path)}: {e}")

    def _patch_camfile_refraction_meta(self, file_path: str, cam_idx: int):
        """Upsert REFRACTION_META block for PINPLATE camera files."""
        try:
            wid = self.cam_to_window.get(int(cam_idx), None)
            if wid is None:
                self._log(
                    f"  Warning: cam{cam_idx} missing cam_to_window mapping; "
                    f"skip REFRACTION_META patch"
                )
                return

            plane = self.window_planes.get(int(wid), None)
            if not plane:
                self._log(
                    f"  Warning: cam{cam_idx} window {wid} missing plane data; "
                    f"skip REFRACTION_META patch"
                )
                return

            if 'plane_pt' not in plane or 'plane_n' not in plane:
                self._log(
                    f"  Warning: cam{cam_idx} window {wid} plane fields incomplete; "
                    f"skip REFRACTION_META patch"
                )
                return

            plane_pt = np.asarray(plane['plane_pt'], dtype=np.float64).reshape(3)
            plane_n = np.asarray(plane['plane_n'], dtype=np.float64).reshape(3)

            if not (np.all(np.isfinite(plane_pt)) and np.all(np.isfinite(plane_n))):
                self._log(
                    f"  Warning: cam{cam_idx} has non-finite plane data; "
                    f"skip REFRACTION_META patch"
                )
                return

            block = (
                "# --- BEGIN_REFRACTION_META ---\n"
                "# VERSION=2\n"
                f"# CAM_ID={int(cam_idx)}\n"
                f"# WINDOW_ID={int(wid)}\n"
                f"# PLANE_PT_EXPORT=[{plane_pt[0]:.4f},{plane_pt[1]:.4f},{plane_pt[2]:.4f}]\n"
                f"# PLANE_N=[{plane_n[0]:.6f},{plane_n[1]:.6f},{plane_n[2]:.6f}]\n"
                "# --- END_REFRACTION_META ---\n"
            )

            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            pattern = re.compile(
                r"#\s*---\s*BEGIN_REFRACTION_META\s*---.*?#\s*---\s*END_REFRACTION_META\s*---\s*\n?",
                flags=re.IGNORECASE | re.DOTALL,
            )

            if pattern.search(text):
                text_new = pattern.sub(block, text, count=1)
            else:
                if text and not text.endswith("\n"):
                    text += "\n"
                text_new = text + block

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_new)
        except Exception as e:
            self._log(f"  Warning: Failed to patch REFRACTION_META for cam{cam_idx}: {e}")
    
    def _update_config(self):
        """Update config.txt to use optimized camera files."""
        config_path = os.path.join(self.proj_dir, "config.txt")
        if not os.path.exists(config_path):
            return
        
        # Backup original config
        backup_path = os.path.join(self.proj_dir, "config_backup.txt")
        if not os.path.exists(backup_path):
            shutil.copy(config_path, backup_path)
            self._log(f"  Backup saved to config_backup.txt")
        
        # Read all lines
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        in_camera_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            if '# Camera File Path' in line_stripped:
                in_camera_section = True
                new_lines.append(line)
                continue
            
            if in_camera_section:
                if line_stripped.startswith('#'):
                    in_camera_section = False
                    new_lines.append(line)
                    continue
                
                # Check if this line defines a camera
                # Look for 'camX.txt' or 'vsc_camX.txt'
                match = re.search(r'(?:vsc_)?cam(\d+)\.txt', line_stripped)
                if match:
                    cam_idx = int(match.group(1))
                    if cam_idx in self.cameras:
                        # Construct new line
                        # Preserve intensity part if exists (e.g. ",255")
                        parts = line_stripped.split(',')
                        suffix = ""
                        if len(parts) > 1:
                            # Reconstruct everything after the first comma
                            suffix = "," + ",".join(parts[1:])
                        
                        # Construct absolute path with forward slashes
                        abs_path = os.path.abspath(os.path.join(self.proj_dir, "camFile_VSC", f"vsc_cam{cam_idx}.txt"))
                        abs_path = abs_path.replace(os.sep, '/')
                        new_line = f"{abs_path}{suffix}\n"
                        new_lines.append(new_line)
                        continue
            
            new_lines.append(line)
        
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        
        self._log(f"  Updated config.txt with optimized camera paths")



    def _update_object_config(self, stats: dict):
        """
        Update ObjectConfig (e.g. tracerConfig.txt) with new tolerances.
        tolerance = Mean + 3 * Std
        """
        config_path = os.path.join(self.proj_dir, "config.txt")
        if not os.path.exists(config_path):
            return
            
        with open(config_path, 'r') as f:
            lines = f.readlines()
            
        # Parse config.txt for Voxel to MM and Object Config path
        voxel_to_mm = 1.0
        obj_cfg_path = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            # Find Voxel to MM factor
            if "Voxel to MM" in line or "voxel, (xmax-xmin)/1000" in line:
                # Usually the NEXT line has the value
                if i + 1 < len(lines):
                    try:
                        val_line = lines[i+1].strip()
                        # Handle comments
                        val_str = val_line.split('#')[0].strip()
                        voxel_to_mm = float(val_str)
                    except:
                        pass
            
            # Find Object Config Path
            if "# STB Config Files" in line:
                # Next few lines might be it. Look for .txt
                # Heuristic: Look for tracerConfig or similar in subsequent lines
                for j in range(1, 4):
                    if i + j < len(lines):
                         sub_line = lines[i+j].strip()
                         # Should be a path
                         if sub_line.endswith(".txt") and "Config" in sub_line:
                             # Check if absolute or relative
                             if os.path.isabs(sub_line):
                                 obj_cfg_path = sub_line
                             else:
                                 obj_cfg_path = os.path.join(self.proj_dir, sub_line)
                             break
        
        if not obj_cfg_path or not os.path.exists(obj_cfg_path):
            self._log(f"  Could not find Object Config file in config.txt")
            return
            
        self._log(f"  Found Object Config: {obj_cfg_path}")
        self._log(f"  Voxel to MM factor: {voxel_to_mm}")
        
        # Calculate new tolerances
        # 2D Tolerance (px) = Proj Mean + 3 * Std
        # 3D Tolerance (voxel) = (Triang Mean + 3 * Std) / Voxel_to_MM
        
        # Note: input stats uses 'tol' keys calculated as Mean + 3*Std
        tol_2d_px = stats.get('proj_tol', 0.5)
        tol_3d_mm = stats.get('triang_tol', 0.1)
        
        tol_3d_voxel = tol_3d_mm / voxel_to_mm if voxel_to_mm > 1e-9 else tol_3d_mm
        
        self._log(f"  New 2D Tolerance: {tol_2d_px:.4f} px")
        self._log(f"  New 3D Tolerance: {tol_3d_voxel:.4f} voxel (from {tol_3d_mm:.4f} mm)")
        
        # Update Object Config
        # Backup first
        base, ext = os.path.splitext(obj_cfg_path)
        backup_cfg = f"{base}_backup{ext}"
        shutil.copy(obj_cfg_path, backup_cfg)
        
        with open(obj_cfg_path, 'r') as f:
            cfg_lines = f.readlines()
            
        # We need to replace specific lines. 
        # Heuristic: 
        # Line with "2D tolerance" comment
        # Line with "3D tolerance" comment
        
        new_lines = []
        for line in cfg_lines:
            if "2D tolerance" in line:
                # Replace number before comment
                parts = line.split('#')
                comment = parts[1] if len(parts) > 1 else " 2D tolerance [px]"
                new_line = f"{tol_2d_px:.4f}\t# {comment.strip()}\n"
                new_lines.append(new_line)
            elif "3D tolerance" in line:
                parts = line.split('#')
                comment = parts[1] if len(parts) > 1 else " 3D tolerance [voxel]"
                new_line = f"{tol_3d_voxel:.4f}\t# {comment.strip()}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
                
        with open(obj_cfg_path, 'w') as f:
            f.writelines(new_lines)
            
        self._log(f"  Updated Object Config with new tolerances.")


# Global context for worker processes
_worker_context = {}

def init_worker(proj_dir: str, params: Dict, cameras_indices: List[int]):
    """
    Initializer function for worker processes.
    Loads resources ONCE when the process starts.
    """
    import os
    import sys
    
    global _worker_context
    _worker_context = {}
    _worker_context['proj_dir'] = proj_dir
    
    try:
        import pyopenlpt as lpt
        _worker_context['lpt'] = lpt
    except ImportError:
        _worker_context['error'] = 'ImportError'
        return

    try:
        # 1. Load Settings
        config_path = os.path.join(proj_dir, "config.txt")
        settings = lpt.BasicSetting()
        settings.readConfig(config_path)
        
        # 2. Load Object Config
        obj_cfg = None
        if settings._object_config_paths and len(settings._object_config_paths) > 0:
            cfg_path = settings._object_config_paths[0]
            if params['obj_type'] == "Tracer":
                obj_cfg = lpt.TracerConfig()
            else:
                obj_cfg = lpt.BubbleConfig()
            
            # TracerConfig/BubbleConfig.readConfig requires (filepath, settings)
            obj_cfg.readConfig(cfg_path, settings)
            
        _worker_context['obj_cfg'] = obj_cfg
        
        # 3. Initialize ImageIOs (don't load images yet)
        image_paths = settings._image_file_paths
        img_ios = {}
        
        for cam_idx in cameras_indices:
            if cam_idx < len(image_paths):
                try:
                    io = lpt.ImageIO()
                    folder_base = os.path.abspath(proj_dir).replace('\\', '/') + '/'
                    io.loadImgPath(folder_base, image_paths[cam_idx])
                    img_ios[cam_idx] = io
                except Exception:
                    pass
        
        _worker_context['img_ios'] = img_ios
        _worker_context['setup_done'] = True
        
    except Exception as e:
        _worker_context['error'] = str(e)


def process_frame_task(frame_id: int, points: List[Tuple], 
                      image_size: Tuple, params: Dict) -> Tuple[List[dict], Dict]:
    """
    Worker function for ProcessPoolExecutor.
    Uses pre-initialized resources from _worker_context.
    """
    import numpy as np
    
    valid_corrs = []
    stats = {
        'processed': 0, 'valid': 0, 
        'roi': 0, 'no_detection': 0, 
        'distance': 0, 'not_isolated': 0, 
        'exception': 0, 'error': 0
    }
    
    global _worker_context
    if 'error' in _worker_context:
        stats['error'] = _worker_context['error']
        return [], stats
        
    if not _worker_context.get('setup_done', False):
        stats['error'] = 'Worker setup failed'
        return [], stats

    try:
        lpt = _worker_context['lpt']
        obj_cfg = _worker_context['obj_cfg']
        img_ios = _worker_context['img_ios']
        
        if obj_cfg is None:
            return [], {'error': 'No Obj Config'}
            
        # Load Images for this frame
        frame_images = {}
        # Only load images for cameras we have IOs for
        # It's quicker to iterate img_ios than loading everything
        for cam_idx, io in img_ios.items():
            try:
                img = io.loadImg(frame_id)
                frame_images[cam_idx] = img
            except:
                pass
        
        if not frame_images:
            return [], stats
            
        # Create Finder (thread/process local is fine, it's light)
        finder = lpt.ObjectFinder2D()
        
        # Use available cameras from initialized IOs
        available_cameras = sorted(img_ios.keys())
        n_cams_total = len(available_cameras)
        
        # Process Points
        for pt in points:
            x, y, z, fid, r3d_mm, cam_2d = pt
            stats['processed'] += 1
            
            # Use total cameras derived from IO availability
            if len(cam_2d) < n_cams_total:
                continue
                
            pts_2d_detected = {}
            all_valid = True
            fail_reason = None
            
            for cam_idx in available_cameras:
                if cam_idx not in cam_2d or cam_idx not in frame_images:
                    all_valid = False
                    break
                
                proj_x, proj_y = cam_2d[cam_idx][0], cam_2d[cam_idx][1]
                point_r = cam_2d[cam_idx][2] if len(cam_2d[cam_idx]) > 2 else params['obj_radius']
                
                margin = params['margin_factor'] * point_r
                
                # ROI
                x0 = max(0, int(proj_x - margin))
                x1 = min(image_size[1], int(proj_x + margin))
                y0 = max(0, int(proj_y - margin))
                y1 = min(image_size[0], int(proj_y + margin))
                
                if x1 <= x0 or y1 <= y0:
                    all_valid = False
                    fail_reason = "roi"
                    break
                
                # Check bounds
                full_img = frame_images[cam_idx]
                n_row = full_img.getDimRow()
                n_col = full_img.getDimCol()
                
                if x0 >= n_col or y0 >= n_row:
                    all_valid = False
                    fail_reason = "roi"
                    break

                try:
                    roi_h = y1 - y0
                    roi_w = x1 - x0
                    roi_img = lpt.Image(roi_h, roi_w, 0.0)
                    
                    # Pixel Copy
                    for i in range(roi_h):
                        for j in range(roi_w):
                            r, c = y0 + i, x0 + j
                            if r < n_row and c < n_col:
                                roi_img[i, j] = full_img[r, c]
                    
                    detections = finder.findObject2D(roi_img, obj_cfg)
                    
                    if len(detections) == 0:
                        all_valid = False
                        fail_reason = "no_detection"
                        break
                        
                    best_det = None
                    if len(detections) > 1:
                        roi_proj_x = proj_x - x0
                        roi_proj_y = proj_y - y0
                        best_dist = float('inf')
                        
                        for det in detections:
                            dx = det._pt_center[0] - roi_proj_x
                            dy = det._pt_center[1] - roi_proj_y
                            d = (dx*dx+dy*dy)**0.5
                            if d < best_dist:
                                best_dist = d
                                best_det = det
                        
                        # Isolation
                        isolated = True
                        is_bubble = (params['obj_type'] == "Bubble")
                        for det in detections:
                            if det is not best_det:
                                dx = det._pt_center[0] - best_det._pt_center[0]
                                dy = det._pt_center[1] - best_det._pt_center[1]
                                sep = (dx*dx+dy*dy)**0.5
                                min_sep = point_r + det._r_px + params['isolation_margin'] if is_bubble else point_r*2 + params['isolation_margin']
                                if sep < min_sep:
                                    isolated = False
                                    break
                        if not isolated:
                            all_valid = False
                            fail_reason = "not_isolated"
                            break
                    else:
                        best_det = detections[0]
                    
                    # Distance check
                    det_x = best_det._pt_center[0] + x0
                    det_y = best_det._pt_center[1] + y0
                    
                    dist_proj = ((det_x - proj_x)**2 + (det_y - proj_y)**2)**0.5
                    
                    tol_mode = params.get('tolerance_mode', 'default')
                    if tol_mode == 'reproj':
                        reproj_thresholds = params.get('reproj_thresholds', {})
                        if cam_idx not in reproj_thresholds:
                            raise KeyError(f"Missing reprojection threshold for camera {cam_idx}")
                        dist_threshold = reproj_thresholds[cam_idx]
                    elif tol_mode == 'custom':
                        dist_threshold = params.get('tolerance_value', 5.0)
                    else:
                        dist_threshold = point_r + 1
                    
                    if dist_proj > dist_threshold:
                        all_valid = False
                        fail_reason = "distance"
                        break
                        
                    pts_2d_detected[cam_idx] = np.array([det_x, det_y])
                    
                except Exception:
                    all_valid = False
                    fail_reason = "exception"
                    break
            
            if all_valid:
                pts_2d_csv = {
                    int(c_idx): np.array([float(vals[0]), float(vals[1])], dtype=np.float64)
                    for c_idx, vals in cam_2d.items()
                    if vals is not None and len(vals) >= 2
                }
                valid_corrs.append({
                    'frame_id': fid,
                    'pt3d': np.array([x, y, z]),
                    'r3d_mm': float(r3d_mm),
                    '2d_per_cam': pts_2d_detected,
                    '2d_csv_per_cam': pts_2d_csv,
                })
                stats['valid'] += 1
            else:
                if fail_reason: stats[fail_reason] += 1
                
    except Exception as e:
        stats['exception'] += 1
        stats['error'] = str(e)
        
    return valid_corrs, stats
