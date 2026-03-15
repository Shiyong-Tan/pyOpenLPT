#!/usr/bin/env python
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false
"""
Test Infrastructure for Refractive Bootstrap Bug Fixes
=======================================================

This module provides synthetic data factories and pytest fixtures to enable
Test-Driven Development (TDD) for all 21 bootstrap bug fixes.

DESIGN PRINCIPLES:
- Synthetic data with REALISTIC parameters (actual camera/wand geometry)
- NO actual test cases yet — fixtures/factories ONLY
- All test cases will be written in W1a-W1e using RED-GREEN-REFACTOR
- Supports 3 bootstrap phases: Phase 1 (essential matrix), Phase 2 (PnP), Phase 3 (BA)

USAGE:
    This is the PREREQUISITE for waves W1a-1e. The actual test cases will be
    added as each bug is fixed. Currently this file contains:
    - 3 synthetic data factory functions
    - 5 pytest fixtures
    - ZERO test cases (tests come later)

Run:
    conda run -n OpenLPT python -m pytest modules/camera_calibration/wand_calibration/test_refractive_bootstrap.py --collect-only -q
"""

from __future__ import annotations

import pytest
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

from modules.camera_calibration.wand_calibration.refractive_bootstrap import (
    PinholeBootstrapP0,
    PinholeBootstrapP0Config,
)

# Import bootstrap classes (ensure these resolve correctly)
# Note: If BootstrapObservations/FrameObservations don't exist as formal classes,
# we'll use dict-based structures matching the actual bootstrap code's expectations


# ============================================================================
# SYNTHETIC DATA FACTORY FUNCTIONS
# ============================================================================

def make_bootstrap_observations(
    n_frames: int = 5,
    n_cameras: int = 4,
    n_points: int = 20,
    wand_length_m: float = 0.3,
    workspace_size_m: float = 0.5,
    projection_noise_px: float = 0.5,
    point_radius_m: float = 0.005,
    random_seed: int = 42
) -> Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Create synthetic bootstrap observations with realistic geometry.
    
    Simulates a 4-camera system observing a wand moving through workspace.
    Each frame contains observations from all cameras (2 wand endpoints visible).
    
    Args:
        n_frames: Number of frames (default: 5)
        n_cameras: Number of cameras (default: 4, arranged in cube)
        n_points: Number of 3D points per frame (default: 20, scattered in workspace)
        wand_length_m: Wand length in meters (default: 0.3m = 300mm)
        workspace_size_m: Workspace cube size (default: 0.5m)
        projection_noise_px: Gaussian projection noise std dev (default: 0.5px)
        point_radius_m: Physical point radius (default: 0.005m = 5mm)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        observations: {frame_id: {cam_id: (uvA, uvB)}} where
            - uvA: (2,) array [u, v] for wand endpoint A in pixels
            - uvB: (2,) array [u, v] for wand endpoint B in pixels
            
    Geometry:
        - Cameras arranged in ~1m cube around origin (realistic lab setup)
        - Wand endpoints separated by wand_length_m
        - Each frame has different wand pose (rotation + translation)
        - Projection uses pinhole camera model with realistic focal length
        
    Example:
        >>> obs = make_bootstrap_observations(n_frames=3, n_cameras=2)
        >>> obs[0][0]  # frame 0, camera 0
        (array([512.3, 384.7]), array([612.1, 390.2]))  # uvA, uvB in pixels
    """
    rng = np.random.RandomState(random_seed)
    
    # Camera intrinsics (realistic values for 1024x1024 sensor)
    focal_length_px = 9000.0  # ~9000 px focal length (common for calibration)
    img_width_px = 1024
    img_height_px = 1024
    cx = img_width_px / 2.0
    cy = img_height_px / 2.0
    
    # Camera extrinsics: arrange cameras in cube around origin
    # Cameras at corners looking inward (realistic multi-camera setup)
    camera_positions = []
    camera_rotations = []
    cube_radius = 0.5  # 0.5m from origin
    
    for i in range(n_cameras):
        # Distribute cameras around a cube
        angle_h = (i / n_cameras) * 2 * np.pi  # horizontal angle
        angle_v = 0.0  # all at same height initially
        
        # Camera position on sphere/cube
        x = cube_radius * np.cos(angle_h)
        y = cube_radius * np.sin(angle_h)
        z = 0.2 + (i % 2) * 0.1  # slight vertical offset
        cam_pos = np.array([x, y, z])
        camera_positions.append(cam_pos)
        
        # Camera rotation: look at origin
        # Simple: align camera Z-axis to point toward origin
        look_dir = -cam_pos / np.linalg.norm(cam_pos)
        # Build rotation matrix (simplified - Z toward target)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, look_dir)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(look_dir, right)
        R = np.column_stack([right, up_corrected, look_dir])
        camera_rotations.append(R)
    
    # Generate observations for each frame
    observations = {}
    
    for fid in range(n_frames):
        observations[fid] = {}
        
        # Generate wand pose for this frame
        # Wand center in workspace (uniformly distributed)
        wand_center = rng.uniform(
            -workspace_size_m / 4, workspace_size_m / 4, size=3
        )
        
        # Wand orientation (random unit vector)
        wand_dir = rng.randn(3)
        wand_dir = wand_dir / np.linalg.norm(wand_dir)
        
        # Wand endpoints A and B
        X_A = wand_center - (wand_length_m / 2.0) * wand_dir
        X_B = wand_center + (wand_length_m / 2.0) * wand_dir
        
        # Project into each camera
        for cam_id in range(n_cameras):
            cam_pos = camera_positions[cam_id]
            R_cam = camera_rotations[cam_id]
            
            # World to camera transformation
            # X_cam = R_cam^T * (X_world - cam_pos)
            X_A_cam = R_cam.T @ (X_A - cam_pos)
            X_B_cam = R_cam.T @ (X_B - cam_pos)
            
            # Project to image plane (pinhole model)
            # u = f * x/z + cx
            # v = f * y/z + cy
            if X_A_cam[2] > 0.01 and X_B_cam[2] > 0.01:  # In front of camera
                uA = focal_length_px * X_A_cam[0] / X_A_cam[2] + cx
                vA = focal_length_px * X_A_cam[1] / X_A_cam[2] + cy
                
                uB = focal_length_px * X_B_cam[0] / X_B_cam[2] + cx
                vB = focal_length_px * X_B_cam[1] / X_B_cam[2] + cy
                
                # Add projection noise
                uvA = np.array([uA, vA]) + rng.randn(2) * projection_noise_px
                uvB = np.array([uB, vB]) + rng.randn(2) * projection_noise_px
                
                observations[fid][cam_id] = (uvA, uvB)
    
    return observations


def make_camera_settings(
    n_cameras: int = 4,
    focal_px: float = 9000.0,
    img_width: int = 1024,
    img_height: int = 1024
) -> Dict[int, dict]:
    """
    Create camera settings dictionary for bootstrap initialization.
    
    Args:
        n_cameras: Number of cameras (default: 4)
        focal_px: Focal length in pixels (default: 9000.0)
        img_width: Image width in pixels (default: 1024)
        img_height: Image height in pixels (default: 1024)
        
    Returns:
        camera_settings: {cam_id: {'focal': float, 'width': int, 'height': int}}
        
    Example:
        >>> settings = make_camera_settings(n_cameras=2)
        >>> settings[0]
        {'focal': 9000.0, 'width': 1024, 'height': 1024}
    """
    camera_settings = {}
    for cam_id in range(n_cameras):
        camera_settings[cam_id] = {
            'focal': focal_px,
            'width': img_width,
            'height': img_height
        }
    return camera_settings


def make_known_geometry(
    n_cameras: int = 4,
    wand_length_m: float = 0.3,
    workspace_bounds_m: float = 0.5
) -> dict:
    """
    Create ground truth geometry for validation tests.
    
    Contains known camera poses, wand length, and workspace bounds that can
    be used to validate bootstrap output against expected values.
    
    Args:
        n_cameras: Number of cameras (default: 4)
        wand_length_m: Wand length in meters (default: 0.3m)
        workspace_bounds_m: Workspace size (default: 0.5m cube)
        
    Returns:
        geometry: dict with keys:
            - 'wand_length_m': float
            - 'workspace_bounds_m': float
            - 'camera_positions': {cam_id: (3,) array}
            - 'camera_rotations': {cam_id: (3,3) rotation matrix}
            - 'expected_baseline_m': Expected baseline between cameras
            
    Example:
        >>> geom = make_known_geometry(n_cameras=2)
        >>> geom['wand_length_m']
        0.3
        >>> geom['camera_positions'][0].shape
        (3,)
    """
    camera_positions = {}
    camera_rotations = {}
    cube_radius = 0.5  # meters
    
    for cam_id in range(n_cameras):
        angle_h = (cam_id / n_cameras) * 2 * np.pi
        x = cube_radius * np.cos(angle_h)
        y = cube_radius * np.sin(angle_h)
        z = 0.2 + (cam_id % 2) * 0.1
        cam_pos = np.array([x, y, z])
        camera_positions[cam_id] = cam_pos
        
        # Simple rotation: look at origin
        look_dir = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, look_dir)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(look_dir, right)
        R = np.column_stack([right, up_corrected, look_dir])
        camera_rotations[cam_id] = R
    
    # Expected baseline (distance between first two cameras)
    if n_cameras >= 2:
        baseline = np.linalg.norm(camera_positions[1] - camera_positions[0])
    else:
        baseline = 0.0
    
    return {
        'wand_length_m': wand_length_m,
        'workspace_bounds_m': workspace_bounds_m,
        'camera_positions': camera_positions,
        'camera_rotations': camera_rotations,
        'expected_baseline_m': baseline,
    }


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def bootstrap_obs_fixture():
    """
    Fixture: Valid bootstrap observations for 5 frames, 4 cameras.
    
    Returns observations dict matching bootstrap expectations:
        {frame_id: {cam_id: (uvA, uvB)}}
        
    Use this for general bootstrap tests requiring valid input data.
    """
    return make_bootstrap_observations(
        n_frames=5,
        n_cameras=4,
        n_points=20,
        random_seed=42
    )


@pytest.fixture
def camera_settings_fixture():
    """
    Fixture: Standard 4-camera configuration with realistic intrinsics.
    
    Returns camera_settings dict:
        {cam_id: {'focal': float, 'width': int, 'height': int}}
        
    Use this for bootstrap initialization and camera setup tests.
    """
    return make_camera_settings(n_cameras=4)


@pytest.fixture
def known_geometry_fixture():
    """
    Fixture: Ground truth geometry for validation.
    
    Returns geometry dict with known camera poses and wand parameters.
    Use this for tests that need to validate bootstrap output against
    expected values.
    """
    return make_known_geometry(n_cameras=4)


@pytest.fixture
def phase1_data_fixture():
    """
    Fixture: Phase 1 specific data (essential matrix estimation).
    
    Phase 1 uses camera pair (cam_i, cam_j) with 8-Point Algorithm.
    Returns dict with:
        - 'observations': synthetic observations
        - 'camera_settings': camera intrinsics
        - 'cam_i': int (camera 0, fixed at origin)
        - 'cam_j': int (camera 1, estimated pose)
        - 'expected_n_frames': int (minimum frames needed)
        
    Use this for Phase 1 tests (essential matrix, recover pose, gauge freedom).
    """
    obs = make_bootstrap_observations(n_frames=10, n_cameras=4, random_seed=100)
    settings = make_camera_settings(n_cameras=4)
    
    return {
        'observations': obs,
        'camera_settings': settings,
        'cam_i': 0,  # Fixed camera
        'cam_j': 1,  # Estimated camera
        'expected_n_frames': 10,
        'min_frames_required': 10,  # 8-Point needs >= 8 correspondences
    }


@pytest.fixture
def phase3_data_fixture():
    """
    Fixture: Phase 3 specific data (bundle adjustment).
    
    Phase 3 optimizes all camera extrinsics with first camera frozen.
    Returns dict with:
        - 'observations': synthetic observations (all cameras)
        - 'camera_settings': camera intrinsics (all cameras)
        - 'n_cameras': int
        - 'n_frames': int
        - 'wand_length_m': float
        - 'frozen_cam_id': int (camera 0 frozen for gauge)
        
    Use this for Phase 3 tests (bundle adjustment, gauge freedom, residual layout).
    """
    n_frames = 15
    n_cameras = 4
    obs = make_bootstrap_observations(
        n_frames=n_frames,
        n_cameras=n_cameras,
        random_seed=200
    )
    settings = make_camera_settings(n_cameras=n_cameras)
    
    return {
        'observations': obs,
        'camera_settings': settings,
        'n_cameras': n_cameras,
        'n_frames': n_frames,
        'wand_length_m': 0.3,
        'frozen_cam_id': 0,  # First camera frozen in Phase 3
    }


# ============================================================================
# HELPER UTILITIES (for future test development)
# ============================================================================

def validate_camera_params(params: np.ndarray, label: str = "camera_params") -> None:
    """
    Validate camera parameter array shape and finite values.
    
    Expected layout: [rvec(3), tvec(3)] for bootstrap (6 params total).
    
    Args:
        params: Camera parameter array
        label: Descriptive label for error messages
        
    Raises:
        AssertionError: If params invalid
    """
    assert params is not None, f"{label}: params is None"
    assert isinstance(params, np.ndarray), f"{label}: params not ndarray"
    assert params.shape == (6,), f"{label}: expected shape (6,), got {params.shape}"
    assert np.all(np.isfinite(params)), f"{label}: contains non-finite values"


def validate_observations(
    observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    min_frames: int = 1,
    min_cameras: int = 2
) -> None:
    """
    Validate observations dictionary structure and content.
    
    Args:
        observations: Bootstrap observations dict
        min_frames: Minimum required frames
        min_cameras: Minimum required cameras per frame
        
    Raises:
        AssertionError: If observations invalid
    """
    assert observations is not None, "observations is None"
    assert isinstance(observations, dict), "observations not dict"
    assert len(observations) >= min_frames, \
        f"Insufficient frames: {len(observations)} < {min_frames}"
    
    for fid, frame_data in observations.items():
        assert isinstance(frame_data, dict), f"Frame {fid} data not dict"
        assert len(frame_data) >= min_cameras, \
            f"Frame {fid}: insufficient cameras {len(frame_data)} < {min_cameras}"
        
        for cam_id, (uvA, uvB) in frame_data.items():
            assert uvA.shape == (2,), f"Frame {fid} cam {cam_id}: uvA shape {uvA.shape}"
            assert uvB.shape == (2,), f"Frame {fid} cam {cam_id}: uvB shape {uvB.shape}"
            assert np.all(np.isfinite(uvA)), f"Frame {fid} cam {cam_id}: uvA non-finite"
            assert np.all(np.isfinite(uvB)), f"Frame {fid} cam {cam_id}: uvB non-finite"


def compute_residual_statistics(residuals: np.ndarray) -> dict:
    """
    Compute RMS and other statistics from residual array.
    
    This will be used in W1a to test RMS computation bug (residual slicing).
    
    Args:
        residuals: Flat residual array from optimizer
        
    Returns:
        stats: dict with 'rms', 'mean', 'std', 'max', 'median'
    """
    assert residuals is not None and len(residuals) > 0, "Empty residuals"
    
    return {
        'rms': float(np.sqrt(np.mean(residuals**2))),
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'max': float(np.max(np.abs(residuals))),
        'median': float(np.median(np.abs(residuals))),
    }


def extract_wand_residuals(
    residuals: np.ndarray,
    n_frames: int,
    n_cameras: int
) -> np.ndarray:
    """
    Extract wand constraint residuals from interleaved layout.
    
    This will be tested in W1a (residual slicing bug).
    
    Bootstrap residual layout (per frame, interleaved):
        [wand_0, reproj_0_camA, reproj_0_camB, ..., wand_1, reproj_1_camA, ...]
        
    Args:
        residuals: Flat residual array
        n_frames: Number of frames
        n_cameras: Number of cameras per frame
        
    Returns:
        wand_residuals: (n_frames,) array of wand constraint residuals
    """
    residuals_per_frame = 1 + 2 * n_cameras  # 1 wand + 2*n_cameras reprojection
    expected_len = n_frames * residuals_per_frame
    assert len(residuals) == expected_len, \
        f"Residual length mismatch: {len(residuals)} != {expected_len}"
    
    # Extract every (1 + 2*n_cameras)-th element (wand residual at start of each block)
    wand_residuals = residuals[::residuals_per_frame]
    return wand_residuals


# ============================================================================
# NO TEST CASES YET
# ============================================================================
# Test cases will be added in W1a-1e when bugs are fixed using RED-GREEN-REFACTOR:
#
# W1a: Test RMS residual slicing (residual layout interleaved per frame)
# W1b: Test homogeneous coordinate division in triangulation
# W1c: Test Phase 1 gauge freedom (cam_i frozen at origin)
# W1d: Test Phase 3 gauge freedom (first camera frozen)
# W1e: Test wand constraint Jacobian sign consistency
#
# Each wave will add ~3-5 test functions to this file.


def test_phase3_rms_residual_slicing(phase3_data_fixture):
    """W1a: Phase 3 RMS must extract interleaved wand residuals per frame."""
    n_frames = phase3_data_fixture['n_frames']
    n_cameras = phase3_data_fixture['n_cameras']

    residuals_per_frame = 1 + 2 * n_cameras

    # Synthetic interleaved residual layout:
    # [wand_0, reproj_0(...), wand_1, reproj_1(...), ...]
    wand_residuals_gt = np.linspace(1.0, float(n_frames), n_frames)
    reproj_residuals = np.full((n_frames, 2 * n_cameras), 50.0, dtype=np.float64)

    residual_blocks = np.column_stack([wand_residuals_gt, reproj_residuals])
    final_res = residual_blocks.reshape(-1)
    assert final_res.shape == (n_frames * residuals_per_frame,)

    # Historical buggy logic (wrong for interleaved layout).
    wrong_rms = float(np.sqrt(np.mean(final_res[n_frames:] ** 2)))

    # Ground truth from helper: every block's first element is wand residual.
    wand_residuals = extract_wand_residuals(final_res, n_frames, n_cameras)
    expected_rms = float(np.sqrt(np.mean(wand_residuals**2)))

    assert not np.isclose(wrong_rms, expected_rms), (
        "Bug reproducer broken: old slicing unexpectedly matched wand RMS"
    )

    fixed_rms = PinholeBootstrapP0._compute_phase3_wand_rms(
        final_res,
        n_frames,
        n_cameras,
    )

    assert np.isclose(fixed_rms, expected_rms), (
        f"Fixed RMS mismatch: got {fixed_rms}, expected {expected_rms}"
    )


def test_homogeneous_division_guarded(monkeypatch):
    """W1b: Homogeneous division must guard near-zero w to avoid inf outputs."""
    bootstrap = PinholeBootstrapP0(config=PinholeBootstrapP0Config())

    observations = {
        0: {
            0: (np.array([100.0, 120.0]), np.array([130.0, 150.0])),
            1: (np.array([102.0, 122.0]), np.array([132.0, 152.0])),
        }
    }
    camera_settings = make_camera_settings(n_cameras=2)
    params_i = np.zeros(6, dtype=np.float64)
    params_j = np.array([0.0, 0.0, 0.0, 120.0, 0.0, 0.0], dtype=np.float64)

    # Scenario 1 (happy path): w > 1e-8 should produce correct Euclidean points.
    happy_seq = [
        np.array([[4.0], [8.0], [12.0], [2.0]], dtype=np.float64),
        np.array([[5.0], [10.0], [15.0], [5.0]], dtype=np.float64),
    ]

    def fake_triangulate_happy(_P1, _P2, _uv1, _uv2):
        return happy_seq.pop(0)

    monkeypatch.setattr(
        "modules.camera_calibration.wand_calibration.refractive_bootstrap.cv2.triangulatePoints",
        fake_triangulate_happy,
    )

    points_happy = bootstrap.triangulate_all_points(
        cam_i=0,
        cam_j=1,
        params_i=params_i,
        params_j=params_j,
        observations=observations,
        camera_settings=camera_settings,
    )

    XA_happy, XB_happy = points_happy[0]
    assert np.allclose(XA_happy, np.array([2.0, 4.0, 6.0]))
    assert np.allclose(XB_happy, np.array([1.0, 2.0, 3.0]))

    # Scenario 2 (edge): w ~= 0 must not produce inf and must be handled as NaN/sentinel.
    edge_seq = [
        np.array([[1.0], [2.0], [3.0], [0.0]], dtype=np.float64),
        np.array([[6.0], [9.0], [12.0], [3.0]], dtype=np.float64),
    ]

    def fake_triangulate_edge(_P1, _P2, _uv1, _uv2):
        return edge_seq.pop(0)

    monkeypatch.setattr(
        "modules.camera_calibration.wand_calibration.refractive_bootstrap.cv2.triangulatePoints",
        fake_triangulate_edge,
    )

    points_edge = bootstrap.triangulate_all_points(
        cam_i=0,
        cam_j=1,
        params_i=params_i,
        params_j=params_j,
        observations=observations,
        camera_settings=camera_settings,
    )

    XA_edge, XB_edge = points_edge[0]
    assert np.all(np.isfinite(XB_edge))
    assert np.isnan(XA_edge).all(), "Near-zero homogeneous w should map to NaN sentinel"
    assert not np.isinf(XA_edge).any(), "Guard must avoid inf from homogeneous divide"


class TestPhase1Gauge:
    def test_phase1_cam_i_frozen(self, phase1_data_fixture, monkeypatch):
        """After Phase 1 BA, cam_i must remain exactly [0,0,0,0,0,0]."""
        bootstrap = PinholeBootstrapP0(config=PinholeBootstrapP0Config())

        class _FakeResult:
            def __init__(self, x):
                self.x = x
                self.cost = float(np.sum(x**2))

        def fake_least_squares(_fun, x0, **_kwargs):
            x = x0.copy()
            # Force first 6 optimized parameters to be non-zero.
            # If cam_i is still inside the state vector, this leaks into output.
            x[:6] = np.array([0.1, -0.2, 0.3, 1.0, -2.0, 3.0], dtype=np.float64)
            return _FakeResult(x)

        # Keep this test focused on gauge anchoring behavior.
        monkeypatch.setattr(
            "modules.camera_calibration.wand_calibration.refractive_bootstrap.least_squares",
            fake_least_squares,
        )
        monkeypatch.setattr(PinholeBootstrapP0, "_validate", lambda self, report: None)

        params_i_opt, _params_j_opt, _report = bootstrap.run(
            cam_i=phase1_data_fixture['cam_i'],
            cam_j=phase1_data_fixture['cam_j'],
            observations=phase1_data_fixture['observations'],
            camera_settings=phase1_data_fixture['camera_settings'],
        )

        assert np.allclose(params_i_opt, np.zeros(6), atol=1e-10)


class TestPhase3Gauge:
    def test_phase3_first_camera_frozen(self, monkeypatch):
        """After Phase 3 BA, first camera must remain exactly [0,0,0,0,0,0]."""
        bootstrap = PinholeBootstrapP0(config=PinholeBootstrapP0Config())

        cam_ids = [0, 1, 2]
        n_frames = 12
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
        for fid in range(n_frames):
            observations[fid] = {}
            for cid in cam_ids:
                u_base = 300.0 + 4.0 * fid + 12.0 * cid
                v_base = 260.0 + 3.0 * fid - 7.0 * cid
                observations[fid][cid] = (
                    np.array([u_base, v_base], dtype=np.float64),
                    np.array([u_base + 15.0, v_base + 8.0], dtype=np.float64),
                )

        camera_settings = make_camera_settings(n_cameras=len(cam_ids))

        cam_params = {
            0: np.zeros(6, dtype=np.float64),
            1: np.array([0.0, 0.02, -0.01, 120.0, 5.0, 10.0], dtype=np.float64),
            2: np.array([0.01, -0.03, 0.02, -90.0, 15.0, 20.0], dtype=np.float64),
        }
        first_cam_id = min(cam_params.keys())

        class _FakeResult:
            def __init__(self, x):
                self.x = x
                self.cost = float(np.sum(x**2))

        def fake_least_squares(_fun, x0, **_kwargs):
            x = x0.copy()
            # Force first camera slot in state vector to non-zero.
            # Before W1d fix this leaks into cam_0 output; after fix cam_0 is frozen.
            x[:6] = np.array([0.25, -0.5, 0.75, 5.0, -6.0, 7.0], dtype=np.float64)
            if x.size >= 12:
                x[6:12] = np.array([-0.15, 0.2, -0.35, -3.0, 4.0, -5.0], dtype=np.float64)
            return _FakeResult(x)

        monkeypatch.setattr(
            "modules.camera_calibration.wand_calibration.refractive_bootstrap.least_squares",
            fake_least_squares,
        )

        cam_params_opt = bootstrap.run_phase3(
            cam_params=cam_params,
            observations=observations,
            camera_settings=camera_settings,
        )

        assert np.allclose(cam_params_opt[first_cam_id], np.zeros(6), atol=1e-10)

        other_cam_ids = [cid for cid in sorted(cam_params_opt.keys()) if cid != first_cam_id]
        assert other_cam_ids, "Expected at least one non-frozen camera"
        for cid in other_cam_ids:
            validate_camera_params(cam_params_opt[cid], label=f"cam_{cid}")
        assert any(
            np.linalg.norm(cam_params_opt[cid]) > 1e-12 for cid in other_cam_ids
        ), "Other cameras should remain optimizable (non-zero extrinsics)"


class TestInlierFiltering:
    def test_ransac_outliers_filtered(self, monkeypatch, capsys):
        """RANSAC inlier masks should filter correspondences and improve robustness."""
        bootstrap = PinholeBootstrapP0(config=PinholeBootstrapP0Config())

        # Build deterministic 10-frame, 2-camera observations (20 correspondences total).
        observations: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
        for fid in range(10):
            base_u = 300.0 + 3.0 * fid
            base_v = 240.0 + 2.0 * fid
            observations[fid] = {
                0: (
                    np.array([base_u, base_v], dtype=np.float64),
                    np.array([base_u + 20.0, base_v + 10.0], dtype=np.float64),
                ),
                1: (
                    np.array([base_u + 2.0, base_v - 1.0], dtype=np.float64),
                    np.array([base_u + 22.0, base_v + 9.0], dtype=np.float64),
                ),
            }

        camera_settings = make_camera_settings(n_cameras=2)

        calls = {}

        def fake_find_essential(pts_i_norm, pts_j_norm, **_kwargs):
            n = len(pts_i_norm)
            calls["find_essential_n"] = n
            assert n == 20, "Expected 20 correspondences before essential-matrix filtering"

            # ~10% essential outliers: last frame's two endpoints are rejected.
            mask_ess = np.ones((n, 1), dtype=np.uint8)
            mask_ess[-2:] = 0
            return np.eye(3, dtype=np.float64), mask_ess

        def fake_recover_pose(_E, pts_i_inliers, pts_j_inliers, **_kwargs):
            n = len(pts_i_inliers)
            calls["recover_pose_n"] = n

            # Keep first 16/18 pose inliers (drops one additional frame's two endpoints).
            mask_pose = np.zeros((n, 1), dtype=np.uint8)
            mask_pose[:16] = 1

            return (
                int(mask_pose.sum()),
                np.eye(3, dtype=np.float64),
                np.array([[1.0], [0.0], [0.0]], dtype=np.float64),
                mask_pose,
            )

        def fake_triangulate_points(_P_i, _P_j, pts_i_inliers_t, _pts_j_inliers_t):
            n = int(pts_i_inliers_t.shape[1])
            calls["triangulate_n"] = n

            # Emit pairwise points with exact 10 mm wand length.
            pts_4d = np.ones((4, n), dtype=np.float64)
            for idx in range(0, n, 2):
                pts_4d[0, idx] = float(idx)
                pts_4d[1, idx] = 0.0
                pts_4d[2, idx] = 100.0
                if idx + 1 < n:
                    pts_4d[0, idx + 1] = float(idx) + 10.0
                    pts_4d[1, idx + 1] = 0.0
                    pts_4d[2, idx + 1] = 100.0
            return pts_4d

        class _FakeResult:
            def __init__(self, x):
                self.x = x
                self.cost = float(np.sum(x**2))

        def fake_least_squares(_fun, x0, **_kwargs):
            return _FakeResult(x0.copy())

        def fake_compute_diagnostics(
            self,
            _cam_i,
            _cam_j,
            _params_i,
            params_j,
            _observations,
            valid_frames,
            _K_i,
            _K_j,
        ):
            return {
                "baseline_mm": float(np.linalg.norm(params_j[3:6])),
                "wand_length_median": self.config.wand_length_mm,
                "wand_length_std": 0.0,
                "wand_length_error": 0.0,
                "reproj_err_mean": 1.0,
                "reproj_err_max": 1.0,
                "valid_frames": len(valid_frames),
            }

        monkeypatch.setattr(
            "modules.camera_calibration.wand_calibration.refractive_bootstrap.cv2.findEssentialMat",
            fake_find_essential,
        )
        monkeypatch.setattr(
            "modules.camera_calibration.wand_calibration.refractive_bootstrap.cv2.recoverPose",
            fake_recover_pose,
        )
        monkeypatch.setattr(
            "modules.camera_calibration.wand_calibration.refractive_bootstrap.cv2.triangulatePoints",
            fake_triangulate_points,
        )
        monkeypatch.setattr(
            "modules.camera_calibration.wand_calibration.refractive_bootstrap.least_squares",
            fake_least_squares,
        )
        monkeypatch.setattr(PinholeBootstrapP0, "_compute_diagnostics", fake_compute_diagnostics)
        monkeypatch.setattr(PinholeBootstrapP0, "_validate", lambda self, report: None)

        _params_i, params_j, report = bootstrap.run(
            cam_i=0,
            cam_j=1,
            observations=observations,
            camera_settings=camera_settings,
        )

        # RED (before fix): recoverPose receives all 20 and triangulation uses all 20.
        # GREEN (after fix): recoverPose sees 18 essential inliers, triangulation sees 16 combined inliers.
        assert calls["recover_pose_n"] == 18
        assert calls["triangulate_n"] == 16

        # Verify inlier-rate reporting and baseline stability from deterministic mock geometry.
        stdout = capsys.readouterr().out
        assert "RANSAC inliers: 16 / 20 (80.0%)" in stdout
        assert np.isclose(np.linalg.norm(params_j[3:6]), 1.0, atol=1e-8)
        assert np.isclose(report["baseline_mm"], 1.0, atol=1e-8)


class TestBarrierConfig:
    def test_barrier_config_fields_exist(self):
        """W2a-T1: Verify barrier continuation config exists in RefractiveBAConfig."""
        from modules.camera_calibration.wand_calibration.refraction_calibration_BA import RefractiveBAConfig
        
        config = RefractiveBAConfig()
        
        # Check all 5 fields exist
        assert hasattr(config, 'margin_side_mm')
        assert hasattr(config, 'alpha_side_gate')
        assert hasattr(config, 'beta_side_dir')
        assert hasattr(config, 'beta_side_soft')
        assert hasattr(config, 'barrier_schedule')
        
        # Check types and defaults
        assert isinstance(config.margin_side_mm, float)
        assert config.margin_side_mm == 0.05
        
        assert isinstance(config.alpha_side_gate, float)
        assert config.alpha_side_gate == 10.0
        
        assert isinstance(config.beta_side_dir, float)
        assert config.beta_side_dir == 1e4
        
        assert isinstance(config.beta_side_soft, float)
        assert config.beta_side_soft == 100.0
        
        assert isinstance(config.barrier_schedule, dict)
        
        # Check schedule has required keys
        assert 'early' in config.barrier_schedule
        assert 'mid' in config.barrier_schedule
        assert 'final' in config.barrier_schedule
        
        # Verify schedule structure (each phase has required fields)
        for phase in ['early', 'mid', 'final']:
            phase_cfg = config.barrier_schedule[phase]
            assert 'gate_scale' in phase_cfg
            assert 'beta_dir_scale' in phase_cfg
            assert 'tau' in phase_cfg
            assert 'soft_on' in phase_cfg


class TestBarrierResiduals:
    @staticmethod
    def _smooth_barrier_residuals(
        sX: float,
        margin_mm: float,
        r_val: float,
        tau: float,
        alpha_side_gate: float,
        beta_side_dir: float,
    ) -> np.ndarray:
        """Mirror wand BA barrier: 2 residual slots per barrier point."""
        gap = (margin_mm + r_val) - sX
        tau_eff = max(tau, 1e-12)
        gap_smooth = tau_eff * np.logaddexp(0.0, gap / tau_eff)
        r_fix_const = np.sqrt(2.0 * alpha_side_gate)
        r_grad_const = np.sqrt(2.0 * beta_side_dir)
        return np.array([
            r_fix_const * (1.0 - np.exp(-gap_smooth / tau_eff)),
            r_grad_const * gap_smooth,
        ], dtype=np.float64)

    def test_feasible_gap_residuals_near_zero(self):
        """Feasible side (gap < 0) should produce near-zero smooth barrier residuals."""
        res = self._smooth_barrier_residuals(
            sX=1.0,
            margin_mm=0.05,
            r_val=0.0,
            tau=0.01,
            alpha_side_gate=10.0,
            beta_side_dir=1e4,
        )

        assert res.shape == (2,)
        assert np.all(np.isfinite(res))
        assert np.all(res >= 0.0)
        assert np.all(res < 1e-8)

    def test_violating_gap_positive_and_smooth_near_zero(self):
        """Violating side (gap > 0) should be positive with smooth no-kink behavior."""
        margin_mm = 0.05
        tau = 0.01
        alpha = 10.0
        beta = 1e4

        # Violating case: preserve point-radius term in gap=(margin+r)-sX.
        sX = 0.0
        res_small_r = self._smooth_barrier_residuals(
            sX=sX,
            margin_mm=margin_mm,
            r_val=0.0,
            tau=tau,
            alpha_side_gate=alpha,
            beta_side_dir=beta,
        )
        res_big_r = self._smooth_barrier_residuals(
            sX=sX,
            margin_mm=margin_mm,
            r_val=0.3,
            tau=tau,
            alpha_side_gate=alpha,
            beta_side_dir=beta,
        )

        assert res_small_r.shape == (2,)
        assert np.all(res_small_r > 0.0)
        assert np.all(res_big_r > res_small_r), "Larger point radius must increase barrier residuals"

        # Smoothness check around gap=0 (no hard kink in value/slope).
        eps = 1e-6
        res_left = self._smooth_barrier_residuals(
            sX=margin_mm + eps,
            margin_mm=margin_mm,
            r_val=0.0,
            tau=tau,
            alpha_side_gate=alpha,
            beta_side_dir=beta,
        )
        res_zero = self._smooth_barrier_residuals(
            sX=margin_mm,
            margin_mm=margin_mm,
            r_val=0.0,
            tau=tau,
            alpha_side_gate=alpha,
            beta_side_dir=beta,
        )
        res_right = self._smooth_barrier_residuals(
            sX=margin_mm - eps,
            margin_mm=margin_mm,
            r_val=0.0,
            tau=tau,
            alpha_side_gate=alpha,
            beta_side_dir=beta,
        )

        d_left = (res_zero - res_left) / eps
        d_right = (res_right - res_zero) / eps
        assert np.all(np.isfinite(d_left)) and np.all(np.isfinite(d_right))
        assert np.max(np.abs(d_left - d_right)) < 1e-2


class TestPinholeBootstrapP0ConfigValidation:
    """W3a: Validate PinholeBootstrapP0Config.__post_init__ rejects invalid configs."""
    
    def test_valid_config_accepted(self):
        """Valid config with all positive finite values should be accepted."""
        config = PinholeBootstrapP0Config(
            wand_length_mm=10.0,
            ui_focal_px=9000.0,
            ftol=1e-6,
            xtol=1e-6,
        )
        assert config.wand_length_mm == 10.0
        assert config.ui_focal_px == 9000.0
        assert config.ftol == 1e-6
        assert config.xtol == 1e-6
    
    def test_negative_wand_length_rejected(self):
        """Negative wand_length_mm should raise ValueError."""
        with pytest.raises(ValueError, match="wand_length_mm must be > 0"):
            PinholeBootstrapP0Config(wand_length_mm=-1.0)
    
    def test_zero_wand_length_rejected(self):
        """Zero wand_length_mm should raise ValueError."""
        with pytest.raises(ValueError, match="wand_length_mm must be > 0"):
            PinholeBootstrapP0Config(wand_length_mm=0.0)
    
    def test_negative_ui_focal_rejected(self):
        """Negative ui_focal_px should raise ValueError."""
        with pytest.raises(ValueError, match="ui_focal_px must be > 0"):
            PinholeBootstrapP0Config(ui_focal_px=-9000.0)
    
    def test_zero_ui_focal_rejected(self):
        """Zero ui_focal_px should raise ValueError."""
        with pytest.raises(ValueError, match="ui_focal_px must be > 0"):
            PinholeBootstrapP0Config(ui_focal_px=0.0)
    
    def test_negative_ftol_rejected(self):
        """Negative ftol should raise ValueError."""
        with pytest.raises(ValueError, match="ftol must be > 0"):
            PinholeBootstrapP0Config(ftol=-1e-6)
    
    def test_zero_ftol_rejected(self):
        """Zero ftol should raise ValueError."""
        with pytest.raises(ValueError, match="ftol must be > 0"):
            PinholeBootstrapP0Config(ftol=0.0)
    
    def test_negative_xtol_rejected(self):
        """Negative xtol should raise ValueError."""
        with pytest.raises(ValueError, match="xtol must be > 0"):
            PinholeBootstrapP0Config(xtol=-1e-6)
    
    def test_zero_xtol_rejected(self):
        """Zero xtol should raise ValueError."""
        with pytest.raises(ValueError, match="xtol must be > 0"):
            PinholeBootstrapP0Config(xtol=0.0)
    
    def test_nan_wand_length_rejected(self):
        """NaN wand_length_mm should raise ValueError."""
        with pytest.raises(ValueError, match="wand_length_mm must be finite"):
            PinholeBootstrapP0Config(wand_length_mm=float('nan'))
    
    def test_inf_wand_length_rejected(self):
        """Inf wand_length_mm should raise ValueError."""
        with pytest.raises(ValueError, match="wand_length_mm must be finite"):
            PinholeBootstrapP0Config(wand_length_mm=float('inf'))
    
    def test_nan_ui_focal_rejected(self):
        """NaN ui_focal_px should raise ValueError."""
        with pytest.raises(ValueError, match="ui_focal_px must be finite"):
            PinholeBootstrapP0Config(ui_focal_px=float('nan'))
    
    def test_inf_ui_focal_rejected(self):
        """Inf ui_focal_px should raise ValueError."""
        with pytest.raises(ValueError, match="ui_focal_px must be finite"):
            PinholeBootstrapP0Config(ui_focal_px=float('inf'))
    
    def test_nan_ftol_rejected(self):
        """NaN ftol should raise ValueError."""
        with pytest.raises(ValueError, match="ftol must be finite"):
            PinholeBootstrapP0Config(ftol=float('nan'))
    
    def test_inf_ftol_rejected(self):
        """Inf ftol should raise ValueError."""
        with pytest.raises(ValueError, match="ftol must be finite"):
            PinholeBootstrapP0Config(ftol=float('inf'))
    
    def test_nan_xtol_rejected(self):
        """NaN xtol should raise ValueError."""
        with pytest.raises(ValueError, match="xtol must be finite"):
            PinholeBootstrapP0Config(xtol=float('nan'))
    
    def test_inf_xtol_rejected(self):
        """Inf xtol should raise ValueError."""
        with pytest.raises(ValueError, match="xtol must be finite"):
            PinholeBootstrapP0Config(xtol=float('inf'))
