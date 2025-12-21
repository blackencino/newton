# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Functional utilities for loading and rendering USD scene geometry.

Core concepts:
- Diegetic: An immutable scene element with geometry and multiple color channels
- ColorExtractor: A function that extracts a color from a USD prim
- parse_diegetics: A fold that maps geometry + color extractors over a USD stage

The pipeline is: USD Stage → list[Diegetic] → RenderContext → pixels
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdShade

from newton._src.sensors.warp_raytrace import ClearData, RenderContext, RenderShapeType

# Optional PIL import for texture loading
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# =============================================================================
# Multi-AOV Render Outputs
# =============================================================================

@dataclass
class RenderOutputs:
    """
    Container for all raw GPU render outputs from a single render pass.
    
    The raytracer outputs multiple AOVs (Arbitrary Output Variables) in one pass:
    - color_image: Lit diffuse render (packed RGBA as uint32)
    - depth_image: Ray hit distance (float32)
    - shape_index_image: Shape index at each pixel (uint32)
    - normal_image: Surface normal at hit point (vec3f)
    
    The shape_index_image is key for generating additional passes like object_id
    and semantic - we use lookup tables to map shape indices to colors in post.
    """
    color_image: wp.array  # (num_worlds, num_cameras, width*height), uint32
    depth_image: wp.array  # (num_worlds, num_cameras, width*height), float32
    shape_index_image: wp.array  # (num_worlds, num_cameras, width*height), uint32
    normal_image: wp.array  # (num_worlds, num_cameras, width*height), vec3f


@dataclass
class ColorLUTs:
    """
    Lookup tables for mapping shape indices to different color channels.
    
    Each LUT is a warp array of packed uint32 RGBA colors, indexed by shape index.
    Used to post-process shape_index_image into different color passes.
    """
    object_id: wp.array  # (num_shapes,), uint32 packed RGBA
    semantic: wp.array   # (num_shapes,), uint32 packed RGBA


@dataclass
class PixelOutputs:
    """
    Container for converted numpy RGB arrays, ready for saving to disk.
    
    All arrays are (height, width, 3) uint8 RGB format.
    """
    color: np.ndarray      # Lit diffuse render
    depth: np.ndarray      # Depth visualization (grayscale as RGB)
    normal: np.ndarray     # Normal visualization ((n+1)/2 mapped to 0-255)
    object_id: np.ndarray  # Object ID colors
    semantic: np.ndarray   # Semantic colors


# =============================================================================
# Warp Kernels for Shape Index → Color Mapping
# =============================================================================

@wp.kernel
def shape_index_to_color_lut(
    shape_indices: wp.array(dtype=wp.uint32, ndim=3),
    color_lut: wp.array(dtype=wp.uint32),
    out_rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    """Map shape indices to colors using a lookup table."""
    world_id, camera_id, pixel_id = wp.tid()
    shape_index = shape_indices[world_id, camera_id, pixel_id]
    if shape_index < wp.uint32(color_lut.shape[0]):
        out_rgba[world_id, camera_id, pixel_id] = color_lut[wp.int32(shape_index)]
    else:
        # Background or invalid - transparent black
        out_rgba[world_id, camera_id, pixel_id] = wp.uint32(0xFF404040)


@wp.kernel
def depth_to_grayscale(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    min_depth: wp.float32,
    max_depth: wp.float32,
    out_rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    """Convert depth values to grayscale visualization (closer = brighter)."""
    world_id, camera_id, pixel_id = wp.tid()
    depth = depth_image[world_id, camera_id, pixel_id]
    
    if depth <= 0.0:
        # No hit - dark gray background
        out_rgba[world_id, camera_id, pixel_id] = wp.uint32(0xFF404040)
        return
    
    # Normalize and invert (closer = brighter)
    denom = wp.max(max_depth - min_depth, 0.001)
    normalized = (depth - min_depth) / denom
    value = wp.uint32((1.0 - normalized) * 255.0)
    value = wp.min(value, wp.uint32(255))
    
    # Pack as grayscale RGBA
    out_rgba[world_id, camera_id, pixel_id] = (
        wp.uint32(0xFF000000) | (value << wp.uint32(16)) | (value << wp.uint32(8)) | value
    )


@wp.kernel
def normal_to_rgb(
    normal_image: wp.array(dtype=wp.vec3f, ndim=3),
    out_rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    """Convert world-space normals to RGB visualization."""
    world_id, camera_id, pixel_id = wp.tid()
    normal = normal_image[world_id, camera_id, pixel_id]
    
    # Check for zero normal (no hit)
    length = wp.length(normal)
    if length < 0.001:
        out_rgba[world_id, camera_id, pixel_id] = wp.uint32(0xFF404040)
        return
    
    # Map from [-1,1] to [0,1] then to [0,255]
    r = wp.uint32((normal[0] * 0.5 + 0.5) * 255.0)
    g = wp.uint32((normal[1] * 0.5 + 0.5) * 255.0)
    b = wp.uint32((normal[2] * 0.5 + 0.5) * 255.0)
    
    r = wp.min(wp.max(r, wp.uint32(0)), wp.uint32(255))
    g = wp.min(wp.max(g, wp.uint32(0)), wp.uint32(255))
    b = wp.min(wp.max(b, wp.uint32(0)), wp.uint32(255))
    
    out_rgba[world_id, camera_id, pixel_id] = (
        wp.uint32(0xFF000000) | (b << wp.uint32(16)) | (g << wp.uint32(8)) | r
    )


@wp.kernel
def find_depth_range(
    depth_image: wp.array(dtype=wp.float32, ndim=3),
    depth_range: wp.array(dtype=wp.float32),
):
    """Find min/max depth values for normalization."""
    world_id, camera_id, pixel_id = wp.tid()
    depth = depth_image[world_id, camera_id, pixel_id]
    if depth > 0.0:
        wp.atomic_min(depth_range, 0, depth)
        wp.atomic_max(depth_range, 1, depth)


# =============================================================================
# Type Aliases
# =============================================================================

RGB = tuple[float, float, float]


# =============================================================================
# Color Extractor Protocol
# =============================================================================

@dataclass(frozen=True)
class ExtractionContext:
    """Immutable context passed to color extractors."""
    usd_file_dir: str
    time_code: Usd.TimeCode
    prim_index: int  # For deterministic random colors


class ColorExtractor(Protocol):
    """Protocol for functions that extract a color from a USD prim."""
    def __call__(self, prim: Usd.Prim, ctx: ExtractionContext) -> RGB: ...


# =============================================================================
# Diegetic: The Core Scene Element
# =============================================================================

@dataclass(frozen=True)
class Diegetic:
    """
    An immutable scene element with geometry and multiple color channels.
    
    "Diegetic" (from film theory): existing within the world of the narrative.
    Each Diegetic represents a visible piece of the scene with pre-computed colors
    for different rendering purposes.

    In film theory, diegesis refers to the world of the story. A "diegetic" object 
    is anything that exists within the characters' reality (a chair, a gun, an actor, a car).

    It specifically excludes "non-diegetic" elements like boom mics, C-stands, film lights, 
    and the camera itself, because the characters do not "see" those things.
    
    Color channels are stored as raw data for use by different render passes:
    - diffuse_albedo: Used by the lit color render pass
    - object_id: Used to create object ID lookup table for post-processing
    - semantic: Used to create semantic lookup table for post-processing
    """
    name: str
    path: str
    vertices: np.ndarray  # World-space positions (N, 3), float32
    faces: np.ndarray     # Triangle indices (M, 3), int32
    diffuse_albedo: RGB   # For photorealistic rendering (used in lit pass)
    object_id: RGB        # For instance segmentation (post-process LUT)
    semantic: RGB         # For semantic segmentation (post-process LUT)


@dataclass(frozen=True)
class ParseOptions:
    """Immutable options for parsing USD stages."""
    time_code: Usd.TimeCode
    path_filter: Callable[[str], bool] | None = None
    skip_invisible: bool = True
    skip_proxy: bool = True


# =============================================================================
# Color Extractor Factory Functions
# =============================================================================

def constant_color(rgb: RGB) -> ColorExtractor:
    """
    Create an extractor that always returns the same color.
    
    Usage:
        extractor = constant_color((0.5, 0.5, 0.5))
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        return rgb
    return extract


def random_color(seed: int = 42, brightness_range: tuple[float, float] = (0.3, 0.9)) -> ColorExtractor:
    """
    Create an extractor that returns a deterministic random color per prim.
    
    Uses prim path hash + seed for reproducibility.
    Colors are in HSV with full saturation, converted to RGB.
    
    Usage:
        extractor = random_color(seed=123)
    """
    lo, hi = brightness_range
    
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        # Hash prim path + seed for deterministic but varied colors
        path_hash = hashlib.md5(f"{prim.GetPath()}{seed}".encode()).digest()
        hue = int.from_bytes(path_hash[:2], 'little') / 65535.0
        
        # HSV to RGB with S=1, V in brightness range
        v = lo + (hi - lo) * (int.from_bytes(path_hash[2:4], 'little') / 65535.0)
        
        # Simple HSV->RGB
        h = hue * 6.0
        c = v  # S=1, so c=v*s=v
        x = c * (1 - abs(h % 2 - 1))
        
        if h < 1: r, g, b = c, x, 0.0
        elif h < 2: r, g, b = x, c, 0.0
        elif h < 3: r, g, b = 0.0, c, x
        elif h < 4: r, g, b = 0.0, x, c
        elif h < 5: r, g, b = x, 0.0, c
        else: r, g, b = c, 0.0, x
        
        return (r, g, b)
    return extract


def primvar_color(primvar_name: str, fallback: RGB) -> ColorExtractor:
    """
    Create an extractor that reads a named primvar (color3f, constant interpolation).
    
    Usage:
        extractor = primvar_color("objectid_color", fallback=(1.0, 0.0, 1.0))
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        primvar = primvars_api.GetPrimvar(primvar_name)
        
        if not primvar or not primvar.HasValue():
            return fallback
        
        if primvar.GetInterpolation() != UsdGeom.Tokens.constant:
            return fallback
        
        value = primvar.Get(ctx.time_code)
        if value is None:
            return fallback
        
        try:
            if hasattr(value, '__getitem__') and len(value) >= 3:
                return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, IndexError):
            pass
        
        return fallback
    return extract


def display_color_extractor(fallback: RGB) -> ColorExtractor:
    """
    Create an extractor that reads the displayColor primvar.
    
    Usage:
        extractor = display_color_extractor(fallback=(0.7, 0.7, 0.7))
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        if not prim.IsA(UsdGeom.Mesh):
            return fallback
        
        mesh = UsdGeom.Mesh(prim)
        display_color_raw = mesh.GetDisplayColorPrimvar().Get(ctx.time_code)
        
        if display_color_raw is None or len(display_color_raw) == 0:
            return fallback
        
        first = display_color_raw[0]
        if hasattr(first, '__getitem__'):
            return (float(first[0]), float(first[1]), float(first[2]))
        
        return fallback
    return extract


def material_diffuse_extractor(fallback: RGB, load_textures: bool = False) -> ColorExtractor:
    """
    Create an extractor that reads UsdPreviewSurface diffuseColor.
    
    If load_textures is True and diffuseColor is connected to a texture,
    attempts to sample the texture's average color.
    
    Usage:
        extractor = material_diffuse_extractor(fallback=(0.5, 0.5, 0.5))
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        materialBinding = UsdShade.MaterialBindingAPI(prim)
        boundMaterial, _ = materialBinding.ComputeBoundMaterial()
        
        if not boundMaterial:
            return fallback
        
        surfaceOutput = boundMaterial.GetSurfaceOutput()
        if not surfaceOutput:
            return fallback
        
        connectedSource = surfaceOutput.GetConnectedSource()
        if not connectedSource:
            return fallback
        
        shaderPrim = connectedSource[0].GetPrim()
        shader = UsdShade.Shader(shaderPrim)
        
        diffuseInput = shader.GetInput("diffuseColor")
        if not diffuseInput:
            return fallback
        
        connSrc = diffuseInput.GetConnectedSource()
        if connSrc:
            # Connected to texture
            connectedShader = UsdShade.Shader(connSrc[0].GetPrim())
            
            if load_textures and _PIL_AVAILABLE:
                fileInput = connectedShader.GetInput("file")
                if fileInput:
                    fileVal = fileInput.Get(ctx.time_code)
                    if fileVal:
                        avg_color = get_texture_average_color(
                            fileVal.path, ctx.usd_file_dir, verbose=False
                        )
                        if avg_color is not None:
                            return avg_color
            
            # Try fallback from texture shader
            fallbackInput = connectedShader.GetInput("fallback")
            if fallbackInput:
                fallbackVal = fallbackInput.Get(ctx.time_code)
                if fallbackVal is not None:
                    return (float(fallbackVal[0]), float(fallbackVal[1]), float(fallbackVal[2]))
        else:
            # Direct value
            val = diffuseInput.Get(ctx.time_code)
            if val is not None and hasattr(val, '__iter__'):
                return (float(val[0]), float(val[1]), float(val[2]))
        
        return fallback
    return extract


def first_of(*extractors: ColorExtractor, fallback: RGB = (0.5, 0.5, 0.5)) -> ColorExtractor:
    """
    Create an extractor that tries each extractor in order, returning the first non-fallback result.
    
    This is a combinator for building fallback chains.
    
    Usage:
        extractor = first_of(
            material_diffuse_extractor((0.5, 0.5, 0.5)),
            display_color_extractor((0.5, 0.5, 0.5)),
            fallback=(0.7, 0.7, 0.7)
        )
    """
    def extract(prim: Usd.Prim, ctx: ExtractionContext) -> RGB:
        for ext in extractors:
            result = ext(prim, ctx)
            if result != fallback:
                return result
        return fallback
    return extract


@dataclass
class CameraData:
    """Container for extracted USD camera data."""
    position: np.ndarray      # World-space position (3,)
    right: np.ndarray         # Camera +X axis in world space (normalized)
    up: np.ndarray            # Camera +Y axis in world space (normalized)
    forward: np.ndarray       # Camera view direction (-Z in local) in world space (normalized)
    fov_vertical: float       # Vertical field of view in radians
    fov_horizontal: float     # Horizontal field of view in radians


# =============================================================================
# Render Configuration
# =============================================================================

@dataclass(frozen=True)
class RenderConfig:
    """Immutable configuration for rendering output."""
    width: int
    height: int
    output_dir: Path
    filename_pattern: str = "render.{frame}.png"
    
    def get_output_path(self, frame: int) -> Path:
        """Get the full output path for a given frame number."""
        return self.output_dir / self.filename_pattern.format(frame=frame)


# =============================================================================
# Geometry Utilities
# =============================================================================

def triangulate(indices: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Convert polygon mesh to triangles via fan triangulation.
    
    Args:
        indices: Face vertex indices array
        counts: Face vertex counts array (e.g., [4, 4, 3] for 2 quads and 1 triangle)
        
    Returns:
        Triangle indices as (N, 3) numpy array
    """
    tris = []
    idx = 0
    for count in counts:
        if count >= 3:
            for i in range(1, count - 1):
                tris.append([indices[idx], indices[idx + i], indices[idx + i + 1]])
        idx += count
    return np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)


def transform_points_to_world(
    points: np.ndarray,
    world_matrix: Gf.Matrix4d
) -> np.ndarray:
    """
    Transform points from local to world space using a 4x4 matrix.
    
    Args:
        points: Local-space points (N, 3)
        world_matrix: USD world transform matrix
        
    Returns:
        World-space points (N, 3)
    """
    M = np.array(world_matrix, dtype=np.float64)
    pts = np.array(points, dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    pts_world_h = pts_h @ M  # Row-vectors * matrix
    return pts_world_h[:, :3]


# =============================================================================
# Geometry Extraction
# =============================================================================

@dataclass(frozen=True)
class ExtractedGeometry:
    """Intermediate result from geometry extraction."""
    name: str
    path: str
    vertices: np.ndarray  # World-space (N, 3) float32
    faces: np.ndarray     # Triangle indices (M, 3) int32


def extract_geometry(
    prim: Usd.Prim,
    xcache: UsdGeom.XformCache,
    time_code: Usd.TimeCode
) -> ExtractedGeometry | None:
    """
    Extract world-space triangulated geometry from a mesh prim.
    
    Handles instanced geometry by getting geometry from prototype
    and transform from the proxy prim.
    
    Returns None if geometry cannot be extracted.
    """
    # Get the actual geometry prim (prototype for instances)
    if prim.IsInstanceProxy():
        geom_prim = prim.GetPrimInPrototype()
    else:
        geom_prim = prim
    
    mesh = UsdGeom.Mesh(geom_prim)
    
    points = mesh.GetPointsAttr().Get(time_code)
    indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)
    counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
    
    if not points or not indices or not counts:
        return None
    
    # Transform using the proxy prim's world transform
    world_mat = xcache.GetLocalToWorldTransform(prim)
    vertices = transform_points_to_world(points, world_mat).astype(np.float32)
    
    # Triangulate
    faces = triangulate(np.array(indices), np.array(counts))
    
    path_str = str(prim.GetPath())
    name = path_str.split("/")[-1]
    
    return ExtractedGeometry(
        name=name,
        path=path_str,
        vertices=vertices,
        faces=faces,
    )


def is_visible_mesh(prim: Usd.Prim, time_code: Usd.TimeCode) -> bool:
    """Check if a prim is a visible mesh (not invisible for rendering)."""
    if not prim.IsA(UsdGeom.Mesh):
        return False
    
    # Get geometry prim for visibility check
    if prim.IsInstanceProxy():
        geom_prim = prim.GetPrimInPrototype()
    else:
        geom_prim = prim
    
    mesh = UsdGeom.Mesh(geom_prim)
    visibility = mesh.ComputeEffectiveVisibility(UsdGeom.Tokens.render, time_code)
    return visibility != UsdGeom.Tokens.invisible


# =============================================================================
# Parse Diegetics: The Fold Function
# =============================================================================

def parse_diegetics(
    stage: Usd.Stage,
    usd_file_path: str,
    options: ParseOptions,
    diffuse_extractor: ColorExtractor,
    object_id_extractor: ColorExtractor,
    semantic_extractor: ColorExtractor,
    verbose: bool = False,
    show_progress: bool = True,
    progress_interval: int = 1000,
) -> list[Diegetic]:
    """
    Parse a USD stage into a list of Diegetic scene elements.
    
    This is the core "fold" function that maps:
    - A geometry extractor (hardcoded for triangle meshes)
    - Three color extractors (one per color channel)
    
    Over the USD stage to produce immutable Diegetics.
    
    Args:
        stage: Opened USD stage
        usd_file_path: Path to USD file (for texture resolution)
        options: ParseOptions controlling traversal behavior
        diffuse_extractor: Extracts diffuse_albedo color
        object_id_extractor: Extracts object_id color
        semantic_extractor: Extracts semantic color
        verbose: Print detailed progress
        show_progress: Print periodic progress updates
        progress_interval: Update interval for progress messages
        
    Returns:
        List of Diegetic objects with geometry and all color channels populated
    """
    usd_file_dir = os.path.dirname(usd_file_path)
    time_code = options.time_code
    xcache = UsdGeom.XformCache(time_code)
    
    diegetics: list[Diegetic] = []
    prims_processed = 0
    prims_skipped = 0
    
    if show_progress:
        print("Parsing USD stage into Diegetics...", flush=True)
    
    for prim_index, prim in enumerate(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())):
        prims_processed += 1
        
        if show_progress and prims_processed % progress_interval == 0:
            print(f"  Processed {prims_processed} prims, found {len(diegetics)} diegetics...", flush=True)
        
        # Filter: must be a mesh
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        path_str = str(prim.GetPath())
        
        # Filter: user-provided path filter
        if options.path_filter is not None and not options.path_filter(path_str):
            continue
        
        # Filter: visibility
        if options.skip_invisible and not is_visible_mesh(prim, time_code):
            if verbose:
                print(f"  SKIP (invisible): {path_str}")
            prims_skipped += 1
            continue
        
        # Filter: proxy geometry
        if options.skip_proxy and is_proxy_path(path_str):
            if verbose:
                print(f"  SKIP (proxy): {path_str}")
            prims_skipped += 1
            continue
        
        # Extract geometry
        geom = extract_geometry(prim, xcache, time_code)
        if geom is None:
            if verbose:
                print(f"  SKIP (no geometry): {path_str}")
            prims_skipped += 1
            continue
        
        # Create extraction context
        ctx = ExtractionContext(
            usd_file_dir=usd_file_dir,
            time_code=time_code,
            prim_index=prim_index,
        )
        
        # Extract all three colors
        diffuse = diffuse_extractor(prim, ctx)
        object_id = object_id_extractor(prim, ctx)
        semantic = semantic_extractor(prim, ctx)
        
        # Create immutable Diegetic
        diegetic = Diegetic(
            name=geom.name,
            path=geom.path,
            vertices=geom.vertices,
            faces=geom.faces,
            diffuse_albedo=diffuse,
            object_id=object_id,
            semantic=semantic,
        )
        
        diegetics.append(diegetic)
        
        if verbose:
            print(f"  PARSED: {path_str} ({len(geom.vertices)} verts, {len(geom.faces)} tris)")
    
    if show_progress:
        print(f"  Done: {prims_processed} prims → {len(diegetics)} diegetics, {prims_skipped} skipped", flush=True)
    
    return diegetics


def make_diegetic_names_unique(diegetics: list[Diegetic]) -> list[Diegetic]:
    """
    Return new list of Diegetics with unique names.
    
    Unlike make_mesh_names_unique, this is a pure function that returns
    new Diegetic objects (since Diegetic is frozen).
    """
    name_counts: dict[str, int] = {}
    result: list[Diegetic] = []
    
    for d in diegetics:
        name = d.name
        if name in name_counts:
            name_counts[name] += 1
            new_name = f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 1
            new_name = name
        
        # Create new Diegetic with updated name (if changed)
        if new_name != name:
            result.append(Diegetic(
                name=new_name,
                path=d.path,
                vertices=d.vertices,
                faces=d.faces,
                diffuse_albedo=d.diffuse_albedo,
                object_id=d.object_id,
                semantic=d.semantic,
            ))
        else:
            result.append(d)
    
    return result


# =============================================================================
# Camera Utilities
# =============================================================================

def get_camera_from_stage(
    stage: Usd.Stage,
    camera_path: str,
    time_code: Usd.TimeCode,
    verbose: bool = False
) -> CameraData:
    """
    Extract camera world-space basis and FOV from a USD camera prim.
    
    USD/RenderMan camera coordinate system:
    - X: right
    - Y: up
    - Z: toward the sensor ("into the lens")
    
    The view direction is -Z (same as OpenGL convention).
    
    Args:
        stage: USD stage
        camera_path: Path to camera prim (e.g., "/World/Camera")
        time_code: Time code for sampling animated cameras
        verbose: If True, print camera info
        
    Returns:
        CameraData with position and orientation in world space
        
    Raises:
        RuntimeError: If camera prim not found or invalid
    """
    import math
    
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Camera not found at {camera_path}")

    usd_camera = UsdGeom.Camera(camera_prim)
    
    # Get intrinsics from Gf.Camera
    gf_camera = usd_camera.GetCamera(time_code)
    fov_v = gf_camera.GetFieldOfView(Gf.Camera.FOVVertical)
    fov_h = gf_camera.GetFieldOfView(Gf.Camera.FOVHorizontal)

    # Get extrinsics using XformCache (same approach as mesh loading)
    xcache = UsdGeom.XformCache(time_code)
    world_mat = xcache.GetLocalToWorldTransform(camera_prim)

    # Transform basis points to world space
    pts_camera = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pts_world = transform_points_to_world(pts_camera, world_mat)

    cam_origin, x_pt, y_pt, z_pt = pts_world

    # Extract and normalize basis vectors
    right = x_pt - cam_origin
    up = y_pt - cam_origin
    z_axis = z_pt - cam_origin  # +Z points toward sensor
    
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)
    forward = -z_axis / np.linalg.norm(z_axis)  # View direction is -Z

    if verbose:
        print(f"Camera at frame {time_code}: FOV={fov_v:.2f}°")
        print(f"  Position: [{cam_origin[0]:.3f}, {cam_origin[1]:.3f}, {cam_origin[2]:.3f}]")
        print(f"  Forward:  [{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}]")

    return CameraData(
        position=cam_origin.astype(np.float32),
        right=right.astype(np.float32),
        up=up.astype(np.float32),
        forward=forward.astype(np.float32),
        fov_vertical=math.radians(fov_v),
        fov_horizontal=math.radians(fov_h),
    )


def camera_rotation_matrix(camera: CameraData) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from camera basis vectors.
    
    Columns are [right, up, forward].
    
    Args:
        camera: CameraData with basis vectors
        
    Returns:
        3x3 orthonormal rotation matrix
    """
    return np.column_stack([camera.right, camera.up, camera.forward]).astype(np.float32)


# =============================================================================
# Warp Kernel
# =============================================================================

@wp.kernel
def generate_world_rays_kernel(
    width: int,
    height: int,
    fov_y: float,
    cam_pos: wp.vec3f,
    cam_right: wp.vec3f,
    cam_up: wp.vec3f,
    cam_forward: wp.vec3f,
    out_rays: wp.array(dtype=wp.vec3, ndim=4),
):
    """Generate rays directly in world space using camera basis vectors."""
    x, y = wp.tid()
    
    if x >= width or y >= height:
        return

    aspect_ratio = float(width) / float(height)
    u = (float(x) + 0.5) / float(width) - 0.5
    v = (float(y) + 0.5) / float(height) - 0.5
    h = wp.tan(fov_y / 2.0)
    
    # Local ray direction (camera looks down -Z)
    x_local = u * 2.0 * h * aspect_ratio
    y_local = -v * 2.0 * h
    
    # Transform to world space
    ray_dir_world = x_local * cam_right + y_local * cam_up + (-1.0) * (-cam_forward)
    
    out_rays[0, y, x, 0] = cam_pos
    out_rays[0, y, x, 1] = wp.normalize(ray_dir_world)


def transforms_and_rays_from_camera_data(camera: CameraData, width: int, height: int) -> tuple[wp.array(dtype=wp.transformf), wp.array(dtype=wp.vec3f)]:
    cam_pos = wp.vec3f(*camera.position)
    cam_right = wp.vec3f(*camera.right)
    cam_up = wp.vec3f(*camera.up)
    cam_forward = wp.vec3f(*camera.forward)
    
    camera_rays = wp.zeros((1, height, width, 2), dtype=wp.vec3f)
    
    wp.launch(
        kernel=generate_world_rays_kernel,
        dim=(width, height),
        inputs=[width, height, camera.fov_vertical, cam_pos, cam_right, cam_up, cam_forward, camera_rays]
    )

    identity_transform = wp.transformf(wp.vec3f(0, 0, 0), wp.quatf(0, 0, 0, 1))
    camera_transforms = wp.array([[identity_transform]], dtype=wp.transformf)

    return camera_transforms, camera_rays


# =============================================================================
# Rendering (Diegetic -> Pixels)
# =============================================================================

def rgb_to_packed_uint32(r: float, g: float, b: float, a: float = 1.0) -> int:
    """Convert RGB floats (0-1) to packed RGBA uint32."""
    return (
        (int(a * 255) << 24) |
        (int(b * 255) << 16) |
        (int(g * 255) << 8) |
        int(r * 255)
    )


def setup_render_context(
    diegetics: list[Diegetic],
    config: RenderConfig,
) -> tuple[RenderContext, ColorLUTs, list]:
    """
    Create RenderContext and ColorLUTs from Diegetics for multi-AOV rendering.
    
    This sets up the scene for a single render pass that outputs multiple AOVs:
    - Lit color render (uses diffuse_albedo)
    - Depth
    - Surface normals
    - Shape indices (for post-processing to object_id and semantic passes)
    
    The ColorLUTs can be used to convert shape_index output to object_id or
    semantic color passes in post-processing.
    
    Args:
        diegetics: List of Diegetic scene elements
        config: Render configuration (resolution, output path)
        
    Returns:
        Tuple of (RenderContext, ColorLUTs, list of warp.Mesh objects)
    """
    num_shapes = len(diegetics)
    print(f"Setting up render context with {num_shapes} diegetics for multi-AOV rendering...")

    warp_meshes = []
    mesh_bounds = np.zeros((num_shapes, 2, 3), dtype=np.float32)
    scene_min = np.full(3, np.inf)
    scene_max = np.full(3, -np.inf)

    for i, d in enumerate(diegetics):
        mesh = wp.Mesh(
            points=wp.array(d.vertices, dtype=wp.vec3f),
            indices=wp.array(d.faces.flatten(), dtype=wp.int32),
        )
        warp_meshes.append(mesh)
        mesh_bounds[i, 0] = d.vertices.min(axis=0)
        mesh_bounds[i, 1] = d.vertices.max(axis=0)
        scene_min = np.minimum(scene_min, mesh_bounds[i, 0])
        scene_max = np.maximum(scene_max, mesh_bounds[i, 1])
    
    print(f"Scene bounds: [{scene_min[0]:.1f}, {scene_min[1]:.1f}, {scene_min[2]:.1f}] to "
          f"[{scene_max[0]:.1f}, {scene_max[1]:.1f}, {scene_max[2]:.1f}]")

    ctx = RenderContext(
        width=config.width, height=config.height,
        enable_textures=False, enable_shadows=True,
        enable_ambient_lighting=True, enable_particles=False,
        num_worlds=1, num_cameras=1,
    )

    ctx.num_shapes = num_shapes
    ctx.mesh_ids = wp.array([m.id for m in warp_meshes], dtype=wp.uint64)
    ctx.mesh_bounds = wp.array(mesh_bounds, dtype=wp.vec3f)
    ctx.shape_types = wp.array([RenderShapeType.MESH] * num_shapes, dtype=wp.int32)
    ctx.shape_enabled = wp.array(list(range(num_shapes)), dtype=wp.uint32)
    ctx.shape_world_index = wp.array([0] * num_shapes, dtype=wp.int32)
    ctx.shape_mesh_indices = wp.array(list(range(num_shapes)), dtype=wp.int32)
    ctx.shape_materials = wp.array([-1] * num_shapes, dtype=wp.int32)

    transforms = [wp.transformf(wp.vec3f(0, 0, 0), wp.quatf(0, 0, 0, 1))] * num_shapes
    ctx.shape_transforms = wp.array(transforms, dtype=wp.transformf)
    ctx.shape_sizes = wp.array([[1, 1, 1]] * num_shapes, dtype=wp.vec3f)

    # Use diffuse_albedo for the lit color render pass
    diffuse_colors = np.array([
        (*d.diffuse_albedo, 1.0) for d in diegetics
    ], dtype=np.float32)
    ctx.shape_colors = wp.array(diffuse_colors, dtype=wp.vec4f)

    ctx.lights_active = wp.array([True], dtype=wp.bool)
    ctx.lights_type = wp.array([1], dtype=wp.int32)
    ctx.lights_cast_shadow = wp.array([True], dtype=wp.bool)
    ctx.lights_position = wp.array([[0, 0, 0]], dtype=wp.vec3f)
    ctx.lights_orientation = wp.array([[-0.577, 0.577, -0.577]], dtype=wp.vec3f)

    # Build ColorLUTs for post-processing shape_index to object_id and semantic
    object_id_lut = np.array([
        rgb_to_packed_uint32(*d.object_id) for d in diegetics
    ], dtype=np.uint32)
    semantic_lut = np.array([
        rgb_to_packed_uint32(*d.semantic) for d in diegetics
    ], dtype=np.uint32)
    
    color_luts = ColorLUTs(
        object_id=wp.array(object_id_lut, dtype=wp.uint32),
        semantic=wp.array(semantic_lut, dtype=wp.uint32),
    )

    return ctx, color_luts, warp_meshes


def render_all_aovs(
    ctx: RenderContext,
    camera: CameraData,
    config: RenderConfig,
) -> RenderOutputs:
    """
    Render the scene from the given camera viewpoint, outputting all AOVs.
    
    This performs a single render pass that outputs:
    - color_image: Lit diffuse render (packed RGBA uint32)
    - depth_image: Ray hit distance (float32)
    - shape_index_image: Shape index at each pixel (uint32)
    - normal_image: Surface normal at hit point (vec3f)
    
    The shape_index_image can be post-processed with ColorLUTs to create
    additional passes like object_id and semantic.
    
    Args:
        ctx: RenderContext configured with scene geometry
        camera: Camera position and orientation
        config: Render configuration (resolution)
        
    Returns:
        RenderOutputs containing all raw GPU output arrays
    """
    camera_transforms, camera_rays = transforms_and_rays_from_camera_data(
        camera, config.width, config.height
    )

    # Create all output arrays
    color_image = ctx.create_color_image_output()
    depth_image = ctx.create_depth_image_output()
    shape_index_image = ctx.create_shape_index_image_output()
    normal_image = ctx.create_normal_image_output()

    # Single render pass outputs all AOVs
    ctx.render(
        camera_transforms=camera_transforms,
        camera_rays=camera_rays,
        color_image=color_image,
        depth_image=depth_image,
        shape_index_image=shape_index_image,
        normal_image=normal_image,
        refit_bvh=True,
        clear_data=ClearData(clear_color=0xFF404040),
    )

    return RenderOutputs(
        color_image=color_image,
        depth_image=depth_image,
        shape_index_image=shape_index_image,
        normal_image=normal_image,
    )


def packed_uint32_to_rgb(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert packed RGBA uint32 array to RGB uint8 array."""
    pixels = packed.reshape(height, width)
    r = (pixels >> 0) & 0xFF
    g = (pixels >> 8) & 0xFF
    b = (pixels >> 16) & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def convert_aovs_to_pixels(
    outputs: RenderOutputs,
    color_luts: ColorLUTs,
    config: RenderConfig,
) -> PixelOutputs:
    """
    Convert raw GPU render outputs to numpy RGB arrays for saving.
    
    Applies ColorLUTs to shape_index_image to create object_id and semantic passes.
    
    Args:
        outputs: Raw RenderOutputs from render_all_aovs
        color_luts: ColorLUTs for object_id and semantic mapping
        config: Render configuration (for dimensions)
        
    Returns:
        PixelOutputs with all passes as (height, width, 3) uint8 arrays
    """
    height, width = config.height, config.width
    
    # Color pass - direct conversion
    color_rgb = packed_uint32_to_rgb(
        outputs.color_image.numpy()[0, 0], height, width
    )
    
    # Depth pass - normalize and convert to grayscale
    depth_range = wp.array([1e10, 0.0], dtype=wp.float32)
    wp.launch(
        find_depth_range,
        outputs.depth_image.shape,
        [outputs.depth_image, depth_range],
    )
    depth_range_np = depth_range.numpy()
    
    depth_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        depth_to_grayscale,
        outputs.depth_image.shape,
        [outputs.depth_image, depth_range_np[0], depth_range_np[1], depth_rgba],
    )
    depth_rgb = packed_uint32_to_rgb(depth_rgba.numpy()[0, 0], height, width)
    
    # Normal pass - convert normals to RGB
    normal_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        normal_to_rgb,
        outputs.normal_image.shape,
        [outputs.normal_image, normal_rgba],
    )
    normal_rgb = packed_uint32_to_rgb(normal_rgba.numpy()[0, 0], height, width)
    
    # Object ID pass - use LUT on shape indices
    object_id_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        shape_index_to_color_lut,
        outputs.shape_index_image.shape,
        [outputs.shape_index_image, color_luts.object_id, object_id_rgba],
    )
    object_id_rgb = packed_uint32_to_rgb(object_id_rgba.numpy()[0, 0], height, width)
    
    # Semantic pass - use LUT on shape indices
    semantic_rgba = wp.zeros_like(outputs.color_image)
    wp.launch(
        shape_index_to_color_lut,
        outputs.shape_index_image.shape,
        [outputs.shape_index_image, color_luts.semantic, semantic_rgba],
    )
    semantic_rgb = packed_uint32_to_rgb(semantic_rgba.numpy()[0, 0], height, width)
    
    return PixelOutputs(
        color=color_rgb,
        depth=depth_rgb,
        normal=normal_rgb,
        object_id=object_id_rgb,
        semantic=semantic_rgb,
    )


def save_pixels_to_png(pixels: np.ndarray, output_path: Path) -> None:
    """Save RGB pixel array to PNG file."""
    if not _PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required for saving PNG files.")
    img = Image.fromarray(pixels, mode="RGB")
    img.save(output_path)
    print(f"Saved: {output_path}")


def save_all_aovs(
    pixel_outputs: PixelOutputs,
    output_dir: Path,
    frame_num: int,
    base_name: str = "render",
    ext: str = "png",
) -> None:
    """
    Save all AOV passes to disk as image files.
    
    Creates files named {base}_{AOV}.{frame:04d}.{ext}:
    - {base}_color.0001.png
    - {base}_depth.0001.png
    - {base}_normal.0001.png
    - {base}_object_id.0001.png
    - {base}_semantic.0001.png
    
    Args:
        pixel_outputs: PixelOutputs from convert_aovs_to_pixels
        output_dir: Directory to save files
        frame_num: Frame number for filename (formatted as 4-digit zero-padded)
        base_name: Base filename (before the AOV suffix)
        ext: File extension (default: "png")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def aov_path(aov: str) -> Path:
        return output_dir / f"{base_name}_{aov}.{frame_num:04d}.{ext}"
    
    save_pixels_to_png(pixel_outputs.color, aov_path("color"))
    save_pixels_to_png(pixel_outputs.depth, aov_path("depth"))
    save_pixels_to_png(pixel_outputs.normal, aov_path("normal"))
    save_pixels_to_png(pixel_outputs.object_id, aov_path("object_id"))
    save_pixels_to_png(pixel_outputs.semantic, aov_path("semantic"))


def render_and_save_all_aovs(
    ctx: RenderContext,
    color_luts: ColorLUTs,
    camera: CameraData,
    config: RenderConfig,
    frame_num: int,
    base_name: str = "render",
    ext: str = "png",
) -> PixelOutputs:
    """
    Render all AOVs and save them to disk.
    
    Convenience function that combines render_all_aovs, convert_aovs_to_pixels,
    and save_all_aovs.
    
    Args:
        ctx: RenderContext configured with scene geometry
        color_luts: ColorLUTs for post-processing
        camera: Camera position and orientation
        config: Render configuration (resolution, output dir)
        frame_num: Frame number for filename (formatted as 4-digit zero-padded)
        base_name: Base filename (before the AOV suffix)
        ext: File extension (default: "png")
        
    Returns:
        PixelOutputs with all passes as numpy arrays
    """
    print(f"Rendering frame {frame_num} (all AOVs)...")
    
    # Single render pass
    outputs = render_all_aovs(ctx, camera, config)
    
    # Convert to pixels
    pixel_outputs = convert_aovs_to_pixels(outputs, color_luts, config)
    
    # Save all passes
    save_all_aovs(pixel_outputs, config.output_dir, frame_num, base_name, ext)
    
    return pixel_outputs


# =============================================================================
# Texture Utilities
# =============================================================================

class TextureColorCache:
    """Cache for texture average colors to avoid reloading."""
    
    def __init__(self):
        self._cache: dict[str, tuple[float, float, float] | None] = {}
    
    def get(self, path: str) -> tuple[float, float, float] | None:
        """Get cached color for path, or None if not cached."""
        return self._cache.get(path)
    
    def set(self, path: str, color: tuple[float, float, float] | None):
        """Cache a color for a path."""
        self._cache[path] = color
    
    def has(self, path: str) -> bool:
        """Check if path is in cache."""
        return path in self._cache
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global cache instance
_texture_color_cache = TextureColorCache()


def resolve_udim_textures(texture_path: str, usd_file_dir: str) -> list[str]:
    """
    Resolve a texture path that may contain <UDIM> placeholder.
    
    Args:
        texture_path: Path that may contain <UDIM> (e.g., "textures/foo.<UDIM>.png")
        usd_file_dir: Directory of the USD file for resolving relative paths
        
    Returns:
        List of resolved file paths (sorted by UDIM number)
    """
    # Resolve relative to USD file directory
    if os.path.isabs(texture_path):
        base_path = texture_path
    else:
        base_path = os.path.join(usd_file_dir, texture_path)
    
    base_path = os.path.normpath(base_path)
    
    # Check for UDIM pattern
    if "<UDIM>" in base_path:
        # Replace <UDIM> with a glob pattern to find all tiles
        glob_pattern = base_path.replace("<UDIM>", "[0-9][0-9][0-9][0-9]")
        matching_files = glob.glob(glob_pattern)
        
        if matching_files:
            # Sort by UDIM number (extract the 4-digit number and sort)
            def extract_udim(path):
                match = re.search(r'\.(\d{4})\.', os.path.basename(path))
                return int(match.group(1)) if match else 9999
            
            matching_files.sort(key=extract_udim)
            return matching_files
        else:
            return []
    else:
        # No UDIM, just return the single path if it exists
        if os.path.exists(base_path):
            return [base_path]
        else:
            return []


def get_texture_average_color(
    texture_path: str,
    usd_file_dir: str,
    cache: TextureColorCache | None = None,
    verbose: bool = False
) -> tuple[float, float, float] | None:
    """
    Load a texture and compute its average RGB color.
    
    Handles <UDIM> patterns by using the lowest UDIM tile (1001).
    
    Args:
        texture_path: Path to the texture (may be relative, may contain <UDIM>)
        usd_file_dir: Directory of the USD file for resolving relative paths
        cache: Optional TextureColorCache for caching results
        verbose: If True, print debug information
        
    Returns:
        tuple (r, g, b) normalized to 0-1, or None if failed
    """
    if not _PIL_AVAILABLE:
        if verbose:
            print("        PIL not available for texture loading")
        return None
    
    if cache is None:
        cache = _texture_color_cache
    
    # Check cache first
    if cache.has(texture_path):
        return cache.get(texture_path)
    
    # Resolve UDIM patterns and get list of texture files
    texture_files = resolve_udim_textures(texture_path, usd_file_dir)
    
    if not texture_files:
        if verbose:
            print(f"        Texture not found: {texture_path}")
            print(f"          (resolved from USD dir: {usd_file_dir})")
        cache.set(texture_path, None)
        return None
    
    # Use the first (lowest UDIM) texture
    resolved_path = texture_files[0]
    
    try:
        if verbose:
            if len(texture_files) > 1:
                print(f"        Found {len(texture_files)} UDIM tiles")
            print(f"        Loading texture: {resolved_path}")
        
        img = Image.open(resolved_path)
        
        # Convert to RGB if necessary (handle RGBA, palette, etc.)
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            # For RGBA, composite over white background to handle transparency
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # Resize to speed up averaging (we don't need full resolution)
        img.thumbnail((64, 64), Image.Resampling.LANCZOS)
        
        # Get average color
        pixels = np.array(img, dtype=np.float32) / 255.0
        avg_color = pixels.mean(axis=(0, 1))
        
        result = tuple(avg_color[:3])
        if verbose:
            print(f"        Average color: RGB({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})")
        
        cache.set(texture_path, result)
        return result
        
    except Exception as e:
        if verbose:
            print(f"        Failed to load texture: {e}")
        cache.set(texture_path, None)
        return None


# =============================================================================
# USD Stage Loading
# =============================================================================

def open_usd_stage(usd_path: str, show_progress: bool = True) -> Usd.Stage:
    """
    Open a USD stage with progress messaging.
    
    Since Usd.Stage.Open() is a blocking call without progress callbacks,
    this prints before/after messages so the user knows parsing is in progress.
    
    Args:
        usd_path: Path to the USD file
        show_progress: If True, print progress messages
        
    Returns:
        Opened USD stage
        
    Raises:
        RuntimeError: If the stage fails to open
    """
    import time
    
    if show_progress:
        print(f"Opening USD stage: {usd_path}", flush=True)
        print("  (This may take a while for large files...)", flush=True)
    
    start_time = time.time()
    stage = Usd.Stage.Open(usd_path)
    elapsed = time.time() - start_time
    
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_path}")
    
    if show_progress:
        print(f"  Stage opened in {elapsed:.1f}s", flush=True)
    
    return stage


def is_proxy_path(path: str) -> bool:
    """Check if a prim path appears to be proxy geometry."""
    return "/proxy/" in path.lower()


# =============================================================================
# Polyscope Visualization (Alternative Consumer)
# =============================================================================

def setup_polyscope(up_direction: str = "z_up") -> None:
    """
    Initialize polyscope with standard settings.
    
    Args:
        up_direction: Up direction ("z_up", "y_up", etc.)
    """
    import polyscope as ps
    ps.init()
    ps.set_up_dir(up_direction)


def register_diegetics_with_polyscope(
    diegetics: list[Diegetic],
    use_diffuse: bool = True,
    verbose: bool = False,
) -> None:
    """
    Register Diegetics with polyscope for interactive visualization.
    
    Args:
        diegetics: List of Diegetic scene elements
        use_diffuse: If True, use diffuse_albedo colors; otherwise use object_id
        verbose: If True, print registration info
    """
    import polyscope as ps
    
    for d in diegetics:
        pm = ps.register_surface_mesh(d.name, d.vertices, d.faces)
        color = d.diffuse_albedo if use_diffuse else d.object_id
        pm.set_color(color)
        
        if verbose:
            print(f"  {d.name}: {color}")


def show_polyscope() -> None:
    """Show the polyscope viewer (blocking)."""
    import polyscope as ps
    ps.show()

