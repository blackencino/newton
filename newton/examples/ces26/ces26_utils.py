# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Reusable utilities for loading and visualizing USD scene geometry.

Provides clean, modular components for:
- Loading meshes from USD stages with proper transform handling
- Resolving UDIM textures and computing average colors
- Extracting material colors from UsdPreviewSurface shaders
- Visualizing meshes with polyscope

These utilities are designed to be used across various test and debug scripts.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade

# Optional PIL import for texture loading
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MeshData:
    """Container for extracted mesh geometry and material information."""
    name: str
    path: str
    vertices: np.ndarray  # World-space vertex positions (N, 3)
    faces: np.ndarray  # Triangle indices (M, 3)
    display_color: tuple[float, float, float] | None = None
    material_color: tuple[float, float, float] | None = None
    
    def get_color(self) -> tuple[float, float, float] | None:
        """Return the best available color (material preferred over display)."""
        return self.material_color or self.display_color


@dataclass
class MeshLoadOptions:
    """Options for controlling mesh loading behavior."""
    time_code: Usd.TimeCode = field(default_factory=lambda: Usd.TimeCode.Default())
    load_material_colors: bool = True
    load_texture_colors: bool = False  # Requires PIL
    path_filter: Callable[[str], bool] | None = None  # Return True to include
    skip_invisible: bool = True
    skip_proxy: bool = True  # Skip paths containing "/proxy/"
    
    def __post_init__(self):
        if self.load_texture_colors and not _PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow is required for texture color loading. "
                "Install with: pip install Pillow"
            )


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
# Material Utilities
# =============================================================================

def get_material_color(
    prim: Usd.Prim,
    time_code: Usd.TimeCode,
    usd_file_dir: str,
    load_textures: bool = False,
    verbose: bool = False
) -> tuple[float, float, float] | None:
    """
    Extract diffuseColor from UsdPreviewSurface material bound to a prim.
    
    If load_textures is True and the diffuseColor is connected to a texture,
    attempts to load the texture and compute its average color.
    
    Args:
        prim: The USD prim (mesh) to get material from
        time_code: USD time code for sampling
        usd_file_dir: Directory of the USD file for resolving texture paths
        load_textures: If True, attempt to load and sample textures
        verbose: If True, print debug information
        
    Returns:
        tuple (r, g, b) or None if no color found
    """
    materialBinding = UsdShade.MaterialBindingAPI(prim)
    boundMaterial, bindingRel = materialBinding.ComputeBoundMaterial()
    
    if not boundMaterial:
        return None
    
    # Try to get the surface shader output
    surfaceOutput = boundMaterial.GetSurfaceOutput()
    if not surfaceOutput:
        return None
    
    # Get connected shader
    connectedSource = surfaceOutput.GetConnectedSource()
    if not connectedSource:
        return None
    
    shaderPrim = connectedSource[0].GetPrim()
    shader = UsdShade.Shader(shaderPrim)
    
    # Check shader type
    shaderId = shader.GetIdAttr().Get()
    if verbose:
        print(f"      Shader ID: {shaderId}")
    
    # Try diffuseColor first (UsdPreviewSurface standard)
    diffuseInput = shader.GetInput("diffuseColor")
    if diffuseInput:
        # Check if it's connected to another shader (e.g., a texture)
        connSrc = diffuseInput.GetConnectedSource()
        if connSrc:
            # It's connected to something (likely a texture reader - UsdUVTexture)
            connectedShader = UsdShade.Shader(connSrc[0].GetPrim())
            connectedShaderId = connectedShader.GetIdAttr().Get()
            
            if verbose:
                texturePrimPath = connSrc[0].GetPrim().GetPath()
                print(f"      diffuseColor is connected to: {texturePrimPath}")
                print(f"        Connected shader ID: {connectedShaderId}")
            
            # Try to get the texture file path
            if load_textures:
                fileInput = connectedShader.GetInput("file")
                if fileInput:
                    fileVal = fileInput.Get(time_code)
                    if fileVal:
                        texturePath = fileVal.path
                        if verbose:
                            print(f"        Texture file (authored): {texturePath}")
                            print(f"        Texture file (resolved by USD): {fileVal.resolvedPath}")
                        
                        avgColor = get_texture_average_color(
                            texturePath, usd_file_dir, verbose=verbose
                        )
                        if avgColor is not None:
                            return avgColor
            
            # Fallback: check for fallback input on the texture shader
            fallbackInput = connectedShader.GetInput("fallback")
            if fallbackInput:
                fallbackVal = fallbackInput.Get(time_code)
                if fallbackVal is not None:
                    if verbose:
                        print(f"        Using fallback from texture: {fallbackVal}")
                    return tuple(fallbackVal)[:3]
        else:
            # Direct value (not connected to texture)
            val = diffuseInput.Get(time_code)
            if val is not None:
                if verbose:
                    print(f"      diffuseColor value: {val}")
                if hasattr(val, '__iter__'):
                    return tuple(val)[:3]
    
    # Try baseColor (some shaders use this)
    baseColorInput = shader.GetInput("baseColor")
    if baseColorInput:
        val = baseColorInput.Get(time_code)
        if val is not None:
            if verbose:
                print(f"      baseColor value: {val}")
            if hasattr(val, '__iter__'):
                return tuple(val)[:3]
    
    # Debug: list all inputs if verbose
    if verbose:
        print(f"      All shader inputs:")
        for inp in shader.GetInputs():
            inpName = inp.GetBaseName()
            inpVal = inp.Get(time_code)
            connSrc = inp.GetConnectedSource()
            if connSrc:
                print(f"        {inpName}: connected to {connSrc[0].GetPrim().GetPath()}")
            else:
                print(f"        {inpName}: {inpVal}")
    
    return None


def extract_display_color(
    display_color_primvar
) -> tuple[float, float, float] | None:
    """
    Extract a single RGB color from a displayColor primvar value.
    
    Args:
        display_color_primvar: Raw value from GetDisplayColorPrimvar().Get()
        
    Returns:
        tuple (r, g, b) or None if no valid color
    """
    if display_color_primvar is None or len(display_color_primvar) == 0:
        return None
    
    first = display_color_primvar[0]
    if hasattr(first, '__getitem__'):
        # It's a Gf.Vec3f or similar - extract components explicitly
        return (float(first[0]), float(first[1]), float(first[2]))
    else:
        # Might be a flat color value
        return (float(display_color_primvar[0]), 
                float(display_color_primvar[1]), 
                float(display_color_primvar[2]))


# =============================================================================
# Mesh Loading
# =============================================================================

def is_proxy_path(path: str) -> bool:
    """Check if a prim path appears to be proxy geometry."""
    return "/proxy/" in path.lower()


def load_meshes_from_stage(
    stage: Usd.Stage,
    usd_file_path: str,
    options: MeshLoadOptions | None = None,
    verbose: bool = False
) -> list[MeshData]:
    """
    Load all visible mesh geometry from a USD stage.
    
    Handles instanced geometry correctly by:
    - Traversing instance proxies
    - Getting geometry from prototype prims
    - Computing world transforms from proxy prims
    
    Args:
        stage: Opened USD stage
        usd_file_path: Path to the USD file (for resolving texture paths)
        options: MeshLoadOptions controlling what to load
        verbose: If True, print progress and debug information
        
    Returns:
        List of MeshData objects with world-space geometry
    """
    if options is None:
        options = MeshLoadOptions()
    
    usd_file_dir = os.path.dirname(usd_file_path)
    time_code = options.time_code
    xcache = UsdGeom.XformCache(time_code)
    
    meshes_found = []
    
    # Traverse including instance proxies
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        path_str = str(prim.GetPath())
        
        # Apply path filter if provided
        if options.path_filter is not None:
            if not options.path_filter(path_str):
                continue
        
        # Get the actual geometry prim (prototype for instances)
        mesh_prim = prim
        if mesh_prim.IsInstanceProxy():
            geom_prim = mesh_prim.GetPrimInPrototype()
        else:
            geom_prim = mesh_prim
        
        mesh = UsdGeom.Mesh(geom_prim)
        
        # Check visibility
        if options.skip_invisible:
            if mesh.ComputeEffectiveVisibility(UsdGeom.Tokens.render, time_code) == UsdGeom.Tokens.invisible:
                if verbose:
                    print(f"  SKIPPING (invisible): {path_str}")
                continue
        
        # Skip proxy meshes
        if options.skip_proxy and is_proxy_path(path_str):
            if verbose:
                print(f"  SKIPPING (proxy): {path_str}")
            continue
        
        # Extract geometry data
        try:
            points = mesh.GetPointsAttr().Get(time_code)
            indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)
            counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
            
            if not points or not indices or not counts:
                if verbose:
                    print(f"  SKIPPING (no geometry data): {path_str}")
                continue
            
            # Transform to world space using the proxy prim's transform
            world_mat = xcache.GetLocalToWorldTransform(mesh_prim)
            pts_world = transform_points_to_world(points, world_mat)
            
            # Triangulate
            tris = triangulate(np.array(indices), np.array(counts))
            
            # Get colors
            display_color = None
            material_color = None
            
            display_color_raw = mesh.GetDisplayColorPrimvar().Get(time_code)
            display_color = extract_display_color(display_color_raw)
            
            if options.load_material_colors:
                material_color = get_material_color(
                    mesh_prim,
                    time_code,
                    usd_file_dir,
                    load_textures=options.load_texture_colors,
                    verbose=verbose
                )
            
            mesh_data = MeshData(
                name=path_str.split("/")[-1],
                path=path_str,
                vertices=pts_world,
                faces=tris,
                display_color=display_color,
                material_color=material_color
            )
            
            if verbose:
                print(f"  LOADED: {path_str}")
                print(f"          {len(pts_world)} verts, {len(tris)} triangles")
            
            meshes_found.append(mesh_data)
            
        except Exception as e:
            if verbose:
                print(f"  ERROR loading {path_str}: {e}")
    
    return meshes_found


def make_mesh_names_unique(meshes: list[MeshData]) -> None:
    """
    Modify mesh names in-place to ensure uniqueness.
    
    Appends _2, _3, etc. to duplicate names.
    """
    name_counts: dict[str, int] = {}
    
    for mesh in meshes:
        name = mesh.name
        if name in name_counts:
            name_counts[name] += 1
            mesh.name = f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 1


# =============================================================================
# Polyscope Visualization
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


def register_meshes_with_polyscope(
    meshes: list[MeshData],
    default_color: tuple[float, float, float] = (0.7, 0.7, 0.7),
    verbose: bool = False
) -> None:
    """
    Register a list of meshes with polyscope for visualization.
    
    Uses material color if available, falls back to display color,
    then to the default color.
    
    Args:
        meshes: List of MeshData objects
        default_color: Color to use if mesh has no color information
        verbose: If True, print which color source is used
    """
    import polyscope as ps
    
    for mesh in meshes:
        pm = ps.register_surface_mesh(mesh.name, mesh.vertices, mesh.faces)
        
        try:
            color = mesh.get_color()
            
            if color is not None:
                if verbose:
                    source = "material" if mesh.material_color else "display"
                    print(f"  {mesh.name}: using {source} color {color}")
                pm.set_color(color)
            else:
                if verbose:
                    print(f"  {mesh.name}: using default color")
                pm.set_color(default_color)
                
        except Exception as e:
            print(f"WARNING: failed to set color for mesh {mesh.name}: {e}")
            pm.set_color(default_color)


def show_polyscope() -> None:
    """Show the polyscope viewer (blocking)."""
    import polyscope as ps
    ps.show()


# =============================================================================
# High-Level Convenience Functions
# =============================================================================

def load_and_visualize_usd(
    usd_path: str,
    time_code: Usd.TimeCode | None = None,
    path_filter: Callable[[str], bool] | None = None,
    load_textures: bool = False,
    up_direction: str = "z_up",
    verbose: bool = True
) -> list[MeshData]:
    """
    High-level function to load a USD file and visualize it with polyscope.
    
    Args:
        usd_path: Path to the USD file
        time_code: USD time code for sampling (default: Usd.TimeCode.Default())
        path_filter: Optional filter function for prim paths
        load_textures: If True, sample textures for material colors
        up_direction: Polyscope up direction
        verbose: If True, print progress information
        
    Returns:
        List of loaded MeshData objects
    """
    if verbose:
        print(f"Loading: {usd_path}")
    
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_path}")
    
    options = MeshLoadOptions(
        time_code=time_code or Usd.TimeCode.Default(),
        load_material_colors=True,
        load_texture_colors=load_textures,
        path_filter=path_filter,
        skip_invisible=True,
        skip_proxy=True
    )
    
    meshes = load_meshes_from_stage(stage, usd_path, options, verbose=verbose)
    
    if verbose:
        print(f"\nFound {len(meshes)} meshes")
    
    if not meshes:
        if verbose:
            print("No meshes found matching criteria.")
        return meshes
    
    make_mesh_names_unique(meshes)
    
    setup_polyscope(up_direction)
    register_meshes_with_polyscope(meshes, verbose=verbose)
    
    if verbose:
        print("\nPolyscope viewer opened. Close window to exit.")
    
    show_polyscope()
    
    return meshes

