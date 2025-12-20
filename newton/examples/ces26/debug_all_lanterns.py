# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Simple USD mesh debug script using polyscope.

Finds all meshes with "Hanging" in their path and displays them.
Shows only "beauty" geometry (skips proxy meshes).

Usage: python newton/examples/ces26/debug_all_lanterns.py
"""

import os
import numpy as np
import polyscope as ps
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdShade, Ar

# Cache for texture average colors (path -> (r, g, b))
_texture_color_cache = {}

# Path to USD file
USD_FILE = r"C:\Users\chorvath\Downloads\20251219_iv060_flat_01\Collected_iv060_flat_01\iv060_flat_01.usd"

# Target pattern to search for (case-insensitive)
TARGET_PATTERN = "Hanging"

# Time code for sampling transforms
TIME_CODE = Usd.TimeCode(2920)


def triangulate(indices, counts):
    """Convert polygon mesh to triangles via fan triangulation."""
    tris = []
    idx = 0
    for count in counts:
        if count >= 3:
            for i in range(1, count - 1):
                tris.append([indices[idx], indices[idx + i], indices[idx + i + 1]])
        idx += count
    return np.array(tris, dtype=np.int32)


def resolve_udim_textures(texture_path: str, usd_file_dir: str) -> list[str]:
    """
    Resolve a texture path that may contain <UDIM> placeholder.
    
    Args:
        texture_path: Path that may contain <UDIM> (e.g., "textures/foo.<UDIM>.png")
        usd_file_dir: Directory of the USD file for resolving relative paths
        
    Returns:
        List of resolved file paths (sorted by UDIM number)
    """
    import glob
    import re
    
    # First resolve relative to USD file directory
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
                # Find 4-digit UDIM number in the filename
                match = re.search(r'\.(\d{4})\.', os.path.basename(path))
                return int(match.group(1)) if match else 9999
            
            matching_files.sort(key=extract_udim)
            print(f"        Found {len(matching_files)} UDIM tiles")
            return matching_files
        else:
            print(f"        No UDIM tiles found for pattern: {glob_pattern}")
            return []
    else:
        # No UDIM, just return the single path if it exists
        if os.path.exists(base_path):
            return [base_path]
        else:
            return []


def get_texture_average_color(texture_path: str, usd_file_dir: str) -> tuple | None:
    """
    Load a texture and compute its average RGB color.
    Handles <UDIM> patterns by using the lowest UDIM tile (1001).
    
    Args:
        texture_path: Path to the texture (may be relative, may contain <UDIM>)
        usd_file_dir: Directory of the USD file for resolving relative paths
        
    Returns:
        tuple (r, g, b) normalized to 0-1, or None if failed
    """
    global _texture_color_cache
    
    # Check cache first
    if texture_path in _texture_color_cache:
        return _texture_color_cache[texture_path]
    
    # Resolve UDIM patterns and get list of texture files
    texture_files = resolve_udim_textures(texture_path, usd_file_dir)
    
    if not texture_files:
        print(f"        Texture not found: {texture_path}")
        print(f"          (resolved from USD dir: {usd_file_dir})")
        _texture_color_cache[texture_path] = None
        return None
    
    # Use the first (lowest UDIM) texture
    resolved_path = texture_files[0]
    
    try:
        print(f"        Loading texture: {resolved_path}")
        img = Image.open(resolved_path)
        
        # Convert to RGB if necessary (handle RGBA, palette, etc.)
        if img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            # For RGBA, composite over white background to handle transparency
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
        
        # Resize to speed up averaging (we don't need full resolution)
        img.thumbnail((64, 64), Image.Resampling.LANCZOS)
        
        # Get average color
        pixels = np.array(img, dtype=np.float32) / 255.0
        avg_color = pixels.mean(axis=(0, 1))
        
        result = tuple(avg_color[:3])
        print(f"        Average color: RGB({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})")
        
        _texture_color_cache[texture_path] = result
        return result
        
    except Exception as e:
        print(f"        Failed to load texture: {e}")
        _texture_color_cache[texture_path] = None
        return None


def get_material_color(prim, time_code, usd_file_dir: str):
    """
    Extract diffuseColor from UsdPreviewSurface material bound to a prim.
    
    If the diffuseColor is connected to a texture, attempts to load the texture
    and compute its average color.
    
    Args:
        prim: The USD prim (mesh) to get material from
        time_code: USD time code for sampling
        usd_file_dir: Directory of the USD file for resolving texture paths
        
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
    
    # Check if it's a UsdPreviewSurface
    shaderId = shader.GetIdAttr().Get()
    print(f"      Shader ID: {shaderId}")
    
    # Try diffuseColor first (UsdPreviewSurface standard)
    diffuseInput = shader.GetInput("diffuseColor")
    if diffuseInput:
        # Check if it's connected to another shader (e.g., a texture)
        connSrc = diffuseInput.GetConnectedSource()
        if connSrc:
            # It's connected to something (likely a texture reader - UsdUVTexture)
            texturePrimPath = connSrc[0].GetPrim().GetPath()
            print(f"      diffuseColor is connected to: {texturePrimPath}")
            
            connectedShader = UsdShade.Shader(connSrc[0].GetPrim())
            connectedShaderId = connectedShader.GetIdAttr().Get()
            print(f"        Connected shader ID: {connectedShaderId}")
            
            # Try to get the texture file path
            fileInput = connectedShader.GetInput("file")
            if fileInput:
                fileVal = fileInput.Get(time_code)
                if fileVal:
                    # fileVal is a Sdf.AssetPath
                    # Use the authored path (.path) rather than resolvedPath, 
                    # since resolvedPath may resolve relative to wrong directory
                    # Our resolve_udim_textures will handle resolution relative to USD file
                    texturePath = fileVal.path
                    print(f"        Texture file (authored): {texturePath}")
                    print(f"        Texture file (resolved by USD): {fileVal.resolvedPath}")
                    
                    # Load texture and get average color
                    avgColor = get_texture_average_color(texturePath, usd_file_dir)
                    if avgColor is not None:
                        return avgColor
            
            # Fallback: check for fallback input on the texture shader
            fallbackInput = connectedShader.GetInput("fallback")
            if fallbackInput:
                fallbackVal = fallbackInput.Get(time_code)
                if fallbackVal is not None:
                    print(f"        Using fallback from texture: {fallbackVal}")
                    return tuple(fallbackVal)[:3]
        else:
            # Direct value (not connected to texture)
            val = diffuseInput.Get(time_code)
            if val is not None:
                print(f"      diffuseColor value: {val}")
                # Could be Gf.Vec3f or tuple
                if hasattr(val, '__iter__'):
                    return tuple(val)[:3]
    
    # Try baseColor (some shaders use this)
    baseColorInput = shader.GetInput("baseColor")
    if baseColorInput:
        val = baseColorInput.Get(time_code)
        if val is not None:
            print(f"      baseColor value: {val}")
            if hasattr(val, '__iter__'):
                return tuple(val)[:3]
    
    # List all inputs for debugging
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


def rotate_points_by_quaternion(points: np.ndarray, quat_wxyz: tuple) -> np.ndarray:
    """
    Rotate points by a quaternion using vectorized operations.
    
    Args:
        points: (N, 3) array of points
        quat_wxyz: quaternion as (w, x, y, z) - USD/GfQuaternion convention
        
    Returns:
        (N, 3) array of rotated points
    """
    w, x, y, z = quat_wxyz
    q_vec = np.array([x, y, z], dtype=np.float64)
    
    # Rodrigues rotation formula via quaternion:
    # t = 2 * (q_xyz × v)
    # v' = v + w*t + (q_xyz × t)
    t = 2.0 * np.cross(q_vec, points)
    rotated = points + w * t + np.cross(q_vec, t)
    return rotated


def main():
    print(f"Loading: {USD_FILE}")
    stage = Usd.Stage.Open(USD_FILE)
    if not stage:
        print("Failed to open USD file")
        return
    
    # Flatten the stage to ensure all layers, references, and overrides are composed
    # print("Flattening stage to resolve all composition arcs...")
    # flattened_layer = stage.Flatten()
    # stage = Usd.Stage.Open(flattened_layer)

    xcache = UsdGeom.XformCache(TIME_CODE)

    # Find all prims containing our target pattern
    print(f"\nSearching for prims containing '{TARGET_PATTERN}'...")
    
    meshes_found = []

    # Traverse including instance proxies
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        path_str = str(prim.GetPath())
        
        # Check if this prim path contains our target pattern (case-insensitive)
        if TARGET_PATTERN.lower() not in path_str.lower():
            continue

        
        meshPrim = prim 
        # If this is an instance proxy, pull geometry from the prototype prim 
        if meshPrim.IsInstanceProxy(): 
            geomPrim = meshPrim.GetPrimInPrototype() 
        else: 
            geomPrim = meshPrim
        mesh = UsdGeom.Mesh(geomPrim)

        if mesh.ComputeEffectiveVisibility(UsdGeom.Tokens.render, TIME_CODE) == UsdGeom.Tokens.invisible:
            print(f"  SKIPPING effectively invisible: {path_str}")
            continue
            
        # Skip proxy meshes - we want beauty geo
        if "/proxy/" in path_str.lower():
            print(f"  SKIP (proxy): {path_str}")
            continue

            
        try:
            mesh = UsdGeom.Mesh(geomPrim)
            points = mesh.GetPointsAttr().Get(TIME_CODE)
            indices = mesh.GetFaceVertexIndicesAttr().Get(TIME_CODE)
            counts = mesh.GetFaceVertexCountsAttr().Get(TIME_CODE)
            displayColor = mesh.GetDisplayColorPrimvar().Get(TIME_CODE)
            
            if not points or not indices or not counts:
                print(f"  SKIP (no data): {path_str}")
                continue
            
            # ---- Get material color from UsdPreviewSurface ----
            print(f"\n  === COLOR for: {path_str.split('/')[-1]} ===")
            usd_dir = os.path.dirname(USD_FILE)
            materialColor = get_material_color(meshPrim, TIME_CODE, usd_dir)
            print(f"    Material color: {materialColor}")
            print(f"    displayColor primvar: {displayColor}")
            print(f"  === END COLOR ===\n")
            
            # IMPORTANT: world transform must be computed on the *proxy prim* (meshPrim) 
            # not on geomPrim (prototype)
            world_mat = xcache.GetLocalToWorldTransform(meshPrim)
            M = np.array(world_mat, dtype=np.float64)
            pts = np.array(points, dtype=np.float64)
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)

            # Row-vectors * matrix
            pts_world_h = pts_h @ M 
            pts_world = pts_world_h[:, :3]
            
            # Triangulate
            tris = triangulate(np.array(indices), np.array(counts))
            
            print(f"  FOUND: {path_str}")
            print(f"         {len(pts)} verts, {len(tris)} triangles")
            
            meshes_found.append({
                "name": path_str.split("/")[-1],
                "path": path_str,
                "vertices": pts_world,
                "faces": tris,
                "displayColor": displayColor,
                "materialColor": materialColor
            })
            
        except Exception as e:
            print(f"  ERROR: {path_str}: {e}")
    
    print(f"\nFound {len(meshes_found)} meshes")
    
    if not meshes_found:
        print("No meshes found. Listing all prims with target pattern...")
        for prim in stage.Traverse():
            if TARGET_PATTERN.lower() in str(prim.GetPath()).lower():
                print(f"  {prim.GetPath()} [{prim.GetTypeName()}]")
        return

    # Check that all mesh names are unique. If not, warn and make unique.
    names = [m["name"] for m in meshes_found]
    if len(names) != len(set(names)):
        print("\nWARNING: Mesh names are not unique. Making names unique.")
        name_counts = {}
        for m in meshes_found:
            name = m["name"]
            if name in name_counts:
                name_counts[name] += 1
                m["name"] = f"{name}_{name_counts[name]}"
            else:
                name_counts[name] = 1  # first instance keeps name as is
    
    # Display with polyscope
    ps.init()
    ps.set_up_dir("z_up")
    
    for m in meshes_found:
        name = m["name"]
        pm = ps.register_surface_mesh(m["name"], m["vertices"], m["faces"])
        
        # Prefer material color, fall back to displayColor
        matCol = m["materialColor"]
        dispCol = m["displayColor"]
        
        if matCol is not None:
            print(f"  {name}: using materialColor {matCol}")
            pm.set_color(tuple(matCol[:3]))
        elif dispCol is not None and len(dispCol) > 0:
            # displayColor might be per-vertex or constant
            if hasattr(dispCol[0], '__iter__'):
                # It's an array of colors, use the first one as constant
                col = tuple(dispCol[0])[:3]
            else:
                col = tuple(dispCol)[:3]
            print(f"  {name}: using displayColor {col}")
            pm.set_color(col)
        else:
            print(f"  {name}: no color found, using default")

    
    print("\nPolyscope viewer opened. Close window to exit.")
    ps.show()


if __name__ == "__main__":
    main()

