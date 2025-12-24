# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Preprocessing pipeline for dynamic segmentation of USD scenes.

This module parses a USD scene and extracts:
- DiageticMetadata: Lightweight per-mesh metadata (no heavy geometry arrays)
- DiageticGroupMetadata: Grouped diegetics by scene graph + objectid_color
- CameraCurve: Pre-baked camera animation over frame range
- Path danger values: Distance from camera to each group's bounding ellipsoid
- Group categories: Ground/Terrain, Unsafe, Safe

All data can be cached to disk as NPZ to avoid re-parsing the large USD file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from pxr import Gf, Usd, UsdGeom

from .utils import (
    BoundingEllipsoid,
    compute_body_centric_measurements,
    usd_to_craig_4x4,
)


# =============================================================================
# Type Aliases
# =============================================================================

RGB = tuple[float, float, float]


# =============================================================================
# DiageticMetadata: Lightweight per-mesh metadata
# =============================================================================


@dataclass(frozen=True)
class DiageticMetadata:
    """
    Lightweight metadata for a single diegetic mesh.

    Does NOT store heavy geometry data (vertices, faces) - those are only used
    during group computation and then discarded or saved separately.

    Attributes:
        path: Full USD prim path (e.g., "/World/Props/Chair/Mesh")
        name: Short name (last component of path)
        objectid_color: RGB color from objectid_color primvar, or None if missing
        world_T_local: 4x4 transform from local to world space (Craig notation)
        vertex_count: Number of vertices (for statistics)
        triangle_count: Number of triangles (for statistics)
    """

    path: str
    name: str
    objectid_color: RGB | None
    world_T_local: np.ndarray  # (4, 4) Craig notation
    vertex_count: int
    triangle_count: int


# =============================================================================
# USD Parsing Utilities
# =============================================================================


def _get_objectid_color(prim: Usd.Prim, time_code: Usd.TimeCode) -> RGB | None:
    """
    Extract objectid_color primvar from a prim.

    Returns None if the primvar doesn't exist or isn't constant interpolation.
    """
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    primvar = primvars_api.GetPrimvar("objectid_color")

    if not primvar or not primvar.HasValue():
        return None

    if primvar.GetInterpolation() != UsdGeom.Tokens.constant:
        return None

    value = primvar.Get(time_code)
    if value is None:
        return None

    try:
        if hasattr(value, "__getitem__") and len(value) >= 3:
            return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, IndexError):
        pass

    return None


def _is_proxy_path(path: str) -> bool:
    """Check if a prim path appears to be proxy geometry."""
    return "/proxy/" in path.lower()


def _is_visible_mesh(prim: Usd.Prim, time_code: Usd.TimeCode) -> bool:
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


def _triangulate(indices: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Convert polygon mesh to triangles via fan triangulation."""
    tris = []
    idx = 0
    for count in counts:
        if count >= 3:
            for i in range(1, count - 1):
                tris.append([indices[idx], indices[idx + i], indices[idx + i + 1]])
        idx += count
    return np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)


def _transform_points_to_world(points: np.ndarray, world_matrix: Gf.Matrix4d) -> np.ndarray:
    """Transform points from local to world space using a 4x4 matrix."""
    M = np.array(world_matrix, dtype=np.float64)
    pts = np.array(points, dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    pts_world_h = pts_h @ M  # USD row-vector convention
    return pts_world_h[:, :3]


@dataclass(frozen=True)
class _ExtractedMeshData:
    """Intermediate result from mesh extraction (includes heavy geometry)."""

    metadata: DiageticMetadata
    vertices_world: np.ndarray  # (N, 3) float32, world space
    faces: np.ndarray  # (M, 3) int32


def _extract_mesh_data(
    prim: Usd.Prim,
    xcache: UsdGeom.XformCache,
    time_code: Usd.TimeCode,
) -> _ExtractedMeshData | None:
    """
    Extract mesh data and metadata from a USD mesh prim.

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

    # Get world transform (using proxy prim's transform for instances)
    world_mat_usd = xcache.GetLocalToWorldTransform(prim)

    # Transform vertices to world space
    vertices_world = _transform_points_to_world(points, world_mat_usd).astype(np.float32)

    # Triangulate
    faces = _triangulate(np.array(indices), np.array(counts))

    # Convert transform to Craig notation
    world_T_local = usd_to_craig_4x4(np.array(world_mat_usd, dtype=np.float64))

    # Extract objectid_color
    objectid_color = _get_objectid_color(prim, time_code)

    path_str = str(prim.GetPath())
    name = path_str.split("/")[-1]

    metadata = DiageticMetadata(
        path=path_str,
        name=name,
        objectid_color=objectid_color,
        world_T_local=world_T_local,
        vertex_count=len(vertices_world),
        triangle_count=len(faces),
    )

    return _ExtractedMeshData(
        metadata=metadata,
        vertices_world=vertices_world,
        faces=faces,
    )


def parse_diagetic_metadata(
    stage: Usd.Stage,
    time_code: Usd.TimeCode,
    path_filter: Callable[[str], bool] | None = None,
    skip_invisible: bool = True,
    skip_proxy: bool = True,
    verbose: bool = False,
    show_progress: bool = True,
    progress_interval: int = 1000,
) -> tuple[list[DiageticMetadata], list[_ExtractedMeshData]]:
    """
    Parse USD stage and extract metadata for all diegetic meshes.

    Returns both lightweight metadata and full mesh data (with geometry).
    The full mesh data is needed for group computation but can be discarded after.

    Args:
        stage: Opened USD stage
        time_code: Time code for sampling (typically start frame)
        path_filter: Optional filter function (path -> bool) to exclude prims
        skip_invisible: If True, skip prims with invisible visibility
        skip_proxy: If True, skip prims with "/proxy/" in path
        verbose: Print detailed progress
        show_progress: Print periodic progress updates
        progress_interval: Update interval for progress messages

    Returns:
        Tuple of (list of DiageticMetadata, list of _ExtractedMeshData)
    """
    xcache = UsdGeom.XformCache(time_code)

    metadata_list: list[DiageticMetadata] = []
    mesh_data_list: list[_ExtractedMeshData] = []
    prims_processed = 0
    prims_skipped = 0

    if show_progress:
        print("Parsing USD stage for diegetic metadata...", flush=True)

    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        prims_processed += 1

        if show_progress and prims_processed % progress_interval == 0:
            print(
                f"  Processed {prims_processed} prims, found {len(metadata_list)} diegetics...",
                flush=True,
            )

        # Filter: must be a mesh
        if not prim.IsA(UsdGeom.Mesh):
            continue

        path_str = str(prim.GetPath())

        # Filter: user-provided path filter
        if path_filter is not None and not path_filter(path_str):
            continue

        # Filter: visibility
        if skip_invisible and not _is_visible_mesh(prim, time_code):
            if verbose:
                print(f"  SKIP (invisible): {path_str}")
            prims_skipped += 1
            continue

        # Filter: proxy geometry
        if skip_proxy and _is_proxy_path(path_str):
            if verbose:
                print(f"  SKIP (proxy): {path_str}")
            prims_skipped += 1
            continue

        # Extract mesh data
        mesh_data = _extract_mesh_data(prim, xcache, time_code)
        if mesh_data is None:
            if verbose:
                print(f"  SKIP (no geometry): {path_str}")
            prims_skipped += 1
            continue

        metadata_list.append(mesh_data.metadata)
        mesh_data_list.append(mesh_data)

        if verbose:
            print(
                f"  PARSED: {path_str} ({mesh_data.metadata.vertex_count} verts, "
                f"{mesh_data.metadata.triangle_count} tris)"
            )

    if show_progress:
        print(
            f"  Done: {prims_processed} prims -> {len(metadata_list)} diegetics, "
            f"{prims_skipped} skipped",
            flush=True,
        )

    return metadata_list, mesh_data_list


# =============================================================================
# DiageticGroupMetadata: Grouped diegetics by scene graph + objectid_color
# =============================================================================


class GroupCategory(Enum):
    """Category for diegetic groups based on semantic meaning."""

    GROUND_TERRAIN = "ground_terrain"
    UNSAFE = "unsafe"  # Collisions must be considered
    SAFE = "safe"  # Collisions possible but not important


@dataclass(frozen=True)
class DiageticGroupMetadata:
    """
    Metadata for a group of diegetics that share the same objectid_color
    and have a common scene graph ancestor.

    Attributes:
        group_id: Unique identifier for this group
        unique_name: Shortest unique name for this group
        common_ancestor_path: USD path of the common ancestor prim
        objectid_color: Shared objectid_color (same for all members)
        member_paths: List of USD paths for all member diegetics
        member_count: Number of diegetics in this group
        total_vertex_count: Sum of vertex counts across all members
        total_triangle_count: Sum of triangle counts across all members
        bbox_min: Bounding box minimum (world space)
        bbox_max: Bounding box maximum (world space)
        bounding_ellipsoid: Body-centric bounding ellipsoid
        path_danger: Minimum distance from camera path to ellipsoid (computed later)
        category: Group category (Ground/Unsafe/Safe)
    """

    group_id: str
    unique_name: str
    common_ancestor_path: str
    objectid_color: RGB
    member_paths: tuple[str, ...]
    member_count: int
    total_vertex_count: int
    total_triangle_count: int
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)
    bounding_ellipsoid: BoundingEllipsoid
    path_danger: float  # Filled in later
    category: GroupCategory  # Filled in later


def _get_objectid_color_at_path(
    stage: Usd.Stage,
    path: str,
    time_code: Usd.TimeCode,
) -> RGB | None:
    """Get objectid_color at a specific prim path, if it exists."""
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    return _get_objectid_color(prim, time_code)


def _find_colored_root_for_mesh(
    stage: Usd.Stage,
    mesh_path: str,
    color: RGB,
    time_code: Usd.TimeCode,
) -> str:
    """
    Find the highest ancestor of a mesh that has the same objectid_color.

    Walks up the hierarchy from the mesh's parent, finding the highest prim
    that has the same objectid_color. This defines the "colored subtree root"
    for this mesh.

    If no ancestor has the color, returns the mesh's immediate parent
    (typically the /geo or asset root).

    Args:
        stage: USD stage
        mesh_path: Path to the mesh prim
        color: The objectid_color to match
        time_code: Time code for sampling

    Returns:
        Path of the highest ancestor with matching color, or mesh parent if none
    """
    # Start from parent of mesh (skip /geo level, go to asset level)
    parts = mesh_path.split("/")

    # Find the asset-level parent (typically 3rd level: /World/AssetName)
    # Walk up looking for colored ancestors
    best_colored_ancestor = None
    current = mesh_path

    while current and current != "/" and current != "/World":
        # Move to parent
        parts = current.rsplit("/", 1)
        current = parts[0] if parts[0] else "/"

        if current == "/" or current == "/World":
            break

        # Check if this ancestor has the same color
        ancestor_color = _get_objectid_color_at_path(stage, current, time_code)
        if ancestor_color is not None and _colors_match(ancestor_color, color):
            best_colored_ancestor = current

    # If we found a colored ancestor, use it; otherwise use the asset-level parent
    if best_colored_ancestor:
        return best_colored_ancestor
    else:
        # Return the first meaningful ancestor (e.g., /World/HangingLanternA_01)
        mesh_parts = mesh_path.split("/")
        if len(mesh_parts) >= 3:
            return "/".join(mesh_parts[:3])  # /World/AssetName
        return mesh_path.rsplit("/", 1)[0]


def _colors_match(c1: RGB, c2: RGB, tolerance: float = 1e-6) -> bool:
    """Check if two colors are equal within tolerance."""
    return all(abs(a - b) < tolerance for a, b in zip(c1, c2))


def _color_to_key(color: RGB) -> str:
    """Convert color to a hashable string key."""
    return f"{color[0]:.6f},{color[1]:.6f},{color[2]:.6f}"


def group_diagetics_by_color_and_ancestor(
    stage: Usd.Stage,
    mesh_data_list: list[_ExtractedMeshData],
    time_code: Usd.TimeCode,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, DiageticGroupMetadata]:
    """
    Group diegetics by shared objectid_color and common scene graph ancestor.

    Two diegetics are in the same group if:
    1. They have the same objectid_color
    2. They belong to the same "colored subtree" (have the same colored root ancestor)

    This properly handles cases where different assets share the same color but
    are in different parts of the scene graph.

    Args:
        stage: USD stage (needed to check ancestor colors)
        mesh_data_list: List of extracted mesh data with geometry
        time_code: Time code for sampling
        verbose: Print detailed progress
        show_progress: Print progress updates

    Returns:
        Dictionary mapping group_id to DiageticGroupMetadata
    """
    if show_progress:
        print("Grouping diegetics by color and ancestor...", flush=True)

    # For each mesh, find its "colored root" - the highest ancestor with same color
    # Group by (color_key, colored_root_path)
    subtree_groups: dict[tuple[str, str], list[_ExtractedMeshData]] = {}
    no_color_meshes: list[_ExtractedMeshData] = []

    for mesh in mesh_data_list:
        color = mesh.metadata.objectid_color
        if color is None:
            no_color_meshes.append(mesh)
        else:
            # Find the colored root for this mesh
            colored_root = _find_colored_root_for_mesh(
                stage, mesh.metadata.path, color, time_code
            )
            color_key = _color_to_key(color)
            group_key = (color_key, colored_root)

            if group_key not in subtree_groups:
                subtree_groups[group_key] = []
            subtree_groups[group_key].append(mesh)

    unique_colors = len(set(k[0] for k in subtree_groups.keys()))
    if show_progress:
        print(f"  Found {unique_colors} unique colors, {len(subtree_groups)} colored subtrees")
        print(f"  {len(no_color_meshes)} meshes without color")

    groups: dict[str, DiageticGroupMetadata] = {}
    group_counter = 0

    for (color_key, colored_root), meshes in subtree_groups.items():
        color = meshes[0].metadata.objectid_color
        assert color is not None

        paths = [m.metadata.path for m in meshes]

        # Concatenate all vertices for body-centric measurements
        all_vertices = np.vstack([m.vertices_world for m in meshes])

        # Compute bounding box
        bbox_min = all_vertices.min(axis=0)
        bbox_max = all_vertices.max(axis=0)

        # Compute body-centric measurements
        body_measurements = compute_body_centric_measurements(all_vertices)

        # Generate group ID and unique name from the colored root
        group_id = f"group_{group_counter:04d}"
        root_name = colored_root.split("/")[-1] if colored_root != "/" else "root"
        unique_name = f"{root_name}_{group_counter}"

        # Statistics
        total_verts = sum(m.metadata.vertex_count for m in meshes)
        total_tris = sum(m.metadata.triangle_count for m in meshes)

        group = DiageticGroupMetadata(
            group_id=group_id,
            unique_name=unique_name,
            common_ancestor_path=colored_root,
            objectid_color=color,
            member_paths=tuple(paths),
            member_count=len(meshes),
            total_vertex_count=total_verts,
            total_triangle_count=total_tris,
            bbox_min=bbox_min.astype(np.float32),
            bbox_max=bbox_max.astype(np.float32),
            bounding_ellipsoid=body_measurements.bounding_ellipsoid,
            path_danger=float("inf"),  # Computed later
            category=GroupCategory.UNSAFE,  # Computed later
        )

        groups[group_id] = group
        group_counter += 1

        if verbose:
            print(
                f"  {group_id}: {unique_name} ({len(meshes)} meshes, "
                f"{total_verts} verts, root={colored_root})"
            )

    # Handle meshes without objectid_color - each becomes its own group
    for mesh in no_color_meshes:
        group_id = f"group_{group_counter:04d}"
        unique_name = f"no_color_{mesh.metadata.name}_{group_counter}"

        body_measurements = compute_body_centric_measurements(mesh.vertices_world)

        group = DiageticGroupMetadata(
            group_id=group_id,
            unique_name=unique_name,
            common_ancestor_path=mesh.metadata.path,
            objectid_color=(0.5, 0.5, 0.5),  # Default gray
            member_paths=(mesh.metadata.path,),
            member_count=1,
            total_vertex_count=mesh.metadata.vertex_count,
            total_triangle_count=mesh.metadata.triangle_count,
            bbox_min=mesh.vertices_world.min(axis=0).astype(np.float32),
            bbox_max=mesh.vertices_world.max(axis=0).astype(np.float32),
            bounding_ellipsoid=body_measurements.bounding_ellipsoid,
            path_danger=float("inf"),
            category=GroupCategory.UNSAFE,
        )

        groups[group_id] = group
        group_counter += 1

    if show_progress:
        print(f"  Created {len(groups)} diegetic groups", flush=True)

    return groups


def make_group_names_unique(
    groups: dict[str, DiageticGroupMetadata],
) -> dict[str, DiageticGroupMetadata]:
    """
    Ensure all group unique_names are actually unique.

    Returns new dict with updated groups (frozen dataclass, so we create new ones).
    """
    # Count name occurrences
    name_counts: dict[str, int] = {}
    for g in groups.values():
        name = g.unique_name
        name_counts[name] = name_counts.get(name, 0) + 1

    # Find duplicates and rename
    name_indices: dict[str, int] = {}
    result: dict[str, DiageticGroupMetadata] = {}

    for group_id, group in groups.items():
        name = group.unique_name
        if name_counts[name] > 1:
            idx = name_indices.get(name, 0)
            name_indices[name] = idx + 1
            new_name = f"{name}_{idx}"
        else:
            new_name = name

        if new_name != group.unique_name:
            # Create new group with updated name
            group = DiageticGroupMetadata(
                group_id=group.group_id,
                unique_name=new_name,
                common_ancestor_path=group.common_ancestor_path,
                objectid_color=group.objectid_color,
                member_paths=group.member_paths,
                member_count=group.member_count,
                total_vertex_count=group.total_vertex_count,
                total_triangle_count=group.total_triangle_count,
                bbox_min=group.bbox_min,
                bbox_max=group.bbox_max,
                bounding_ellipsoid=group.bounding_ellipsoid,
                path_danger=group.path_danger,
                category=group.category,
            )

        result[group_id] = group

    return result


# =============================================================================
# CameraCurve: Pre-baked camera animation
# =============================================================================


@dataclass(frozen=True)
class CameraCurve:
    """
    Pre-baked camera animation over a frame range.

    Stores world-space eye positions for each frame, which are used to
    compute proximity ("path danger") to diegetic groups.

    Attributes:
        camera_path: USD path to the camera prim
        frames: Array of frame numbers (N,)
        positions: World-space eye positions (N, 3)
        forwards: World-space forward vectors (N, 3) - view direction
    """

    camera_path: str
    frames: np.ndarray  # (N,) int
    positions: np.ndarray  # (N, 3) float32
    forwards: np.ndarray  # (N, 3) float32


def extract_camera_curve(
    stage: Usd.Stage,
    camera_path: str,
    frames: list[int],
    verbose: bool = False,
    show_progress: bool = True,
) -> CameraCurve:
    """
    Extract camera animation over a range of frames.

    Args:
        stage: USD stage
        camera_path: Path to camera prim
        frames: List of frame numbers to sample
        verbose: Print detailed progress
        show_progress: Print progress updates

    Returns:
        CameraCurve with positions for each frame
    """
    if show_progress:
        print(f"Extracting camera curve for {len(frames)} frames...", flush=True)

    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Camera not found at {camera_path}")

    positions = []
    forwards = []

    for i, frame in enumerate(frames):
        time_code = Usd.TimeCode(frame)
        xcache = UsdGeom.XformCache(time_code)
        world_mat = xcache.GetLocalToWorldTransform(camera_prim)

        # Transform origin and -Z direction to world space
        M = np.array(world_mat, dtype=np.float64)
        pts_local = np.array([[0, 0, 0], [0, 0, -1]], dtype=np.float64)
        pts_h = np.concatenate([pts_local, np.ones((2, 1), dtype=np.float64)], axis=1)
        pts_world = (pts_h @ M)[:, :3]

        origin = pts_world[0]
        forward_pt = pts_world[1]
        forward = forward_pt - origin
        forward = forward / np.linalg.norm(forward)

        positions.append(origin)
        forwards.append(forward)

        if verbose and (i % 50 == 0 or i == len(frames) - 1):
            print(f"  Frame {frame}: pos=[{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}]")

    if show_progress:
        print(f"  Extracted {len(frames)} camera positions", flush=True)

    return CameraCurve(
        camera_path=camera_path,
        frames=np.array(frames, dtype=np.int32),
        positions=np.array(positions, dtype=np.float32),
        forwards=np.array(forwards, dtype=np.float32),
    )


# =============================================================================
# Path Danger Computation
# =============================================================================


def compute_path_danger(
    group: DiageticGroupMetadata,
    camera_curve: CameraCurve,
) -> tuple[np.ndarray, float]:
    """
    Compute the distance curve from camera to group's bounding ellipsoid.

    Args:
        group: Diegetic group with bounding ellipsoid
        camera_curve: Pre-baked camera animation

    Returns:
        Tuple of (distance_curve, min_distance):
        - distance_curve: (N,) array of distances for each frame
        - min_distance: Minimum distance over all frames ("path danger")
    """
    ellipsoid = group.bounding_ellipsoid
    distances = np.array(
        [ellipsoid.distance_to_point(pos) for pos in camera_curve.positions],
        dtype=np.float32,
    )
    min_distance = float(distances.min())
    return distances, min_distance


def update_groups_with_path_danger(
    groups: dict[str, DiageticGroupMetadata],
    camera_curve: CameraCurve,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, DiageticGroupMetadata]:
    """
    Update all groups with path danger values.

    Returns new dict with updated groups (frozen dataclass).
    """
    if show_progress:
        print("Computing path danger for all groups...", flush=True)

    result: dict[str, DiageticGroupMetadata] = {}

    for group_id, group in groups.items():
        _, min_distance = compute_path_danger(group, camera_curve)

        # Create new group with updated path_danger
        updated_group = DiageticGroupMetadata(
            group_id=group.group_id,
            unique_name=group.unique_name,
            common_ancestor_path=group.common_ancestor_path,
            objectid_color=group.objectid_color,
            member_paths=group.member_paths,
            member_count=group.member_count,
            total_vertex_count=group.total_vertex_count,
            total_triangle_count=group.total_triangle_count,
            bbox_min=group.bbox_min,
            bbox_max=group.bbox_max,
            bounding_ellipsoid=group.bounding_ellipsoid,
            path_danger=min_distance,
            category=group.category,
        )

        result[group_id] = updated_group

        if verbose:
            print(f"  {group.unique_name}: path_danger={min_distance:.2f}")

    if show_progress:
        # Find most and least dangerous
        sorted_groups = sorted(result.values(), key=lambda g: g.path_danger)
        if sorted_groups:
            closest = sorted_groups[0]
            print(f"  Closest to camera: {closest.unique_name} (dist={closest.path_danger:.2f})")

    return result


# =============================================================================
# Group Categorization
# =============================================================================


def _is_ground_or_terrain(group: DiageticGroupMetadata) -> bool:
    """
    Check if a group is ground or terrain for shot IV060.

    Specific patterns for this shot:
    - /World/StarWarsSet_01/assembly/Terrain_01 (terrain meshes)
    - /World/SimGravelLrg_01 (gravel ground cover)

    Checks both the common_ancestor_path and actual member mesh paths.
    """
    # Check common ancestor
    path_lower = group.common_ancestor_path.lower()
    if "/terrain_01" in path_lower or "/simgravellrg" in path_lower:
        return True

    # Check member paths for terrain-related content
    for member_path in group.member_paths:
        member_lower = member_path.lower()
        # Specific terrain paths in this shot
        if "/terrain_01/" in member_lower:
            return True
        if "/simgravellrg" in member_lower:
            return True
        # Match mesh names containing 'terrain'
        if "/terrain" in member_lower and "/geo/" in member_lower:
            return True

    return False


def _is_safe_object(group: DiageticGroupMetadata) -> bool:
    """
    Check if a group is a "safe" object (collisions not important) for shot IV060.

    Specific patterns for this shot:
    - /World/HangingLanternX_YY (all lantern variants A, B, C, D, E)
    - /World/HangingLanternChainX_YY (chain links)
    - /World/HangingLanternChainNLinkA (chain link variants)
    """
    path_lower = group.common_ancestor_path.lower()

    # Match any HangingLantern or chain variant
    if "hanginglantern" in path_lower:
        return True

    # Also check member paths (in case ancestor is too high)
    for member_path in group.member_paths:
        if "hanginglantern" in member_path.lower():
            return True

    return False


def categorize_group(group: DiageticGroupMetadata) -> GroupCategory:
    """
    Categorize a group based on naming heuristics.

    Priority:
    1. Ground/Terrain - checked first
    2. Safe - hanging lanterns, chains, etc.
    3. Unsafe - everything else (collisions must be considered)
    """
    if _is_ground_or_terrain(group):
        return GroupCategory.GROUND_TERRAIN
    elif _is_safe_object(group):
        return GroupCategory.SAFE
    else:
        return GroupCategory.UNSAFE


def update_groups_with_categories(
    groups: dict[str, DiageticGroupMetadata],
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, DiageticGroupMetadata]:
    """
    Update all groups with category classifications.

    Returns new dict with updated groups.
    """
    if show_progress:
        print("Categorizing groups...", flush=True)

    result: dict[str, DiageticGroupMetadata] = {}
    category_counts: dict[GroupCategory, int] = {c: 0 for c in GroupCategory}

    for group_id, group in groups.items():
        category = categorize_group(group)
        category_counts[category] += 1

        # Create new group with updated category
        updated_group = DiageticGroupMetadata(
            group_id=group.group_id,
            unique_name=group.unique_name,
            common_ancestor_path=group.common_ancestor_path,
            objectid_color=group.objectid_color,
            member_paths=group.member_paths,
            member_count=group.member_count,
            total_vertex_count=group.total_vertex_count,
            total_triangle_count=group.total_triangle_count,
            bbox_min=group.bbox_min,
            bbox_max=group.bbox_max,
            bounding_ellipsoid=group.bounding_ellipsoid,
            path_danger=group.path_danger,
            category=category,
        )

        result[group_id] = updated_group

        if verbose:
            print(f"  {group.unique_name}: {category.value}")

    if show_progress:
        for cat, count in category_counts.items():
            print(f"  {cat.value}: {count} groups")

    return result


# =============================================================================
# NPZ Cache Save/Load
# =============================================================================


def save_preprocessing_cache(
    output_path: Path,
    metadata_list: list[DiageticMetadata],
    groups: dict[str, DiageticGroupMetadata],
    camera_curve: CameraCurve,
    mesh_data_list: list[_ExtractedMeshData] | None = None,
    verbose: bool = False,
) -> None:
    """
    Save preprocessing results to NPZ file.

    Args:
        output_path: Path to save the NPZ file
        metadata_list: List of DiageticMetadata
        groups: Dictionary of DiageticGroupMetadata
        camera_curve: Pre-baked camera animation
        mesh_data_list: Optional list of mesh data with geometry (can be large!)
        verbose: Print detailed progress
    """
    print(f"Saving preprocessing cache to {output_path}...", flush=True)

    data: dict[str, np.ndarray] = {}

    # Save metadata list (as structured arrays where possible)
    data["metadata_paths"] = np.array([m.path for m in metadata_list], dtype=object)
    data["metadata_names"] = np.array([m.name for m in metadata_list], dtype=object)
    data["metadata_vertex_counts"] = np.array([m.vertex_count for m in metadata_list], dtype=np.int32)
    data["metadata_triangle_counts"] = np.array(
        [m.triangle_count for m in metadata_list], dtype=np.int32
    )

    # Save objectid colors (with None handling)
    colors = []
    for m in metadata_list:
        if m.objectid_color is not None:
            colors.append(list(m.objectid_color) + [1.0])  # RGB + valid flag
        else:
            colors.append([0.0, 0.0, 0.0, 0.0])  # Invalid
    data["metadata_objectid_colors"] = np.array(colors, dtype=np.float32)

    # Save transforms
    transforms = np.array([m.world_T_local for m in metadata_list], dtype=np.float64)
    data["metadata_transforms"] = transforms

    # Save groups
    group_ids = list(groups.keys())
    data["group_ids"] = np.array(group_ids, dtype=object)
    data["group_unique_names"] = np.array([groups[gid].unique_name for gid in group_ids], dtype=object)
    data["group_ancestor_paths"] = np.array(
        [groups[gid].common_ancestor_path for gid in group_ids], dtype=object
    )
    data["group_colors"] = np.array(
        [list(groups[gid].objectid_color) for gid in group_ids], dtype=np.float32
    )
    data["group_member_counts"] = np.array(
        [groups[gid].member_count for gid in group_ids], dtype=np.int32
    )
    data["group_vertex_counts"] = np.array(
        [groups[gid].total_vertex_count for gid in group_ids], dtype=np.int32
    )
    data["group_triangle_counts"] = np.array(
        [groups[gid].total_triangle_count for gid in group_ids], dtype=np.int32
    )
    data["group_bbox_min"] = np.array([groups[gid].bbox_min for gid in group_ids], dtype=np.float32)
    data["group_bbox_max"] = np.array([groups[gid].bbox_max for gid in group_ids], dtype=np.float32)
    data["group_path_danger"] = np.array(
        [groups[gid].path_danger for gid in group_ids], dtype=np.float32
    )
    data["group_categories"] = np.array(
        [groups[gid].category.value for gid in group_ids], dtype=object
    )

    # Save ellipsoid data for each group
    data["group_ellipsoid_centers"] = np.array(
        [groups[gid].bounding_ellipsoid.center_world for gid in group_ids], dtype=np.float32
    )
    data["group_ellipsoid_radii"] = np.array(
        [groups[gid].bounding_ellipsoid.radii for gid in group_ids], dtype=np.float32
    )
    data["group_ellipsoid_axes"] = np.array(
        [groups[gid].bounding_ellipsoid.body_axes_world for gid in group_ids], dtype=np.float32
    )

    # Save member paths as single concatenated array with offsets
    all_member_paths = []
    member_offsets = [0]
    for gid in group_ids:
        all_member_paths.extend(groups[gid].member_paths)
        member_offsets.append(len(all_member_paths))
    data["group_member_paths"] = np.array(all_member_paths, dtype=object)
    data["group_member_offsets"] = np.array(member_offsets, dtype=np.int32)

    # Save camera curve
    data["camera_path"] = np.array([camera_curve.camera_path], dtype=object)
    data["camera_frames"] = camera_curve.frames
    data["camera_positions"] = camera_curve.positions
    data["camera_forwards"] = camera_curve.forwards

    # Optionally save mesh geometry (can be very large!)
    if mesh_data_list is not None:
        print("  Including mesh geometry (this may be large)...", flush=True)

        # Concatenate all vertices with offsets
        all_vertices = []
        vertex_offsets = [0]
        all_faces = []
        face_offsets = [0]

        for mesh in mesh_data_list:
            # Offset face indices by current vertex count
            offset_faces = mesh.faces + len(all_vertices)
            all_vertices.extend(mesh.vertices_world.tolist())
            all_faces.extend(offset_faces.tolist())
            vertex_offsets.append(len(all_vertices))
            face_offsets.append(len(all_faces))

        data["mesh_vertices"] = np.array(all_vertices, dtype=np.float32)
        data["mesh_vertex_offsets"] = np.array(vertex_offsets, dtype=np.int32)
        data["mesh_faces"] = np.array(all_faces, dtype=np.int32)
        data["mesh_face_offsets"] = np.array(face_offsets, dtype=np.int32)

    # Save compressed
    np.savez_compressed(output_path, **data)

    # Report file size
    file_size = os.path.getsize(output_path)
    size_mb = file_size / (1024 * 1024)
    print(f"  Saved {size_mb:.1f} MB to {output_path}", flush=True)


def load_preprocessing_cache(
    input_path: Path,
    verbose: bool = False,
) -> tuple[list[DiageticMetadata], dict[str, DiageticGroupMetadata], CameraCurve]:
    """
    Load preprocessing results from NPZ file.

    Args:
        input_path: Path to the NPZ file
        verbose: Print detailed progress

    Returns:
        Tuple of (metadata_list, groups, camera_curve)
    """
    print(f"Loading preprocessing cache from {input_path}...", flush=True)

    data = np.load(input_path, allow_pickle=True)

    # Load metadata list
    paths = data["metadata_paths"]
    names = data["metadata_names"]
    vertex_counts = data["metadata_vertex_counts"]
    triangle_counts = data["metadata_triangle_counts"]
    objectid_colors = data["metadata_objectid_colors"]
    transforms = data["metadata_transforms"]

    metadata_list = []
    for i in range(len(paths)):
        color_data = objectid_colors[i]
        objectid_color = None if color_data[3] < 0.5 else tuple(color_data[:3])

        metadata = DiageticMetadata(
            path=str(paths[i]),
            name=str(names[i]),
            objectid_color=objectid_color,
            world_T_local=transforms[i],
            vertex_count=int(vertex_counts[i]),
            triangle_count=int(triangle_counts[i]),
        )
        metadata_list.append(metadata)

    # Load groups
    group_ids = data["group_ids"]
    unique_names = data["group_unique_names"]
    ancestor_paths = data["group_ancestor_paths"]
    colors = data["group_colors"]
    member_counts = data["group_member_counts"]
    group_vertex_counts = data["group_vertex_counts"]
    group_triangle_counts = data["group_triangle_counts"]
    bbox_mins = data["group_bbox_min"]
    bbox_maxs = data["group_bbox_max"]
    path_dangers = data["group_path_danger"]
    categories = data["group_categories"]
    ellipsoid_centers = data["group_ellipsoid_centers"]
    ellipsoid_radii = data["group_ellipsoid_radii"]
    ellipsoid_axes = data["group_ellipsoid_axes"]
    member_paths_all = data["group_member_paths"]
    member_offsets = data["group_member_offsets"]

    groups = {}
    for i, gid in enumerate(group_ids):
        gid = str(gid)

        # Extract member paths for this group
        start = member_offsets[i]
        end = member_offsets[i + 1]
        member_paths = tuple(str(p) for p in member_paths_all[start:end])

        # Reconstruct ellipsoid
        ellipsoid = BoundingEllipsoid(
            center_world=ellipsoid_centers[i],
            radii=ellipsoid_radii[i],
            body_axes_world=ellipsoid_axes[i],
        )

        group = DiageticGroupMetadata(
            group_id=gid,
            unique_name=str(unique_names[i]),
            common_ancestor_path=str(ancestor_paths[i]),
            objectid_color=tuple(colors[i]),
            member_paths=member_paths,
            member_count=int(member_counts[i]),
            total_vertex_count=int(group_vertex_counts[i]),
            total_triangle_count=int(group_triangle_counts[i]),
            bbox_min=bbox_mins[i],
            bbox_max=bbox_maxs[i],
            bounding_ellipsoid=ellipsoid,
            path_danger=float(path_dangers[i]),
            category=GroupCategory(str(categories[i])),
        )
        groups[gid] = group

    # Load camera curve
    camera_curve = CameraCurve(
        camera_path=str(data["camera_path"][0]),
        frames=data["camera_frames"],
        positions=data["camera_positions"],
        forwards=data["camera_forwards"],
    )

    print(
        f"  Loaded {len(metadata_list)} diegetics, {len(groups)} groups, "
        f"{len(camera_curve.frames)} camera frames",
        flush=True,
    )

    return metadata_list, groups, camera_curve


# =============================================================================
# High-Level Preprocessing Pipeline
# =============================================================================


def run_preprocessing_pipeline(
    stage: Usd.Stage,
    camera_path: str,
    frames: list[int],
    output_cache_path: Path | None = None,
    include_geometry: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
) -> tuple[list[DiageticMetadata], dict[str, DiageticGroupMetadata], CameraCurve]:
    """
    Run the complete preprocessing pipeline.

    Steps:
    1. Parse USD for diegetic metadata
    2. Group diegetics by color and ancestor
    3. Make group names unique
    4. Extract camera curve
    5. Compute path danger for all groups
    6. Categorize groups
    7. Optionally save to cache

    Args:
        stage: Opened USD stage
        camera_path: Path to camera prim
        frames: List of frame numbers
        output_cache_path: Optional path to save NPZ cache
        include_geometry: If True, include geometry in cache (large!)
        verbose: Print detailed progress
        show_progress: Print progress updates

    Returns:
        Tuple of (metadata_list, groups, camera_curve)
    """
    time_code = Usd.TimeCode(frames[0])

    # Step 1: Parse USD
    metadata_list, mesh_data_list = parse_diagetic_metadata(
        stage=stage,
        time_code=time_code,
        verbose=verbose,
        show_progress=show_progress,
    )

    if not metadata_list:
        raise RuntimeError("No diegetics found in USD stage")

    # Step 2: Group by color and ancestor
    groups = group_diagetics_by_color_and_ancestor(
        stage=stage,
        mesh_data_list=mesh_data_list,
        time_code=time_code,
        verbose=verbose,
        show_progress=show_progress,
    )

    # Step 3: Make names unique
    groups = make_group_names_unique(groups)

    # Step 4: Extract camera curve
    camera_curve = extract_camera_curve(
        stage=stage,
        camera_path=camera_path,
        frames=frames,
        verbose=verbose,
        show_progress=show_progress,
    )

    # Step 5: Compute path danger
    groups = update_groups_with_path_danger(
        groups=groups,
        camera_curve=camera_curve,
        verbose=verbose,
        show_progress=show_progress,
    )

    # Step 6: Categorize groups
    groups = update_groups_with_categories(
        groups=groups,
        verbose=verbose,
        show_progress=show_progress,
    )

    # Step 7: Optionally save cache
    if output_cache_path is not None:
        save_preprocessing_cache(
            output_path=output_cache_path,
            metadata_list=metadata_list,
            groups=groups,
            camera_curve=camera_curve,
            mesh_data_list=mesh_data_list if include_geometry else None,
            verbose=verbose,
        )

    return metadata_list, groups, camera_curve

