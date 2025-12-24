# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Dynamic segmentation preprocessing utilities for USD scenes.

This package provides tools for:
- Body-centric measurements of point clouds
- Transform conversions (USD row-vector to Craig column-vector notation)
- Preprocessing USD scenes into semantic group metadata
- Camera curve extraction and path danger computation
- Group categorization (Ground/Terrain, Unsafe, Safe)
"""

from .preprocess import (
    CachedGeometry,
    CameraCurve,
    DiageticGroupMetadata,
    DiageticMetadata,
    GroupCategory,
    extract_camera_curve,
    group_diagetics_by_color_and_ancestor,
    load_geometry_from_cache,
    load_preprocessing_cache,
    parse_diagetic_metadata,
    run_preprocessing_pipeline,
    save_preprocessing_cache,
    update_groups_with_categories,
    update_groups_with_path_danger,
)
from .utils import (
    BodyCentricMeasurements,
    BoundingEllipsoid,
    compose_transforms,
    compute_body_centric_measurements,
    make_rotation_4x4,
    make_translation_4x4,
    transform_points_craig,
    usd_to_craig_3x3,
    usd_to_craig_4x4,
)

__all__ = [
    # Utils
    "BodyCentricMeasurements",
    "BoundingEllipsoid",
    "compose_transforms",
    "compute_body_centric_measurements",
    "make_rotation_4x4",
    "make_translation_4x4",
    "transform_points_craig",
    "usd_to_craig_3x3",
    "usd_to_craig_4x4",
    # Preprocess
    "CachedGeometry",
    "CameraCurve",
    "DiageticGroupMetadata",
    "DiageticMetadata",
    "GroupCategory",
    "extract_camera_curve",
    "group_diagetics_by_color_and_ancestor",
    "load_geometry_from_cache",
    "load_preprocessing_cache",
    "parse_diagetic_metadata",
    "run_preprocessing_pipeline",
    "save_preprocessing_cache",
    "update_groups_with_categories",
    "update_groups_with_path_danger",
]

