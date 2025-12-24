# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Dynamic segmentation preprocessing utilities for USD scenes.

This package provides tools for:
- Body-centric measurements of point clouds
- Transform conversions (USD row-vector to Craig column-vector notation)
- Preprocessing USD scenes into semantic group metadata
"""

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
    "BodyCentricMeasurements",
    "BoundingEllipsoid",
    "compose_transforms",
    "compute_body_centric_measurements",
    "make_rotation_4x4",
    "make_translation_4x4",
    "transform_points_craig",
    "usd_to_craig_3x3",
    "usd_to_craig_4x4",
]

