# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for dynamic segmentation preprocessing.

Core concepts:
- Craig notation: Transformations named as output_T_input, column vectors (V_new = M @ V_old)
- USD uses row vectors (V_new = V_old @ M), so we transpose when converting
- Body-centric measurements: Centroid, covariance, SVD-based body frame, bounding ellipsoid

All functions are pure/functional with immutable outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =============================================================================
# Transform Conversion: USD (row-vector) to Craig Notation (column-vector)
# =============================================================================


def usd_to_craig_3x3(usd_matrix_3x3: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 USD rotation/scale matrix from row-vector to column-vector convention.

    USD convention: V_new = V_old @ M  (row vectors)
    Craig convention: V_new = M @ V_old  (column vectors)

    The conversion is simply a transpose.

    Args:
        usd_matrix_3x3: 3x3 matrix in USD row-vector convention

    Returns:
        3x3 matrix in Craig column-vector convention
    """
    return np.asarray(usd_matrix_3x3, dtype=np.float64).T


def usd_to_craig_4x4(usd_matrix_4x4: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 USD transform matrix from row-vector to column-vector convention.

    USD convention: [x, y, z, 1] @ M = [x', y', z', 1]
    Craig convention: M @ [x, y, z, 1]^T = [x', y', z', 1]^T

    Args:
        usd_matrix_4x4: 4x4 homogeneous transform matrix in USD row-vector convention

    Returns:
        4x4 homogeneous transform matrix in Craig column-vector convention
    """
    return np.asarray(usd_matrix_4x4, dtype=np.float64).T


def make_translation_4x4(offset: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 translation matrix in Craig notation.

    Args:
        offset: 3D translation vector (what to add to go from input to output frame)

    Returns:
        4x4 homogeneous translation matrix
    """
    M = np.eye(4, dtype=np.float64)
    M[:3, 3] = offset
    return M


def make_rotation_4x4(R: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 rotation/scale matrix in Craig notation from a 3x3 matrix.

    Args:
        R: 3x3 rotation (or rotation+scale) matrix

    Returns:
        4x4 homogeneous matrix with R in upper-left 3x3
    """
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    return M


# =============================================================================
# Body-Centric Measurements
# =============================================================================


@dataclass(frozen=True)
class BoundingEllipsoid:
    """
    Axis-aligned bounding ellipsoid in the body frame, transformable to world space.

    The ellipsoid is centered at `center_world` and has semi-axis lengths
    `radii` along the body frame axes (defined by `body_axes_world`).

    Attributes:
        center_world: Ellipsoid center in world space (3,)
        radii: Semi-axis lengths along body frame axes (3,) - ordered max to min
        body_axes_world: 3x3 matrix where columns are body frame axes in world space
                         Column 0 = max radius direction, Column 2 = min radius direction
    """

    center_world: np.ndarray  # (3,)
    radii: np.ndarray  # (3,) - semi-axis lengths (max, mid, min)
    body_axes_world: np.ndarray  # (3, 3) - columns are axes in world space

    def distance_to_point(self, point_world: np.ndarray) -> float:
        """
        Compute approximate distance from a world-space point to the ellipsoid surface.

        Uses the "spherical cow" normalization: transforms the point to a space
        where the ellipsoid is a unit sphere, computes distance there, then
        scales back. This is an approximation but works well for proximity testing.

        Args:
            point_world: Point in world space (3,)

        Returns:
            Approximate distance to ellipsoid surface (negative if inside)
        """
        # Vector from ellipsoid center to point
        delta = np.asarray(point_world, dtype=np.float64) - self.center_world

        # Transform to body frame
        # body_axes_world columns are the axes, so to go from world to body
        # we use the transpose (assuming orthonormal axes)
        delta_body = self.body_axes_world.T @ delta

        # Scale to unit sphere space (divide by radii)
        delta_unit = delta_body / self.radii

        # Distance in unit sphere space
        dist_unit = np.linalg.norm(delta_unit)

        # The distance from the surface of a unit sphere
        surface_dist_unit = dist_unit - 1.0

        # Scale back to world space using average radius as approximation
        # (This is the "spherical cow" approximation)
        avg_radius = np.mean(self.radii)
        return float(surface_dist_unit * avg_radius)

    def contains_point(self, point_world: np.ndarray) -> bool:
        """
        Check if a world-space point is inside the ellipsoid.

        Args:
            point_world: Point in world space (3,)

        Returns:
            True if point is inside (or on surface of) the ellipsoid
        """
        delta = np.asarray(point_world, dtype=np.float64) - self.center_world
        delta_body = self.body_axes_world.T @ delta
        delta_unit = delta_body / self.radii
        return float(np.linalg.norm(delta_unit)) <= 1.0


@dataclass(frozen=True)
class BodyCentricMeasurements:
    """
    Immutable container for body-centric measurements of a point cloud.

    All transforms follow Craig notation (column vectors, output_T_input naming).

    Attributes:
        centroid: Mean position of all vertices in world space (3,)
        centered_T_world: 4x4 transform from world to centered space (translates by -centroid)
        covariance: 3x3 covariance matrix of centered positions
        eigenvalues: Eigenvalues from SVD, sorted descending (max, mid, min) (3,)
        eigenvectors: 3x3 matrix where columns are eigenvectors matching eigenvalues
        body_T_centered: 3x3 rotation from centered to body frame (axes aligned to eigenvectors)
        sphericalCow_T_body: 3x3 scale matrix that normalizes to roughly spherical
        unitSphericalCow_T_sphericalCow: 3x3 uniform scale to unit radius
        unitSphericalCow_T_centered: 3x3 composed transform (for convenience)
        max_body_radius: Maximum radius in sphericalCow space before unit normalization
        bounding_ellipsoid: World-space bounding ellipsoid representation
    """

    centroid: np.ndarray  # (3,) world space
    centered_T_world: np.ndarray  # (4, 4)
    covariance: np.ndarray  # (3, 3)
    eigenvalues: np.ndarray  # (3,) sorted descending
    eigenvectors: np.ndarray  # (3, 3) columns are eigenvectors
    body_T_centered: np.ndarray  # (3, 3) rotation matrix
    sphericalCow_T_body: np.ndarray  # (3, 3) scale matrix
    unitSphericalCow_T_sphericalCow: np.ndarray  # (3, 3) uniform scale
    unitSphericalCow_T_centered: np.ndarray  # (3, 3) composed
    max_body_radius: float
    bounding_ellipsoid: BoundingEllipsoid


def compute_body_centric_measurements(vertices_world: np.ndarray) -> BodyCentricMeasurements:
    """
    Compute body-centric measurements from a set of world-space vertices.

    This function analyzes the spatial distribution of vertices to find:
    - The centroid (center of mass)
    - Principal axes via SVD of the covariance matrix
    - A body frame aligned to these principal axes
    - Transforms to normalize the shape to a "spherical cow" and then to unit scale
    - A bounding ellipsoid that encompasses all vertices

    The body frame is right-handed with:
    - Axis 0 (X): parallel to max eigenvalue eigenvector (longest extent)
    - Axis 2 (Z): parallel to min eigenvalue eigenvector (shortest extent)
    - Axis 1 (Y): parallel to remaining eigenvector
    - cross(axis0, axis1) = axis2 (right-handed)

    Args:
        vertices_world: (N, 3) array of vertex positions in world space

    Returns:
        BodyCentricMeasurements with all computed transforms and the bounding ellipsoid
    """
    vertices = np.asarray(vertices_world, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got shape {vertices.shape}")

    n_verts = vertices.shape[0]
    if n_verts == 0:
        raise ValueError("Cannot compute body-centric measurements for empty vertex array")

    # Step 1: Centroid
    centroid = vertices.mean(axis=0)

    # Step 2: Centered offsets and centered_T_world transform
    centered_T_world = make_translation_4x4(-centroid)
    centered_vertices = vertices - centroid

    # Step 3: Covariance matrix
    # Covariance = (1/N) * sum(v @ v.T) for centered v
    # Using numpy: cov = centered.T @ centered / N
    covariance = (centered_vertices.T @ centered_vertices) / n_verts

    # Step 4: Eigendecomposition via SVD
    # For a symmetric positive semi-definite matrix, SVD gives us eigendecomposition
    # U, S, Vh = svd(C) where S are singular values (= eigenvalues for symmetric)
    # and U columns are eigenvectors
    U, S, Vh = np.linalg.svd(covariance)

    # S is already sorted descending by numpy.linalg.svd
    eigenvalues = S
    eigenvectors = U  # Columns are eigenvectors

    # Step 5: Body frame rotation (body_T_centered)
    # We want:
    # - axis0 (body X) = max eigenvalue eigenvector
    # - axis2 (body Z) = min eigenvalue eigenvector
    # - axis1 (body Y) = remaining eigenvector
    # - Ensure right-handed: cross(axis0, axis1) = axis2

    axis0 = eigenvectors[:, 0]  # Max eigenvalue
    axis1 = eigenvectors[:, 1]  # Mid eigenvalue
    axis2 = eigenvectors[:, 2]  # Min eigenvalue

    # Ensure right-handedness: if cross(axis0, axis1) points opposite to axis2, flip axis2
    cross_01 = np.cross(axis0, axis1)
    if np.dot(cross_01, axis2) < 0:
        axis2 = -axis2

    # Build rotation matrix: columns are the body axes expressed in centered coords
    # This is centered_T_body (from body to centered), so we need the transpose for body_T_centered
    centered_R_body = np.column_stack([axis0, axis1, axis2])

    # To go from centered to body, we use the inverse (transpose for orthonormal)
    body_T_centered = centered_R_body.T

    # Step 6: SphericalCow transform (sphericalCow_T_body)
    # Scale each body axis by 1/sqrt(eigenvalue) to make roughly spherical
    # Eigenvalues represent variance along each axis, sqrt gives standard deviation
    # We want to normalize by the extent, so divide positions by sqrt(eigenvalue)

    # Handle near-zero eigenvalues to avoid division by zero
    eps = 1e-10
    safe_eigenvalues = np.maximum(eigenvalues, eps)
    scale_factors = 1.0 / np.sqrt(safe_eigenvalues)

    # The scale matrix (diagonal)
    sphericalCow_T_body = np.diag(scale_factors)

    # Step 7: Transform vertices to sphericalCow space and find max radius
    # sphericalCow = sphericalCow_T_body @ body_T_centered @ centered
    body_vertices = (body_T_centered @ centered_vertices.T).T  # (N, 3)
    sphericalCow_vertices = (sphericalCow_T_body @ body_vertices.T).T  # (N, 3)

    radii_sphericalCow = np.linalg.norm(sphericalCow_vertices, axis=1)
    max_body_radius = float(radii_sphericalCow.max()) if n_verts > 0 else 1.0

    # Avoid division by zero
    if max_body_radius < eps:
        max_body_radius = 1.0

    # Step 8: Unit spherical cow normalization
    unitSphericalCow_T_sphericalCow = np.eye(3) / max_body_radius

    # Step 9: Composed transform
    unitSphericalCow_T_centered = unitSphericalCow_T_sphericalCow @ sphericalCow_T_body @ body_T_centered

    # Step 10: Bounding ellipsoid
    # The ellipsoid has semi-axis lengths equal to max_body_radius * sqrt(eigenvalues)
    # (inverse of the sphericalCow scaling, times the unit sphere radius)
    ellipsoid_radii = max_body_radius * np.sqrt(safe_eigenvalues)

    # The body axes in world space are the eigenvectors (they're already in world-aligned centered space)
    body_axes_world = centered_R_body  # Columns are axes

    bounding_ellipsoid = BoundingEllipsoid(
        center_world=centroid.copy(),
        radii=ellipsoid_radii,
        body_axes_world=body_axes_world,
    )

    return BodyCentricMeasurements(
        centroid=centroid,
        centered_T_world=centered_T_world,
        covariance=covariance,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        body_T_centered=body_T_centered,
        sphericalCow_T_body=sphericalCow_T_body,
        unitSphericalCow_T_sphericalCow=unitSphericalCow_T_sphericalCow,
        unitSphericalCow_T_centered=unitSphericalCow_T_centered,
        max_body_radius=max_body_radius,
        bounding_ellipsoid=bounding_ellipsoid,
    )


# =============================================================================
# Additional Utility Functions
# =============================================================================


def transform_points_craig(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Transform points using a 4x4 matrix in Craig notation.

    Args:
        M: 4x4 homogeneous transform matrix (Craig notation)
        points: (N, 3) array of points

    Returns:
        (N, 3) array of transformed points
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]

    # Add homogeneous coordinate
    pts_h = np.concatenate([pts, np.ones((n, 1), dtype=np.float64)], axis=1)

    # Transform: M @ pts_h.T gives (4, N), we want (N, 3)
    result_h = (M @ pts_h.T).T

    return result_h[:, :3]


def compose_transforms(*matrices: np.ndarray) -> np.ndarray:
    """
    Compose multiple transform matrices in Craig notation (left to right).

    The result is: matrices[0] @ matrices[1] @ ... @ matrices[-1]

    This means the rightmost matrix is applied first, matching Craig notation
    where output_T_input chains as: C_T_A = C_T_B @ B_T_A

    Args:
        *matrices: Variable number of 4x4 (or 3x3) matrices

    Returns:
        Composed matrix
    """
    if len(matrices) == 0:
        raise ValueError("At least one matrix required")

    result = np.asarray(matrices[0], dtype=np.float64)
    for M in matrices[1:]:
        result = result @ np.asarray(M, dtype=np.float64)

    return result

