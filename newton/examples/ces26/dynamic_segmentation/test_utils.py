# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""
Test script for dynamic_segmentation/utils.py

Verifies:
1. USD to Craig transform conversion (transpose)
2. Body-centric measurements on simple test shapes
3. Bounding ellipsoid distance computations

Run with: uv run python newton/examples/ces26/dynamic_segmentation/test_utils.py
"""

import numpy as np

from utils import (
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


def test_usd_to_craig_conversion():
    """Test that USD to Craig conversion is a transpose."""
    print("Testing USD to Craig conversion...")

    # Create a known matrix
    usd_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

    craig_mat = usd_to_craig_3x3(usd_mat)

    # Should be transposed
    expected = usd_mat.T
    assert np.allclose(craig_mat, expected), f"Expected transpose, got:\n{craig_mat}"

    # Test 4x4 version
    usd_4x4 = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [10, 20, 30, 1]], dtype=np.float64
    )

    craig_4x4 = usd_to_craig_4x4(usd_4x4)
    expected_4x4 = usd_4x4.T
    assert np.allclose(craig_4x4, expected_4x4), f"Expected transpose, got:\n{craig_4x4}"

    print("  PASS: USD to Craig conversion works correctly")


def test_transform_points():
    """Test point transformation in Craig notation."""
    print("Testing point transformation...")

    # Translation
    T = make_translation_4x4(np.array([10, 20, 30]))
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    result = transform_points_craig(T, points)
    expected = points + np.array([10, 20, 30])
    assert np.allclose(result, expected), f"Translation failed:\n{result}"

    # Rotation (90 degrees around Z)
    c, s = 0, 1  # cos(90), sin(90)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    R_4x4 = make_rotation_4x4(R)

    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = transform_points_craig(R_4x4, points)
    expected = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(result, expected), f"Rotation failed:\n{result}\nExpected:\n{expected}"

    print("  PASS: Point transformation works correctly")


def test_compose_transforms():
    """Test transform composition."""
    print("Testing transform composition...")

    # Translate then rotate = different result than rotate then translate
    T = make_translation_4x4(np.array([1, 0, 0]))
    c, s = 0, 1
    R = make_rotation_4x4(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))

    # R @ T: first translate, then rotate
    RT = compose_transforms(R, T)
    p = np.array([[0, 0, 0]])
    result = transform_points_craig(RT, p)
    # Translate [0,0,0] by [1,0,0] -> [1,0,0], then rotate 90deg -> [0,1,0]
    expected = np.array([[0, 1, 0]])
    assert np.allclose(result, expected), f"Compose RT failed:\n{result}\nExpected:\n{expected}"

    # T @ R: first rotate, then translate
    TR = compose_transforms(T, R)
    result = transform_points_craig(TR, p)
    # Rotate [0,0,0] -> [0,0,0], then translate -> [1,0,0]
    expected = np.array([[1, 0, 0]])
    assert np.allclose(result, expected), f"Compose TR failed:\n{result}\nExpected:\n{expected}"

    print("  PASS: Transform composition works correctly")


def test_body_centric_sphere():
    """Test body-centric measurements on a unit sphere point cloud."""
    print("Testing body-centric measurements on sphere...")

    # Generate points on a unit sphere
    np.random.seed(42)
    n = 1000
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(np.random.uniform(-1, 1, n))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    sphere_points = np.column_stack([x, y, z])

    measurements = compute_body_centric_measurements(sphere_points)

    # Centroid should be near origin
    assert np.allclose(measurements.centroid, [0, 0, 0], atol=0.1), (
        f"Sphere centroid should be near origin, got: {measurements.centroid}"
    )

    # Eigenvalues should be roughly equal (isotropic)
    ev = measurements.eigenvalues
    ratio = ev.max() / ev.min()
    assert ratio < 1.5, f"Sphere eigenvalues should be similar, got ratio: {ratio}"

    # Bounding ellipsoid should have roughly equal radii
    ellipsoid = measurements.bounding_ellipsoid
    radii_ratio = ellipsoid.radii.max() / ellipsoid.radii.min()
    assert radii_ratio < 1.5, f"Ellipsoid radii should be similar, got ratio: {radii_ratio}"

    # All sphere points should be inside or on the ellipsoid (allowing numerical tolerance)
    for p in sphere_points[:10]:  # Test a sample
        dist = ellipsoid.distance_to_point(p)
        assert dist <= 0.1, f"Sphere point should be inside ellipsoid, dist: {dist}"

    print(f"  Centroid: {measurements.centroid}")
    print(f"  Eigenvalues: {measurements.eigenvalues}")
    print(f"  Ellipsoid radii: {ellipsoid.radii}")
    print("  PASS: Sphere body-centric measurements are isotropic")


def test_body_centric_elongated():
    """Test body-centric measurements on an elongated shape."""
    print("Testing body-centric measurements on elongated shape...")

    # Create an elongated point cloud (box: 10 x 2 x 1)
    np.random.seed(42)
    n = 1000
    points = np.random.uniform(-0.5, 0.5, (n, 3))
    points[:, 0] *= 10  # X is longest
    points[:, 1] *= 2  # Y is medium
    points[:, 2] *= 1  # Z is shortest

    # Shift to non-origin
    offset = np.array([100, 200, 300])
    points_shifted = points + offset

    measurements = compute_body_centric_measurements(points_shifted)

    # Centroid should be at offset
    assert np.allclose(measurements.centroid, offset, atol=0.5), (
        f"Centroid should be at offset, got: {measurements.centroid}"
    )

    # Eigenvalues should reflect the 10:2:1 ratio (as variance)
    ev = measurements.eigenvalues
    # Variance is proportional to (length/2)^2 / 3 for uniform box
    # Ratios should be roughly 100:4:1 for lengths 10:2:1
    ev_normalized = ev / ev.min()
    print(f"  Eigenvalue ratios (normalized to min): {ev_normalized}")

    # Max eigenvalue should be much larger than min
    assert ev_normalized[0] > 10, "Max eigenvalue should be much larger than min"
    assert ev_normalized[1] > 2, "Mid eigenvalue should be larger than min"

    # Body frame axis 0 should be roughly aligned with X
    body_axes = measurements.eigenvectors
    axis0 = body_axes[:, 0]
    x_alignment = abs(np.dot(axis0, [1, 0, 0]))
    assert x_alignment > 0.9, f"Axis 0 should align with X, got alignment: {x_alignment}"

    # Bounding ellipsoid should contain all points
    ellipsoid = measurements.bounding_ellipsoid
    for p in points_shifted[:20]:
        inside = ellipsoid.contains_point(p)
        assert inside, f"Point should be inside ellipsoid: {p}"

    # Test distance: point well outside should have positive distance
    far_point = offset + np.array([100, 0, 0])
    dist = ellipsoid.distance_to_point(far_point)
    assert dist > 0, f"Far point should have positive distance, got: {dist}"

    print(f"  Centroid: {measurements.centroid}")
    print(f"  Ellipsoid radii: {ellipsoid.radii}")
    print("  PASS: Elongated shape body-centric measurements are correct")


def test_bounding_ellipsoid_distance():
    """Test ellipsoid distance calculations."""
    print("Testing bounding ellipsoid distance calculations...")

    # Create a simple axis-aligned ellipsoid at origin
    ellipsoid = BoundingEllipsoid(
        center_world=np.array([0, 0, 0]),
        radii=np.array([10, 5, 2]),  # Semi-axes: 10, 5, 2
        body_axes_world=np.eye(3),  # Aligned with world axes
    )

    # Point at origin should be inside (negative distance)
    dist_origin = ellipsoid.distance_to_point([0, 0, 0])
    assert dist_origin < 0, f"Origin should be inside, dist: {dist_origin}"

    # Point on the X axis at x=10 should be on surface (distance ~0)
    dist_surface = ellipsoid.distance_to_point([10, 0, 0])
    assert abs(dist_surface) < 0.5, f"Surface point should have near-zero distance, got: {dist_surface}"

    # Point far along X axis should have positive distance
    dist_far = ellipsoid.distance_to_point([20, 0, 0])
    assert dist_far > 0, f"Far point should have positive distance, got: {dist_far}"

    print(f"  Distance at origin: {dist_origin:.3f}")
    print(f"  Distance at surface: {dist_surface:.3f}")
    print(f"  Distance far away: {dist_far:.3f}")
    print("  PASS: Ellipsoid distance calculations work correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing dynamic_segmentation/utils.py")
    print("=" * 60)
    print()

    test_usd_to_craig_conversion()
    print()
    test_transform_points()
    print()
    test_compose_transforms()
    print()
    test_body_centric_sphere()
    print()
    test_body_centric_elongated()
    print()
    test_bounding_ellipsoid_distance()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

