# Fancy Dynamic Semantics - Implementation Progress

This log tracks implementation progress for the preprocessing pipeline described in
`README_FancySemanticsPlan.md`.

---

## Step 0: Utils ✓

**Status:** Complete  
**Date:** 2025-12-23  
**Files Created:**
- `dynamic_segmentation/utils.py` - Core utilities
- `dynamic_segmentation/__init__.py` - Package init with exports
- `dynamic_segmentation/test_utils.py` - Test script

### Implemented:

**Transform Conversion (USD row-vector → Craig column-vector):**
- `usd_to_craig_3x3()` / `usd_to_craig_4x4()` - Matrix transpose conversion
- `make_translation_4x4()` / `make_rotation_4x4()` - Homogeneous matrix builders
- `transform_points_craig()` - Apply transforms to point arrays
- `compose_transforms()` - Chain transforms left-to-right

**Body-Centric Measurements:**
- `BoundingEllipsoid` dataclass:
  - `center_world`, `radii`, `body_axes_world`
  - `distance_to_point()` - Approximate distance using spherical cow normalization
  - `contains_point()` - Membership test
  
- `BodyCentricMeasurements` dataclass:
  - `centroid` - Mean vertex position
  - `centered_T_world` - Translation to centered space
  - `covariance` - 3×3 covariance matrix
  - `eigenvalues` / `eigenvectors` - SVD decomposition (sorted descending)
  - `body_T_centered` - Rotation to body frame (right-handed, axis0=max, axis2=min)
  - `sphericalCow_T_body` - Scale to roughly spherical
  - `unitSphericalCow_T_sphericalCow` - Scale to unit radius
  - `unitSphericalCow_T_centered` - Composed transform
  - `max_body_radius` - Radius before unit normalization
  - `bounding_ellipsoid` - World-space ellipsoid

- `compute_body_centric_measurements()` - Main computation from world-space vertices

### Tests:
All tests pass:
- USD to Craig conversion (transpose verification)
- Point transformation (translation, rotation)
- Transform composition (order verification)
- Sphere point cloud (isotropic eigenvalues)
- Elongated shape (correct axis alignment, eigenvalue ratios)
- Ellipsoid distance calculations (inside/surface/outside)

---

## Step 1: Pre-processing USD into semantic setup

**Status:** Not started

### Pending Tasks:
1. `DiageticMetadata` dataclass and parse function
2. `DiageticGroupMetadata` and grouping logic (by scene graph + objectid_color)
3. `CameraCurve` extraction for full frame range (2920-3130)
4. Path danger computation (camera to ellipsoid distance curve)
5. Group categorization (Ground/Terrain, Unsafe, Safe)
6. NPZ cache save/load for preprocessing results

---

