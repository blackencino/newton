# Fancy Dynamic Semantics - Implementation Progress

This log tracks implementation progress for the preprocessing pipeline described in
`README_FancySemanticsPlan.md`.

**USD File:** `C:\Users\chorvath\Downloads\20251223_iv060_flat_04\Collected_20251223_iv060_flat_04\20251223_iv060_flat_04.usd`  
**Camera:** `/World/TD060`  
**Frames:** 2920-3130 (211 frames)

---

## Step 0: Utils ✓

**Status:** Complete  
**Date:** 2025-12-23  
**Files:**
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
- `BoundingEllipsoid` dataclass with `distance_to_point()`, `contains_point()`
- `BodyCentricMeasurements` dataclass with centroid, covariance, SVD, body frame transforms
- `compute_body_centric_measurements()` - Main computation from world-space vertices

---

## Step 1: Pre-processing USD into semantic setup

**Status:** Partially Complete (functional but needs refinement)  
**Date:** 2025-12-23  
**Files:**
- `dynamic_segmentation/preprocess.py` - Main preprocessing module
- `dynamic_segmentation/test_preprocess.py` - Test on real USD (slow ~30s)
- `run_preprocessing.py` - Script to run full pipeline

### Implemented:

1. **DiageticMetadata** ✓ - Lightweight per-mesh metadata
2. **parse_diagetic_metadata()** ✓ - Parses USD, returns 2449 diegetics
3. **DiageticGroupMetadata** ✓ - Grouped diegetics with body measurements
4. **group_diagetics_by_color_and_ancestor()** ✓ - Groups by (color, colored_root), creates 68 groups
5. **CameraCurve** ✓ - Pre-baked camera animation
6. **extract_camera_curve()** ✓ - Extracts camera positions over frame range
7. **compute_path_danger()** ✓ - Distance from camera to ellipsoid
8. **update_groups_with_path_danger()** ✓ - Batch update all groups
9. **GroupCategory** ✓ - Enum: GROUND_TERRAIN, UNSAFE, SAFE
10. **categorize_group()** ✓ - Pattern matching on paths
11. **update_groups_with_categories()** ✓ - Batch categorization
12. **save_preprocessing_cache()** / **load_preprocessing_cache()** ✓ - NPZ serialization
13. **run_preprocessing_pipeline()** ✓ - High-level pipeline function

### Shot-Specific Scene Structure (IV060):

```
/World/
├── StarWarsSet_01/assembly/          # Main set (1894 meshes, 18 color-groups)
│   ├── Terrain_01/geo/               # Ground terrain (2 meshes)
│   ├── ClothModuleX_YY/              # Fabric/cloth elements
│   ├── CrateX_YY/                    # Crates and boxes
│   ├── PotteryX_YY/                  # Pottery objects
│   ├── TentMainA_YY/                 # Tent structures
│   └── ... many more props
├── HangingLanternA_01/               # Lantern type A (67 meshes each)
├── HangingLanternA_02/
├── HangingLanternA_03/
├── HangingLanternB_01/               # Lantern type B (6 meshes each)
├── HangingLanternC_01/               # Lantern type C (33 meshes each)
├── HangingLanternChainA_01/          # Chain links (1-3 meshes each)
├── SimGravelLrg_01/                  # Gravel simulation (not rendered)
├── BDXDroid_01/                      # Droid character
├── TD060                             # Camera
└── ... other props
```

### Categorization Heuristics (Shot-Specific):

**GROUND_TERRAIN:** Paths containing `/Terrain_01/` in member_paths  
**SAFE:** Paths containing `hanginglantern` (any variant)  
**UNSAFE:** Everything else (default)

Note: `SimGravelLrg_01` is a point simulation cache and doesn't render in this pass.

### Current Test Results:
- 2449 diegetics parsed (skipping proxy geometry)
- 68 groups created (by color + ancestor subtree)
- 26 SAFE groups (all lanterns/chains correctly identified)
- 1 GROUND_TERRAIN group (terrain meshes in StarWarsSet_01)
- 41 UNSAFE groups (props, crates, structures)

### Known Issues / Next Steps:

1. **Grouping granularity**: Multiple assets within `StarWarsSet_01` share the same 
   `objectid_color`, causing them to be grouped together by color. The terrain meshes
   end up in a large group with other objects that share the same color. The current
   categorization uses member_paths to detect terrain within these mixed groups.

2. **Gravel not handled**: `SimGravelLrg_01` is a simulation point cache that doesn't
   produce rendered geometry in our raytracer pass - it can be ignored.

3. **Tests are slow**: The test script parses the real 12GB USD file and takes ~30s.
   Consider caching the preprocessing results to disk for faster iteration.

---

## Cache Size Test Results (2025-12-24)

Tested saving preprocessing data to `D:\ces26_data\td060\v04\`:

### Geometry Statistics:
- **36,093,730 vertices** (36M)
- **70,301,071 triangles** (70M) 
- **Estimated uncompressed: 1.19 GB**

### Cache File Sizes:
| Cache Type | Size | Save Time | Notes |
|------------|------|-----------|-------|
| Metadata-only | **60 KB** | instant | Perfect for fast iteration |
| With geometry | **633 MB** | **252s** | Too slow, not practical |

### Verdict: **Do NOT store geometry in cache**

The 4+ minute save time makes this impractical. Parsing the USD (67s) is faster than 
saving/loading the geometry cache. The metadata-only cache (60 KB) is sufficient for
most preprocessing needs.

### Grouping Analysis:

The `StarWarsSet_01_0` group (categorized as GROUND_TERRAIN) contains:
- **444 total meshes**, but only **3 are terrain**
- Breakdown: 155 tents, 7 cloth, 7 pottery, 2 crates, 270 other
- All share `objectid_color` RGB(0.184, 0.153, 0.604)
- This same color is also used by `HangingLanternA_03`!

The `objectid_color` primvar isn't unique per semantic object - many different asset
types share the same color. The grouping is correct per the plan (color + ancestor),
but semantic meaning must come from path analysis within groups.

### Path Danger Analysis:

Camera passes through **11 group ellipsoids** during the shot:
- Expected for a camera moving through a detailed set
- Largest: `Sphere` group with -421 path danger (camera very deep inside)
- This is normal behavior for interior shots

### To Continue This Work:

```bash
# Run from newton/examples/ces26 directory
cd newton/examples/ces26

# Run utils tests (fast)
uv run python -m dynamic_segmentation.test_utils

# Run preprocessing test on real USD (slow, ~30s)
uv run python -m dynamic_segmentation.test_preprocess

# Run full preprocessing and test save/load
uv run python run_preprocess_test.py

# Examine groups and cache contents
uv run python run_preprocess_test2.py
```

---

