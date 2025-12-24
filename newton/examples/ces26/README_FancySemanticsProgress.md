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

## Step 1: Pre-processing USD into semantic setup ✓

**Status:** Complete  
**Date:** 2025-12-24  
**Files:**
- `dynamic_segmentation/preprocess.py` - Main preprocessing module
- `run_preprocess_v2.py` - Script to run updated pipeline
- Cache: `D:\ces26_data\td060\v04\preprocess_v2.npz` (65 KB)

### Grouping Strategy: "Parent of geo"

The flattened USD follows a consistent pattern:
```
/World/{AssetName}/geo/{meshes...}
/World/StarWarsSet_01/assembly/{AssetName}/geo/{meshes...}
```

The **asset root** is the PARENT of the `geo` prim. This gives semantically-correct
groups where each lantern, crate, terrain, etc. is its own group.

### Implemented:

1. **DiageticMetadata** ✓ - Lightweight per-mesh metadata
2. **parse_diagetic_metadata()** ✓ - Parses USD, returns 2442 diegetics
3. **DiageticGroupMetadata** ✓ - Grouped diegetics with body measurements
4. **group_diagetics_by_asset_root()** ✓ - Groups by asset root, creates 132 groups
5. **_find_asset_root()** ✓ - Finds parent of 'geo' prim
6. **_is_excluded_path()** ✓ - Filters out PaintTool and Sphere
7. **CameraCurve** ✓ - Pre-baked camera animation
8. **extract_camera_curve()** ✓ - Extracts camera positions over frame range
9. **compute_path_danger()** ✓ - Distance from camera to ellipsoid
10. **update_groups_with_path_danger()** ✓ - Batch update all groups
11. **GroupCategory** ✓ - Enum: GROUND_TERRAIN, UNSAFE, SAFE
12. **categorize_group()** ✓ - Pattern matching on paths
13. **update_groups_with_categories()** ✓ - Batch categorization
14. **save_preprocessing_cache()** / **load_preprocessing_cache()** ✓ - NPZ serialization
15. **run_preprocessing_pipeline()** ✓ - High-level pipeline function

### Shot-Specific Scene Structure (IV060):

```
/World/
├── StarWarsSet_01/assembly/          # Main set (90 assets inside)
│   ├── Terrain_01/geo/               # Ground terrain (3 meshes)
│   ├── TentMainA_06/geo/             # Tent (155 meshes)
│   ├── GearPartsShelfA_01/geo/       # Shelf (75 meshes)
│   ├── CommsA_03/geo/                # Comms (55 meshes)
│   └── ... 86 more assets
├── HangingLanternA/geo/              # Lantern type A (67 meshes)
├── HangingLanternA_01/geo/           # Lantern type A instance (67 meshes)
├── HangingLanternA_02/geo/           # Lantern type A instance (67 meshes)
├── HangingLanternB/geo/              # Lantern type B (6 meshes)
├── HangingLanternC/geo/              # Lantern type C (33 meshes)
├── HangingLanternChain*/geo/         # Chain links (1-3 meshes each)
├── CrateA/geo/                       # Crate (11 meshes)
├── GearA/geo/                        # Gear (1 mesh)
├── TD060                             # Camera (not rendered)
└── ... 42 top-level assets total
```

### Grouping Strategy:

**Asset Root = Parent of 'geo' prim**

This correctly groups each asset (lantern, crate, tent) as a single semantic unit,
regardless of how many sub-meshes it contains.

### Categorization Heuristics (Shot-Specific):

**GROUND_TERRAIN:** Asset root contains `Terrain`  
**SAFE:** Asset root contains `hanginglantern` (any variant)  
**UNSAFE:** Everything else (default)

### Excluded Paths:
- Anything not under `/World/` - non-scene content (look-dev, materials, utilities)
- `/World/PaintTool/*` - Point instancer scatter (not renderable)
- `/World/Sphere` - Debug geometry

### Current Test Results (V2):
- 2435 diegetics parsed
- 131 asset groups created (by hierarchy)
- 26 SAFE groups (all lanterns/chains correctly identified)
- 1 GROUND_TERRAIN group (Terrain_01 with 3 meshes)
- 104 UNSAFE groups (props, crates, structures)

### Validation: objectid_color Consistency ✓

Verified that all meshes within each group share the same `objectid_color`:
- **131/131 groups** have uniform objectid_color values
- Zero groups have mixed colors

This confirms the hierarchy-based grouping doesn't accidentally combine meshes with
different artist-assigned colors.

### Resolved Issues:

1. **Grouping granularity** ✓ - Fixed by switching from color-based to hierarchy-based
   grouping. Now each asset (lantern, crate, terrain) is its own group.

2. **Excluded geometry** ✓ - PaintTool and Sphere are now excluded from processing.

### Notes:

- Preprocessing takes ~70s (mostly USD parsing + body measurements)
- Cache file is only 65 KB (metadata only, no geometry)
- Geometry cache (633 MB) is NOT recommended - re-parsing USD is faster

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

# Run V2 preprocessing (hierarchy-based, ~70s)
uv run python run_preprocess_v2.py

# Analyze USD hierarchy and asset structure
uv run python analyze_assets.py
uv run python dump_hierarchy.py
```

### Cache Files:
- `D:\ces26_data\td060\v04\preprocess_v2.npz` - Latest preprocessing cache (65 KB)

---

## Abandoned Approaches

These approaches were implemented and tested but abandoned in favor of simpler solutions.

### Color-Based Grouping (Removed 2025-12-23)

**Original approach:** Group meshes by `(objectid_color, colored_root_ancestor)` - meshes
with the same color AND a common ancestor with that color would be grouped together.

**Functions removed:**
- `group_diagetics_by_color_and_ancestor()` - Main grouping function
- `_find_colored_root_for_mesh()` - Walk up hierarchy to find colored ancestor
- `_get_objectid_color_at_path()` - Check color at a specific prim path
- `_colors_match()` - Compare colors with tolerance
- `_color_to_key()` - Convert color to hashable string

**Why abandoned:**
1. The `objectid_color` primvar is NOT unique per semantic object - many different 
   assets share the same color (e.g., `StarWarsSet_01` and `HangingLanternA_03` both
   have RGB(0.184, 0.153, 0.604))
2. Required traversing the USD hierarchy to check ancestor colors, which is slow
3. The flattened USD already has a clean hierarchy pattern (`/World/{Asset}/geo/`)
   that makes hierarchy-based grouping trivial and fast

**Better solution:** `group_diagetics_by_asset_root()` - simply use the parent of the
`geo` prim as the asset root. This is fast (no USD queries) and semantically correct.

---

