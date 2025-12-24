# Fancy Dynamic Semantics Plan.

We have a USD scene consisting of static geometry (so far) which is rendered 
using a programmatic renderer, given a number of different viewing types. We parse
the USD file, which is quite large (about 12 Gb) and look for what we call
"Diagetics", which are objects that exist within the world being visualized, and 
separately lights and cameras. One of the outputs we render over about 290 frames
is called "semantic", and it is a flat-colored rgb value per separate, semantic "object".

There's an artist-generated primvar on all of the diagetic renderable meshes which 
indicates an artist-assigned color to use. We'll be using this color for one version
of the semantic coloring. The name of that primvar is: "objectid_color". 

We're going to attempt to implement a somewhat complex quasi-simulation of
semantic segmentation colors for groups of diagetic elements in this scene. Because the 
USD file is so big, we want to first create a utility to parse through the USD and
extract metadata about the diagetic elements.

## Preamble:
All code is functional, uses immutable value semantics. Rely on the reference-counted
nature of underlying tensors and arrays, don't sweat guaranteeing that the underlying
arrays aren't accidentally clobbered.

If a pass requires intermediate results, create immutable intermediate dataclasses 
as helpers, or just use tuples. For example - when we gather diagetic group metadata,
the names of each are not necessarily unique at first before a uniqification pass
is done. Don't do this by mutating the struct, do it by sweeping over them and either
producing new ones, or by having a partial struct.

Use Craig notation for transformations. Transpose USD transformations as needed so that
transformation is `Vnew = M * V` (column vectors). Transformations are named so that
the output frame is on the left, the input frame on the right, so compositions are 
visually debuggable. For example:

```
# Get screen-space points from object-space points
pt_screen = screen_T_camera @ camera_T_world @ world_T_object @ pt_object
```

## Step 0: Utils

We need some utils before we even get started. These should go into the file: 
`dynamic_segmentation/utils.py`. 

1. USD transformations (row-vector, `Vnew = Vold * M`) into craig notation col vectors,
`Vnew = M * Vold`

2. Body-centric measurements
Given a set of vertices created by concatenating together all the verts from multiple 
objects (assuming world space coordinates), find the following:
  2. Centroid (average of all verts)
  2. Centered offsets (positions minus centroid) craig: centered_T_world
  2. Covariance of centered offsets
  2. Eigenvalues, Eigenvectors (use SVD) of covariance of centered offsets
  2. rotation from centered offsets to body frame, which is a right-handed coordinate frame such that
     axis 0 is parallel to max eigenvalue eigenvector, axis 2 is parallel to the min eignvalue eignvector,
     axis 1 is parallel to remaining eigenvector, and cross(axis0, axis1) = axis2
     craig notation: body_T_centered
  2. Create a coordinate frame which divides each body axis by the eigenvalue of that axis, so that
     the resulting coordinate frame is non-uniformly scaled to be sphere-like
     craig notation: sphericalCow_T_body
  2. Use points in sphericalCow space to determine maximum radius, which is then used to normalize the
     sphericalCow space to produce unitSphericalCow.
  2. Save the composed complete 3x3 matrix unitSphericalCow_T_centered, along with centered_T_world,
     and body_T_centered.
  2. Use all of the above to formulate a representation of the bounding ellipsoid that is aligned
     to the body coordinate frame and encompasses all the points. This bounding ellipsoid should
     be able to be understood in world space.

## Step 1: Pre-processing USD into semantic setup.

This work will go into the file: `dynamic_segmentation/preprocess.py`

1. Use the tools already built into the `ces26_utils.py` module to parse the USD, looking
for diagetic elements. Group into a single semantic diagetic group each collection of 
diagetics that are part of the same assembly (have same scene graph grouping) and share
the same objectid_color attribute. This should obey exactly the same rules as before - 
not using proxy geometry, observing the visibility attributes, ignoring lights and cameras
and non-diagetic prims.

2. Implement DiageticMetadata, which stores all of the information gathered about 
a Diagetic except for the actual (heavy) geometry verts and tris. We will want to parse the
USD to generate this data, so in addition do defining the frozen dataclass for it, write the
function which returns these as a `list[DiageticMetadata]`
  2. Verts in world space (these won't be saved, but are used for group info later)
  2. Local to world transform as computed by the xform cache at the start frame (2920)
  2. objectid color (from primvar)
  2. full USD path

3. Implement DiageticGroupMetadata - once we've got the diagetic metadata list, we need to
loop over it and group together diagetics which are in the same scene graph subtree that 
shares the exact same objectid color. If two diagetics have the same objectid color, but they
don't have a common scene graph ancestor that shares that same objectid color, they're 
different groups. The DiageticGroupMetadata is very similar to the DiageticMetadata, it
should contain as below, this function returns `dict[string, DiageticGroupMetatdata]`
  3. `list[DiageticMetadata]`: All the diagetics (metadata) in this group
  3. Bounding box in world space
  3. Use the body-centric tools described in the preamble to compute the body-centric
     measurements of this whole diagetic group. This will require concatenating the 
     world space vertex positions of all the included diagetic elements
  3. objectid color (same for all children in this group)
  3. full USD path for top-level common ancestor prim
  3. shortest unique name for this DiageticGroupMetatdata

4. Extract camera with full animation. Right now we extract a camera world-space eye position
at a given time code, along with a 3x3 rotation matrix. We also extract camera intrinsics. 
We later use this extrinsic+intrinsic info to create world-space rays to trace through the
scene with. Pre-filter the USD through the whole given frame range (still 2920 to 3130) and
bake the whole camera curve into a form we can extract later. The world space eye position
of this camera is a curve through the world which we'll be computing proximity to.

5. Bake into each diagetic group metadata a time-sampled curve which contains the distance from the
world space eye position of the camera to the closest point on the enclosing ellipsoid of the 
group. Then find the minimum value of this whole curve, so that each diagetic group ends up 
with a single "path danger" value.

6. Categorize each diagetic group as one of the following classes, based solely on some heuristics
of objectid color and naming that I'll explain below. (Not based on the path danger we computed in 5)
  6. Ground/Terrain
  6. Unsafe object (collisions must be considered)
  6. Safe object (collisions possible but not important)
For this specific shot, the classification heuristic is as follows. First, try to find the ground 
and terrain objects via name inspection. Do actual parsing of the USD file and look for likely 
candidates, but only of diagetic groups. Look at the element names or the group names or whatever
works. Then: any group who has hanging lantern or hanging chain (or any variant thereof) in the
name is a safe object. Any group that isn't safe and isn't terrain, is unsafe.

These 6 parts represent the end of pre-processing. Nothing in them should change unless the USD
changes. If it is possible to save the actual world-space vertices (concatenated) of the
groups, and it isn't prohibitively huge on disk, this would spare us having to re-parse the USD. 
Test this, and use numpy to save an npz of either just the DiageticMetadata list and DiageticGroupMetadata dict
and baked camera - or, if it isn't prohibitive, include the world space concatenated vertex positions
and triangles (with appropriate index offsets to compensate for vertex concatenation) per group.