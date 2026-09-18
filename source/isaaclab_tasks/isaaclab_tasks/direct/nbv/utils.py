import numpy as np
import os
import gc
import omni.usd
import open3d as o3d
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Sdf, UsdLux
from pxr import Usd, UsdGeom, Gf, UsdPhysics
from pxr import Sdf

def get_random_usd(stage, usd_parent, prim_path):
    models = os.listdir(usd_parent)
    model = np.random.choice(models)

    model_path = os.path.join(usd_parent, model, model + ".usd")

    # Create root Xform
    model_prim = stage.DefinePrim(prim_path, "Xform")

    # Reference GSO USD
    model_prim.GetReferences().AddReference(model_path)

    # Create scale op
    xform = UsdGeom.Xformable(model_prim)

    scale_op = None

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break

    if scale_op is None:
        scale_op = xform.AddScaleOp()

    # Make sure there is no previous scale
    scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # Compute bounding box in LOCAL space
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )

    bbox = bbox_cache.ComputeLocalBound(model_prim)
    bbox_range = bbox.ComputeAlignedRange()

    min_bound = bbox_range.GetMin()
    max_bound = bbox_range.GetMax()

    dimensions = max_bound - min_bound
    largest_dimension = max(dimensions)

    # Normalize largest dimension to 0.5 m
    scale = 0.5 / float(largest_dimension)

    if scale > 1.0:
        scale = np.floor(scale)

    scale_op.Set(Gf.Vec3f(float(scale), float(scale), float(scale)))

    # Collision on meshes
    for prim in Usd.PrimRange(model_prim):
        if prim.IsA(UsdGeom.Mesh):
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr(True)

            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

            # Static object -> triangle mesh collision is okay
            mesh_collision_api.CreateApproximationAttr("none")

    return model_prim


def replace_random_usd(stage, usd_parent, model_prim):

    models = os.listdir(usd_parent)
    model = np.random.choice(models)

    model_path = os.path.join(
        usd_parent,
        model,
        model + ".usd",
    )

    # Get existing scale operation
    xform = UsdGeom.Xformable(model_prim)

    scale_op = None

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break

    if scale_op is None:
        scale_op = xform.AddScaleOp()

    # --------------------------------------------------
    # IMPORTANT:
    # Remove previous object's scale before measuring
    # --------------------------------------------------
    scale_op.Set(
        Gf.Vec3f(
            1.0,
            1.0,
            1.0,
        )
    )

    # --------------------------------------------------
    # Replace reference
    # --------------------------------------------------
    model_prim.GetReferences().ClearReferences()
    model_prim.GetReferences().AddReference(model_path)

    # --------------------------------------------------
    # Compute bounding box of NEW object
    # --------------------------------------------------
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )

    bbox = bbox_cache.ComputeLocalBound(model_prim)
    bbox_range = bbox.ComputeAlignedRange()

    min_bound = bbox_range.GetMin()
    max_bound = bbox_range.GetMax()

    dimensions = max_bound - min_bound
    largest_dimension = max(dimensions)

    # --------------------------------------------------
    # Normalize largest dimension to 0.5 m
    # --------------------------------------------------
    scale = 0.5 / float(largest_dimension)

    # Don't enlarge very small objects
    if scale > 1.0:
        scale = np.floor(scale)

    # Apply new scale
    scale_op.Set(
        Gf.Vec3f(
            float(scale),
            float(scale),
            float(scale),
        )
    )

    # Re-apply collision API to new meshes
    for prim in Usd.PrimRange(model_prim):

        if prim.IsA(UsdGeom.Mesh):

            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            collision_api.CreateCollisionEnabledAttr(True)

            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.CreateApproximationAttr("none")

    return model_prim


def get_camera(stage):
    d555_path = (
            f"{ISAAC_NUCLEUS_DIR}/Sensors/RealSense/D555/rsd555.usd"
        )

    ee_path = "/World/envs/env_0/UR10e/wrist_3_link"
    camera_path = ee_path + "/D555"

    camera_mount = stage.DefinePrim(camera_path, "Xform")
    camera_mount.GetReferences().AddReference(d555_path)

    xform = UsdGeom.Xformable(camera_mount)

    translate_op = None
    rotate_op = None

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp()

    if rotate_op is None:
        rotate_op = xform.AddRotateXYZOp()

    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.04))
    rotate_op.Set(Gf.Vec3f(0.0, 90.0, 0.0))

    return camera_mount