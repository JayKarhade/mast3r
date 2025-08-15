#!/usr/bin/env python3
import argparse
import os
import numpy as np
import trimesh
import copy
import tempfile
import torch
from scipy.spatial.transform import Rotation

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs

try:
    from mast3r.retrieval.processor import Retriever
    has_retrieval = True
except Exception:
    has_retrieval = False

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes


class SparseGAState:
    """State container for sparse global alignment results"""
    def __init__(self, sparse_ga, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name


def convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                               cam_color=None, as_pointcloud=False, transparent_cams=False):
    """Convert scene output to GLB file"""
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # Full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # Add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    print(f'Exporting 3D scene to {outfile}')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(scene_state, min_conf_thr=2, as_pointcloud=False, 
                           mask_sky=False, clean_depth=False, transparent_cams=False, 
                           cam_size=0.05, TSDF_thresh=0):
    """Extract 3D model (glb file) from a reconstructed scene"""
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # Get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    
    return convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, 
                                      as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, 
                                      cam_size=cam_size)


def get_reconstructed_scene(image_dir, output_dir, model, device='cuda', image_size=512,
                           retrieval_model=None, optim_level='refine+depth', 
                           lr1=0.07, niter1=300, lr2=0.01, niter2=300,
                           min_conf_thr=1.5, matching_conf_thr=0.0,
                           as_pointcloud=True, mask_sky=False, clean_depth=True,
                           transparent_cams=False, cam_size=0.2,
                           scenegraph_type='complete', winsize=1, win_cyclic=False, 
                           refid=0, TSDF_thresh=0.0, shared_intrinsics=False):
    """
    Main function to reconstruct scene from images using MASt3R
    
    Args:
        image_dir: Path to directory containing images
        output_dir: Directory to save outputs
        model: Loaded MASt3R model
        device: Device to run on ('cuda' or 'cpu')
        image_size: Size to resize images to
        retrieval_model: Optional retrieval model for image matching
        optim_level: Optimization level ('coarse', 'refine', 'refine+depth')
        lr1: Learning rate for coarse alignment
        niter1: Number of iterations for coarse alignment
        lr2: Learning rate for refinement
        niter2: Number of iterations for refinement
        min_conf_thr: Minimum confidence threshold
        matching_conf_thr: Matching confidence threshold
        as_pointcloud: Export as pointcloud instead of mesh
        mask_sky: Apply sky masking
        clean_depth: Clean up depth maps
        transparent_cams: Use transparent cameras
        cam_size: Camera size in visualization
        scenegraph_type: Type of scene graph ('complete', 'swin', 'logwin', 'oneref', 'retrieval')
        winsize: Window size for scene graph
        win_cyclic: Whether to use cyclic sequence
        refid: Reference ID for scene graph
        TSDF_thresh: TSDF threshold for post-processing
        shared_intrinsics: Whether to use shared intrinsics
    
    Returns:
        scene_state: Reconstructed scene state object
        outfile: Path to output 3D model file
    """
    # Load images
    if os.path.isdir(image_dir):
        filelist = [os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir))]
    else:
        filelist = [image_dir]  # Single image or video file
    
    imgs = load_images(filelist, size=image_size, verbose=True)
    
    # Handle single image case
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    # Set up scene graph parameters
    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    # Handle retrieval-based matching
    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        if not has_retrieval:
            raise ValueError("Retrieval functionality not available")
        if retrieval_model is None:
            raise ValueError("Retrieval model required for retrieval scene graph")
        
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)
        
        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    # Create pairs
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, 
                      symmetrize=True, sim_mat=sim_matrix)
    
    # Adjust iterations based on optimization level
    if optim_level == 'coarse':
        niter2 = 0

    # Create output directory and cache
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Run sparse global alignment
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                   model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, 
                                   device=device, opt_depth='depth' in optim_level, 
                                   shared_intrinsics=shared_intrinsics,
                                   matching_conf_thr=matching_conf_thr)
    
    # Create output file
    outfile_name = os.path.join(output_dir, 'scene.glb')
    scene_state = SparseGAState(scene, cache_dir, outfile_name)
    
    # Generate 3D model
    outfile = get_3D_model_from_scene(scene_state, min_conf_thr, as_pointcloud, 
                                     mask_sky, clean_depth, transparent_cams, 
                                     cam_size, TSDF_thresh)
    
    return scene_state, outfile


def main():
    parser = argparse.ArgumentParser(description='MASt3R Scene Reconstruction')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to input images directory')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for results')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to model weights')
    parser.add_argument('--model_name', type=str, 
                       default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                       help='Model name (alternative to weights)')
    parser.add_argument('--retrieval_model', type=str, default=None,
                       help='Path to retrieval model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for processing')
    
    # Optimization parameters
    parser.add_argument('--optim_level', type=str, default='refine+depth',
                       choices=['coarse', 'refine', 'refine+depth'],
                       help='Optimization level')
    parser.add_argument('--lr1', type=float, default=0.07,
                       help='Learning rate for coarse alignment')
    parser.add_argument('--niter1', type=int, default=300,
                       help='Number of iterations for coarse alignment')
    parser.add_argument('--lr2', type=float, default=0.01,
                       help='Learning rate for refinement')
    parser.add_argument('--niter2', type=int, default=300,
                       help='Number of iterations for refinement')
    
    # Scene graph parameters
    parser.add_argument('--scenegraph_type', type=str, default='oneref',
                       choices=['complete', 'swin', 'logwin', 'oneref', 'retrieval'],
                       help='Scene graph type')
    parser.add_argument('--winsize', type=int, default=1,
                       help='Window size for scene graph')
    parser.add_argument('--win_cyclic', action='store_true',
                       help='Use cyclic sequence for scene graph')
    parser.add_argument('--refid', type=int, default=0,
                       help='Reference ID for scene graph')
    
    # Post-processing parameters
    parser.add_argument('--min_conf_thr', type=float, default=1.5,
                       help='Minimum confidence threshold')
    parser.add_argument('--matching_conf_thr', type=float, default=0.0,
                       help='Matching confidence threshold')
    parser.add_argument('--TSDF_thresh', type=float, default=0.0,
                       help='TSDF threshold for post-processing')
    parser.add_argument('--as_pointcloud', action='store_true', default=True,
                       help='Export as pointcloud instead of mesh')
    parser.add_argument('--mask_sky', action='store_true',
                       help='Apply sky masking')
    parser.add_argument('--clean_depth', action='store_true', default=True,
                       help='Clean up depth maps')
    parser.add_argument('--shared_intrinsics', action='store_true',
                       help='Use shared intrinsics for all views')
    
    args = parser.parse_args()
    
    # Load model
    if args.weights:
        from mast3r.model import AsymmetricMASt3R
        print(f"Loading model from {args.weights}")
        model = AsymmetricMASt3R.from_pretrained(args.weights).to(args.device)
    elif args.model_name:
        from mast3r.model import AsymmetricMASt3R
        print(f"Loading model {args.model_name}")
        model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)
    else:
        raise ValueError("Either --weights or --model_name must be provided")
    
    # Load retrieval model if specified
    retrieval_model = None
    if args.retrieval_model:
        if not has_retrieval:
            raise ValueError("Retrieval functionality not available")
        print(f"Loading retrieval model {args.retrieval_model}")
        # Load retrieval model here based on your specific implementation
        retrieval_model = args.retrieval_model
    
    # Run reconstruction
    print(f"Processing images from {args.input_dir}")
    scene_state, outfile = get_reconstructed_scene(
        image_dir=args.input_dir,
        output_dir=args.output_dir,
        model=model,
        device=args.device,
        image_size=args.image_size,
        retrieval_model=retrieval_model,
        optim_level=args.optim_level,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        min_conf_thr=args.min_conf_thr,
        matching_conf_thr=args.matching_conf_thr,
        as_pointcloud=args.as_pointcloud,
        mask_sky=args.mask_sky,
        clean_depth=args.clean_depth,
        scenegraph_type=args.scenegraph_type,
        winsize=args.winsize,
        win_cyclic=args.win_cyclic,
        refid=args.refid,
        TSDF_thresh=args.TSDF_thresh,
        shared_intrinsics=args.shared_intrinsics
    )
    
    print(f"Reconstruction completed!")
    print(f"3D model saved to: {outfile}")
    print(f"Cache directory: {scene_state.cache_dir}")


if __name__ == '__main__':
    main()