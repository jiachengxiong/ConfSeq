import numpy as np
from rdkit.Chem import rdMolTransforms
from rdkit import Geometry
import oddt
import oddt.surface
from oddt import toolkit

import torch
from rdkit import Chem
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes


def get_mesh(mol, center_pos=False, confId=-1, scaling=1.0, probe_radius=1.4):
    if center_pos:
        conformer = mol.GetConformer(confId)
        center = rdMolTransforms.ComputeCentroid(conformer)
        if max(np.abs([center.x, center.y, center.z])) > 0.1:
            pos = np.array(conformer.GetPositions(), dtype=np.float64)
            offset = np.mean(pos, axis=0)
            pos = pos - offset
            for i in range(pos.shape[0]):
                point = Geometry.Point3D(pos[i, 0], pos[i, 1], pos[i, 2])
                conformer.SetAtomPosition(i, point)

    oddtconf = Chem.MolToMolBlock(mol, confId=confId)
    oddtconftool = toolkit.readstring('sdf', oddtconf)
    oddtconftool.calccharges()
    verts, faces = oddt.surface.generate_surface_marching_cubes(oddtconftool, scaling=scaling, probe_radius=probe_radius)
    return verts, faces


def get_pointcloud_from_mesh(mesh, num_samples, return_mesh=False, return_normals=True):
    """Sample points from a mesh to generate a point cloud."""
    mesh = Meshes(verts=[torch.FloatTensor(mesh[0].copy())], faces=[torch.FloatTensor(mesh[1].copy())])
    point_clouds = sample_points_from_meshes(mesh, num_samples, return_normals=return_normals)
    if not return_mesh:
        return point_clouds
    else:
        return point_clouds, mesh
