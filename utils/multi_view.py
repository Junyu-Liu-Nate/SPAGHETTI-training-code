from custom_types import *
import torch
from utils import files_utils, image_utils
from PIL import Image
import cv2
from pypoisson import poisson_reconstruction
import plyfile
import os

def load_image(path: str) -> TS:
    im = cv2.imread(path, -1)
    mask = ((im == 65535).sum(-1) == 4)
    mask = torch.from_numpy(mask)
    return torch.from_numpy(im.astype(np.float64)), mask


def save_ply(points, filename, colors=None, normals=None):
    points = points.numpy()
    vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats='f4, f4, f4')
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        normals = normals.numpy()
        vertex_normal = np.core.records.fromarrays(normals.transpose(), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.core.records.fromarrays(colors.transpose() * 255, names='red, green, blue',
                                                  formats='u1, u1, u1')
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def get_view_rotation(view, ind):
    eye = view[ind]
    center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    axis = nnf.normalize(center - eye, 2, -1)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    s = torch.cross(axis, up)
    if s.norm(2, -1) == 0:
        up = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
        s = torch.cross(axis, up)
    s = nnf.normalize(s, 2, -1)
    up = torch.cross(s, axis)
    r = torch.eye(4, dtype=torch.float64)
    t = torch.eye(4, dtype=torch.float64)
    t[:3, 3] = -eye
    r[0, :3] = s
    r[1, :3] = up
    r[2, :3] = -axis
    mm = r.matmul(t)
    # r[:3, 3] = -eye

    # files_utils.export_mesh(up.unsqueeze(0), "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/up")
    # files_utils.export_mesh(eye.unsqueeze(0), "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/eye")

    return torch.inverse(mm)


def main(path: str):
    points_all = []
    normals_all = []
    vs, faces = files_utils.load_mesh("/home/amir/projects/sdf_gmm/assets/cache/3mv/Character/view/view.off")
    files_utils.export_mesh((vs, faces ), "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/view")
    vs = vs.to(torch.float64)[2:]

    for i in range(len(vs)):
        image, mask = load_image(f"/home/amir/projects/sdf_gmm/assets/cache/3mv/Character/dn/12520_Ferb/dn-256-{i}.png")
        h, w, c = image.shape
        v_arr, u_arr = torch.meshgrid(torch.arange(h), torch.arange(w))
        u_arr, v_arr = u_arr.to(dtype=torch.float64), v_arr.to(dtype=torch.float64)
        # files_utils.imshow(image)
        xyzd = image / 32768. - 1.
        xyzd[mask] = -1
        depth = xyzd[:, :, -1]
        # files_utils.imshow((xyzd + 1) / 2)

        mask_ = ~mask
        rot = get_view_rotation(vs, i)

        points = torch.zeros((~mask).sum(), 4, dtype=torch.float64)
        normals = torch.zeros((~mask).sum(), 3, dtype=torch.float64)
        points_fast = torch.zeros((~mask).sum(), 3, dtype=torch.float64)
        points_fast[:, 0] = ((u_arr[mask_] * 2.0 + 1.0 - w) / w)
        points_fast[:, 1] = ((h - v_arr[mask_] * 2.0 - 1.0) / h)
        points_fast[:, 2] = depth[mask_].flatten()
        point_id = 0
        l, r, b, t, n, f = -2.5, 2.5, -2.5, 2.5, .1, 5.
        proj = torch.tensor([
            [2.0/(r-l),  0.0,        0.0,         -(r+l)/(r-l)],
            [0.0,        2.0/(t-b),  0.0,         -(t+b)/(t-b)],
            [0.0,        0.0,        -2.0/(f-n),  -(f+n)/(f-n)],
            [0.0,        0.0,        0.0,         1.0]
        ], dtype=torch.float64)
        proj_inv = torch.inverse(proj)

        for u in range(w):
            for v in range(h):
                if mask_[v, u]:
                    points[point_id] = torch.tensor([(u * 2.0 + 1.0 - w) / w, (h - v * 2.0 - 1.0) / h, depth[v, u], 1],
                                                    dtype=torch.float64)
                    normals[point_id] = xyzd[v, u, :-1]
                    point_id += 1
        save_ply(points[:, :3], "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial_before.ply", normals=normals)
        # files_utils.export_mesh(points_fast, "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trialb")
        transform = torch.matmul(rot, proj_inv)
        # points = torch.einsum('ad,nd->na', proj_inv, points)
        points = torch.einsum('ad,nd->na', transform, points)
        normals = torch.einsum('ad,nd->na', rot[:3, :3], normals)
        points_all.append(points[:, :3])
        normals_all.append(normals)

        save_ply(points[:, :3], "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial_after.ply", normals=normals)
        return
        # files_utils.export_mesh(points, f"/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial_{i}")
    points_all = torch.cat(points_all)
    normals_all = torch.cat(normals_all)
    # select = torch.rand(len(points_all)).topk(10000)[1]
    # faces, vertices = poisson_reconstruction(points_all.numpy(), normals_all.numpy(), depth=5)
    # files_utils.export_mesh(points_all[select], "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial", normals=normals_all[select])
    # vs, _, normals = files_utils.load_ply("/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial.ply",
    #                                       get_normals=True)
    # save_ply(points_all, "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial.ply", normals=normals_all)
    # plyfile.
    # files_utils.export_mesh((vertices, faces), "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial_poisson")


def marching_pc(vs, normals):

    def forward(points):
        dist = ((points[:, None] - vs[None, :]) ** 2).sum(-1)
        idx = dist.argmin(1)
        vec = points - vs[idx]
        normals_vec = normals[idx]
        dot = (vec * normals_vec).sum(-1)
        out = torch.sign(dot)
        return out

    return forward
    # out = 2 * out.sigmoid_() - 1

def marching():
    device = CPU
    vs, _, normals = files_utils.load_ply("/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial.ply",
                                          get_normals=True)
    vs, _ = mesh_utils.to_unit_sphere((vs, None), scale=.9)
    vs, normals = vs.to(device), normals.to(device)
    mcubes = mcubes_meshing.MarchingCubesMeshing(device, max_batch=16**3)
    # mesh = mcubes.occ_meshing(marching_pc(vs, normals))
    decoder = marching_pc(vs, normals)
    points = 2 * torch.rand(5000, 3, device=device) - 1
    label = decoder(points)
    mask = label > 0
    files_utils.export_mesh(points[mask], "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial_mc")
    # if mesh is not None:
    #     files_utils.export_mesh(mesh, "/home/amir/projects/sdf_gmm/assets/cache/3mv_processed/trial_mc")
    # print(mesh is not None)
    return


if __name__ == '__main__':
    from utils import mcubes_meshing, mesh_utils
    # marching()
    main("/home/amir/projects/sdf_gmm/assets/cache/3mv/Character/dn/12517_Phineas/dn-256-0.png")
