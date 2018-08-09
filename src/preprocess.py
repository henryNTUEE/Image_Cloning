import time
import triangle
import triangle.plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import cv2
from GetPatchInterface import GetPatchInterface 


EPSILON     = 1e-8
ARCCOS_LUT  = np.arccos(np.linspace(-1., 1., 8192))
TAN_LUT     = np.tan(np.linspace(0., np.pi/2., 8192))


def GetAdaptiveMesh(boundary, show=True):
    num_pnt = boundary.shape[0]
    segments = np.stack([np.arange(num_pnt), (np.arange(num_pnt)+1)%num_pnt], axis=1)
    patch = dict(vertices=boundary, segments=segments)
    tri_mesh = triangle.triangulate(patch, 'pq')
    scipy_tri_mesh= Delaunay(tri_mesh['vertices'].astype('int32'))
    if show:
        triangle.plot.compare(plt, patch, tri_mesh)
        plt.show()
        plt.triplot(tri_mesh['vertices'][:,0], tri_mesh['vertices'][:,1], scipy_tri_mesh.simplices.copy())
        plt.show()
    return tri_mesh, scipy_tri_mesh

def CalcBCCoordinates(tri_mesh, patch_pnts):
    simplex_idxs = tri_mesh.find_simplex(patch_pnts)
    T_invs = tri_mesh.transform[simplex_idxs, :2]
    bc_coords_ij = np.einsum('ijk,ik->ij', T_invs, (patch_pnts - tri_mesh.transform[simplex_idxs, 2]))
    bc_coords = np.hstack([bc_coords_ij, 1 - np.sum(bc_coords_ij, axis=1, keepdims=True)])
    return simplex_idxs, bc_coords


class MVCSolver:
    def __init__(self, config):
        self.hierarchic = config['hierarchic']
        self.base_angle_Th = config['base_angle_Th']
        self.base_angle_exp = config['base_angle_exp']
        self.base_length_Th = config['base_length_Th']
        self.adaptiveMeshShapeCriteria = config['adaptiveMeshShapeCriteria']
        self.adaptiveMeshSizeCriteria = config['adaptiveMeshSizeCriteria']
        self.min_h_res = config['min_h_res']

    def CalcMVCoordinates(self, mesh_vertices, boundary):
        """
        Calculate Mean-Value Coordinates for each mesh vertex
        Args:
          mesh_vertices:    vertices of the triangular mesh without the boundary, numpy array of shape [N1, 2].
          boundary:         vertices of the boundary, numpy array of shape [N2, 2].
          hierarchic:       whether performing hierarchical boundary sampling.
        Returns
          MVCoords:         Mean-Value Coordinates for each vertex, numpy array of shape [N1, N2]
        """
        num_vertices = mesh_vertices.shape[0]
        num_boundary = boundary.shape[0]
        max_h_depth = int(np.log2(num_boundary / self.min_h_res))
        max_h_step = int(np.power(2, max_h_depth))
        CalcMVCoordinates = np.zeros((num_vertices, num_boundary), dtype='float32')
        for i, vertex in enumerate(mesh_vertices):
            if self.hierarchic:
                CalcMVCoordinates[i, :] = self.CalcHCoord(vertex, boundary, num_boundary, max_h_depth, max_h_step)
            else:
                CalcMVCoordinates[i, :] = self.CalcCoord(vertex, boundary, num_boundary)
        return CalcMVCoordinates

    def CalcHCoord(self, vertex, boundary, num_boundary, max_depth, max_step):
        """
        Performing hierarchical boundary sampling
        """
        coord = np.zeros(num_boundary, dtype='float32')
        handled = np.zeros(num_boundary, dtype='bool')
        indices_stack = [i for i in range(0, num_boundary, max_step)]
        depths_stack = [False]*num_boundary
        while not len(indices_stack) == 0:
            idx = indices_stack.pop()
            depth = depths_stack.pop()
            if handled[idx]:
                continue
            step = int(np.power(2, max_depth - depth))
            prev_idx = (idx - step + num_boundary) % num_boundary
            next_idx = (idx + step) % num_boundary
            # set ref_vertices = [prev_pnt, curr_pnt, next_pnt] #
            ref_vertices = np.array([boundary[prev_idx], boundary[idx], boundary[next_idx]], dtype='float32')
            ref_vectors = ref_vertices - vertex
            lengths = np.maximum(np.linalg.norm(ref_vectors, axis=1), EPSILON)
            prev_cos = np.clip(np.sum(np.product(ref_vectors[:2], axis=0)) / (lengths[0]*lengths[1]), -1., 1.)
            next_cos = np.clip(np.sum(np.product(ref_vectors[1:], axis=0)) / (lengths[1]*lengths[2]), -1., 1.)
            prev_angle = ARCCOS_LUT[int( (prev_cos+1.) / 2. * (8192-1) )]
            next_angle = ARCCOS_LUT[int( (next_cos+1.) / 2. * (8192-1) )]
            # adjusted threshold #
            length_Th = max_step / np.power(self.base_length_Th, depth)
            angle_Th = self.base_angle_Th * np.power(self.base_angle_exp, depth)
            # test current boundary point #
            if (step == 1) or (lengths[1] >= length_Th and prev_angle <= angle_Th and next_angle <= angle_Th):
                handled[idx] = True
                # calc MV coord #
                prev_tan = TAN_LUT[int( (prev_angle/2.) / (np.pi/2.) * (8192-1) )]
                next_tan = TAN_LUT[int( (next_angle/2.) / (np.pi/2.) * (8192-1) )]
                coord[idx] = (prev_tan + next_tan) / lengths[1]
            else:
                # insert curr index #
                indices_stack.append(idx)
                depths_stack.append(depth+1)
                # insert prev index #
                if not handled[prev_idx]:
                    indices_stack.append(prev_idx)
                    depths_stack.append(depth+1)
                # insert finer prev index #
                finer_prev_idx = int((prev_idx + step / 2) % num_boundary)
                if not handled[finer_prev_idx]:
                    indices_stack.append(finer_prev_idx)
                    depths_stack.append(depth+1)
                # insert next index #
                if not handled[next_idx]:
                    indices_stack.append(next_idx)
                    depths_stack.append(depth+1)
                # insert finer next index #
                finer_next_idx = int((next_idx - step / 2 + num_boundary) % num_boundary)
                if not handled[finer_next_idx]:
                    indices_stack.append(finer_next_idx)
                    depths_stack.append(depth+1)
        return coord / np.sum(coord)

    def CalcCoord(self, vertex, boundary, num_boundary):
        coord = np.zeros(num_boundary, dtype='float32')
        ref_vectors = boundary - vertex
        lengths = np.maximum(np.linalg.norm(ref_vectors, axis=1), EPSILON)
        idx_less_than_1 = np.where(lengths < 1.)[0]
        if not idx_less_than_1.shape[0] == 0:
            coord[idx_less_than_1[0]] == 1.
            return coord
        ref_unit_vectors = ref_vectors / lengths[..., None]
        cos_thetas = np.clip(np.sum(ref_unit_vectors*np.roll(ref_unit_vectors, -1, axis=0), axis=1), -1., 1.)
        #thetas = ARCCOS_LUT[( (cos_thetas+1.) / 2. * (8192-1) ).astype('int32')]
        thetas = np.arccos(cos_thetas)
        #tans = TAN_LUT[( (thetas/2.) / (np.pi/2.) * (8192-1) ).astype('int32')]
        tans = np.tan(thetas/2.)
        coord = (tans + np.roll(tans, 1)) / lengths
        return coord / np.sum(coord)    


if __name__ == "__main__":
    mvc_config = {'hierarchic': True,
                  'base_angle_Th': 0.75,
                  'base_angle_exp': 0.8,
                  'base_length_Th': 2.5,
                  'adaptiveMeshShapeCriteria': 0.125,
                  'adaptiveMeshSizeCriteria': 0.,
                  'min_h_res': 16.}
    src_img = cv2.imread('../img/Dog.jpg')
    GetPatchUI = GetPatchInterface(src_img)
    GetPatchUI.run()
    boundary, boundary_values, patch_pnts, patch_values = GetPatchUI.GetPatch(sample_step=2)
    num_pnt = boundary.shape[0]
    print("num_pnt:", num_pnt)

    tri_mesh, scipy_tri_mesh = GetAdaptiveMesh(boundary, show=False)
    print("# of mesh vertices:", tri_mesh['vertices'].shape[0])

    # vertices except boundary #
    mesh_vertices = tri_mesh['vertices'][num_pnt:].astype('int32')
    # Calc MV Coords #
    mvc = MVCSolver(mvc_config)
    MVCoords = mvc.CalcMVCoordinates(mesh_vertices, boundary)
    print("MVCoords shape:", MVCoords.shape)
    print("num sampling boundary points of %d vertices:\n" % np.sum(MVCoords > 0, axis=1).shape[0], np.sum(MVCoords > 0, axis=1))

    plt.scatter(scipy_tri_mesh.points[:, 0], scipy_tri_mesh.points[:, 1], color='blue', s=4)
    #plt.triplot(tri_mesh['vertices'][:,0], tri_mesh['vertices'][:,1], scipy_tri_mesh.simplices.copy())
    """
    # show sampling boundary points of a random vertex #
    vertex_idx = np.random.randint(mesh_vertices.shape[0])
    plt.scatter(np.append(boundary[:, 0][MVCoords[vertex_idx] > 0], mesh_vertices[vertex_idx][0]),
                np.append(boundary[:, 1][MVCoords[vertex_idx] > 0], mesh_vertices[vertex_idx][1]),
                color='red', s=16)
    """
    simplex_idxs, BCCoords = CalcBCCoordinates(scipy_tri_mesh, patch_pnts)
    """
    # plot points outside the simplices #
    cnt = 0
    for i, p in enumerate(patch_pnts):
        if np.all(np.any(scipy_tri_mesh.points != p, axis=1)):
            if np.any(BCCoords[i] > 1.-EPSILON):
                cnt += 1
                print(p, BCCoords[i])
                plt.scatter(p[0], p[1], color='red', s=64)
    print('outliers num:', cnt)
    """
    plt.show()
