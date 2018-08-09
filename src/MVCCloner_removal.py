import os
import time
from itertools import product
import numpy as np
import cv2
from GetPatchInterface import GetPatchInterface 
from  preprocess import MVCSolver, GetAdaptiveMesh, CalcBCCoordinates
#from poisson_blending import PoissonBlendingInterface


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception("Failed to load the image from "+path)
    return img


class MVCCloner_removal:
    def __init__(self, src_img_path, target_img_path, output_path, mvc_config,grab):
        self.src_img = load_img(src_img_path)
        self.target_img = load_img(src_img_path)
        self.output_path = output_path
        self.GetPatchUI = GetPatchInterface(self.src_img,grab)
        self.mvc_solver = MVCSolver(mvc_config)
        # source patch attributes #
        self.lefttop = None
        self.rightbottom = None
        self.boundary = None
        self.boundary_values = None
        self.patch_pnts = None
        self.patch_values = None
        # UI attributes #
        self.moving = False
        self.anchor = None
        self.win_X = self.target_img.shape[1]
        self.win_Y = self.target_img.shape[0]
        self.theta = 0.
        self.ratio = 1.
        # Cloning attributes #
        self.MVCoords = None
        self.BCCoords = None
        self.triangles_vertices = None
        self.mesh_diffs = None
        self.num_boundary = None

        self.target_boundary_values = None
        self.original_patch_ptn = None
        

    def GetPatch(self):
        # get source patch from UI #
        self.GetPatchUI.run()
        start_t = time.time()
        self.boundary, self.boundary_values, self.patch_pnts, self.patch_values = self.GetPatchUI.GetPatch(sample_step=2)
        self.target_boundary_values = self.target_img[self.boundary[:, 1], self.boundary[:, 0], :]
        self.boundary = self.boundary.astype('float32')
        self.patch_pnts = self.patch_pnts.astype('float32')
        self.lefttop = np.min(self.boundary, axis=0)
        self.rightbottom = np.max(self.boundary, axis=0)
        print("GetPatch:", time.time() - start_t)
        start_t = time.time()
        # get adaptive triangular mesh #
        mesh, scipy_mesh = GetAdaptiveMesh(self.boundary, show=False)
        print("GetAdaptiveMesh:", time.time() - start_t)
        start_t = time.time()
        # vertices except boundary #
        self.num_boundary = self.boundary.shape[0]
        mesh_inner_vertices = scipy_mesh.points[self.num_boundary:]
        # Calc MV Coords #
        self.MVCoords = self.mvc_solver.CalcMVCoordinates(mesh_inner_vertices, self.boundary)
        print("MVCCoords:", time.time() - start_t)
        start_t = time.time()
        simplex_idxs, self.BCCoords = CalcBCCoordinates(scipy_mesh, self.patch_pnts)
        inliners_idxs = ~np.any(self.BCCoords < 0.-1e-8, axis=1)
        # filter outliers #
        self.patch_pnts = self.patch_pnts[inliners_idxs]
        self.original_patch_ptn = self.patch_pnts.copy()
        self.patch_values = self.patch_values[inliners_idxs]
        simplex_idxs = simplex_idxs[inliners_idxs]
        self.BCCoords = self.BCCoords[inliners_idxs]
        # find simplex vertices of mesh points #
        self.triangles_vertices = scipy_mesh.simplices[simplex_idxs]
        # create space for storing mesh diffs #
        self.mesh_diffs = np.zeros((len(scipy_mesh.points), 3), dtype='float32')
        print("BCCoords:", time.time() - start_t)

    # mouse callback function
    def mouse_on_patch(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if np.all([x, y] > self.lefttop.astype('int32')) and np.all([x, y] < self.rightbottom.astype('int32')):
                self.moving = True
                self.anchor = np.array([x, y], dtype='int32')

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.moving:
                self.move_patch(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.moving = False
            self.anchor = None

    def keyboard_on_patch(self, k):
        if k == ord('>'):
            #self.rotate_patch(10.)
            pass
        elif k == ord('<'):
            #self.rotate_patch(-10.)
            pass
        elif k == 0: # UP
            print('UP')
        elif k == 1: # DOWN
            print('DOWN')
        elif k == 2: # LEFT
            #self.zoom_patch(1./np.power(2, 1./5))
            pass
        elif k == 3: # RIGHT
            #self.zoom_patch(np.power(2, 1./5))
            pass

    def move_patch(self, x, y):
        max_corner = [self.win_X, self.win_Y]
        displacement = [x, y] - self.anchor
        self.anchor = np.array([x, y], dtype='int32')
        future_lefttop = (self.lefttop + displacement).astype('int32')
        future_rightbottom = (self.rightbottom + displacement).astype('int32')
        if np.any(future_lefttop < 0) or np.any(future_rightbottom + [1, 1] >= max_corner):
            return
        self.set_patch_pos(displacement)

    def rotate_patch(self, d):
        # transpose of the rotation matrix #
        rad = np.pi/180.*(d)
        M = np.array([[    np.cos(rad), np.sin(rad)],
                      [-1.*np.sin(rad), np.cos(rad)]], dtype='float32')
        patch_center = (self.rightbottom + self.lefttop) / 2.
        self.boundary = np.dot( (self.boundary - patch_center) , M) + patch_center
        self.patch_pnts = np.dot( (self.patch_pnts - patch_center) , M) + patch_center
        self.lefttop = np.min(self.boundary, axis=0)
        self.rightbottom = np.max(self.boundary, axis=0)
        self.theta = (self.theta + d) % 360

    def zoom_patch(self, ratio):
        if 0.5 < self.ratio * ratio and self.ratio * ratio < 2.:
            self.ratio *= ratio
            patch_center = (self.rightbottom + self.lefttop) / 2.
            self.boundary *=ratio
            self.patch_pnts *= ratio
            self.lefttop *= ratio
            self.rightbottom *= ratio
            new_patch_center = (self.rightbottom + self.lefttop) / 2.
            self.set_patch_pos(patch_center - new_patch_center)

    def set_patch_pos(self, displacement):
        self.boundary += displacement
        self.patch_pnts += displacement
        self.lefttop += displacement
        self.rightbottom += displacement


    def reset(self):
        screen_center = np.array([self.win_X >> 1 , self.win_Y >> 1], dtype='int32')
        self.rotate_patch(-1*self.theta)
        self.zoom_patch(1./self.ratio)
        #self.lefttop = np.min(self.boundary, axis=0) # calc in rotate patch
        #self.rightbottom = np.max(self.boundary, axis=0)
        patch_center = ((self.rightbottom + self.lefttop) / 2).astype('int32')
        self.set_patch_pos(screen_center - patch_center)
        self.moving = False
        self.anchor = None

    def run(self):
        assert not self.boundary is None, "Source Patch is not selected yet!"
        self.reset()
        cv2.namedWindow('MVCCloner')
        cv2.setMouseCallback('MVCCloner', self.mouse_on_patch)
        clone_time = []
        patch_time = []
        while True:
            img = self.target_img.copy()
            start_t = time.time()
            clone_values = self.CalcCloningValues()
            clone_time.append(time.time() - start_t)
            start_t = time.time()
            self.patch_img(img, clone_values)
            patch_time.append(time.time() - start_t)
            disp_img = self.draw_boundary(img)
            cv2.imshow('MVCCloner', disp_img)
            k = cv2.waitKey(5) & 0xFF
            if k == 32:     # space
                self.reset()
            elif k == ord('s'):
                cv2.imwrite(self.output_path, img)
                #poisson_output = PoissonBlendingInterface(self.target_img.copy(), self.boundary, 
                #                        self.boundary_values, self.patch_pnts, self.patch_values)
                #cv2.imwrite('Poisson_output.png', poisson_output)
            elif k == 13 or k == 27:   # enter or esc
                cv2.imwrite(self.output_path, img)
                print("Clone time:", np.mean(clone_time))
                print("Patch time:", np.mean(patch_time))
                break
            else:
                self.keyboard_on_patch(k)
        cv2.destroyAllWindows()

    def draw_boundary(self, img):
        disp_img = img.copy()
        boundary = self.boundary.astype('int32')
        cv2.drawContours(disp_img, [boundary], 0, (0, 255, 0), 2)
        return disp_img

    def CalcCloningValues(self):
        patch_pnts_int = self.patch_pnts.astype('int32')
        self.patch_values = self.target_img[patch_pnts_int[:, 1], patch_pnts_int[:, 0], :].astype('float32')
        boundary_int = self.boundary.astype('int32')
        self.boundary_values = self.target_img[boundary_int[:, 1], boundary_int[:, 0], :].astype('float32')
        diffs = self.target_boundary_values - self.boundary_values
        interpolants = np.dot( self.MVCoords , diffs)
        self.mesh_diffs[:self.num_boundary, :] = diffs
        self.mesh_diffs[self.num_boundary:, :] = interpolants
        BCinterps = self.mesh_diffs[self.triangles_vertices]
        clone_values = np.einsum('ijk,ij->ik', BCinterps, self.BCCoords) + self.patch_values
        return np.clip(clone_values, 0., 255.).astype('uint8')

    def patch_img(self, img, set_values):
        tmp = np.zeros_like(self.target_img, dtype='float32')
        weights = np.zeros((self.win_Y, self.win_X), dtype='float32')
        patch_pnts_int = self.original_patch_ptn.astype('int32')
        for dx, dy in product(range(2), range(2)):
            if dy == 0 and dx == 0:
                weight = 0.97
            else:
                weight = 0.01
            gause_pnts = patch_pnts_int + [dx, dy]
            tmp[gause_pnts[:, 1], gause_pnts[:, 0], :] += set_values * weight
            weights[gause_pnts[:, 1], gause_pnts[:, 0]] += weight
        patch_mask = weights > 0.
        img[patch_mask] = tmp[patch_mask] / weights[patch_mask][..., None]
        #img[patch_pnts_int[:, 1], patch_pnts_int[:, 0], :] = set_values



if __name__ == "__main__":
    mvc_config = {'hierarchic': True,
                  'base_angle_Th': 0.75,
                  'base_angle_exp': 0.8,
                  'base_length_Th': 2.5,
                  'adaptiveMeshShapeCriteria': 0.125,
                  'adaptiveMeshSizeCriteria': 0.,
                  'min_h_res': 16.}
    src_img_path = './target.jpg'
    target_img_path = './target.jpg'
    output_path = './out.jpg'

    mvc_cloner = MVCCloner_removal(src_img_path, target_img_path, output_path, mvc_config)
    mvc_cloner.GetPatch()
    mvc_cloner.run()

