import cv2
import numpy as np
from grabcut import GCManager


def unique_row(data):
    sorted_idx = np.lexsort(data.T)
    sorted_data =  data[sorted_idx]
    diffs = np.diff(sorted_data, axis=0)
    uniqueness = np.ones(len(data), dtype='bool')
    uniqueness[1:] = np.any(diffs, axis=1)
    uniqueness[sorted_idx] = uniqueness.copy()
    return data[uniqueness]

def GetPatchPnts(boundary, mask):
    min_y = np.min(boundary[:, 1])
    max_y = np.max(boundary[:, 1])
    min_x = np.min(boundary[:, 0])
    max_x = np.max(boundary[:, 0])
    # generate candidates grid points (x, y) #
    cand_pnts = np.stack(np.meshgrid(range(min_x, max_x+1), range(min_y, max_y+1)), axis=-1).reshape(-1, 2)
    return cand_pnts[mask[cand_pnts[:, 1], cand_pnts[:, 0]], :]


class GetPatchInterface:
    def __init__(self, src_img=None,grab=None):
        if src_img is None:
            self.src_img = np.zeros((512, 512, 3), np.uint8)
        else:
            self.src_img = src_img
        self.drawing = False # true if mouse is pressed
        self.add = False
        self.boundary = np.empty([0, 2], dtype='int32')
        self.track = np.empty([0, 2], dtype='int32')
        self.first_idx = 0
        self.grab = grab
        self.grabCut = GCManager(src_img.copy(),self.grab)
        

    # mouse callback function
    def draw_boundary(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.add = True
            self.boundary = np.empty([0, 2], dtype='int32')
            self.track = np.empty([0, 2], dtype='int32')
            self.boundary = np.append(self.boundary, [[x, y]], axis=0)
            self.track = np.append(self.track, [[x, y]], axis=0)

        elif event == cv2.EVENT_MOUSEMOVE:
            pnt = np.array([x, y], dtype='int32')
            if self.drawing == True:
                self.track = np.append(self.track, [pnt], axis=0)
            if self.add:
                dist2origs = np.linalg.norm(pnt - self.boundary[:-20], axis=1)
                if self.boundary.shape[0] > 40 and np.any(dist2origs < 5):
                    self.add = False
                    self.first_idx = np.argmin(dist2origs[:20])
                elif not np.all(pnt == self.boundary[-1]):
                    line_pnts = self.fix_boundary(self.boundary[-1], pnt)
                    self.boundary = np.append(self.boundary, line_pnts, axis=0)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.boundary.shape[0] > 20:
                pnt = np.array([x, y], dtype='int32')
                dist2origs = np.linalg.norm(pnt - self.boundary[:-20], axis=1)
                self.track = np.append(self.track, [pnt], axis=0)
                if self.add:
                    if not(self.boundary.shape[0] > 40 and np.any(dist2origs < 5)):
                        if not np.all(pnt == self.boundary[-1]):
                            line_pnts = self.fix_boundary(self.boundary[-1], pnt)
                            self.boundary = np.append(self.boundary, line_pnts, axis=0)
                self.first_idx = np.argmin(dist2origs[:20])
                self.boundary = self.boundary[self.first_idx:]
                line_pnts = self.fix_boundary(self.boundary[-1], self.boundary[0])
                self.boundary = np.append(self.boundary, line_pnts[:-1], axis=0)
            else:
                self.boundary = np.empty([0, 2], dtype='int32')
                self.track = np.empty([0, 2], dtype='int32')
            self.drawing = False
            self.add = False
            self.first_idx = 0

    def fix_boundary(self, start_pnt, end_pnt):
        step_x = np.abs(end_pnt[0] - start_pnt[0])
        step_y = np.abs(end_pnt[1] - start_pnt[1])
        step = step_x if step_x >= step_y else step_y
        ratio = np.linspace(1, step, step)/step
        line_pnts = np.round((end_pnt*ratio[..., None] + start_pnt*(1-ratio[..., None]))).astype('int32')
        return line_pnts # return with end point

    def reset(self):
        self.drawing = False # true if mouse is pressed
        self.add = False
        self.boundary = np.empty([0, 2], dtype='int32')
        self.track = np.empty([0, 2], dtype='int32')
        self.first_idx = 0

    def run(self):
        self.reset()
        cv2.namedWindow('GetPatch')
        cv2.setMouseCallback('GetPatch', self.draw_boundary)
        while True:
            img = self.src_img.copy()
            if not len(self.boundary) == 0:
                if self.drawing:
                    #cv2.polylines(img, [self.track], False, (0, 255, 0), 3)
                    cv2.polylines(img, [self.boundary], False, (0, 255, 0), 3)
                else:
                    cv2.drawContours(img, [self.boundary], 0, (0,255,0), 3)
            cv2.imshow('GetPatch', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:
                self.reset()
            elif k == 13:
                if self.boundary.shape[0] == 0:
                    print("UserWarning: The Boundary is not chosen yet!")
                else:
                    # remove duplicate points #
                    self.boundary = unique_row(self.boundary)
                    break
        cv2.destroyWindow('GetPatch')
        self.boundary = self.grabCut.interactive_session(self.boundary)

    def GetPatch(self, sample_step=2):
        assert not self.boundary.shape[0] == 0, "The Boundary is not chosen yet!"
        #approx_boundary = cv2.approxPolyDP(self.boundary, 0.001, True)
        #approx_boundary = approx_boundary.reshape(-1, 2)
        sample_boundary = self.boundary[::sample_step].copy()
        img = np.zeros(self.src_img.shape[:2], dtype='uint8')
        cv2.drawContours(img, [sample_boundary], 0, 255, -1)
        mask = img.astype('bool')
        patch_pnts = GetPatchPnts(sample_boundary, mask)
        values = self.src_img.copy()
        boundary_values = values[sample_boundary[:, 1], sample_boundary[:, 0], :].astype('float32')
        patch_values = values[patch_pnts[:, 1], patch_pnts[:, 0], :].astype('float32')
        return sample_boundary, boundary_values, patch_pnts, patch_values


if __name__ == "__main__":
    src_img = cv2.imread('../img/Dog.jpg')
    shape = src_img.shape if not src_img is None else (512, 512, 3)
    GetPatchUI = GetPatchInterface(src_img)
    GetPatchUI.run()
    boundary, boundary_values, patch_pnts, patch_values = GetPatchUI.GetPatch(sample_step=2)
    print("Boundary:\n", boundary)
    """
    print("not in pat:\n")
    for p in boundary:
        if not p in patch_pnts:
            print(p)
    """
    patch_img = np.ones(shape, dtype='uint8')
    #cv2.drawContours(patch_img, [boundary], 0, (0,255,0), 3)
    #patch_img[patch_pnts[:, 1], patch_pnts[:, 0], :] = patch_values
    #patch_img[boundary[:, 1], boundary[:, 0], :] = boundary_values
    cv2.imshow('img', patch_img)
    cv2.waitKey(0)

