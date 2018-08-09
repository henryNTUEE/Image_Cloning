import numpy as np
import scipy.sparse
import pyamg
import cv2
import gc
import sys

def PoissonBlendingInterface(tar, boundary, boundary_values, patch_pnts, patch_values):
    """
        interface to poisson blending from MVCCloner
    """
    boundary = boundary.astype('int32')
    patch_pnts = patch_pnts.astype('int32')
    ## reconstruct img and mask from pnts ##
    src_img = np.zeros_like(tar, dtype=np.uint8)
    src_img[boundary[:, 1], boundary[:, 0], :] = boundary_values
    src_img[patch_pnts[:, 1], patch_pnts[:, 0], :] = patch_values
    mask = np.zeros(src_img.shape[:2], dtype=bool)
    mask[boundary[:, 1], boundary[:, 0]] = True 
    mask[patch_pnts[:, 1], patch_pnts[:, 0]] = True

    output = PoissonBlending(src_img, tar, mask)

    return output

def PoissonBlending(src, tar, mask):
    H, W = tar.shape[:2]
    blending_mask = mask
    fill_mask = np.ones_like(mask, dtype=bool)
    loc = blending_mask.nonzero()
    loc_map = {} # mapping from coordinate to variable
    for i_loc, (j, i) in enumerate(zip(loc[0], loc[1])):
        loc_map[(j, i)] = i_loc
    #w_l, w_r = BlendingWeights(src_mask, tar_mask, src_img.shape[0])
    N = np.count_nonzero(blending_mask)
    y_min = np.min(loc[0])
    y_max = np.max(loc[0])
    x_min = np.min(loc[1])
    x_max = np.max(loc[1])
    res = np.zeros((N, 3))
    size = np.prod((y_max-y_min+1, x_max-x_min+1))
    print('solving...N: {}'.format(N))
    stride = x_max - x_min + 1
    A = scipy.sparse.identity(N, format='lil')
    b = np.zeros((N, 3), dtype=np.float32)
    for (j, i) in zip(loc[0], loc[1]):
        alpha = 0.0 #w_l[j, i]
        cur_ptr = loc_map[(j, i)]
        if(blending_mask[j, i]):
            N_p = 0.0
            v_pq = np.zeros((1,3), dtype=np.float32)
            f_p = tar[j, i, :].astype(np.float32)
            g_p = src[j, i, :].astype(np.float32)
            if(j > 0):
                if(fill_mask[j - 1, i]): #upper neighbor exists
                    f_q = tar[j-1, i, :].astype(np.float32)
                    g_q = src[j-1, i, :].astype(np.float32)
                    if(blending_mask[j - 1, i]): # in the omega
                        v_pq += np.dot([alpha, 1-alpha], np.array([(f_p-f_q), (g_p-g_q)]))
                        A[cur_ptr, loc_map[(j-1, i)]] = -1.0
                    else: # on the boundary
                        # known function f*_p + v_pq
                        # here we choose gradient image of original image with its
                        # pixel value exists.
                        v_pq += tar[j-1, i, :].astype(np.float32) #+ (f_p-f_q) 
                    N_p += 1.0
            if(j < H - 1):
                if(fill_mask[j + 1, i]): #lower neighbor exists
                    f_q = tar[j+1, i, :].astype(np.float32)
                    g_q = src[j+1, i, :].astype(np.float32)
                    if(blending_mask[j + 1, i]): # in the omega
                        v_pq +=  np.dot([alpha, 1-alpha], np.array([(f_p-f_q), (g_p-g_q)]))
                        A[cur_ptr, loc_map[(j+1, i)]] = -1.0
                    else: # on the boundary
                        v_pq +=tar[j+1, i, :].astype(np.float32) #+ (f_p-f_q)
                    N_p += 1.0
            if(fill_mask[j, i - 1]): #left neighbor exists
                f_q = tar[j, i-1, :].astype(np.float32)
                g_q = src[j, i-1, :].astype(np.float32)
                if(blending_mask[j, i-1]): # in the omega
                    v_pq += np.dot([alpha, 1-alpha], np.array([(f_p-f_q), (g_p-g_q)]))
                    A[cur_ptr, loc_map[(j, i-1)]] = -1.0
                else: # on the boundary
                    v_pq +=tar[j, i-1, :].astype(np.float32) #+ (f_p-f_q)
                N_p += 1.0
            if(fill_mask[j, i + 1]): #right neighbor exists
                f_q = tar[j, i+1, :].astype(np.float32)
                g_q = src[j, i+1, :].astype(np.float32)
                if(blending_mask[j, i+1]): # in the omega
                    v_pq += np.dot([alpha, 1-alpha], np.array([(f_p-f_q), (g_p-g_q)]))
                    A[cur_ptr, loc_map[(j, i+1)]] = -1.0
                else: # on the boundary
                    v_pq +=tar[j, i+1, :].astype(np.float32) #+ (f_p-f_q)
                N_p += 1.0
            A[cur_ptr, cur_ptr] = N_p
            b[cur_ptr, :] = v_pq.astype(np.float32)
        else: # not in blending region
            raise Exception('Illegal image!!')
    gc.collect()
    A = A.tocsr()
    for c in range(3):
        x = pyamg.solve(A, b[:, c], verb=True, tol=1e-5)
        x = np.clip(x, 0, 255)
        res[:, c] = x.astype('uint8')
    tar = tar.copy()
    tar[blending_mask, :] = res
    return tar

def onmouse(event, x, y, flags, param):
    global ix, iy, com_img, warp

    if event == cv2.EVENT_LBUTTONDOWN and ~warp:
        ix, iy = x, y
        cv2.circle(com_img, (ix, iy), 3, (255, 255, 0), -1)
        warp = True

if __name__ == '__main__':
    warp = False
    blend = False
    if len(sys.argv) == 4:
        src_name = sys.argv[1]
        mask_name = sys.argv[2]
        tar_name = sys.argv[3]
    src = cv2.imread(src_name)
    mask = cv2.imread(mask_name)
    tar = cv2.imread(tar_name)
    com_img = tar.copy()
    new_src = np.zeros_like(tar)
    new_src[:src.shape[0], :src.shape[1], :] = src[:, :, :]
    new_mask = np.zeros_like(tar)
    new_mask[:mask.shape[0], :mask.shape[1], :] = mask[:, :, :]
    output = new_src.copy()
    cv2.namedWindow('tar')
    cv2.namedWindow('out')
    cv2.setMouseCallback('tar',onmouse)


    while(1):
        cv2.imshow('tar', com_img)
        cv2.imshow('out', output)
        k = cv2.waitKey(1) & 0xFF
            # key bindings
        if k == 27:         # esc to exit
            break
        if warp and k == ord('n'):
            
            print(ix, iy)
            cur_src = cv2.warpAffine(new_src.copy(), np.array([[1, 0, ix], [0, 1, iy]], dtype=np.np.float3232), (new_src.shape[::-1][1:]))
            cur_mask = cv2.warpAffine(new_mask.copy(), np.array([[1, 0, ix], [0, 1, iy]], dtype=np.np.float3232), (new_src.shape[::-1][1:]))
            cur_mask = (cur_mask / 255).astype(bool)
            output =  PoissonBlending(cur_src, tar, cur_mask[:, :, 0])
            warp = False
            blend = True

    cv2.destroyAllWindows()
