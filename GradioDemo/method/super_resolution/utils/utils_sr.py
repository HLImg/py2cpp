import numpy as np
import scipy
from scipy import fftpack
from scipy.interpolate import interp2d
import torch
import cv2


def imsave(img, img_path):
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def downsample_np(x, sf=3, center=False):
    st = (sf-1)//2 if center else 0
    return x[st::sf, st::sf, ...]


def uint2single(img):
    return np.float32(img/255.)


def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


def wrap_boundary_liu(img, img_size):

    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    if img.ndim == 2:
        ret = wrap_boundary(img, img_size)
    elif img.ndim == 3:
        ret = [wrap_boundary(img[:, :, i], img_size) for i in range(3)]
        ret = np.stack(ret, 2)
    return ret


def wrap_boundary(img, img_size):

    """
    python code from:
    https://github.com/ys-koshelev/nla_deblur/blob/90fe0ab98c26c791dcbdf231fe6f938fca80e2a0/boundaries.py
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H, W) = np.shape(img)
    H_w = int(img_size[0]) - H
    W_w = int(img_size[1]) - W

    # ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1
    HG = img[:, :]

    r_A = np.zeros((alpha*2+H_w, W))
    r_A[:alpha, :] = HG[-alpha:, :]
    r_A[-alpha:, :] = HG[:alpha, :]
    a = np.arange(H_w)/(H_w-1)
    # r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
    r_A[alpha:-alpha, 0] = (1-a)*r_A[alpha-1, 0] + a*r_A[-alpha, 0]
    # r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end)
    r_A[alpha:-alpha, -1] = (1-a)*r_A[alpha-1, -1] + a*r_A[-alpha, -1]

    r_B = np.zeros((H, alpha*2+W_w))
    r_B[:, :alpha] = HG[:, -alpha:]
    r_B[:, -alpha:] = HG[:, :alpha]
    a = np.arange(W_w)/(W_w-1)
    r_B[0, alpha:-alpha] = (1-a)*r_B[0, alpha-1] + a*r_B[0, -alpha]
    r_B[-1, alpha:-alpha] = (1-a)*r_B[-1, alpha-1] + a*r_B[-1, -alpha]

    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha-1:, :])
        B2 = solve_min_laplacian(r_B[:, alpha-1:])
        r_A[alpha-1:, :] = A2
        r_B[:, alpha-1:] = B2
    else:
        A2 = solve_min_laplacian(r_A[alpha-1:-alpha+1, :])
        r_A[alpha-1:-alpha+1, :] = A2
        B2 = solve_min_laplacian(r_B[:, alpha-1:-alpha+1])
        r_B[:, alpha-1:-alpha+1] = B2
    A = r_A
    B = r_B

    r_C = np.zeros((alpha*2+H_w, alpha*2+W_w))
    r_C[:alpha, :] = B[-alpha:, :]
    r_C[-alpha:, :] = B[:alpha, :]
    r_C[:, :alpha] = A[:, -alpha:]
    r_C[:, -alpha:] = A[:, :alpha]

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha-1:, alpha-1:])
        r_C[alpha-1:, alpha-1:] = C2
    else:
        C2 = solve_min_laplacian(r_C[alpha-1:-alpha+1, alpha-1:-alpha+1])
        r_C[alpha-1:-alpha+1, alpha-1:-alpha+1] = C2
    C = r_C
    # return C
    A = A[alpha-1:-alpha-1, :]
    B = B[:, alpha:-alpha]
    C = C[alpha:-alpha, alpha:-alpha]
    ret = np.vstack((np.hstack((img, B)), np.hstack((A, C))))
    return ret


def solve_min_laplacian(boundary_image):
    (H, W) = np.shape(boundary_image)

    # Laplacian
    f = np.zeros((H, W))
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0
    j = np.arange(2, H) - 1
    k = np.arange(2, W) - 1
    f_bp = np.zeros((H, W))
    f_bp[np.ix_(j, k)] = -4 * boundary_image[np.ix_(j, k)] + boundary_image[np.ix_(j, k + 1)] + boundary_image[np.ix_(j, k - 1)] + boundary_image[np.ix_(j - 1, k)] + boundary_image[np.ix_(j + 1, k)]

    del (j, k)
    f1 = f - f_bp  # subtract boundary points contribution
    del (f_bp, f)

    # DST Sine Transform algo starts here
    f2 = f1[1:-1, 1:-1]
    del (f1)

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2, type=1, axis=0) / 2
    else:
        tt = fftpack.dst(f2, type=1) / 2

    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1, axis=0) / 2)
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt), type=1) / 2)
    del (f2)

    # compute Eigen Values
    [x, y] = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    denom = (2 * np.cos(np.pi * x / (W - 1)) - 2) + (2 * np.cos(np.pi * y / (H - 1)) - 2)

    # divide
    f3 = f2sin / denom
    del (f2sin, x, y)

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3 * 2, type=1, axis=1) / (2 * (f3.shape[1] + 1))
    else:
        tt = fftpack.idst(f3 * 2, type=1, axis=0) / (2 * (f3.shape[0] + 1))
    del (f3)
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt) * 2, type=1) / (2 * (tt.shape[0] + 1)))
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt) * 2, type=1, axis=0) / (2 * (tt.shape[1] + 1)))
    del (tt)

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image
    img_direct[1:-1, 1:-1] = 0
    img_direct[1:-1, 1:-1] = img_tt
    return img_direct

def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH, image or kernel
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x