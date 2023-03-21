import json
import numpy as np
import cv2
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.nn.functional import grid_sample, interpolate
from torchvision.utils import draw_keypoints
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.linalg import inv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils import rot_x, rot_y, rot_z, deg2rad


def app_board_area(board, total_area):
    mins = board.min(0)[0].tolist()
    maxs = board.max(0)[0].tolist()
    p1 = np.array(mins)
    p2 = np.array([mins[0], maxs[1]])
    p3 = np.array(maxs)
    v1 = p1 - p2
    v2 = p3 - p2
    area = np.linalg.norm(np.cross(v1, v2)) / 4
    return area


def get_nhat(alpha1, alpha2):
    n = np.array([[0, 0, 1]]).T
    nhat = rot_y(alpha2)@rot_x(alpha1)@n
    nhat /= np.linalg.norm(nhat)
    return nhat


def get_camera(alpha1, alpha2):
    cam = np.eye(3)
    cam = rot_y(alpha2)@rot_x(alpha1)@cam
    nhat = cam[:, -1:]
    return cam, nhat


def get_projective(alpha1, alpha2, alpha3, scale):
    #nhat = -get_nhat(alpha1, alpha2)
    #nhat1 = np.array([[1, 0, 0]]).T - nhat[0, 0]*nhat
    #nhat1 = nhat1 / np.linalg.norm(nhat1)
    #nhat2 = np.cross(nhat.T, nhat1.T).T
    #CR = np.concatenate([nhat1, nhat2, nhat], axis=1)

    # World 2 Camera
    CR, nhat = get_camera(alpha1, alpha2)
    a12_alph = np.array(max(abs(alpha1), abs(alpha2))) #2.0*np.arcsin(np.linalg.norm(CR-np.eye(3), ord='fro') / np.sqrt(8))
    RA = rot_z(alpha3)
    R = CR@RA
    P = scale*nhat
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, -1:] = -R.T@P
    T = torch.from_numpy(T).float()

    return T, R, P, a12_alph


def get_cv_image_residuals(image, corners):
    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    print('findChessboardCorners')
    retval, cv_corners = cv2.findChessboardCorners(image.squeeze().numpy(), (image_size[1]-1, image_size[0]-1), chessboard_flags)
    corners = corners.view(1, -1, 2)

    if retval:
        print('cornerSubPix')
        cv_corners = cv2.cornerSubPix(image.squeeze().numpy(), cv_corners, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1))
        cv_corners += 0.5
        cv_corners = torch.from_numpy(cv_corners).view(1, -1, 2)
        residual = (corners - cv_corners)
        if residual.flatten().max() > 2:
            print('match corners')
            residual_m2m = corners.view(1, 1, -1, 2) - cv_corners.view(1, -1, 1, 2)
            ridx = residual_m2m.norm(dim=-1).argmin(dim=1)
            cv_corners = cv_corners[:, ridx].view(1, -1, 2)
            residual = (corners - cv_corners)
        return residual, cv_corners
    return None, None


class Smoothing(object):
    def __init__(self, in_channels=1, sigma=2.0, kernel_size=7, device=torch.device('cpu')):
        self.device = device
        self.in_channels = in_channels
        dxf, dyf, g = self.init_derivative_filters(sigma, kernel_size, in_channels)
        self.dx_filt = dxf.to(device)
        self.dy_filt = dyf.to(device)
        self.g = g.to(device)

    def init_derivative_filters(self, sigma, kernel_size, in_channels):
        """ _init_derivative_filters(), initialize the derivative filters
        Args:
            sigma, for gaussian spread
            kernel_size, size of the derivative filter
            in_channels, rgb=3 or intensity image=1
        Returns:
            None
        """
        lowpass = np.atleast_2d(np.exp(-0.5*(np.arange(-kernel_size//2+1, kernel_size//2+1, 1)/sigma)**2))
        lowpass = lowpass/np.sum(lowpass)
        lowpass = lowpass.transpose() * lowpass
        dx = -1.0 / np.square(sigma) * np.arange(-kernel_size//2+1, kernel_size//2+1, 1)
        dx_filter = dx * lowpass
        dy_filter = dx_filter.transpose()

        # To pytorch
        g = torch.Tensor(lowpass)
        g = g.view(1, 1, kernel_size, kernel_size).repeat(in_channels, in_channels, 1, 1).to(self.device)

        dx_filter = torch.Tensor(dx_filter)
        dx_filter = dx_filter.view(1, 1, kernel_size, kernel_size).repeat(in_channels, in_channels, 1, 1).to(self.device)

        dy_filter = torch.Tensor(dy_filter)
        dy_filter = dy_filter.view(1, 1, kernel_size, kernel_size).repeat(in_channels, in_channels, 1, 1).to(self.device)

        return dx_filter, dy_filter, g

    def get_smoothed(self, tensor):
        """ _get_smoothed(), do convolution with input image
        Args:
            tensor (FloatTensor): of size [1,c,h,w]
        Returns:
            tl (FloatTensor): smoothed image
        """
        tl = F.conv2d(tensor, self.g, padding=self.g.size(-1)//2)
        return tl


def expmr(mtr):
    n = np.array([mtr[2,1],mtr[0,2],mtr[1,0]])
    phi=np.sqrt(np.sum(n**2))
    if phi==0:
        R=np.eye(3)
    else:
        #Sv=thetaS/Sn
        #Nx=cross_matrix(Sv)
        Nx = mtr/phi
        R=np.eye(3)+Nx*np.sin(phi)+Nx.dot(Nx).dot(1-np.cos(phi))
    return R

def cross_matrix(theta):
    return np.array([[    0,    -theta[2], theta[1]],
                     [ theta[2],        0,-theta[0]],
                     [-theta[1], theta[0],       0]])


def logmr(R):
    """
    Matrix logarithm for rotation matrices
    """
    n=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    # n has length 2sin(phi)
    nn=np.sqrt(np.sum(n.flatten()**2))
    if (nn>0):
        # Make axis n a unit vector
        n=n/nn
        # tr(R) = 1+2cos(phi)
        tn=R[0,0]+R[1,1]+R[2,2]-1
        phi=np.arctan2(nn,tn);
    else:
        phi=0

    # Axis-angle vector
    # return n*phi;
    return cross_matrix(n*phi)

def apply_lidar_noise(points_3d, R, t, axis_aligned_noise):
    cam_3d = points_3d@R.T + t.T
    np3 = cam_3d / torch.norm(cam_3d, dim=0)
    new_x = torch.cross(torch.tensor([[0., 1., 0.]]), np3)
    new_x = new_x / torch.norm(new_x, dim=1, keepdim=True)
    new_y = torch.cross(np3, new_x)
    new_y = new_y / torch.norm(new_y, dim=1, keepdim=True)
    sys = torch.stack([new_x, new_y, np3], dim=-1)

    sigma = torch.tensor(axis_aligned_noise).diag().float()
    m = MultivariateNormal(torch.zeros(3), sigma)
    cnoise = m.sample([points_3d.shape[0]])

    new_noise = torch.einsum('bij,bi->bi', sys, cnoise)

    #out = ((cam_3d + new_noise) - t.T)@R
    out = points_3d + new_noise@R

    return out

image_size = (16, 16)
render_factor = 300
num_test_imgs = 0
num_imgs = 60
alpha = 0.25
#min_scale, max_scale = (1.6, 3.5)
min_scale, max_scale = (1.0, 3.0)
out_size = (np.array(image_size)*100).tolist()
in_channels = 1
sigma = 6.75
kernel_size = 11
max_alpha12 = 15     # degrees
min_alpha12 = -15    # degrees
max_alpha3 = 5     # degrees
min_alpha3 = -5    # degrees
plot_corners = False

image = np.zeros((2, image_size[1]))
image[0, ::2] = 1
image[1, 1::2] = 1
image = torch.from_numpy(image)
image = image.repeat(image_size[0] // 2, 1).float()

# corner coordinates
corners_x = torch.linspace(-1, 1, image.shape[1] + 1)
corners_y = torch.linspace(-1, 1, image.shape[0] + 1)
corners_x, corners_y = torch.meshgrid(corners_x[1:-1], corners_y[1:-1])
corners = torch.stack([corners_x, corners_y, torch.zeros_like(corners_y), torch.ones_like(corners_y)], dim=-1)

coners_noise_params = (0., 0.0004)
corners_noise = torch.stack([corners_x, corners_y, torch.zeros_like(corners_y)], dim=-1)
#corners_noise += torch.normal(coners_noise_params[0], coners_noise_params[1], size=corners_noise.shape)
corners_noise = corners_noise.view(-1, 3)

# image grid coordinates
gx = torch.linspace(-1, 1, image.shape[1]*render_factor)
gy = torch.linspace(-1, 1, image.shape[0]*render_factor)
grid_x, grid_y = torch.meshgrid(gx, gy)
grid = torch.stack([grid_x, grid_y], dim=-1).float()
#grid = torch.stack([grid_x, grid_y, torch.zeros_like(grid_y)], dim=-1).float()
height, width, _ = grid.shape
#a = grid[:,:] - corners[3:4,3:4]
#na = torch.norm(a, dim=-1) < 0.01
#grid_d = grid[na]

# smooth class init
smooth = Smoothing(in_channels=in_channels, sigma=sigma, kernel_size=kernel_size, device='cuda')

# sample camera parameters
fx = 1.0
fy = fx
K = np.array([[out_size[0]/2, 0., out_size[0]/2], [0., out_size[1]/2, out_size[1]/2], [0., 0., 1.]])
dist_coeffs = np.array([-0.0684573,  0.0100024, 0.0, 0.0, 0.000603611])
print('dist_coeffs', dist_coeffs)

scale_lin1 = np.linspace(max_scale, min_scale, num_imgs-num_test_imgs)
scale_lin2 = np.linspace(max_scale, min_scale, 10)
alpha_lin = np.linspace(min_alpha12, max_alpha12, num_imgs)


residuals = []
meta = []
matches = []
matches_ideal = []
i = 0
for j in range(num_imgs):
    print(f'Try create board {j}')

    # parameters
    view_point = torch.zeros(4)
    if True:
        # scale = 1.65
        # alpha12 = deg2rad(np.random.rand(2)*(max_alpha12 - min_alpha12) + min_alpha12).tolist()
        # alpha3 = deg2rad(np.random.rand(1)*(max_alpha3 - min_alpha3) + min_alpha3)

        if j < num_imgs-num_test_imgs:
            scale = scale_lin1[j]
            #alpha12 = [deg2rad(alpha_lin[j]), 0.]
            #alpha12 = [deg2rad(0.), 0.]
            #alpha3 = deg2rad(0.0)
            alpha12 = deg2rad(np.random.rand(2)*(max_alpha12 - min_alpha12) + min_alpha12).tolist()
            alpha3 = deg2rad(np.random.rand(1)*(max_alpha3 - min_alpha3) + min_alpha3)
        else:
            scale = scale_lin2[j-(num_imgs-num_test_imgs)]
            alpha12 = [deg2rad(0.), 0.]
            alpha3 = deg2rad(0.0)
    else:
        scale = np.random.rand(1)*(max_scale - min_scale) + min_scale
        alpha12 = deg2rad(np.random.rand(2)*(max_alpha12 - min_alpha12) + min_alpha12).tolist()
        alpha3 = deg2rad(np.random.rand(1)*(max_alpha3 - min_alpha3) + min_alpha3)

    # W2C
    T, R, t, a12_mag = get_projective(alpha12[0], alpha12[1], alpha3, scale)

    # grid distort
    if True:
        grid_cam = grid.view(-1, 2).numpy()
        grid_cam = cv2.undistortPoints(grid_cam, np.eye(3), dist_coeffs)
        grid_cam = torch.from_numpy(grid_cam).squeeze()
        grid_cam = torch.cat([grid_cam, torch.ones(grid_cam.shape[0], 1)], 1)
    else:
        grid_cam = grid

    # grid project
    TH = inv(T[:3, (0, 1, 3)])
    grid_cam = TH@grid_cam.view(-1, 3).transpose(1, 0)
    grid_cam = grid_cam[:3, :]
    grid_cam = grid_cam / grid_cam[-1, :]
    grid_cam = grid_cam.transpose(1, 0)
    grid_cam = grid_cam[:, :2]
    #ch_area = ((grid_cam >= -1) & (grid_cam <= 1)).all(-1).sum()/(width*height)
    #print(f'Checkerboard area: {ch_area}')

    # grid sample
    print('Grid sample')
    grid_cam = grid_cam.view(1, height, width, 2)
    image_warp = grid_sample(image.view(1, 1, *image_size), grid_cam, mode='nearest')
    image_warp = (~image_warp.bool()).float().to('cuda')

    # smoothing image
    print('Smoothing')
    image_warp = smooth.get_smoothed(image_warp)

    # interpolate small
    # mean tensor(0.0008) std tensor(0.0658) var tensor(0.0043)   no noise
    # mean tensor(-0.0015) std tensor(0.0842) var tensor(0.0071)     noise
    image_warp = (interpolate(image_warp, size=out_size, mode='bicubic') + torch.randn([1, 1, *out_size]).to('cuda')*(14.78/255.0)).clamp(0.0, 1.0)
    image_warp = (image_warp*255).byte().to('cpu')

    # ideal corners
    corners_cam = T@corners.view(-1, 4).transpose(1, 0)
    corners_cam = corners_cam[:3, :]
    corners_cam = corners_cam / corners_cam[-1, :]
    corners_cam = corners_cam.transpose(1, 0)
    #corners_cam = corners_cam[:, :2]

    cn = apply_lidar_noise(corners_noise, T[:3, :3], T[:3, -1:], [(scale)*1.6e-7, 0.9*(scale)*1.6e-7, (scale**2)*1.6e-7])

    if ((corners_cam < -1) | (corners_cam > 1)).any():
        print('board outside image')
        continue
    else:
        i += 1

    ch_area = app_board_area(corners_cam[:, :2], out_size[0]*out_size[1])
    print(f'Checkerboard area: {ch_area}')

    # add meta parameters
    meta.append({'R': T[:3, :3].tolist(), 't': T[:3, -1].tolist(), 'K': K.tolist(), 'dist_coeffs': dist_coeffs.tolist(),
                 'a12_mag': a12_mag.item(),
                 'coners_noise_params': coners_noise_params, 'ch_area': ch_area.item(), 'width': out_size[0], 'height': out_size[1]})

    # undistort
    if True:
        corners_cam = corners_cam.numpy()
        corners_cam, _ = cv2.projectPoints(corners_cam,
                                           np.array([[0., 0., 0.]]),  np.array([[0., 0., 0.]]),
                                           np.eye(3), dist_coeffs)
        corners_cam = torch.from_numpy(corners_cam).squeeze()

    save_image(image_warp.float()/255, f'{i:02d}.png')

    # open cv corner detector
    print('OpenCv detect')
    #ideal_corners_cv = (torch.stack([corners_cam[:, 1], corners_cam[:, 0]], dim=-1) + 1) / 2
    #ideal_corners_cv = ideal_corners_cv * torch.tensor([[image_warp.shape[3], image_warp.shape[2]]])
    ideal_corners_cv = torch.stack([corners_cam[:, 1], corners_cam[:, 0], torch.ones_like(corners_cam[:, 0])], dim=-1)@torch.from_numpy(K).float().T
    ideal_corners_cv = ideal_corners_cv[:, :2]

    #image_residuals, points_2d = (None, None)
    image_residuals, cv_points_2d = get_cv_image_residuals(image_warp, ideal_corners_cv)

    if image_residuals is not None:
        residuals.append(image_residuals)
        # sigma = torch.tensor([(scale)*1.6e-7, 0.9*(scale)*1.6e-7, (scale**2)*1.6e-7]).diag().float()
        # m = MultivariateNormal(torch.zeros(3), sigma)
        # cnoise = m.sample([corners_noise.shape[0]])
        # print(cnoise.max())
        # cn = corners_noise + cnoise

        matches.append({'2D': cv_points_2d.view(-1, 2).tolist(), '3D': cn.tolist(),
                        'C': [[K[0, -1].item(), K[1, -1].item()]]})
        matches_ideal.append({'2D': ideal_corners_cv.view(-1, 2).tolist(), '3D': corners[:, :, :3].view(-1, 3).tolist(),
                              'C': [[K[0, -1].item(), K[1, -1].item()]]})
    else:
        matches.append(None)
        matches_ideal.append(None)

    # plot stuff
    image_warp = image_warp.repeat(1, 3, 1, 1)
    if False:
        ideal_corners = ideal_corners_cv.view(1, -1, 2)
        image_warp = draw_keypoints(image_warp.squeeze(0).byte(), ideal_corners, colors="red", radius=1, width=0).unsqueeze(0)
        cv_points_2d = cv_points_2d.squeeze().view(1, -1, 2)
        image_warp = draw_keypoints(image_warp.squeeze(0).byte(), cv_points_2d, colors="yellow", radius=1, width=0).unsqueeze(0)

    save_image(image_warp.float()/255, f'{i:02d}.png')

residuals = torch.cat(residuals, dim=0)
json.dump(residuals.tolist(), open('render_detector_residuals.json', 'w'))
resuduals = residuals.flatten()
print('mean', residuals.mean(), 'std', residuals.std(), 'var', residuals.var())
json.dump(meta, open('render_cb_meta.json', 'w'))
json.dump(matches, open('render_cb_matches.json', 'w'))
json.dump(matches_ideal, open('render_cb_matches_ideal.json', 'w'))
