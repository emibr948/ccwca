import numpy as np
import torch
import struct
import json
from os.path import join
from os import listdir
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def g_func(dist_coeffs, points_2d):
	k1, k2, k3 = dist_coeffs
	r2 = (points_2d[:, :2]**2).sum(axis=1)
	mtpl = 1 + k1*r2 + k2*(r2**2) + k3*(r2**3)
	mtpl = mtpl.reshape(-1, 1)
	points_2d[:, :2] = points_2d[:, :2] * mtpl
	return points_2d


def reprojection(points_3d, fx, fy, cx, cy, dist_coeffs, R, t):
	K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
	t = t.reshape(1, 3)
	p3d = points_3d@R.T + t
	p3d /= p3d[:, -1:]
	p2d_hat = g_func(dist_coeffs, p3d)
	p2d_hat = p2d_hat@K.T
	p2d_hat = p2d_hat[:, :2]
	return p2d_hat


def load_json(path):
	return json.load(open(path))


def write_pointcloud(filename, xyz_points, rgb_points=None):

	""" creates a .pkl file of the point clouds generated
	"""

	assert xyz_points.shape[1] == 3, 'Input XYZ points should be Nx3 float array'
	if rgb_points is None:
		rgb_points = np.ones(xyz_points.shape).astype(np.uint8) #*255
	assert xyz_points.shape == rgb_points.shape, 'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

	# Write header of .ply file
	fid = open(filename, 'wb')
	fid.write(bytes('ply\n', 'utf-8'))
	fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
	fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
	fid.write(bytes('property float x\n', 'utf-8'))
	fid.write(bytes('property float y\n', 'utf-8'))
	fid.write(bytes('property float z\n', 'utf-8'))
	fid.write(bytes('property uchar red\n', 'utf-8'))
	fid.write(bytes('property uchar green\n', 'utf-8'))
	fid.write(bytes('property uchar blue\n', 'utf-8'))
	fid.write(bytes('end_header\n', 'utf-8'))

	# Write 3D points to .ply file
	for i in range(xyz_points.shape[0]):
		fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
										rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
										rgb_points[i,2].tostring())))
	fid.close()

def plane_fit(coords):
	# barycenter of the points
	# compute centered coordinates
	G = coords.sum(axis=0) / coords.shape[0]

	# run SVD
	u, s, vh = np.linalg.svd(coords - G)

	# unitary normal vector
	# u_norm = vh[0, :]
	return vh[0, :], vh[1, :], vh[2, :]


class MegaSfmDataset(object):

	def __init__(self, scene_path):
		self.scene_path = scene_path

		anno_lst = listdir(join(self.scene_path, 'sparse/manhattan'))
		self.anno_path = join(self.scene_path, 'sparse/manhattan', anno_lst[0])

		self.cameras = {}
		read_cam = np.loadtxt(join(self.anno_path, 'cameras.txt'), delimiter=' ', dtype=np.ndarray)
		if len(read_cam.shape) == 1:
			cam_id, _, width, height, f, cx, cy, k1 = read_cam
			self.cameras.update({int(cam_id): {'width': int(width),
										   'height': int(height),
										   'f': float(f),
										   'cxcy': [float(cx), float(cy)], 'k': float(k1)}})
		else:
			for cam in read_cam:
				cam_id, _, width, height, f, cx, cy, k1 = cam
				self.cameras.update({int(cam_id): {'width': int(width),
											  'height': int(height),
											  'f': float(f),
											  'cxcy': [float(cx), float(cy)], 'k': float(k1)}})

		self.images = {}
		read_images = np.genfromtxt(join(self.anno_path, 'images.txt'), delimiter='\t', dtype=np.ndarray)
		#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
		#   POINTS2D[] as (X, Y, POINT3D_ID)
		for img_info, coords in zip(*[iter(read_images)]*2):
			img_info = img_info.split(b' ')
			img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, filename = img_info
			img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, filename = int(img_id), float(qw), float(qx), float(qy), \
																   float(qz), float(tx), float(ty), float(tz), \
																   int(cam_id), filename.decode('utf8')

			coords = np.fromstring(coords, dtype=float, sep=' ')
			coords = coords.reshape(-1, 3)
			point3d_id = coords[:, -1].astype(int)
			coords = coords[:, :2]
			#R = Rotation.from_quat([qx, qy, qz, qw]).as_dcm()
			R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
			self.images.update({filename.split('.')[0]: {'R': R.tolist(),
										   't': [tx, ty, tz],
										   'img_id': img_id,
										   'cam_id': cam_id,
										   'points2d': coords.tolist(),
										   'points3d_id': point3d_id.tolist()}})

		self.points_3d_track = {}
		self.points_3d = []
		self.points_3d_rgb = []
		read_points_3d = np.genfromtxt(join(self.anno_path, 'points3D.txt'), delimiter='\t', dtype=np.ndarray)
		#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
		for p3d in read_points_3d:
			data = np.fromstring(p3d, dtype='float', sep=' ')
			xyz = data[1:4].tolist()
			self.points_3d.append(xyz)
			rgb = data[4:7].tolist()
			self.points_3d_rgb.append(rgb)
			point3d_id = data[0].astype('int').item()
			tracks = data[8:].astype('int').reshape(-1, 2).tolist()
			self.points_3d_track.update({point3d_id: {'tracks': tracks, 'point3d': xyz, 'rgb': rgb}})

		print('initialized dataset')

	def save_point_cloud(self):
		xyz = np.array(self.points_3d).astype('float32')
		rgb = np.array(self.points_3d_rgb).astype('uint8')

		write_pointcloud(join(self.scene_path, 'pointcloud.ply'), xyz, rgb)


# megadeph1500
if True:
	root_path = 'datasets/megadepth/megadepth_sfm_v1/MegaDepth_v1_SfM'
	proj_path = 'projects/ccwca'
	out_path = f'{proj_path}/mega_1500'

	scene_cache = ["0015_0.1_0.3.npz", "0015_0.3_0.5.npz", "0022_0.1_0.3.npz", "0022_0.3_0.5.npz", "0022_0.5_0.7.npz"]
	scene_cache = [np.load(f"{proj_path}/{scene}", allow_pickle=True) for scene in scene_cache]
	scenes1500 = {}
	for scene in scene_cache:
		for img in scene['image_paths']:
			if img is not None:
				_, sc, _, name = img.split('/')
				if sc in scenes1500:
					scenes1500[sc].append(name.split('.')[0])
				else:
					scenes1500.update({sc: [name.split('.')[0]]})

	for scene, files in scenes1500.items():
			print(scene, len(files))
			dataset = MegaSfmDataset(join(root_path, scene))
			di = []
			mi = []
			for fi in files:
					cam_id = dataset.images[fi]['cam_id']
					width = dataset.cameras[cam_id]['width']
					height = dataset.cameras[cam_id]['height']
					cx, cy = dataset.cameras[cam_id]['cxcy']
					f = dataset.cameras[cam_id]['f']
					k = dataset.cameras[cam_id]['k']
					R = dataset.images[fi]['R']
					t = dataset.images[fi]['t']
					img_id = dataset.images[fi]['img_id']
					points3d_id = dataset.images[fi]['points3d_id']
					p2d = []
					p3d = []
					for i, p3d_id in enumerate(points3d_id):
							if p3d_id > -1:
									xy = dataset.images[fi]['points2d'][i]
									xy = [xy[0], xy[1]]
									p2d.append(xy)
									p3d.append(dataset.points_3d_track[p3d_id]['point3d'])

					p2d_hat = reprojection(np.array(p3d), f, f, cx, cy, [k, 0, 0], np.array(R), np.array(t))
					residuals = (p2d_hat - np.array(p2d)).T.tolist()
					di.append({'img': fi, 'k': k, 'f': f, 'R': R, 't': t, 'residuals': residuals})
					mi.append({'img': fi, '2D': p2d, '3D': p3d, 'C': [[cx, cy]]})
			print(len(di), len(mi))
			json.dump(mi, open(join(out_path, f'matches_{scene}.json'), 'w'))
			json.dump({'sfm': {'t1': di}}, open(join(out_path, f'matches_{scene}_sfmres.json'), 'w'))


if False:
	root_path = 'projects/ccwca/results/mega'
	out_path = 'projects/ccwca/results/mega_tmp'

	scenes = listdir(root_path)

	for scene in scenes:
		dataset = MegaSfmDataset(join(root_path, scene))
		test_images = list(dataset.images.keys())[:500]
		di = []
		mi = []
		for ti in test_images:
			cam_id = dataset.images[ti]['cam_id']
			width = dataset.cameras[cam_id]['width']
			height = dataset.cameras[cam_id]['height']
			cx, cy = dataset.cameras[cam_id]['cxcy']
			f = dataset.cameras[cam_id]['f']
			k = dataset.cameras[cam_id]['k']
			R = dataset.images[ti]['R']
			t = dataset.images[ti]['t']
			img_id = dataset.images[ti]['img_id']
			points3d_id = dataset.images[ti]['points3d_id']
			p2d = []
			p3d = []
			for i, p3d_id in enumerate(points3d_id):
				if p3d_id > -1:
					xy = dataset.images[ti]['points2d'][i]
					xy = [xy[0], xy[1]]
					p2d.append(xy)
					p3d.append(dataset.points_3d_track[p3d_id]['point3d'])

			p2d_hat = reprojection(np.array(p3d), f, f, cx, cy, [k, 0, 0], np.array(R), np.array(t))
			residuals = (p2d_hat - np.array(p2d)).T.tolist()
			di.append({'k': k, 'f': f, 'R': R, 't': t, 'residuals': residuals})
			mi.append({'2D': p2d, '3D': p3d, 'C': [[cx, cy]]})

		json.dump(mi, open(join(out_path, f'matches_{scene}.json'), 'w'))
		json.dump({'sfm': {'t1': di}}, open(join(out_path, f'matches_{scene}_sfmres.json'), 'w'))



if False:
	root_path = 'projects/ccwca/results/mega/5005'
	test_images = ['00036.jpg', '00044.jpg', '00121.jpg', '00189.jpg', '00202.jpg', '00236.jpg']
	dataset = MegaSfmDataset(root_path)

	for ti in test_images:
		cam_id = dataset.images[ti]['cam_id']
		width = dataset.cameras[cam_id]['width']
		height = dataset.cameras[cam_id]['height']
		cx, cy = dataset.cameras[cam_id]['cxcy']
		f = dataset.cameras[cam_id]['f']
		k = dataset.cameras[cam_id]['k']
		R = dataset.images[ti]['R']
		t = dataset.images[ti]['t']
		img_id = dataset.images[ti]['img_id']
		points3d_id = dataset.images[ti]['points3d_id']
		p2d = []
		p3d = []
		for i, p3d_id in enumerate(points3d_id):
			if p3d_id > -1:
				xy = dataset.images[ti]['points2d'][i]
				xy = [xy[0], xy[1]]
				p2d.append(xy)
				p3d.append(dataset.points_3d_track[p3d_id]['point3d'])

		p2d_hat = reprojection(np.array(p3d), f, f, cx, cy, [k, 0, 0], np.array(R), np.array(t))
		residuals = (p2d_hat - np.array(p2d)).T.tolist()
		json.dump({'sfm': {'t1': [{'k': k, 'f': f, 'R': R, 't': t, 'residuals': residuals}]}}, open(join(root_path, f'matches_{img_id:05d}_sfmres.json'), 'w'))


if False:
	root_path = 'projects/ccwca/results/mega/5005'
	test_images = ['00036.jpg', '00044.jpg', '00121.jpg', '00189.jpg', '00202.jpg', '00236.jpg']
	dataset = MegaSfmDataset(root_path)

	for ti in test_images:
		cam_id = dataset.images[ti]['cam_id']
		width = dataset.cameras[cam_id]['width']
		height = dataset.cameras[cam_id]['height']
		cx, cy = dataset.cameras[cam_id]['cxcy']
		img_id = dataset.images[ti]['img_id']
		points3d_id = dataset.images[ti]['points3d_id']
		p2d = []
		p3d = []
		for i, p3d_id in enumerate(points3d_id):
			if p3d_id > -1:
				xy = dataset.images[ti]['points2d'][i]
				xy = [xy[0], xy[1]]
				p2d.append(xy)
				p3d.append(dataset.points_3d_track[p3d_id]['point3d'])
		json.dump({'2D': p2d, '3D': p3d, 'C': [[cx, cy]]}, open(join(root_path, f'matches_{img_id:05d}.json'), 'w'))


# estimate of sift distribution using Essential Matrix, not correct!
if False:
	def cross_matrix(theta):
		return np.array([[    0,    -theta[2], theta[1]],
						 [ theta[2],        0,-theta[0]],
						 [-theta[1], theta[0],       0]])

	root_path = 'projects/ccwca/results/mega/5005'
	dataset = MegaSfmDataset(root_path)

	residuals = []
	for data3d in dataset.points_3d_track.values():
		tmp_points = []
		tmp_T = []
		for img_id, point_id in data3d['tracks']:
			img_key = f'{img_id:05d}.jpg'
			cam_id = dataset.images[img_key]['cam_id']
			pt = torch.tensor(dataset.images[img_key]['points2d'][point_id] + [1.]).float()
			cx, cy = dataset.cameras[cam_id]['cxcy']
			f = dataset.cameras[cam_id]['f']
			k = dataset.cameras[cam_id]['k']
			K = torch.tensor([[f, 0., cx],[0., f, cy], [0., 0., 1.]]).float()
			pt = torch.inverse(K)@pt.view(3, 1)
			R = torch.tensor(dataset.images[img_key]['R']).float()
			t = torch.tensor(dataset.images[img_key]['t']).float()
			tmp_points.append(pt)
			tmp_T.append((R, t))

		H = [torch.eye(3).float()]
		R0, t0 = tmp_T[0]
		for i in range(1, len(tmp_T)):
			Rx, tx = tmp_T[i]
			E = torch.from_numpy(cross_matrix(t0)).float()@R0@torch.inverse(Rx)
			H.append(E)

		e_lines = [U@p for p, U in zip(tmp_points, H)]
		point = e_lines[0][:2]

		e_lines = torch.stack(e_lines[1:]).squeeze(-1)
		e_lines = e_lines[:, :2]
		e_lines /= torch.norm(e_lines, 1)
		tmp_res = torch.norm(point.view(1,2) - (point.T*e_lines).sum(1).view(-1, 1)*e_lines, dim=1)
		#tmp_res = ((point/torch.norm(point)).view(1, 3)*e_lines).sum(1).abs()
		residuals.append(tmp_res.flatten())

	residuals = torch.cat(residuals)
	torch.save(residuals, join(root_path, 'sfm_E_sift_distribution.pt'))
	print(f'sfm mean {residuals.mean()}, std {residuals.std()}')

# bad estimate of sift distribution? Not correct!
if False:
	root_path = 'projects/ccwca/results/mega/5005'
	dataset = MegaSfmDataset(root_path)

	residuals = []
	for data3d in dataset.points_3d_track.values():
		tmp_points = []
		tmp_T = []
		for img_id, point_id in data3d['tracks']:
			img_key = f'{img_id:05d}.jpg'
			cam_id = dataset.images[img_key]['cam_id']
			pt = torch.tensor(dataset.images[img_key]['points2d'][point_id] + [1.])
			cx, cy = dataset.cameras[cam_id]['cxcy']
			f = dataset.cameras[cam_id]['f']
			k = dataset.cameras[cam_id]['k']
			K = torch.tensor([[f, 0., cx],[0., f, cy], [0., 0., 1.]])
			pt = torch.inverse(K)@pt.view(3, 1)
			pt = torch.cat([pt, torch.ones(1, 1)], 0)
			R = torch.tensor(dataset.images[img_key]['R'])
			t = torch.tensor(dataset.images[img_key]['t'])
			T = torch.eye(4)
			T[:3, :3] = R
			T[:3, -1] = t
			tmp_points.append(pt)
			tmp_T.append(T)

		H = [torch.eye(4)]
		for i in range(1, len(tmp_T)):
			U = tmp_T[0]@torch.inverse(tmp_T[i])
			H.append(U)

		h_points = [U@p for p, U in zip(tmp_points, H)]
		h_points = torch.stack(h_points)

		tmp_res = h_points.view(-1, 1, 2) - h_points.view(1, -1, 2)
		tidx = torch.tril_indices(tmp_res.shape[0], tmp_res.shape[0],  offset=-1)
		tmp_res = tmp_res[tidx.split(1, 0)].squeeze(0)
		residuals.append(tmp_res.flatten())

	residuals = torch.cat(residuals)
	torch.save(residuals, join(root_path, 'stm_sift_distribution.pt'))
	print(f'sfm mean {residuals.mean()}, std {residuals.std()}')

	#residuals = residuals.tolist()
	#json.dump(residuals, open(join(root_path, 'sfm_sift_distribution.json'), 'w'))


if False:
	dataset = MegaSfmDataset('projects/ccwca/results/mega/5005')
	dataset.save_point_cloud()

# Esitamte noise
if False:
	from plyfile import PlyData
	root_path = 'projects/ccwca/results/mega/5005'
	plane_data = PlyData.read(join(root_path, 'pointcloud_plane.ply'))
	x = plane_data['vertex']['x'].tolist()
	y = plane_data['vertex']['y'].tolist()
	z = plane_data['vertex']['z'].tolist()
	xyz = np.array([x, y, z]).T
	v1, v2, v3 = plane_fit(xyz)
	distances = xyz@v3.reshape(3, 1).flatten()
	print(distances.mean(), distances.std())
