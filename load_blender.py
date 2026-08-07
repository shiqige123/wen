import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
#用于加载Blender数据集并对图像和位姿进行预处理的

#平移矩阵，表示沿 z 轴平移 t 单位。即相机的位置沿 z 轴的移动。
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()
#旋转矩阵，表示绕 y 轴旋转 φ 角度。旋转的角度是弧度制，因此要将传入的 φ 转换为弧度。
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()
#转矩阵，表示绕 x 轴旋转 θ 角度，同样需要将 θ 转换为弧度。
rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


#生成相机的位姿矩阵（4x4）
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

#basedir是基路径；half_res是否将图像分辨率缩小默认为False，同时lego数据集中的默认训练图片规格为800*800，当该参数为True，则需要将图片规格转化为400*400
#testskip的值为N，则会从测试集和验证集中挑选 1\n数量作为测试集和验证集，相当于跳跃步长。
#加载Blender渲染数据集，包括训练集、验证集和测试集。其主要功能是读取存储在JSON文件中的数据，并加载对应的图像文件及其位姿矩阵。
def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)   #将训练、验证、测试数据对应的json文件都读入进来，并用一个名为metas的字典存储

    all_imgs = []
    all_poses = []
    counts = [0]
    #循环训练、验证、测试
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA) 所有图像数据，归一化至 [0, 1] 区间，并转换为 float32 类型。
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    #输出结果 imgs:所有图像数据，poses:所有位姿矩阵，render_poses:渲染的位姿矩阵，[H, W, focal]:图像高度、宽度和焦距，i_split:训练、验证、测试集的索引。
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    #根据性能决定是否降低分辨率：True 时，宽、高、焦距减半
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


