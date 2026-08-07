import numpy as np
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original
#_minify 函数用于检查给定目录下是否已经存在指定缩放比例的图像，如果没有，就会使用mogrify工具来批量缩小图像并保存为PNG格式
def _minify(basedir, factors=[], resolutions=[]): #basedir: 作为基目录;factors: 一个列表，表示缩小的因子;resolutions: 一个列表，表示目标分辨率的大小
    needtoload = False #用于判断是否需要加载并处理图像。
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True  #遍历 factors 列表中的每个缩放因子，对于每个因子，构造一个对应的图像目录路径 imgdir，如果该目录不存在，则设置 needtoload 为 True，表示需要加载图像。
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True #遍历 resolutions 列表中的每个分辨率元组 (宽度, 高度)。构造一个对应的图像目录路径 imgdir，如果该目录不存在，则同样设置 needtoload 为 True
    if not needtoload:
        return #如果 needtoload 为 False，说明所有需要的图像目录都已经存在，直接返回，不需要进一步处理。
    
    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images') #构造原始图像目录 imgdir 的路径，
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))] #并列出该目录下所有文件的路径。
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])] #然后筛选出扩展名为 JPG、PNG、JPEG 等常见图像格式的文件
    imgdir_orig = imgdir #保存原始图像目录的路径，用于后续复制图像文件
    
    wd = os.getcwd() #保存当前工作目录 wd，以便在后续操作结束后能够恢复到原始工作目录

    for r in factors + resolutions: #遍历 factors 和 resolutions 的合并列表，对于每个元素：
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r) #如果 r 是一个整数（表示缩小因子），构造一个目录名 name，并设置 resizearg 为缩小的百分比
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0]) #如果 r 是一个元组（表示分辨率），则构造一个目录名 name，并设置 resizearg 为目标分辨率
        imgdir = os.path.join(basedir, name) #构造目标图像目录路径，如果该目录已存在，跳过当前迭代。
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir) #创建目标图像目录。
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)#使用cp命令复制原始图像目录下的所有文件到目标图像目录中
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)]) #构造一个mogrify命令，该命令将图像调整为resizearg指定的尺寸，并将其格式转换为 png。
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True) #打印构造的 mogrify 命令，然后将当前工作目录切换到目标图像目录，执行 mogrify 命令
        os.chdir(wd) #恢复到原始工作目录。
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')#如果原始图像格式不是 PNG，使用rm命令删除原始格式的图像文件，保留 PNG 格式的文件。打印已删除的文件
        print('Done')
            
        
        
#它负责加载图像数据及其相关的姿势（poses）和边界（bounds）信息
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) #使用 np.load 从 basedir 路径下加载姿势和边界数据 (poses_bounds.npy)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) #poses_arr[:, :-2] 提取姿势数据（前面所有列，去掉最后两列），poses 被重新形状化为 (3, 5, N)，表示每个姿势矩阵（3×5）对应一个图像，N 是图像的数量
    bds = poses_arr[:, -2:].transpose([1,0])#poses_arr[:, -2:] 提取边界数据（最后两列），并转置为 bds，包含图像的边界信息
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0] #获取 basedir/images 目录下所有图像文件的路径，过滤出以 JPG、jpg、png 结尾的文件，按文件名排序
    sh = imageio.imread(img0).shape  #选择第一个图像文件（img0），并用 imageio.imread 读取该图像，获取其尺寸（sh），以便后续计算图像的比例和目标大小。
    
    sfx = '' #初始化一个字符串 sfx，用于存储生成的目录名称后缀（例如：_200 或 _1920x1080
    
    if factor is not None:
        sfx = '_{}'.format(factor) #如果提供了 factor 参数，设置 sfx 为 factor 的字符串表示（例如，_2 表示缩小为原来的一半）。
        _minify(basedir, factors=[factor]) #调用 _minify 函数以缩小图像，使用提供的因子进行处理。
        factor = factor
    elif height is not None: #如果提供了 height 参数，计算缩放因子 factor，使得图像的高度为目标 height。宽度 width 会按比例计算
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]]) #调用 _minify 函数使用计算出的分辨率（height, width）来调整图像大小，并设置 sfx 为目标分辨率
        sfx = '_{}x{}'.format(width, height)
    elif width is not None: #如果提供了 width 参数，计算缩放因子 factor，使得图像的宽度为目标 width。高度 height 按比例计算
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]]) #调用 _minify 函数使用计算出的分辨率（height, width）来调整图像大小，并设置 sfx 为目标分辨率
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1 #如果没有提供 factor、height 或 width，则将 factor 设置为 1，表示不进行任何缩放
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return #根据计算出的 sfx 创建图像目录路径 imgdir，如果该目录不存在，打印错误信息并返回。
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] #获取 imgdir 目录下所有图像文件的路径，并过滤出以 JPG、jpg、png 结尾的文件，按文件名排序。
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return #检查加载的图像文件数量与姿势数据中的图像数量是否匹配。如果不匹配，打印错误信息并返回。

    sh = imageio.imread(imgfiles[0]).shape #读取第一张图像的尺寸，并更新 poses 中存储图像尺寸的部分
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) #储的是图像的高度和宽度
    poses[2, 4, :] = poses[2, 4, :] * 1./factor #存储的是缩放因子。
    
    if not load_imgs:
        return poses, bds #如果 load_imgs 为 False，只返回姿势和边界数据，不加载图像。
    
    def imread(f): #定义一个辅助函数 imread，根据文件扩展名来决定是否忽略 gamma 校正。对于 PNG 格式的图像，使用 ignoregamma=True 参数避免色彩空间的转换。
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)   #读取所有图像文件，将每张图像归一化到 [0, 1] 的范围，并将所有图像堆叠成一个数组，形成一个 (height, width, 3, N) 的数组，其中 N 是图像的数量
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x): #这个函数接受一个向量 x，并返回它的单位向量。
    return x / np.linalg.norm(x) #np.linalg.norm(x) 计算向量 x 的模（长度），然后将 x 除以其模，以得到单位向量

def viewmatrix(z, up, pos): #生成一个相机的视图矩阵（也叫做相机变换矩阵），用于将世界坐标系转换到相机坐标系
    vec2 = normalize(z) #对视线方向进行归一化。z是视线方向
    vec1_avg = up #将 up 向量（相机的上方向，y轴）保存为 vec1_avg。
    vec0 = normalize(np.cross(vec1_avg, vec2)) #计算 up 和 z 的叉积，得到相机的右方向（x 轴方向）。
    vec1 = normalize(np.cross(vec2, vec0)) # 计算 z 和 x 轴的叉积，得到相机的真正的上方向（y 轴方向）
    m = np.stack([vec0, vec1, vec2, pos], 1) #将这三个方向向量和位置向量组合成一个视图矩阵 m，矩阵的第一列是右方向，第二列是上方向，第三列是视线方向，第四列是相机位置。
    return m

def ptstocam(pts, c2w): #将世界坐标系中的点 pts 转换到相机坐标系中
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0] #pts - c2w[:3, 3]: 计算点相对于相机的位置，c2w[:3, 3] 提取 c2w 矩阵中的位置部分，表示相机的位置。np.matmul(c2w[:3, :3].T, ...): 将点的位置向量与相机的旋转部分（c2w[:3, :3]）的转置矩阵相乘，将点从世界坐标系转换到相机坐标系
    return tt                                                             #[..., np.newaxis]: 增加一个新的轴，以便进行矩阵乘法。 [... , 0]: 从结果中移除额外的维度，得到最终的三维点坐标。返回结果是一个形状为 (N, 3) 的数组，表示点在相机坐标系中的坐标。

def poses_avg(poses): #计算所有相机姿势的平均姿势，并返回一个 c2w 矩阵，表示将世界坐标系转化为相机坐标系的变换矩阵。

    hwf = poses[0, :3, -1:] #提取第一帧姿势中的焦距和相机尺寸信息（通常为图像的宽、高和焦距）。

    center = poses[:, :3, 3].mean(0) #计算所有相机位置（即 poses 中的前三列第四列部分）的均值，作为相机的中心位置。
    vec2 = normalize(poses[:, :3, 2].sum(0)) #计算所有相机的视线方向（z 轴方向）的平均值，并进行归一化
    up = poses[:, :3, 1].sum(0) #计算所有相机的上方向（y 轴方向）的平均值。
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1) # 使用 viewmatrix 函数生成视图矩阵，并将焦距和相机尺寸信息与视图矩阵合并，生成最终的 c2w 矩阵。
    
    return c2w #返回的 c2w 矩阵表示相机的平均位置和方向。



#render_path_spiral用于生成一个螺旋形的相机路径，围绕一个固定点（通常是场景的中心）进行旋转，并以指定的步长和焦距进行相机位置的变化。
#其目的是生成一系列沿螺旋路径运动的相机视角（render_poses）。这些视角将被用于渲染场景的不同角度，通常是为了生成 3D 场景的多视角图像或视频。
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = [] #创建一个空的列表 render_poses 用来存储每一帧相机的位置和方向
    rads = np.array(list(rads) + [1.]) #这行代码将rads列表中的每个半径值（例如 x, y, z）扩展为三维，且在末尾添加了一个额外的 1。这使得每个相机位置都有一个适应不同轴向的比例
    hwf = c2w[:,4:5] #提取相机的焦距和其他相关参数
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]: #循环计算相机的位置和视角
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)  #计算相机的新位置
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.]))) #计算相机的视线方向
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1)) #生成相机的变换矩阵
    return render_poses
    

#将一组相机姿势（poses）重新定位，使其以某个平均位置为中心
#它首先计算一个平均相机位置，然后使用变换矩阵将所有相机姿势转换到以该参考点为中心的坐标系中。
def recenter_poses(poses):

    poses_ = poses+0 #poses_ 是 poses 数组的一个独立副本，它会用于存储最终的结果。
    bottom = np.reshape([0,0,0,1.], [1,4]) #这行代码创建了一个 1x4 的行向量 [0, 0, 0, 1]，这是一个齐次坐标中的“底部”向量，通常用来表示在齐次坐标中的“平移”部分。在 4x4 的变换矩阵中，这一行通常用于表示平移分量。
    c2w = poses_avg(poses) #计算并返回一个表示所有相机姿势平均位置的变换矩阵
    c2w = np.concatenate([c2w[:3,:4], bottom], -2) #取出 c2w 矩阵的前 3 行和前 4 列，这部分是旋转矩阵（前 3 列）和位移（最后一列）。然后用 bottom 向量将其拼接，得到一个完整的 4x4 的变换矩阵。这样 c2w 就变成了一个 4x4 的变换矩阵，可以直接应用于其他姿势。
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4] #将变换后的姿势矩阵的旋转部分和位移部分（前三列和前四列）赋值给 poses_。poses_ 就保存了已经重新定位的相机姿势。
    poses = poses_
    return poses


#####################

#将相机的姿势（poses）进行变换，使得相机运动轨迹变为一个圆形轨道（如球面运动)
def spherify_poses(poses, bds):
    #将 3x4 的变换矩阵扩展为 4x4 的矩阵的简化函数
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3] #获取光线的方向（旋转矩阵的第三列）
    rays_o = poses[:,:3,3:4] #获取光线的起点（旋转矩阵的第四列）

    def min_line_dist(rays_o, rays_d): #计算所有相机姿势的最小距离点（pt_mindist）。通过最小化从相机位置到光线的垂直距离，来确定所有相机的中心位置。
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1]) #A_i 是一个旋转矩阵，减去光线方向的外积，它用于计算最短路径。
        b_i = -A_i @ rays_o #b_i 是与相机位置的关系，通过矩阵运算得到距离。
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1))) #计算了所有相机位置到原点的平均距离，作为一个半径值。
    
    sc = 1./rad #缩放因子
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    #计算球面上新的相机距离
    centroid = np.mean(poses_reset[:,:3,3], 0) #所有相机位置的中心点
    zh = centroid[2] #该点的 z 坐标。
    radcircle = np.sqrt(rad**2-zh**2) #球面投影的半径
    new_poses = [] #新的相机姿势列表，其中相机沿着一个圆形轨道移动
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    #将 new_poses 和 poses_reset 的最后一列（即相机的平移部分）与原始的平移向量拼接，确保新的姿势矩阵包含正确的齐次坐标。
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

#数据加载函数，用于加载和处理从指定目录 basedir 中获取的相机姿势（poses）、边界（bds）、图像数据（imgs)
def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    
#加载数据
    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0 修正旋转矩阵的顺序
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided 根据 bd_factor 缩放数据
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:# 进行重心对齐
        poses = recenter_poses(poses)
        
    if spherify: #将相机姿势进行球面化，即让相机沿着球面轨迹进行运动。
        poses, render_poses, bds = spherify_poses(poses, bds)

    else: #如果不进行球面化（即 spherify=False），则计算一个平均相机姿势，并基于这个姿势生成一个螺旋路径：
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path  生成螺旋路径的相机姿势
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1) #计算每个相机姿势与平均姿势之间的距离 dists。
    i_test = np.argmin(dists) #i_test 是距离平均姿势最远的相机的索引，作为测试集视图（即验证集
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test



