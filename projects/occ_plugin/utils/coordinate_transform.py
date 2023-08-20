
import torch

# @info 将粗糙的体素坐标转换为细粒度的体素坐标，注意这里的体素坐标是体素网格中的坐标，而不是空间坐标
# @param coarse_cor 粗糙的体素坐标
# @param ratio 粗糙体素与细粒度体素的比例，这里是4
# @param topk 细粒度体素的最大数量
def coarse_to_fine_coordinates(coarse_cor, ratio, topk=30000):
    """
    Args:
        coarse_cor (torch.Tensor): [3, N]"""

    # 粗糙体素坐标为什么要乘以ratio？？？
    # 答：因为粗糙体素坐标是在粗糙体素网格中的坐标，而细粒度体素坐标是在细粒度体素网格中的坐标
    # 例如粗糙体素网格的尺寸为[4, 4, 4]，细粒度体素网格的尺寸为[16, 16, 16]，则ratio=4
    fine_cor = coarse_cor * ratio
    # fine_cor[None]表示对fine_cor进行维度扩展，从[3, N]扩展为[1, 3, N]，即在第0维上进行扩展1次
    # repeat(ratio**3, 1, 1)表示在第0维上进行扩展ratio**3次，即64次，以覆盖多个细粒度位置
    fine_cor = fine_cor[None].repeat(ratio**3, 1, 1)  # [64, 3, N]

    device = fine_cor.device
    # 当ratio=4时，相当于是1分64，即将一个粗体素网格划分为64个细体素网格
    # 下面构造这64个小体素网格的局部坐标，从[0, 0, 0]到[3, 3, 3]，总共64个坐标
    value = torch.meshgrid([torch.arange(ratio).to(device), torch.arange(ratio).to(device), torch.arange(ratio).to(device)])
    value = torch.stack(value, dim=3).reshape(-1, 3)

    # 将这64个小体素网格的局部坐标与粗糙体素坐标相加，得到64个小体素网格的全局坐标
    fine_cor = fine_cor + value[:,:,None]

    # 将体素网格坐标恢复为[3, N]的形状
    if fine_cor.shape[-1] < topk:
        return fine_cor.permute(1,0,2).reshape(3,-1)
    else:
        # 如果坐标总数超过了上限topk，则从中随机抽取topk个坐标
        fine_cor = fine_cor[:,:,torch.randperm(fine_cor.shape[-1])[:topk]]
        return fine_cor.permute(1,0,2).reshape(3,-1)


# @info 将体素坐标转换为像素坐标
# @param points 体素坐标，形状是[1, 30000, 3]
# @param rots 外参旋转
# @param trans 外参平移
# @param intrins 内参
# @param post_rots 后处理旋转
# @param post_trans 后处理平移
# @param bda_mat 坐标变换矩阵
# @param pts_range 点云坐标范围
# @param W_img 图像宽度
# @param H_img 图像高度
# @param W_occ 体素宽度
# @param H_occ 体素高度
# @param D_occ 体素深度
# @return points_uv 像素坐标，形状是[1, 30000, 1, 2]
def project_points_on_img(points, rots, trans, intrins, post_rots, post_trans, bda_mat, pts_range,
                        W_img, H_img, W_occ, H_occ, D_occ):
    with torch.no_grad():
        voxel_size = ((pts_range[3:] - pts_range[:3]) / torch.tensor([W_occ-1, H_occ-1, D_occ-1])).to(points.device)
        points = points * voxel_size[None, None] + pts_range[:3][None, None].to(points.device)

        # project 3D point cloud (after bev-aug) onto multi-view images for corresponding 2D coordinates
        inv_bda = bda_mat.inverse()
        points = (inv_bda @ points.unsqueeze(-1)).squeeze(-1)
        
        # from lidar to camera
        points = points.view(-1, 1, 3)
        points = points - trans.view(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / (points_d + 1e-5)

        # points_uv的形状是[1, 30000, 2]
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[..., :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)

        points_uv[..., 0] = (points_uv[..., 0] / (W_img-1) - 0.5) * 2
        points_uv[..., 1] = (points_uv[..., 1] / (H_img-1) - 0.5) * 2

        mask = (points_d[..., 0] > 1e-5) \
            & (points_uv[..., 0] > -1) & (points_uv[..., 0] < 1) \
            & (points_uv[..., 1] > -1) & (points_uv[..., 1] < 1)
    
    # FIXME: points的形状是[1, 30000, 3]，那么points_uv的形状应该是[1, 30000, 2]，但是从返回语句来看，points_uv是一个四维张量，问题出在哪里？？？

    return points_uv.permute(2,1,0,3), mask