import torch
import numpy as np
# print(torch.Tensor([3]))
# print(torch.sqrt(torch.Tensor([3])))

# all_points = torch.Tensor([[3,-1,1],[1,2,3],[2,1,-3]])
# min_coord = all_points.min(dim = -2)
# max_coord = all_points.max(dim = -2)
# print(min_coord)
# print(max_coord)
# gt_lidar_pose = torch.eye(4)
# print(gt_lidar_pose)

# timestamps = torch.zeros((10), dtype=torch.float32)
# print(timestamps)
# angles=[[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],]
# angles = np.pad(angles,((0,0),(0,2)),'constant',constant_values=(0,0))
# print(angles)
# rot = [1,2,3,4]
# quat = torch.Tensor([rot[3],rot[0],rot[1],rot[2]])
# dirs = torch.tensor([[-1.], [1.]])
# dir = dirs[..., None]
# pts_o = torch.tensor([[1,2,3]])
# pts_o = pts_o.repeat(10,1)
# pts_d = torch.tensor([[1,2,3]])
# pts_d = pts_d.repeat(10,1)
# t = (dirs[..., None] - pts_o[:, [0,1,2]]) / pts_d[:, [0,1,2]]
# print(dirs[..., None] - pts_o[:, [0,1,2]])
t =  torch.Tensor([[5,-1,-2,5],[4,-1,5,-2],[5,-1,-2,5],[3,5,5,-3]])
# print(t.clamp(min=0))
# print(t.clamp(min=0).max(dim = 0)[0])
out = torch.unique(t,sorted = False, dim = 0)
print(out)
