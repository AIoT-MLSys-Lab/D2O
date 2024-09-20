import torch

# 示例张量
tensor_A_1 = torch.randn(1, 8, 1024, 128)
print(tensor_A_1)
tensor_A = tensor_A_1[:, :, None,  :].expand(1, 8, 4 ,1024, 128)
# 变换后的形状为 (1, 8*4, 1048, 128)
print(tensor_A)

tensor_c = tensor_A.reshape(1, 32, 1024, 128)
# 切片操作，变回 (1, 8, 1048, 128)
# print(tensor_c)
# print(tensor_c[:, [0, 4, 8, 12, 16, 20, 24, 28], :])

print("是否一致：", torch.all(torch.eq(tensor_A_1, tensor_c[:, [0, 4, 8, 12, 16, 20, 24, 28], :, :])))
# 检查是否与初始张量一致
# print("是否一致：", torch.all(torch.eq(tensor_A, restored_tensor)))