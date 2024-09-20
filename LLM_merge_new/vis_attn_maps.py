import matplotlib.pyplot as plt
import torch
import seaborn as sns

def visualize_attention_maps(attention_maps, save_directory):
    # 设置Seaborn样式
    sns.set()

    # 遍历每一层的注意力图并进行可视化
    for i, attention_map in enumerate(attention_maps):
        # 创建一个新的图形对象
        fig = plt.figure(figsize=(6, 6))

        # 绘制热力图
        # breakpoint()
        attention_map = attention_map.cpu().mean(1).squeeze(0)
        column_sums = attention_map.sum(dim=0)
        column_var = column_sums.var()
        print(column_var)

        # 使用Seaborn绘制热力图（使用'jet'颜色映射）
        sns.heatmap(attention_map, cmap='Reds', vmin=0, vmax=0.001, square=True, cbar=True)

        # 在图片上方显示方差分布
        plt.text(0.5, 1.05, f'Column Variance: {column_var:.2f}', ha='center', va='bottom', transform=plt.gca().transAxes)

        # 关闭坐标轴
        plt.axis('off')

        # 保存图形为文件
        save_path = f"{save_directory}/coqa_0th_heatmap_{i}.png"
        plt.savefig(save_path)

        # 关闭图形对象
        plt.close(fig)

attention_maps = torch.load('statrack/coqa_promprt_attentions_0.pkl')
# breakpoint()
# attention_maps = torch.load('temp.pkl')
visualize_attention_maps(attention_maps, 'statrack')
