import trimesh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# 读取STL文件
mesh = trimesh.load_mesh(r'D:\Desktop\小论文\\小论文7-钢丝腐蚀坑大小与时间的威布尔分布\20250114钢筋\3-11.stl')
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

def calculate_angle(x, y):
    # 计算从原点(0, 0)到点(x, y)的角度（弧度）
    angle_rad = np.arctan2(y, x)

    # 将弧度转换为度数
    angle_deg = np.degrees(angle_rad)

    # 确保角度在 0 到 360 度之间
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg


vertices = mesh.vertices

# 定义z区间，0.01
z_interval = 20

# 创建一个新数组来存储计算结果
new_array = []

# 先将所有z1值提取出来，以便进行分组
z1_values = vertices[:, 2]  # 提取所有点的z坐标

# 获取z1值的最小值和最大值
z_min = np.min(z1_values)
z_max = np.max(z1_values)

# 计算z区间
z_bins = np.arange(z_min, z_max, z_interval)

# 对每个z区间计算中心点和处理每个散点
for i in range(len(z_bins) - 1):
    # 选择当前区间内的点
    mask = (z1_values >= z_bins[i]) & (z1_values < z_bins[i + 1])
    selected_points = vertices[mask]

    if selected_points.shape[0] == 0:
        continue  # 如果该区间没有点，跳过

    # 计算中心点（这里以平均值作为中心点）
    center_point = np.mean(selected_points, axis=0)

    # 计算散点到中心点的距离，以及中心点的角度
    for vertex in selected_points:
        x1, y1, z1 = vertex
        center_x, center_y, center_z = center_point

        # 计算散点到中心点的距离
        distance = np.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2 ) - 3

        # 计算从中心点到散点的角度（基于x, y平面）
        angle = (np.degrees(np.arctan2(y1 - center_y, x1 - center_x)) + 150) % 360
        if angle < 0:
            angle += 360  # 确保角度在0-360之间

        # 新数组的x为z1，y为角度，z为散点到中心点的距离
        new_array.append([z1, angle, distance])



# 将数据转换为numpy数组，方便处理
new_array = np.array(new_array)

# 提取z1, angle, distance
z1 = new_array[:, 0]
angle = new_array[:, 1]
distance = new_array[:, 2]

# 获取所有z坐标
z_coords = vertices[:, 2]

# 获取z轴的范围
z_min = np.min(z_coords)
z_max = np.max(z_coords)

# 去除两端的 30000 个点
num_points_to_remove = 100000
z_sorted = np.sort(z_coords)

# 计算去除后剩下的点的范围
new_z_min = z_sorted[num_points_to_remove]
new_z_max = z_sorted[-num_points_to_remove]

# 筛选出新的点集（去除两端的点）
filtered_vertices = vertices[(z_coords >= new_z_min) & (z_coords <= new_z_max)]

# 设置步长，每隔5单位进行切割
step_size = 1
# 获取新的z坐标范围
z_coords_filtered = filtered_vertices[:, 2]

# 定义一个空列表存储每个切面的中心点
center_points = []

# 对每个切面进行处理
for z_val in np.arange(np.min(z_coords_filtered), np.max(z_coords_filtered), step_size):
    # 筛选出z坐标为z_val的所有点
    sliced_points = filtered_vertices[np.abs(filtered_vertices[:, 2] - z_val) < step_size / 2]  # 限制一个小范围内的点

    if sliced_points.shape[0] > 0:
        # 计算该切面的几何中心
        center = np.mean(sliced_points, axis=0)
        center_points.append(center)

# 将计算得到的中心点转换为numpy数组
center_points = np.array(center_points)

# 可视化部分点和计算得到的中心曲线
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制随机选择的部分点
sample_points = filtered_vertices[::100]  # 每100个点选择一个，减小可视化数据量
ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], c='b', label='Sampled Points')

# 绘制计算得到的中心曲线
ax.plot(center_points[:, 0], center_points[:, 1], center_points[:, 2], c='r', label='Center Line (every 5 units)')

# 设置坐标轴范围，以避免自动缩放
ax.set_xlim([np.min(filtered_vertices[:, 0]), np.max(filtered_vertices[:, 0])])
ax.set_ylim([np.min(filtered_vertices[:, 1]), np.max(filtered_vertices[:, 1])])
ax.set_zlim([np.min(filtered_vertices[:, 2]), np.max(filtered_vertices[:, 2])])

# 设置图形标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()


# 输出部分新数组
print("部分新数组：")
print(new_array[:10])  # 输出前10个结果

color_values = distance

# 创建一个散点图
plt.figure(figsize=(12, 4))

custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_cmap", ["#4A7B77", "#E8D58C", "#F4A666"]
)
# 绘制散点图，横坐标是z1，纵坐标是angle，颜色用color_values表示
scatter = plt.scatter(z1, angle, c=color_values, cmap=custom_cmap, vmin=-1, vmax=0, s=12)

# 添加颜色条
# 添加颜色条并设置字体大小
cbar = plt.colorbar(scatter)
cbar.set_label('Depth (mm)', fontsize=14)  # 调整颜色条标签的字体大小
cbar.ax.tick_params(labelsize=14)  # 颜色条刻度字体大小


# 设置图形的标签
plt.xlabel('Lengths (mm)', fontsize=14)
plt.ylabel('Angle (degrees)', fontsize=14)
plt.title('')

# **调整布局，确保 x 轴标签不会被遮挡**
plt.subplots_adjust(bottom=0.25)

# 显示图形
plt.show()



from scipy.ndimage import minimum_filter

# 设定窗口大小，用于识别局部极小值
window_size = 5  # 例如 5 个点作为局部窗口

# 重新组织数据，以便处理极小值
z_values = np.sort(np.unique(z1))  # 获取唯一的z值
angle_values = np.sort(np.unique(angle))  # 获取唯一的角度值
z_index_map = {z: i for i, z in enumerate(z_values)}  # z值索引映射
angle_index_map = {a: i for i, a in enumerate(angle_values)}  # 角度索引映射

# 创建深度矩阵
depth_matrix = np.full((len(z_values), len(angle_values)), np.inf)  # 先填充极大值

# 填充深度矩阵
for i in range(len(z1)):
    depth_matrix[z_index_map[z1[i]], angle_index_map[angle[i]]] = distance[i]

# 进行局部最小值筛选
local_min = (depth_matrix == minimum_filter(depth_matrix, size=window_size))

# 提取极小值点
z_minima, angle_minima = np.where(local_min)
minima_points = [(z_values[z], angle_values[a], depth_matrix[z, a]) for z, a in zip(z_minima, angle_minima)]

# 提取极小值数据集
min_z1 = np.array([p[0] for p in minima_points])
min_angle = np.array([p[1] for p in minima_points])
min_distance = np.array([p[2] for p in minima_points])

# 绘制局部极小值点
plt.figure(figsize=(12, 4))
plt.scatter(min_z1, min_angle, c='red', s=20, label='Local Minima')

# 设置图形的标签
plt.xlabel('Lengths (mm)', fontsize=14)
plt.ylabel('Angle (degrees)', fontsize=14)
plt.title('Local Minima Points in Depth Map', fontsize=16)
plt.legend()
plt.show()


