import trimesh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist

# ----------------------------
# 读取 STL 文件及预处理
# ----------------------------
mesh = trimesh.load_mesh(r'D:\Desktop\小论文\\小论文7-钢丝腐蚀坑大小与时间的威布尔分布\20250114钢筋\3-11.stl')
plt.rcParams['font.family'] = 'Times New Roman'


def calculate_angle(x, y):
    angle_rad = np.arctan2(y, x)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


vertices = mesh.vertices

# ----------------------------
# 计算每个区间内的中心点及各点到中心点的距离（减去3）
# ----------------------------
new_array = []
z1_values = vertices[:, 2]
z_min = np.min(z1_values)
z_max = np.max(z1_values)
z_bins = np.arange(z_min, z_max, 10)

for i in range(len(z_bins) - 1):
    mask = (z1_values >= z_bins[i]) & (z1_values < z_bins[i + 1])
    selected_points = vertices[mask]
    if selected_points.shape[0] == 0:
        continue
    center_point = np.mean(selected_points, axis=0)
    for vertex in selected_points:
        x1, y1, z1 = vertex
        center_x, center_y, _ = center_point
        distance = np.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2) - 3
        angle = (np.degrees(np.arctan2(y1 - center_y, x1 - center_x)) + 150) % 360
        if angle < 0:
            angle += 360
        new_array.append([z1, angle, distance])

new_array = np.array(new_array)
z1 = new_array[:, 0]
angle = new_array[:, 1]
distance = new_array[:, 2]

# ----------------------------
# 筛选数据并绘制二维散点图（图2）
# ----------------------------
# 筛选深度在 -1 到 -0.5 之间的数据
filtered_array = np.array([item for item in new_array if -1 <= item[2] <= -0.5])
z_filtered = filtered_array[:, 0]
angle_filtered = filtered_array[:, 1]
distance_filtered = filtered_array[:, 2]

norm = plt.Normalize(vmin=-1, vmax=-0.5)
cmap = plt.cm.Greys_r

plt.figure(figsize=(12, 4))
scatter_filtered = plt.scatter(z_filtered * 0.1, angle_filtered, c=distance_filtered,
                               cmap=cmap, norm=norm, s=10)
cbar = plt.colorbar(scatter_filtered)
cbar.set_label('Depth (mm)', fontsize=14)
cbar.ax.tick_params(labelsize=14)
plt.xlabel('Lengths (mm)', fontsize=14)
plt.ylabel('Angle (degrees)', fontsize=14)
plt.title('Points with Depth between -1 and -0.5 (mm)', fontsize=16)
plt.subplots_adjust(bottom=0.25)
plt.show()

# ----------------------------
# 利用网格法提取局部极小值点（代表蚀坑最深处）
# ----------------------------
points = filtered_array  # 每行：[z1, angle, depth]
z_scaled = points[:, 0] * 0.1
angle_points = points[:, 1]
depth = points[:, 2]

num_bins_z = 100
num_bins_angle = 100
z_min_plot, z_max_plot = z_scaled.min(), z_scaled.max()
angle_min, angle_max = angle_points.min(), angle_points.max()
z_bins_grid = np.linspace(z_min_plot, z_max_plot, num_bins_z + 1)
angle_bins_grid = np.linspace(angle_min, angle_max, num_bins_angle + 1)

local_minima_points = []
for i in range(num_bins_z):
    for j in range(num_bins_angle):
        in_bin = (z_scaled >= z_bins_grid[i]) & (z_scaled < z_bins_grid[i + 1]) & \
                 (angle_points >= angle_bins_grid[j]) & (angle_points < angle_bins_grid[j + 1])
        bin_points = points[in_bin]
        if bin_points.shape[0] > 0:
            min_idx = np.argmin(bin_points[:, 2])
            local_minima_points.append(bin_points[min_idx])
local_minima_points = np.array(local_minima_points)

plt.figure(figsize=(12, 4))
scatter_min = plt.scatter(local_minima_points[:, 0] * 0.1, local_minima_points[:, 1],
                          c=local_minima_points[:, 2], cmap=cmap, norm=norm, s=30, edgecolor='k')
cbar = plt.colorbar(scatter_min)
cbar.set_label('Depth (mm)', fontsize=14)
plt.xlabel('Lengths (mm)', fontsize=14)
plt.ylabel('Angle (degrees)', fontsize=14)
plt.title('Local Minimum Depth Points', fontsize=16)
plt.subplots_adjust(bottom=0.25)
plt.show()

# ----------------------------
# 利用 DBSCAN 对局部极小值点进行聚类，确定各蚀坑的中心
# ----------------------------
# 这里将二维坐标定义为： x = z1*0.1, y = angle
X = np.column_stack((local_minima_points[:, 0] * 0.1, local_minima_points[:, 1]))
db = DBSCAN(eps=1.0, min_samples=2).fit(X)
labels = db.labels_
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.discard(-1)


# ----------------------------
# 定义函数：基于从蚀坑中心向外扩展时深度变化的梯度来确定蚀坑边界
# ----------------------------
def compute_pit_diameter_by_gradient(pit_center, data_points, gradient_threshold=0.01, max_radius=20, nbins=20):
    """
    pit_center: (x, y) 蚀坑中心（单位：mm）
    data_points: 原始筛选后的数据，每行 [z1, angle, depth]，其中二维坐标为 x = z1*0.1, y = angle
    gradient_threshold: 当深度梯度低于此值时认为边界到达（单位：mm/mm）
    max_radius: 最大考虑半径（mm）
    nbins: 分箱数
    返回：(pit_diameter, bin_centers, depth_means, gradients)
    """
    x_center, y_center = pit_center
    # 转换二维坐标
    x_points = data_points[:, 0] * 0.1
    y_points = data_points[:, 1]
    depths = data_points[:, 2]

    # 计算各点相对于pit_center的径向距离
    r = np.sqrt((x_points - x_center) ** 2 + (y_points - y_center) ** 2)
    mask = r <= max_radius
    r = r[mask]
    depths = depths[mask]

    if len(r) == 0:
        return None

    # 将半径范围分箱
    bins = np.linspace(0, max_radius, nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    depth_means = []
    for i in range(nbins):
        bin_mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.sum(bin_mask) > 0:
            depth_means.append(np.mean(depths[bin_mask]))
        else:
            depth_means.append(np.nan)
    depth_means = np.array(depth_means)

    # 去除缺失值
    valid = ~np.isnan(depth_means)
    if np.sum(valid) < 2:
        return None
    bin_centers = bin_centers[valid]
    depth_means = depth_means[valid]

    # 计算深度相对于半径的梯度
    gradients = np.gradient(depth_means, bin_centers)
    # 由于蚀坑中心深度最深，向外深度变浅，梯度通常为正
    # 找到第一个梯度低于阈值的bin，认为此处深度变化平缓，即为边界
    idx = np.where(gradients < gradient_threshold)[0]
    if len(idx) == 0:
        pit_radius = bin_centers[-1]
    else:
        pit_radius = bin_centers[idx[0]]
    pit_diameter = pit_radius * 2
    return pit_diameter, bin_centers, depth_means, gradients


# ----------------------------
# 对每个蚀坑计算直径（基于深度梯度平缓处）
# ----------------------------
pit_diameters = {}
for label in unique_labels:
    cluster_points = X[labels == label]
    pit_center = np.mean(cluster_points, axis=0)  # 取局部极小值点聚类的均值作为蚀坑中心
    # 使用原始筛选数据（filtered_array）来构造径向深度分布
    result = compute_pit_diameter_by_gradient(pit_center, filtered_array,
                                              gradient_threshold=0.01, max_radius=20, nbins=20)
    if result is not None:
        pit_diameter, bin_centers, depth_means, gradients = result
        pit_diameters[label] = pit_diameter
        print(f"Pit {label}: diameter = {pit_diameter:.2f} mm")

        # 可选：绘制径向深度分布及梯度示意图（用于调试或观察效果）
        plt.figure()
        plt.plot(bin_centers, depth_means, marker='o', label='Mean Depth')
        plt.plot(bin_centers, gradients, marker='x', label='Gradient')
        plt.axvline(x=pit_diameter / 2, color='r', linestyle='--', label='Pit radius')
        plt.xlabel('Radial Distance (mm)')
        plt.ylabel('Depth (mm) / Gradient (mm/mm)')
        plt.title(f'Pit {label} Radial Profile')
        plt.legend()
        plt.show()
    else:
        print(f"Pit {label}: 无足够数据计算直径")

# ----------------------------
# 可视化各蚀坑及其边界（以计算出的直径标注）
# ----------------------------
plt.figure(figsize=(12, 4))
colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
for label, color in zip(unique_labels, colors):
    cluster_points = X[labels == label]
    pit_center = np.mean(cluster_points, axis=0)
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=color, label=f'Pit {label}', s=30, edgecolor='k')
    if label in pit_diameters:
        plt.text(pit_center[0], pit_center[1], f'{pit_diameters[label]:.1f}',
                 fontsize=12, color='k')
        # 绘制以pit_center为圆心，pit_radius为半径的圆（边界）
        pit_radius = pit_diameters[label] / 2
        theta = np.linspace(0, 2 * np.pi, 100)
        x_circle = pit_center[0] + pit_radius * np.cos(theta)
        y_circle = pit_center[1] + pit_radius * np.sin(theta)
        plt.plot(x_circle, y_circle, color='r', linestyle='--')
plt.xlabel('Lengths (mm)', fontsize=14)
plt.ylabel('Angle (degrees)', fontsize=14)
plt.title('Detected Pits with Boundary Based on Depth Gradient', fontsize=16)
plt.legend()
plt.show()
