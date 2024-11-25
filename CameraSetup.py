import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

class CameraSetup:
    def __init__(self, camera_center, focal_length, image_size, pixel_size):
        """
        初始化相机设置
        """
        self.camera_center = camera_center
        self.focal_length = focal_length
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.pixel_grid = self.generate_pixel_grid()

    def generate_pixel_grid(self):
        """
        生成像素平面的均匀网格 (像素坐标)
        """
        image_width, image_height = self.image_size
        x = np.linspace(-image_width / 2, image_width / 2, image_width) * self.pixel_size
        y = np.linspace(-image_height / 2, image_height / 2, image_height) * self.pixel_size
        xx, yy = np.meshgrid(x, y)
        pixel_points_3d = np.vstack((xx.ravel(), yy.ravel(), np.full(xx.size, -self.focal_length)))  # 位于焦平面上
        return pixel_points_3d

    def transform_points(self, points, extrinsic_matrix):
        """
        将点进行旋转和平移变换
        """
        points_homogeneous = np.vstack((points, np.ones((1, points.shape[1]))))
        transformed_points = extrinsic_matrix @ points_homogeneous
        return transformed_points[:3, :]

    def set_transform(self, translation, euler_angles):
        """
        设置旋转和平移矩阵 (旋转和平移相机)
        """
        rotation_matrix = R.from_euler('xyz', np.deg2rad(euler_angles)).as_matrix()
        extrinsic_matrix = np.hstack((rotation_matrix, translation.reshape(3, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
        self.transformed_camera_center = self.transform_points(self.camera_center.reshape(3, 1), extrinsic_matrix).flatten()
        self.transformed_pixel_grid = self.transform_points(self.pixel_grid, extrinsic_matrix)
        self.rotated_normal_vector = rotation_matrix @ np.array([0, 0, 1])
        self.center_pixel = np.mean(self.transformed_pixel_grid, axis=1)
        return self.transformed_camera_center,self.transformed_pixel_grid,self.center_pixel

    def plot_camera_setup(self):
        """
        绘制相机光心、像素平面和像素点与光心的连线
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 初始相机光心 (未旋转/平移前)
        ax.scatter(*self.camera_center, c='r', marker='o', s=5, label='Original Camera Center')

        # 绘制原始像素平面上的点 (未旋转/平移前)
        ax.scatter(self.pixel_grid[0, :], self.pixel_grid[1, :], self.pixel_grid[2, :], c='b', s=1, label='Original Pixel Points')

        # 变换后的相机光心
        ax.scatter(*self.transformed_camera_center, c='g', marker='o', s=5, label='Transformed Camera Center')

        # 绘制变换后的像素平面上的点
        ax.scatter(self.transformed_pixel_grid[0, :], self.transformed_pixel_grid[1, :], self.transformed_pixel_grid[2, :], c='cyan', s=1, label='Transformed Pixel Points')

        # 绘制法向量
        ax.quiver(self.center_pixel[0], self.center_pixel[1], self.center_pixel[2],
                  self.rotated_normal_vector[0], self.rotated_normal_vector[1], self.rotated_normal_vector[2],
                  length=15, color='k', label='Normal Vector', arrow_length_ratio=0.1)

        # 绘制变换后的相机光心与像素平面的每个点的连线
        for i in range(self.transformed_pixel_grid.shape[1]):
            ax.plot([self.transformed_camera_center[0], self.transformed_pixel_grid[0, i]],
                    [self.transformed_camera_center[1], self.transformed_pixel_grid[1, i]],
                    [self.transformed_camera_center[2], self.transformed_pixel_grid[2, i]], 'g-', lw=0.5)

        # 找到像素平面中心点
        center_pixel = np.mean(self.pixel_grid, axis=1)

        # 绘制原始相机光心与像素平面中心的连线
        ax.plot([self.camera_center[0], center_pixel[0]],
                [self.camera_center[1], center_pixel[1]],
                [self.camera_center[2], center_pixel[2]], 'r-', lw=2, label='Original Optical Axis')

        # 绘制变换后的相机光心与像素平面中心的连线
        ax.plot([self.transformed_camera_center[0], self.center_pixel[0]],
                [self.transformed_camera_center[1], self.center_pixel[1]],
                [self.transformed_camera_center[2], self.center_pixel[2]], 'g-', lw=2, label='Transformed Optical Axis')

        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Camera Setup with Focal Length: {self.focal_length}')
        ax.legend()

        # 设置坐标轴比例相等
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]

        max_range = np.max([x_range, y_range, z_range])

        ax.set_xlim([x_limits[0], x_limits[0] + max_range])
        ax.set_ylim([y_limits[0], y_limits[0] + max_range])
        ax.set_zlim([z_limits[0], z_limits[0] + max_range])

        plt.show()

if __name__=="__main__":

    # 初始化相机设置
    camera_setup = CameraSetup(camera_center=np.array([0, 0, 0]),
                               focal_length=5,
                               image_size=(15, 15),
                               pixel_size=1)

    # 设置旋转和平移矩阵
    translation = np.array([0, 0, 20])
    euler_angles = [180, 45, 0]  # roll, pitch, yaw
    camera_setup.set_transform(translation, euler_angles)

    # 绘制相机设置
    camera_setup.plot_camera_setup()
