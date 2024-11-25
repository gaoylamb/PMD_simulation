from CameraSetup import CameraSetup
from southwell_grid import hfli2q
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tracer import *




"setting"
iter_times=6
x_range = (-40,40)
y_range = (-25,25)

FD = {'k': -1, 'cv': 0.004}
obj_position =0
object_size = 50

image_size=[720,720]
pixel_size=0.0035
focal_length=15


"Initialize CCD settings"
camera_setup = CameraSetup(camera_center=np.array([0, 0, 0]),
                           focal_length=15,
                           image_size=image_size,
                           pixel_size=pixel_size)


translation = np.array([150, 0, 400])
euler_angles = [180, np.arctan(translation[0]/translation[2])*180/np.pi, np.arctan(translation[1]/translation[2])*180/np.pi]  # roll, pitch, yaw
pinhole_position, CCD_position,center_pixel=camera_setup.set_transform(translation, euler_angles)



"Define light propagation"
incident_CCD2pinhole=CCD_incident_ray_lmn(CCD_position, pinhole_position)
obj_intersection_points =intersection_CCDpoint(pinhole_position,incident_CCD2pinhole,obj_position)
incident_pinhole2obj = np.hstack([obj_intersection_points, incident_CCD2pinhole])
surf_position=newton_method(incident_pinhole2obj,obj_position,FD,x_range ,y_range)
n_surf=calculate_normal(FD, surf_position,x_range ,y_range)
reflected_direction = reflect(surf_position,incident_pinhole2obj[:,3:],n_surf)


"caculate screen"
screen_z=100
screen_size=[-500, 500]
screen_intersection_points =intersection_point(surf_position,reflected_direction,screen_z,screen_size)
screen_position = np.array([0, 0, screen_z])
screen_normal = np.array([0, 0, -1])

def get_fringe_on_screen( position_x, frequency, amplitude, screen_position):
    fringe = amplitude * np.sin(2 * np.pi * frequency * (position_x - screen_position[0]))
    fringe_x = fringe.reshape(image_size)
    return fringe_x
screen_pattern_frequency = 1 / 32
screen_pattern_amplitude = 10
fringer_x = get_fringe_on_screen(screen_intersection_points[:, 0], screen_pattern_frequency,
                                                    screen_pattern_amplitude, screen_position)
#################


def show_setup(CCD_position):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(pinhole_position[0], pinhole_position[1], pinhole_position[2], color='yellow', s=10, label='CCD_pinhole')


    CCD_position = CCD_position.T
    ax.plot_surface(CCD_position[:, 0].reshape(image_size), CCD_position[:, 1].reshape(image_size),
                    CCD_position[:, 2].reshape(image_size), alpha=0.2, label='CCD')
    ax.plot([pinhole_position[0], center_pixel[0]],
            [pinhole_position[1], center_pixel[1]],
            [pinhole_position[2], center_pixel[2]], 'r-', lw=2, label='Optical Axis')
    for point in CCD_position:
        x_ccd, y_ccd, z_ccd = point
        ax.plot([pinhole_position[0], x_ccd], [pinhole_position[1], y_ccd], [pinhole_position[2], z_ccd], color='gray',
                alpha=0.1)


    # ax.quiver(surf_position[:, 0], surf_position[:, 1], surf_position[:, 2],
    #           reflected_direction[:, 0], reflected_direction[:, 1], reflected_direction[:, 2],
    #           color='b', length=15, alpha=0.3,normalize=True, label='Reflected direction')

    ax.quiver(surf_position[:, 0], surf_position[:, 1], surf_position[:, 2],
              -n_surf[:, 0], -n_surf[:, 1], -n_surf[:, 2], color='g', length=10, alpha=0.3, normalize=True,
              label='surf  normal direction')


    # ax.plot_surface(obj_intersection_points[:,0].reshape(image_size), obj_intersection_points[:,1].reshape(image_size), obj_intersection_points[:,2].reshape(image_size), color='r', alpha=0.2, label='Reflection Plane')


    ax.plot_surface(surf_position[:, 0].reshape(image_size), surf_position[:, 1].reshape(image_size),
                    surf_position[:, 2].reshape(image_size), color='b', alpha=0.5, label='Reflection Surf')
    # ax.plot_surface(screen_intersection_points[:,0].reshape(image_size), screen_intersection_points[:,1].reshape(image_size), screen_intersection_points[:,2].reshape(image_size), color='r', alpha=0.2, label='Reflection Plane')


    for i in range(np.size(screen_intersection_points, 0)):
        x_obj, y_obj, z_obj = surf_position[i, :]
        x_screen, y_screen, z_screen = screen_intersection_points[i, :]

        ax.plot([x_screen, x_obj], [y_screen, y_obj], [z_screen, z_obj], color='gray', linewidth=1, alpha=0.5)

    for i in range(np.size(screen_intersection_points, 0)):
        x_obj, y_obj, z_obj = surf_position[i, :]
        ax.plot([pinhole_position[0], x_obj], [pinhole_position[1], y_obj], [pinhole_position[2], z_obj], color='red',
                linewidth=1, alpha=0.1)


    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reflected Rays and Intersection with Screen')

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
# show_setup(CCD_position)



################################

def show_newton_surf():
    fig = plt.figure()
    plt.imshow(surf_position[:,2].reshape(image_size[1],image_size[0]))
    plt.colorbar()
    fig = plt.figure()
    plt.imshow((surf_position[:,0].reshape(image_size[1],image_size[0])))
    fig = plt.figure()
    plt.imshow((surf_position[:,1].reshape(image_size[1],image_size[0])))
    plt.colorbar()

    plt.show()

# show_newton_surf()


def iter(pinhole_translation, Point_OnMr_WdCo, Point_InSc_WdCo):
    Opticalcenter = pinhole_translation
    iter = 0
    h_n = np.ones_like(Point_OnMr_WdCo[:, 2])

    # while np.all(np.abs(h_n-Point_OnMr_WdCo[:, 2]) > 1e-6):
    while iter < iter_times:
        z_m2screen = Point_OnMr_WdCo[:, 2] - Point_InSc_WdCo[:, 2]
        z_m2camera = Point_OnMr_WdCo[:, 2] - Opticalcenter[2]
        d_m2screen = np.sqrt(np.power(Point_OnMr_WdCo[:, 0] - Point_InSc_WdCo[:, 0], 2) +
                             np.power(Point_OnMr_WdCo[:, 1] - Point_InSc_WdCo[:, 1], 2) + np.power(z_m2screen, 2))
        d_m2camera = np.sqrt(np.power((Point_OnMr_WdCo[:, 0] - Opticalcenter[0]), 2) +
                             np.power((Point_OnMr_WdCo[:, 1] - Opticalcenter[1]), 2) + np.power(z_m2camera, 2))

        #
        sx = ((Point_OnMr_WdCo[:, 0] - Point_InSc_WdCo[:, 0]) / d_m2screen +
              (Point_OnMr_WdCo[:, 0] - Opticalcenter[0]) / d_m2camera) / \
             ((Point_InSc_WdCo[:, 2] - Point_OnMr_WdCo[:, 2]) / d_m2screen +
              (Opticalcenter[2] - Point_OnMr_WdCo[:, 2]) / d_m2camera)

        sy = ((Point_OnMr_WdCo[:, 1] - Point_InSc_WdCo[:, 1]) / d_m2screen +
              (Point_OnMr_WdCo[:, 1] - Opticalcenter[1]) / d_m2camera) / \
             ((Point_InSc_WdCo[:, 2] - Point_OnMr_WdCo[:, 2]) / d_m2screen +
              (Opticalcenter[2] - Point_OnMr_WdCo[:, 2]) / d_m2camera)

        """integrate"""
        mask = np.zeros_like(z_m2screen.reshape(image_size[1], image_size[0]))
        mask[~np.isnan(z_m2screen.reshape(image_size[1], image_size[0]))] = 1
        """southwell"""
        temp=hfli2q(sx.reshape(image_size[1],image_size[0]),sy.reshape(image_size[1],image_size[0]),Point_OnMr_WdCo[:,0].reshape(image_size[1],image_size[0]),Point_OnMr_WdCo[:,1].reshape(image_size[1],image_size[0]))
        """iteration"""
        REF=surf_position[:, 2].reshape(image_size[1],image_size[0])
        DIS=temp[360,360]-REF[360,360]
        temp=temp-DIS
        h_n = temp.reshape(Point_OnMr_WdCo[:, 2].shape)
        Point_OnMr_WdCo[:, 2] = h_n

        if iter < iter_times-1:
            prob_ray = Point_OnSUT - Opticalcenter
            Point_OnMr_WdCo[:, 0] = (h_n - Opticalcenter[2]) * (prob_ray[:, 0] / prob_ray[:, 2]) + Opticalcenter[0]
            Point_OnMr_WdCo[:, 1] = (h_n - Opticalcenter[2]) * (prob_ray[:, 1] / prob_ray[:, 2]) + Opticalcenter[1]
        iter += 1
        # show_(temp,mask)


    return h_n, Point_OnMr_WdCo,mask


def show_(temp,mask):
    fig = plt.figure()
    plt.imshow(
        temp * mask)
    plt.colorbar()
    plt.title("h")
    plt.show()




Point_OnSUT=np.copy(surf_position)
Point_OnMr_WdCo=np.copy(obj_intersection_points)
h,Point_OnMr_WdCo_C,mask=iter(pinhole_position,Point_OnMr_WdCo,screen_intersection_points)


"plot"
fig = plt.figure()
plt.imshow(
    surf_position[:,2].reshape(image_size[1],image_size[0]))
plt.colorbar()
plt.title("original")


fig = plt.figure()
plt.imshow(
   Point_OnMr_WdCo_C[:,0].reshape(image_size[1],image_size[0])- Point_OnSUT[:,0].reshape(image_size[1],image_size[0]))
plt.colorbar()
plt.title("X vs ")

fig = plt.figure()
plt.imshow(Point_OnSUT[:,0].reshape(image_size[1],image_size[0]))
plt.colorbar()
plt.title("X original ")


fig = plt.figure()
plt.imshow(Point_OnMr_WdCo_C[:,0].reshape(image_size[1],image_size[0]))
plt.colorbar()
plt.title("X NEW ")



error=h.reshape(image_size[1],image_size[0]) - Point_OnSUT[:,2].reshape(image_size[1],image_size[0])

h=h.reshape(image_size[1],image_size[0])


fig = plt.figure()
plt.imshow(\
    h, cmap='viridis', interpolation='nearest',\
    vmin=np.nanmin(surf_position[:,2]), vmax=np.nanmax(surf_position[:,2]))
plt.colorbar()
plt.title("caculate")
pv_value = np.nanmax(error) - np.nanmin(error)
rms_value = np.sqrt(np.nanmean(error**2))

fig = plt.figure()
img = plt.imshow(error, vmin=-2e-3, vmax=5e-3)
plt.title("error")
plt.colorbar()
plt.show()
