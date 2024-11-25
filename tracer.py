import numpy as np
import sympy as sp

"define SUT"
def mirror_func(x, y,k,cv,x_range = (-200, 200),y_range = (-200, 200)):
    z = np.zeros_like(x, dtype=float)*np.nan
    r2 = x * x + y * y
    alpha = 1 - (1 + k) * cv * cv * r2
    z=(cv * r2) / (1 + np.sqrt(alpha))
    return z

def mirror_func1(x, y,k,cv,x_range = (-200, 200),y_range = (-200, 200)):


    z = np.zeros_like(x, dtype=float)*np.nan

    z= 3*np.cos(x/20) + 5*np.sin(y/20)
    return z

"Newton method"
def newton_method(data, obj_position,FD,x_range ,y_range,tol=1e-9, max_iter=1500):
    k=FD['k']
    cv = FD['cv']
    t = np.full(data.shape[0], 10)
    l = data[:, 3]
    m = data[:, 4]
    n = data[:, 5]#
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    for _ in range(max_iter):
        x = x + t * l
        y = y + t * m
        z = z + t * n
        dz = mirror_func(x, y,k,cv,x_range ,y_range) - z
        dz = dz * np.abs(n)
        t = dz / n
        if np.all(np.abs(t) < tol):
            break


    mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
    x[~mask]=np.nan
    y[~mask] = np.nan
    z = mirror_func(x, y, k, cv, x_range, y_range)
    surf_position=np.array([ x, y, z]).T
    return surf_position
"Define normal vector"
def calculate_normal(FD, surf_position,x_range = (-200, 200),y_range = (-200, 200)):
    k = FD['k']
    cv = FD['cv']
    x=surf_position[:,0]
    y=surf_position[:,1]
    r2 = x * x + y * y
    r = np.sqrt(r2)

    alpha = 1.0 - (1.0 + k) * cv * cv * r2
    sqrt_alpha = np.sqrt(alpha)

    mm = (cv * r / (1.0 + sqrt_alpha)) * (2.0 + (cv * cv * r2 * (1.0 + k)) / (alpha * (1.0 + sqrt_alpha)))
    rad = mm / np.sqrt(1.0 + mm * mm)

    ln = (x / r) * rad
    mn = (y / r) * rad

    nn_squared = ln * ln + mn * mn

    mask = nn_squared> 1.0

    nn_squared[mask] =0
    nn = -np.sqrt(1.0 - nn_squared)
    n_surf=np.array([ln,mn,nn]).T

    n_plane = (np.zeros_like(n_surf, dtype=float)+np.array([0,0,1]))
    mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
    n_surf[~mask] = [0,0,1]
    return n_surf

def calculate_normal1(FD,surf_position,x_range = (-200, 200),y_range = (-200, 200)):

    x, y, z = sp.symbols('x y z')
    f = 3*sp.cos(x/20) + 5*sp.sin(y/20) -z


    gradient = sp.Matrix([f.diff(var) for var in (x, y, z)])

    gradient_func = sp.lambdify((x, y, z), gradient, 'numpy')

    x = surf_position[:, 0]
    y = surf_position[:, 1]
    z = surf_position[:, 2]

    grad_values = np.array([gradient_func(x[i], y[i], z[i]) for i in range(len(x))])

    grad_values = np.squeeze(grad_values)


    grad_magnitudes = np.linalg.norm(grad_values, axis=1)


    n_surf = grad_values / grad_magnitudes[:, np.newaxis]


    # n_surf = np.array(grad_values).T
    n_plane = (np.zeros_like(n_surf, dtype=float)+np.array([0,0,1]))

    mask = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
    n_surf[~mask] = [0,0,1]
    return n_surf




"Define reflect ray"
def reflect(incident_ray_position,incident_ray_direction,surf_normal):
    x=incident_ray_position[:,0]
    y=incident_ray_position[:,1]
    z=incident_ray_position[:,2]

    normal=surf_normal

    dot_products = np.sum(incident_ray_direction * normal, axis=1)


    reflected_direction = incident_ray_direction - 2 * dot_products[:, np.newaxis] * normal

    return reflected_direction

"Define ray direction"
def incident_ray_lmn(object_position, pinhole_position, object_size=0, pixel_number=0):
    half_size = object_size // 2

    if object_position.size == 1:
        x = np.linspace(0, object_size, pixel_number) - half_size
        y = np.linspace(0, object_size, pixel_number) - half_size
        z = np.full((pixel_number, pixel_number), object_position)
        X, Y = np.meshgrid(x, y)
        points = np.vstack([X.ravel(), Y.ravel(), z.ravel()]).T
    else:
        points=object_position

    rays =   points -pinhole_position
    norms = np.linalg.norm(rays, axis=1).reshape(-1, 1)
    ray_lmn_normalized = rays / norms

    data = np.hstack([points, ray_lmn_normalized])

    return data

def CCD_incident_ray_lmn(CCD_position, pinhole_position):

    ray_lmn = pinhole_position[:, np.newaxis] - CCD_position
    ray_lmn_normalized = ray_lmn / np.linalg.norm(ray_lmn)

    return ray_lmn_normalized.T

"Define screen points"
def intersection_point(ray_position,ray_direction,screen_z,screen_size):

    screen_position = np.array([0, 0, screen_z])
    screen_normal = np.array([0, 0, 1])


    t = (screen_position[2] - ray_position[:, 2]) / ray_direction[:, 2]
    intersection_points = ray_position[:, :3] + t[:, np.newaxis] * ray_direction
    mask = (intersection_points[:, 0] >= screen_size[0]) & (intersection_points[:, 0] <= screen_size[1]) & (
                intersection_points[:, 1] >= screen_size[0]) & (intersection_points[:, 1] <= screen_size[1])
    intersection_points[~mask] = [0, 0, np.nan]
    return intersection_points
"Define world points"
def intersection_CCDpoint(ray_position,ray_direction,plane_z):
    screen_position = np.array([0, 0, plane_z])
    screen_normal = np.array([0, 0, 1])


    t = (screen_position[2] - ray_position[2]) / ray_direction[:, 2]
    intersection_points = ray_position + t[:, np.newaxis] * ray_direction
    return intersection_points
