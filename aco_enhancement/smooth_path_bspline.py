import numpy as np
from scipy.interpolate import splrep, splev

def smooth_path_bspline(path_list, num_samples=None):
    path_arr = np.array(path_list)
    if len(path_arr) < 3: return path_arr
    
    x = path_arr[:, 0]
    y = path_arr[:, 1]

    # parameterize using cumulative distance to handle sharp turns
    dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    dist = np.concatenate(([0], dist))
    t = np.cumsum(dist)

    # cubic b-spline interpolation
    try:
        tck_x = splrep(t, x, k=3, s=0.35) # s controls smoothness
        tck_y = splrep(t, y, k=3, s=0.5)
    except:
        return np.column_stack((x, y))

    if num_samples is None:
        num_samples = len(path_list) * 5 # upsample points for better resolution

    t_new = np.linspace(t[0], t[-1], num_samples)
    x_new = splev(t_new, tck_x)
    y_new = splev(t_new, tck_y)

    return np.column_stack((x_new, y_new))

