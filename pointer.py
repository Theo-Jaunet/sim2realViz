import numpy
from habitat.config.default import get_config

##### From ROS
#color_intrinsic = [607.0703735351562,   0.0,             332.4421691894531,
#                     0.0,             607.4769897460938, 238.6558837890625,
#                     0.0,               0.0,               1.0]
#color_pos = [0.046, 0.019, 0.593]
#color_size = [640, 480]
#
#depth_intrinsic = [382.2403564453125,   0.0,             320.8781433105469,
#                     0.0,             382.2403564453125, 237.9492950439453,
#	             0.0,               0.0,               1.0]
#depth_pos = [0.047, 0.004, 0.593]
#depth_size = [640, 480]
#
#
#
#color_hfov = numpy.degrees(2 * numpy.atan(0.5 * color_size[0] / color_intrinsic[0]))
#color_vfov = numpy.degrees(2 * numpy.atan(0.5 * color_size[1] / color_intrinsic[4]))
#
#depth_hfov = numpy.degrees(2 * numpy.atan(0.5 * depth_size[0] / depth_intrinsic[0]))
#depth_vfov = numpy.degrees(2 * numpy.atan(0.5 * depth_size[1] / depth_intrinsic[4]))
#
#print("color_hfov", color_hfov)
#print("color_vfov", color_vfov)
#print("depth_hfov", depth_hfov)
#print("depth_vfov", depth_vfov)


def get_intrinsics_from_config(cfg, sensor_type):
    """
    Retrieve intrinsic parameters of the rgb or depth camera in habitat simulator.

    :param cfg: configuration of habitat simulator
    :param sensor_type: Which sensor to retrieve parameters for, either 'rgb' or 'depth'
    :type cfg: habitat.Config
    :type sensor_type: str
    :returns: the focal lengths (fx, fy) and optical centers (cx, cy) of the sensor
    :rtype: dict(str: float)
    """
    KNOWN_SENSORS = ("rgb", "depth")
    if sensor_type not in  KNOWN_SENSORS:
        raise ValueError(f"Invalid sensor type '{sensor_type}',"
                         + f" expected one of '{KNOWN_SENSORS}'")

    sensor_cfg = getattr(cfg.SIMULATOR, sensor_type.upper() + "_SENSOR")
    half_w = 0.5 * sensor_cfg.WIDTH
    half_h = 0.5 * sensor_cfg.HEIGHT
    f = half_w / numpy.tan(numpy.radians(0.5 * sensor_cfg.HFOV))
    return {"fx": f, "fy": f, "cx": half_w, "cy": half_h}


def get_rel_sensor_pos_from_config(cfg):
    """
    Retrieve position of the depth sensor relative to the rgb sensor in habitat simulator.

    :param cfg: configuration of habitat simulator
    :type cfg: habitat.Config
    :returns: the position (dx, dy, dz) of the depth sensor relative to the rgb sensor
    :rtype: tuple(float)
    """
    return tuple(d - r for d, r in zip(cfg.SIMULATOR.DEPTH_SENSOR.POSITION,
                                       cfg.SIMULATOR.RGB_SENSOR.POSITION))


def project_depth_pixel_to_rel_pos(pixel_coords, depth, intrinsics):
    """
    Reproject a depth d associated with pixel coordinates (u, v)
    to the coordinates (xr, yr, zr) of the corresponding point in the real world,
    relative to the sensor (x-axis pointing right, y-axis up, z-axis backward).

    :param pixel_coords: (u, v) coordinates of the pixel in the image, in pixels
    :param depth: depth corresponding to this pixel, in meters
    :param intrinsics: focal lengths (fx, fy) and optical centers (cx, cy) of the depth sensor,
        in (fractions of) pixels
    :type pixel_coords: tuple(int)
    :type depth: float
    :type intrinsics: dict(str: float)
    :returns: the (xr, yr, zr) coordinates of the pixel relative to the camera, in meters
    :rtype: tuple(float)
    """
    x = depth * (pixel_coords[0] - intrinsics["cx"]) / intrinsics["fx"]
    y = depth * (pixel_coords[1] - intrinsics["cy"]) / intrinsics["fy"]
    return x, -y, -depth


def tranform_rel_pos_to_abs(rel_pos, sensor_pos, sensor_heading):
    """
    Convert relative coordinates (xr, yr, zr) w.r.t. the sensor
    to absolute coordinates (x, y, z) in the world
    given the sensor position (xs, ys, zs) and heading.

    :returns: the absolute (x, y, z) of the point in the world, in meters
    :rtype: tuple(float)
    """
    x = sensor_pos[0] + numpy.cos(sensor_heading) * rel_pos[0] \
            + numpy.sin(sensor_heading) * rel_pos[2]
    y = sensor_pos[1] + rel_pos[1]
    z = sensor_pos[2] - numpy.sin(sensor_heading) * rel_pos[0] \
            + numpy.cos(sensor_heading) * rel_pos[2]
    return x, y, z


def project_rel_pos_to_pixel(rel_pos, intrinsics):
    u = intrinsics["fx"] * rel_pos[0] / -rel_pos[2] + intrinsics["cx"]
    v = intrinsics["fy"] * rel_pos[1] /  rel_pos[2] + intrinsics["cy"]
    return int(u), int(v)


def project_abs_pos_to_map(pos, map_origin, map_resolution):
    u = (pos[0] - map_origin[0]) / map_resolution[0]
    v = (pos[2] - map_origin[2]) / map_resolution[1]
    return int(u), int(v)


def align_depth_to_rgb(depth_img, delta_pos, depth_intrinsics, rgb_intrinsics, rgb_size):
    # make z point forward and y down
    delta_pos = numpy.array([delta_pos[0], -delta_pos[1], -delta_pos[2]])
    
    # put everything in matrices to handle all computation in matrix operations
    # warn: y is first...
    f = numpy.array([depth_intrinsics["fy"], depth_intrinsics["fx"]])
    c = numpy.array([depth_intrinsics["cy"], depth_intrinsics["cx"]])
    # Get a meshgrid of image coordinates in depth image
    coords = numpy.mgrid[:depth_img.shape[0], :depth_img.shape[1]]

    # compute 3d coordinates relative to depth sensor
    xy = depth_img[None, :, :] * (coords - c[:, None, None]) / f[:, None, None]
    coords_3d = numpy.vstack((xy, depth_img[None, :, :]))

    # move to the frame attached to rgb sensor
    # warn: only translation is currently supported! no rotation
    coords_3d += delta_pos[:, None, None]

    # reproject the 3d points on the rgb image plane
    f = numpy.array([rgb_intrinsics["fy"], rgb_intrinsics["fx"]])
    c = numpy.array([rgb_intrinsics["cy"], rgb_intrinsics["cx"]])
    rgb_coords = f[:, None, None] * coords_3d[:2] / coords_3d[2:] + c[:, None, None]
    rgb_coords = rgb_coords.astype(numpy.int64)

    aligned_depth = numpy.full(rgb_size, numpy.nan)
    for (i, j), d in numpy.ndenumerate(coords_3d[2]):
        i, j = rgb_coords[:, i, j]
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ii = i + di
                jj = j + dj
                if ii < 0 or ii >= rgb_size[0] or jj < 0 or jj >= rgb_size[1]:
                    continue
                if numpy.isnan(aligned_depth[ii, jj]) or d < aligned_depth[ii, jj]:
                    aligned_depth[ii, jj] = d

    return aligned_depth


if __name__ == "__main__":
    import cv2
    import random
    import glob

    cv2.namedWindow("Map")
    cv2.namedWindow("Obs")
    clk_loc = (10,10)
    # def on_mouse(event, x, y, flags, param):
    #     global clk_loc
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         clk_loc = (x, y)

    # cv2.setMouseCallback("Obs", on_mouse)

    topdown = cv2.imread("static/assets/images/map.jpg")
    topdown = cv2.resize(topdown, None, fx=0.5, fy=0.5)
    map_bounds = (-1.6, -1.3, 18.93, 19.81)
    map_height = 0.047
    map_origin = (map_bounds[0], map_height, map_bounds[1])
    map_resolution = ((map_bounds[2] - map_origin[0]) / topdown.shape[1],
                      (map_bounds[3] - map_origin[1]) / topdown.shape[0])

    depth_file = random.choice(glob.glob("data_source/out/sim/expe_man_ctrl/*_depth.jpeg"))
    disp_obs = cv2.imread(depth_file)
    *sensor_pos, deg = (float(s[1:]) for s in depth_file.split('/')[-1].split('_')[:4])
    sensor_heading = numpy.radians(deg)

    rgb_obs = cv2.imread(depth_file.replace("_depth", "_rgb"))

    # Scale depth
    cfg = get_config("sim_config_citi_256_40cm.yaml")
    intrinsics = get_intrinsics_from_config(cfg, "depth")
    rgb_intrinsics = get_intrinsics_from_config(cfg, "rgb")
    rgb_size = (cfg.SIMULATOR.RGB_SENSOR.HEIGHT, cfg.SIMULATOR.RGB_SENSOR.WIDTH)
    delta_pos = get_rel_sensor_pos_from_config(cfg)
    obs = cv2.cvtColor(disp_obs, cv2.COLOR_BGR2GRAY)
    obs = obs.astype(float) / 255 * cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH \
            + cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH

    aligned_depth = align_depth_to_rgb(obs, delta_pos, intrinsics, rgb_intrinsics, rgb_size)
    disp_align = aligned_depth.copy()
    disp_align -= numpy.nanmin(disp_align)
    disp_align /= numpy.nanmax(disp_align)
    cv2.imshow("Aligned depth", disp_align)
    cv2.imshow("RGB", rgb_obs)

    map_pos = project_abs_pos_to_map(sensor_pos, map_origin, map_resolution)
    cv2.circle(topdown, map_pos, 10, (255, 0, 0), -1)
    cv2.line(topdown, map_pos, (int(map_pos[0] - 20 * numpy.sin(sensor_heading)),
                                int(map_pos[1] - 20 * numpy.cos(sensor_heading))),
             (255, 0, 0), 5)

    c = -1
    while c < 0:
        if clk_loc is not None:
            cv2.circle(disp_obs, clk_loc, 5, (0, 0, 255), -1)
            rel_clk_pos = project_depth_pixel_to_rel_pos(clk_loc, obs[clk_loc[1], clk_loc[0]],
                                                         intrinsics)
            clk_pos = tranform_rel_pos_to_abs(rel_clk_pos, sensor_pos, sensor_heading)
            map_pos = project_abs_pos_to_map(clk_pos, map_origin, map_resolution)
            cv2.circle(topdown, map_pos, 10, (0, 0, 255), -1)

        cv2.imshow("Map", topdown)
        cv2.imshow("Obs", disp_obs)
        c = cv2.waitKey(30)
