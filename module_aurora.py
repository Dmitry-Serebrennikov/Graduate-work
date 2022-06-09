import os
import sys
import cv2
import h5py
import os.path
import aacgmv2
import glob as gl
import numpy as np
import pandas as pd
import datetime as dt
import cartopy.crs as ccrs
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from ellipse import LsqEllipse
import geopy.distance
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings("ignore")

DEBUG = False

IONOSPHERIC_HEIGHT = 300
EARTHS_RADIUS = 6371.228
MIN_LAT = 50.0
RESOLUTION = 0.1
MASK_WIDTH = 1.0

DILATE_KERNEL_SIZE = 7
DILATE_ITERATIONS = 10

CLOSE_WIDTH = 300
CLOSE_HEIGHT = 100

PARAM_FILE_PATH = 'parameters.csv'


def kmeans(X):
    model = KMeans(n_clusters=2).fit(X.reshape(-1, 1))
    threshold = (model.cluster_centers_[0][0]+model.cluster_centers_[1][0])/2
    return threshold


def otsu(X):
    weight = 1.0/len(X)
    y, x = np.histogram(X)
    x = x[:-1]
    threshold = -1000

    max_value = -1

    for i in range(1,len(x)-1):
        pcb = np.sum(y[:i])
        pcf = np.sum(y[i:])
        Wb = pcb * weight
        Wf = pcf * weight
        mub = np.sum(x[:i]*y[:i]) / float(pcb)
        muf = np.sum(x[i:]*y[i:]) / float(pcf)
        value = Wb * Wf * (mub - muf)**2

        if value > max_value:
            threshold = x[i]
            max_value = value

    return threshold


def median(X):
    threshold = np.median(X)
    
    return threshold


def quantile(X, level):
    X_min = X.min()
    X_max = X.max()
    bins = 20
    width = (X_max - X_min) / bins
    y, x = np.histogram(X, range=(X_min, X_max), bins=bins, density=True)

    summa = 0
    for i in range(len(y)):
        summa += y[i]*width
        if summa > level:
            return x[i]

    return 0


def binarize(X, threshold):
    Y = np.zeros(X.shape)
    Y[X >= threshold] = 1

    return Y


def make_ellipse_points(center, width, height, phi):
    t = np.linspace(0, 2*np.pi, 1000)
    ellipse_x = center[0] + width*np.cos(t)*np.cos(phi) - height*np.sin(t)*np.sin(phi)
    ellipse_y = center[1] + width*np.cos(t)*np.sin(phi) + height*np.sin(t)*np.cos(phi)

    ellipse_x = np.append(ellipse_x, ellipse_x[0])
    ellipse_y = np.append(ellipse_y, ellipse_y[0])

    return ellipse_x, ellipse_y


def to_lat_lon(x, y, projection):
    lon_lat = ccrs.Geodetic().transform_point(x, y, projection)
    return lon_lat[1], lon_lat[0]


def get_border_points(time, lats, lons, rotis, threshold, border_type='outer'):
    lats, lons, rotis = convert_latlon(lats, lons, height=IONOSPHERIC_HEIGHT, time=time, values=rotis, method_code='G2A')

    lons_lats = list(zip(lons, lats))

    lons_range = np.arange(-180.0, 180.0-RESOLUTION, RESOLUTION)
    lats_range = np.arange(-90.0, 90.0, RESOLUTION)
    lons_matrix, lats_matrix = np.meshgrid(lons_range, lats_range)

    rotis_matrix = np.zeros(lats_matrix.shape)
    for i in range(len(lons_lats)):
        lon, lat = lons_lats[i]
        x = np.argmin(np.abs(lons_range - lon))
        y = np.argmin(np.abs(lats_range - lat))
        rotis_matrix[y][x] = rotis[i]

    min_id = int((MIN_LAT + 90.0) / RESOLUTION)
    rotis_matrix[:min_id,:] = np.zeros((min_id, rotis_matrix.shape[1]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
    rotis_matrix_aug = cv2.dilate(rotis_matrix, kernel, iterations=DILATE_ITERATIONS)

    ret, bin_matrix = cv2.threshold(rotis_matrix_aug, threshold, 1, cv2.THRESH_BINARY)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_WIDTH, CLOSE_HEIGHT))
    bin_matrix_aug = cv2.morphologyEx(bin_matrix, cv2.MORPH_CLOSE, close_kernel)

    mask_1, mask_2, L = get_mask_and_level(RESOLUTION, MASK_WIDTH)

    if border_type == 'outer':
        mask_upper = mask_2
        mask_lower = mask_1
    else:
        mask_upper = mask_1
        mask_lower = mask_2

    borders_upper = cv2.filter2D(bin_matrix_aug, -1, mask_upper, borderType=cv2.BORDER_CONSTANT)
    _, borders_upper = cv2.threshold(borders_upper, L, 1, cv2.THRESH_BINARY)

    bin_matrix_inv = 1 - bin_matrix_aug
    borders_lower = cv2.filter2D(bin_matrix_inv, -1, mask_lower, borderType=cv2.BORDER_CONSTANT)
    _, borders_lower = cv2.threshold(borders_lower, L, 1, cv2.THRESH_BINARY)

    borders = (borders_upper * borders_lower).astype(dtype=np.int8)

    border_lats = []
    border_lons = []
    for y in range(borders.shape[0]):
        for x in range(borders.shape[1]):
            if borders[y, x] == 1:
                border_lats.append(lats_matrix[y, x])
                border_lons.append(lons_matrix[y, x])

    border_lats = np.array(border_lats)
    border_lons = np.array(border_lons)

    border_lats, border_lons, _ = convert_latlon(border_lats, border_lons, height=IONOSPHERIC_HEIGHT, time=time, method_code='A2G')

    return border_lats, border_lons


def convert_matrix_to_vector(values_matrix, lats_matrix, lons_matrix):
    values_vector = []
    lats_vector = []
    lons_vector = []
    for y in range(values_matrix.shape[0]):
        for x in range(values_matrix.shape[1]):
            if values_matrix[y, x] > 0:
                values_vector.append(values_matrix[y, x])
                lats_vector.append(lats_matrix[y, x])
                lons_vector.append(lons_matrix[y, x])

    return np.array(values_vector), np.array(lats_vector), np.array(lons_vector)


def get_ellipse_parameters(border_lats, border_lons):
    border_lons_lats = np.array(list(zip(border_lons.tolist(), border_lats.tolist())))

    globe = ccrs.Globe(ellipse=None, semimajor_axis=EARTHS_RADIUS*1000, semiminor_axis=EARTHS_RADIUS*1000)
    projection = ccrs.LambertAzimuthalEqualArea(central_latitude=90, central_longitude= -90, globe=globe)

    border_xy = projection.transform_points(ccrs.Geodetic(), border_lons.flatten(), border_lats.flatten())[:,:2]
    border_x = border_xy[:, 0]
    border_y = border_xy[:, 1]

    model = LsqEllipse().fit(border_xy)

    center, width, height, phi = model.as_parameters()

    ellipse_x, ellipse_y = make_ellipse_points(center, width, height, phi)

    ellipse_lons_lats = ccrs.Geodetic().transform_points(projection, ellipse_x.flatten(), ellipse_y.flatten())
    ellipse_lats = ellipse_lons_lats[:,1]
    ellipse_lons = ellipse_lons_lats[:,0]

    ellipse_center_lon_lat = ccrs.Geodetic().transform_point(center[0], center[1], projection)
    ellipse_center_lat = ellipse_center_lon_lat[1]
    ellipse_center_lon = ellipse_center_lon_lat[0]

    ellipse_width_x = center[0] + width*np.cos(phi)
    ellipse_width_y = center[1] + width*np.sin(phi)
    ellipse_width_lat, ellipse_width_lon = to_lat_lon(ellipse_width_x, ellipse_width_y, projection)

    t = np.pi / 2
    ellipse_height_x = center[0] + width*np.cos(t)*np.cos(phi) - height*np.sin(t)*np.sin(phi)
    ellipse_height_y = center[1] + width*np.cos(t)*np.sin(phi) + height*np.sin(t)*np.cos(phi)
    ellipse_height_lat, ellipse_height_lon = to_lat_lon(ellipse_height_x, ellipse_height_y, projection)

    ellipse_width = distance_haversine(ellipse_center_lat, ellipse_center_lon, ellipse_width_lat, ellipse_width_lon)
    ellipse_height = distance_haversine(ellipse_center_lat, ellipse_center_lon, ellipse_height_lat, ellipse_height_lon)

    ellipse_angle = phi/np.pi*180 - 90

    return ellipse_center_lat, ellipse_center_lon, ellipse_width, ellipse_height, ellipse_angle, ellipse_lats, ellipse_lons, border_lats, border_lons


def process_dat_file(file_path, method):

    file_name = file_path.split('/')[-2]+'_'+file_path.split('/')[-1]
    file_name = ' '.join(file_name.split('.')[:-1])

    df = pd.read_fwf(file_path, widths = [9, 8, 8, 8, 8], names=['lon', 'lat', 'roti', '1', '2'])

    time = dt.datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

    lats = df['lat'].to_numpy()
    lons = df['lon'].to_numpy()
    rotis = df['roti'].to_numpy()

    process_data(time, lats, lons, rotis, file_name, method)

    return


def process_hdf5_file(file_path, method, hour_minute_second=''):

    file_name = file_path.split('/')[-1]
    file_name = ' '.join(file_name.split('.')[:-1])

    all_time_steps = False

    try:
        hour = int(hour_minute_second.split(':')[0])
        minute = int(hour_minute_second.split(':')[1])
        second = int(hour_minute_second.split(':')[2])
    except:
        all_time_steps = True

    with h5py.File(file_path, 'r') as f:
        key = 'data'
        if key in f.keys():
            group = f[key]
            dates = list(group.keys())
            first_day = dt.datetime.strptime(dates[0], '%Y-%m-%d %H:%M:%S.%f').day

            for date in dates:

                time = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

                if all_time_steps or (time.day== first_day and time.hour==hour and time.minute==minute and time.second==second):

                    data = list(group[date][()])

                    lats = np.zeros(len(data))
                    lons = np.zeros(len(data))
                    rotis = np.zeros(len(data))

                    for i, value in enumerate(data):
                        lats[i] = value[0]
                        lons[i] = value[1]
                        rotis[i] = value[2]

                    process_data(time, lats, lons, rotis, file_name, method)

    return


def convert_latlon(lats, lons, height=0, time=dt.datetime.now(), values=np.array([]), method_code='G2A'):
    if values.shape[0] == 0:
        values = np.zeros(lats.shape)

    if method_code == 'G2A':
        mlats, mlons, _ = aacgmv2.convert_latlon_arr(lats, lons, height, time, method_code='G2A')
    else:
        mlats, mlons, _ = aacgmv2.convert_latlon_arr(lats, lons, height, time, method_code='A2G')

    lats_nan_ids = np.where(np.isnan(mlats))[0].tolist()
    lons_nan_ids = np.where(np.isnan(mlons))[0].tolist()

    nan_ids = np.array(list(set(lats_nan_ids + lons_nan_ids)))

    try:
        mlats = np.delete(mlats, nan_ids, axis=0)
        mlons = np.delete(mlons, nan_ids, axis=0)
        values = np.delete(values, nan_ids, axis=0)
    except:
        pass

    return mlats, mlons, values


def process_data(time, lats, lons, rotis, file_name, method):
    if method == 'kmeans':
        threshold = kmeans(rotis)
    elif method == 'otsu':
        threshold = otsu(rotis)
    elif method == 'median':
        threshold = median(rotis)
    else:
        threshold = quantile(rotis, 0.99)

    rotis_bin = binarize(rotis, threshold)

    inner_border_lats, inner_border_lons = get_border_points(time, lats, lons, rotis, threshold, border_type='inner')
    inner_ellipse_center_lat, inner_ellipse_center_lon, inner_ellipse_width, inner_ellipse_height, inner_ellipse_angle, inner_ellipse_lats, inner_ellipse_lons, inner_border_lats, inner_border_lons = get_ellipse_parameters(inner_border_lats, inner_border_lons)
    print('inner: %s, %.6f, %.6f, %.0f, %.0f, %.1f' % (file_name, inner_ellipse_center_lat, inner_ellipse_center_lon, inner_ellipse_width, inner_ellipse_height, inner_ellipse_angle))

    outer_border_lats, outer_border_lons = get_border_points(time, lats, lons, rotis, threshold, border_type='outer')
    outer_ellipse_center_lat, outer_ellipse_center_lon, outer_ellipse_width, outer_ellipse_height, outer_ellipse_angle, outer_ellipse_lats, outer_ellipse_lons, outer_border_lats, outer_border_lons = get_ellipse_parameters(outer_border_lats, outer_border_lons)
    print('outer: %s, %.6f, %.6f, %.0f, %.0f, %.1f' % (file_name, outer_ellipse_center_lat, outer_ellipse_center_lon, outer_ellipse_width, outer_ellipse_height, outer_ellipse_angle))

    distance = []
    for j in range(0,len(outer_ellipse_lats),100):
        short_distance = []
        for i in range(0,len(inner_ellipse_lats),100):
            try:
                short_distance.append(distance_haversine(inner_ellipse_lats[i], inner_ellipse_lons[i], outer_ellipse_lats[j], outer_ellipse_lons[j]))
            except:
                pass
        distance.append(np.min(np.array(short_distance)))

    distance_min = np.min(distance)
    distance_modal = stats.mode(distance)[0][0]
    distance_mean = np.mean(distance)
    distance_max = np.max(distance)
    print('%.0f, %.0f, %.0f, %.0f' % (distance_min, distance_modal, distance_mean, distance_max))

    if not os.path.isfile(PARAM_FILE_PATH):
        with open(PARAM_FILE_PATH, 'w') as f:
            f.write('file_name,method,inner_ellipse_center_lat,inner_ellipse_center_lon,inner_ellipse_width,inner_ellipse_height,inner_ellipse_angle,outer_ellipse_center_lat,outer_ellipse_center_lon,outer_ellipse_width,outer_ellipse_height,outer_ellipse_angle,distance_min,distance_modal,distance_mean,distance_max\r\n')

    with open(PARAM_FILE_PATH, 'a') as f:
        f.write('%s, %s, %.6f, %.6f, %.0f, %.0f, %.1f, %.6f, %.6f, %.0f, %.0f, %.1f, %.0f, %.0f, %.0f, %.0f\r\n' % (file_name, method, inner_ellipse_center_lat, inner_ellipse_center_lon, inner_ellipse_width, inner_ellipse_height, inner_ellipse_angle, outer_ellipse_center_lat, outer_ellipse_center_lon, outer_ellipse_width, outer_ellipse_height, outer_ellipse_angle, distance_min, distance_modal, distance_mean, distance_max))

    import matplotlib.pyplot as plt

    projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0)
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=projection)
    ax.set_title('Карта значений ROTI (файл %s, метод %s)'%(file_name, method))
    ax.gridlines(draw_labels=True, dms=True, color='gray', alpha=0.5, linestyle='--')
    plt.scatter(lons, lats, c=rotis, cmap='jet', marker='o', lw=0.5, s=2, transform=ccrs.PlateCarree())
    plt.savefig('images/map-1-%s-%s.png'%(file_name, method), dpi=120)

    projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0)
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=projection)
    ax.set_title('Карта значений ROTI (файл %s, метод %s)'%(file_name, method))
    ax.gridlines(draw_labels=True, dms=True, color='gray', alpha=0.5, linestyle='--')
    plt.scatter(lons, lats, c=rotis_bin, cmap='binary', marker='o', lw=0.5, s=2, transform=ccrs.PlateCarree())
    plt.savefig('images/map-2-%s-%s.png'%(file_name, method), dpi=120)

    projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0)
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=projection)
    ax.set_title('Карта значений ROTI (файл %s, метод %s)'%(file_name, method))
    ax.gridlines(draw_labels=True, dms=True, color='gray', alpha=0.5, linestyle='--')
    plt.scatter(lons, lats, c=rotis_bin, cmap='binary', marker='o', lw=0.5, s=2, transform=ccrs.PlateCarree())
    plt.scatter(inner_border_lons, inner_border_lats, color='red', marker='o', s=1, transform=ccrs.PlateCarree())
    plt.scatter(outer_border_lons, outer_border_lats, color='red', marker='o', s=1, transform=ccrs.PlateCarree())
    plt.savefig('images/map-3-%s-%s.png'%(file_name, method), dpi=120)

    projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0)
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=projection)
    ax.set_title('Карта значений ROTI (файл %s, метод %s)'%(file_name, method))
    ax.gridlines(draw_labels=True, dms=True, color='gray', alpha=0.5, linestyle='--')
    plt.scatter(lons, lats, c=rotis_bin, cmap='binary', marker='o', lw=0.5, s=2, transform=ccrs.PlateCarree())
    plt.scatter(inner_border_lons, inner_border_lats, color='red', marker='o', s=1, transform=ccrs.PlateCarree())
    plt.scatter(outer_border_lons, outer_border_lats, color='red', marker='o', s=1, transform=ccrs.PlateCarree())
    plt.scatter(inner_ellipse_lons, inner_ellipse_lats, color='yellow', s=7, transform=ccrs.PlateCarree())
    plt.scatter(outer_ellipse_lons, outer_ellipse_lats, color='yellow', s=7, transform=ccrs.PlateCarree())
    plt.savefig('images/map-4-%s-%s.png'%(file_name, method), dpi=120)

    projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0)
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=projection)
    ax.set_title('Карта значений ROTI (файл %s, метод %s)'%(file_name, method))
    ax.gridlines(draw_labels=True, dms=True, color='gray', alpha=0.5, linestyle='--')
    plt.scatter(lons, lats, c=rotis, cmap='jet', marker='o', lw=0.5, s=2, transform=ccrs.PlateCarree())
    plt.scatter(inner_ellipse_lons, inner_ellipse_lats, color='yellow', s=7, transform=ccrs.PlateCarree())
    plt.scatter(outer_ellipse_lons, outer_ellipse_lats, color='yellow', s=7, transform=ccrs.PlateCarree())
    plt.savefig('images/map-5-%s-%s.png'%(file_name, method), dpi=120)

    roti_min = rotis.min()
    roti_max = rotis.max()
    y, X = np.histogram(rotis, range=(roti_min, roti_max), density=True)
    X = X[:-1]
    width = X[1] - X[0]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), dpi=120, facecolor='w', edgecolor='k')
    ax.bar(X, y, width=width*0.9, align='center', color='brown', alpha=0.8, zorder=100)
    ax.set_title('Гистограмма распределения ROTI (файл %s, метод %s)'%(file_name, method), fontsize=11)
    ax.set_xlabel('ROTI, TECu/мин')
    ax.set_ylabel('Плотность вероятности')
    ax.axvline(x=threshold, color='grey', linestyle='--', zorder=1000)
    ax.grid(dashes=(5, 2, 1, 2), zorder=0)
    plt.savefig('images/hist-%s-%s.png'%(file_name, method), dpi=120)

    sys.modules.pop('matplotlib.pyplot')
    
    return



def process_file(file_path, method='otsu', hour_minute_second=''):
    file_ext = file_path.split('.')[-1]

    if file_ext == 'dat':
        process_dat_file(file_path, method)
    elif file_ext == 'h5' or file_ext == 'hdf5':
        process_hdf5_file(file_path, method, hour_minute_second=hour_minute_second)
    else:
        print('Unknown file extension %s' % file_ext)

    return


def distance_geopy(ellipse_center_lat, ellipse_center_lon, oval_center_lat, oval_center_lon):
    return geopy.distance.geodesic((ellipse_center_lat, ellipse_center_lon), (oval_center_lat, oval_center_lon)).km


def distance_haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * (EARTHS_RADIUS + IONOSPHERIC_HEIGHT)


def rmse(dX):
    Z = np.zeros(dX.shape)
    rmse = np.sqrt(mean_squared_error(dX, Z))
    return rmse


def get_mask_and_level(resolution, width):
    N = int(width / resolution) // 2 * 2 + 1

    M = N//2
    kernel = np.zeros((N, N))
    ones = np.ones((M, N))
    mask_1 = kernel.copy()
    mask_1[:M, :] = ones

    mask_2 = kernel.copy()
    mask_2[M+1:, :] = ones

    level = int(np.round(0.6*np.sum(mask_1)))

    return mask_1, mask_2, level


def print_help():
    print('usage: python3 aurora.py [-h] [-i [INPUT]] [-m [METHOD]] [-t [TIME]]')
    print()
    print('optional arguments:')
    print('  -h, --help            show this help message and exit')
    print('  -i [INPUT], --input [INPUT]')
    print('                        input file name')
    print('  -m [METHOD], --method [METHOD]')
    print('                        method for determining the ROTI threshold value (otsu, kmeans, median, quantile)')
    print('  -t [HH:MM:SS], --time [HH:MM:SS]')
    print('                        UTC time in format HH:MM:SS')

    return



if __name__ == '__main__':

    try:
        if len(sys.argv)==2 and ('-h' in sys.argv[1] or '--help' in sys.argv[1] or '/?' in sys.argv[1]):
            print_help()
            sys.exit()

        parser = ArgumentParser()
        parser.add_argument('-i', '--input', nargs='?', help='input file path')
        parser.add_argument('-m', '--method', nargs='?', help='method for determining the ROTI threshold value')
        parser.add_argument('-t', '--time', nargs='?', help='UTC time in format HH:MM:SS')
        args = parser.parse_args()

        if args.input and os.path.isfile(args.input):
            file_path = args.input
        else:
            print_help()
            sys.exit()

        if args.method:
            method = args.method
        else:
            method = 'otsu'

        if args.time:
            hour_minute_second = args.time
        else:
            hour_minute_second = ''

        process_file(file_path, method, hour_minute_second=hour_minute_second)

    except KeyboardInterrupt:

        sys.exit()

