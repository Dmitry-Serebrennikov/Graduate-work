from argparse import ArgumentParser
import sys
import os

import os
import sys
import h5py
import aacgmv2
import numpy as np
import datetime as dt

import cartopy.crs as ccrs

import matplotlib
import matplotlib.pyplot as plt

import io
import cv2
from sklearn.cluster import KMeans

from scipy.signal import convolve2d
from skimage.measure import EllipseModel




R_EARTH = 6371 # радиус Земли, км
IONOSPHERIC_HEIGHT = 400 # высота ионосферы в однослойной модели, км
R = R_EARTH + IONOSPHERIC_HEIGHT
S = 900 # константа, используется (будет использоваться) при переводе пиксели -> геомагнитные к-ты

FILTER_SIZE = 23
BORDER_SIZE = (FILTER_SIZE - 1) // 2
L = int(np.round(0.68*FILTER_SIZE*BORDER_SIZE, 0))


def preprocess_input(file_name: str, time: str = '00:00'):
    """
    Обработка данных из HDF5-файла
    """

    # открываем HDF5-файл
    with h5py.File(file_name, 'r') as f:

        # задаем название группы данных в файле
        key = 'data'

        # если такая группа данных есть в файле
        if key in f.keys():

            # выбираем группу данных в файле
            group = f[key]

            # загружаем список всех моментов времени
            dates = list(group.keys())

            # перебираем все моменты времени
            time = dt.datetime.strptime(time, '%H:%M')
            for date in dates:

                # преобразуем дату в объект datetime
                file_time = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

                # если текущий момент совпадает с заданным
                if file_time.hour == time.hour and file_time.minute == time.minute:

                    # загружаем в список данные для данного момента времени
                    data = list(group[date][()])

                    # создаем массивы для хранения широт, долгот, высот и значений ROTI
                    lats = np.zeros(len(data))
                    lons = np.zeros(len(data))
                    heights = np.full(lats.shape, IONOSPHERIC_HEIGHT)
                    rotis = np.zeros(len(data))

                    # перебираем все элементы списка с данными
                    for i, value in enumerate(data):
                        # раскладываем значения в три отдельные массива
                        lats[i] = value[0]
                        lons[i] = value[1]
                        rotis[i] = value[2]

                    # освобождаем память
                    del data

                    # преобразуем географические координаты в геомагнитные координаты
                    mlats, mlons, _ = aacgmv2.convert_latlon_arr(lats, lons, IONOSPHERIC_HEIGHT, file_time, method_code='G2A')
                    
                    return mlats, mlons, rotis


def fig2img(fig: matplotlib.figure.Figure) -> np.ndarray:
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw', dpi=100)
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = data.reshape((int(h), int(w), -1))
    return img


def get_rotis_img(
    mlats: np.ndarray, 
    mlons: np.ndarray, 
    rotis: np.ndarray
) -> np.ndarray:
    fig = plt.figure(figsize=(9, 9), dpi=100)
    ax = plt.axes(projection=ccrs.Orthographic(0, 90))
    ax.scatter(mlons, mlats, c=rotis, cmap='jet', marker=',', lw=0.5, s=1, transform=ccrs.PlateCarree())
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    plt.close(fig)
    
    img = fig2img(fig)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def get_kmeans_level(
    rotis: np.ndarray,
    min_value: int = 0,
    max_value: int = 255
) -> int:
    # получаем rotis между 0 и 255 чтобы потом использовать level для изображения. Метод Otsu сразу дает оценку между 0 и 255
    normalized_rotis = (rotis - rotis.max()) / (rotis.min() - rotis.max()) # переделанная minmax нормализация (scaling), т.к. 0 - черный, 255 - белый 
    normalized_rotis = normalized_rotis * (max_value - min_value) + min_value # сделано для универсальности, если значения будут не в отрезке [0;255]
    normalized_rotis = normalized_rotis.reshape((-1, 1))
    
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(normalized_rotis)
    
    # берем минимальное значение кластеров, чтобы получить порог
    kemans_level = kmeans.cluster_centers_.min(axis=0)[0] # т.к. черный -> 0, то min
    return int(kemans_level)


def get_otsu_level(rotis_img: np.ndarray) -> int:
    otsu_level, _ = cv2.threshold(
        rotis_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return otsu_level


def get_level(
    rotis: np.ndarray, rotis_img: np.ndarray,
    method: str = 'otsu'
):
    if method == 'otsu':
        return get_otsu_level(rotis_img)
    return get_kmeans_level(rotis)


if __name__ == '__main__':

    try:
        parser = ArgumentParser()
        parser.add_argument(
            '-i', '--input',
            nargs='?',
            help='Input HDF5 file path',
            default='data/roti_2007_237_-85_86_N_-176_179_E_0ba6.h5'
        )
        parser.add_argument(
            '-t', '--time', 
            nargs='?', 
            help='UTC time in format HH:MM',
            default='00:00'
        )
        parser.add_argument(
            '-m', '--method',
            nargs='?', 
            help='Binarization method',
            default='otsu' # или "kmeans"
        )
        args = parser.parse_args()

        if not os.path.isfile(args.input):
            sys.exit()

        if not args.method in ['otsu', 'kmeans']:
            sys.exit()
    
        file_path = args.input
        time = args.time
        method = args.method

        mlats, mlons, rotis = preprocess_input(
            file_path,
            time
        )
        rotis_img = get_rotis_img(mlats, mlons, rotis)

        level = get_level(rotis, rotis_img, method=method)

        # метод наращивания, делаем через инвертирование изображения, так как метод наращивает белые пиксели, а нам нужны черные
        circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        rotis_img_morph = 255 - cv2.morphologyEx(255 - rotis_img, cv2.MORPH_DILATE, circle_kernel) 
        
        # Отбираем кандидатов для построения эллипса
        rotis_upper_filter = (rotis_img_morph < level).astype('uint')
        rotis_lower_filter = (rotis_img_morph > level).astype('uint')

        upper_kernel = np.zeros((FILTER_SIZE, FILTER_SIZE))
        upper_kernel[0:BORDER_SIZE, :] = 1

        lower_kernel = np.zeros((FILTER_SIZE, FILTER_SIZE))
        lower_kernel[BORDER_SIZE+1:, :] = 1

        conv_upper = convolve2d(rotis_upper_filter, upper_kernel, mode='same', fillvalue=0)
        conv_lower = convolve2d(rotis_lower_filter, lower_kernel, mode='same', fillvalue=0)

        candidates_cond = (conv_upper >= L) & (conv_lower >= L)

        rotis_candidates = cv2.cvtColor(rotis_img_morph, cv2.COLOR_GRAY2BGR)
        rotis_candidates[candidates_cond] = (255, 0, 0)

        # Накладываем эллипс
        candidates = np.argwhere(candidates_cond == True)
        candidates = candidates[:, [1, 0]]

        ellipse = cv2.fitEllipse(candidates)

        rotis_candidates_cvellipse = rotis_candidates.copy()
        rotis_candidates_cvellipse = cv2.ellipse(rotis_candidates_cvellipse, ellipse, (255, 255, 0), 2)

        # блок для перевода из пикселей в MAG
        ## в процессе
        
        # Сохраняем результаты в файлы
        file_name = file_path
        if '/' in file_path:
            file_name = file_path.split('/')[-1]
        matplotlib.image.imsave(
            f'Result-{file_name}-{time}.png',
            rotis_candidates_cvellipse
        )
        # также сохранить параметры эллипса в геомагнитных координатах - в процессе
        

    except KeyboardInterrupt:
        sys.exit()
