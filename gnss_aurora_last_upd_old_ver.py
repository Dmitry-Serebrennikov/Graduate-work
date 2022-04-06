import h5py
import numpy as np
import aacgmv2
import datetime
import math

import matplotlib.pyplot as plt

import pandas as pd

import os

from sklearn.cluster import KMeans
from skimage.measure import EllipseModel
from skimage.filters import threshold_otsu

from PIL import Image, ImageFont, ImageDraw

class ellipsis_estimator():
    def __init__(self):
        '''
        Конструктор класса:
        Инициализируются поля:
            Параметры для визуализации данных
            Пути для сохранения чертежей
        Назначение путей:
            STR_MAIN_DATA_PATH - корневой каталог для данных
            STR_INITIAL_DATA_PATH - каталог для изображений с начальными данными
            STR_AREA_EXPANSION_PATH - каталог для изображений с дополненными данными
            STR_CDF_PATH - каталог для изображений с распределением бинаризованных данных
            STR_BINARY_PATH - каталог для изображений с бинаризованными дополненными данными
            STR_PROJECTION_PATH - каталог для изображений с начерченным найденным овалом на проекции
            STR_ALL_PATH - каталог для изображений, объединяющих описанные выше
            STR_KMEANS_COS_PATH - каталог для изображений, визуализирующих объединение данных на кластеры
        '''

        self.cm = plt.cm.get_cmap('bwr')
        self.color_line_plus = '#333333'
        self.color_line_minus = '#00FF00'
        self.color_line_plus_2 = '#FFFF00'
        
        self.STR_CURRENT_PATH = os.getcwd()
        
        STR_MAIN_DATA_FOLDER = 'temp'
        self.STR_MAIN_DATA_PATH = os.path.join(self.STR_CURRENT_PATH, STR_MAIN_DATA_FOLDER)

        STD_SIMURG_FOLDER = os.path.join(self.STR_CURRENT_PATH, 'data')

        STR_INITIAL_DATA = 'initial_data'
        self.STR_INITIAL_DATA_PATH = os.path.join(self.STR_MAIN_DATA_PATH, STR_INITIAL_DATA)

        STR_AREA_EXPANSION = 'area_expansion'
        self.STR_AREA_EXPANSION_PATH = os.path.join(self.STR_MAIN_DATA_PATH, STR_AREA_EXPANSION)

        STR_CDF = 'cdf'
        self.STR_CDF_PATH = os.path.join(self.STR_MAIN_DATA_PATH, STR_CDF)

        STR_BINARY_DATA = 'binary_data'
        self.STR_BINARY_PATH = os.path.join(self.STR_MAIN_DATA_PATH, STR_BINARY_DATA)

        STR_PROJECTION = 'projection'
        self.STR_PROJECTION_PATH = os.path.join(self.STR_MAIN_DATA_PATH, STR_PROJECTION)

        STR_ALL = 'all'
        self.STR_ALL_PATH = os.path.join(self.STR_MAIN_DATA_PATH, STR_ALL)

        STR_KMEANS_COS = 'kmeans_cos'
        self.STR_KMEANS_COS_PATH = os.path.join(self.STR_MAIN_DATA_PATH, STR_KMEANS_COS)

        self.data_folders = [
            self.STR_INITIAL_DATA_PATH,
            self.STR_AREA_EXPANSION_PATH,
            self.STR_CDF_PATH,
            self.STR_BINARY_PATH,
            self.STR_PROJECTION_PATH,
            self.STR_ALL_PATH,
            self.STR_KMEANS_COS_PATH
        ]
        
        for folder in self.data_folders:
            if not(os.path.exists(folder)):
                os.makedirs(folder)
        
    def read_file_and_times(self, file_path):
        '''
        Открытие файла и считывание находящихся в нем записей
        '''
        self.data_file = h5py.File(file_path, 'r')
        self.frame_times = list(self.data_file['data'].keys())
        return self.frame_times
        
    def read_file_data(self, frame_time):
        '''
        Чтение данных из одной записи в файле.
        Поскольку записи имеют ключ - время в формате гггг-мм-дд ЧЧ:ММ:СС.мС,
        то их удобно сохранить для использования при переводе координат.
        '''
        self.frame_time_curr = frame_time
        self.frame_dtime = datetime.datetime.strptime(frame_time, '%Y-%m-%d %H:%M:%S.%f')
        self.frame_name = str(self.frame_dtime.hour) + '-' + str(self.frame_dtime.minute)
        self.geodetic_latitudes = self.data_file['data'][frame_time]['lat']
        self.geodetic_longitudes = self.data_file['data'][frame_time]['lon']
        self.values = self.data_file['data'][frame_time]['vals']
        self.heights = np.array([0]*len(self.values))
        
        geo_data = {'geo_longitudes': self.geodetic_longitudes, 'geo_latitudes': self.geodetic_latitudes, 'values': self.values}
        
        self.df_geo_data = pd.DataFrame(data=geo_data)
        self.df_geo_data = self.df_geo_data.sort_values(by=['geo_longitudes', 'geo_latitudes'], ascending=False, ignore_index=True)
        
    def coordinates_transition_and_augmentation(self):
        '''
        1. Перевод координат из географической в геомагнитную систему координат.
        Используется библиотека AACGMv2
        2. Расширение областей методом усреднения значений в соседних точках.
        3. Составление таблиц с переведенными сырыми данными и с дополненными данными.
        '''
        mag_latitudes, mag_longitudes, _ = aacgmv2.wrapper.get_aacgm_coord_arr(self.geodetic_latitudes, self.geodetic_longitudes, self.heights, dtime=self.frame_dtime)
        self.augmented_data = np.zeros((180, 360))
        self.neighbor_points = np.zeros((180, 360))

        buf_longitudes = []
        buf_latitudes = []
        self.only_averaged_values = []
        only_averages_lons = []
        self.only_averages_lats = []

        for index in range(0, len(mag_latitudes)):
            if not(np.isnan(mag_longitudes[index])) and not(np.isnan(mag_latitudes[index])) and not(np.isnan(self.values[index])):
                longitude_buf = int(mag_longitudes[index])
                buf_longitudes.append(longitude_buf)

                latitude_buf = int(mag_latitudes[index])
                buf_latitudes.append(latitude_buf)

                self.augmented_data[latitude_buf][longitude_buf] += self.values[index]
                self.neighbor_points[latitude_buf][longitude_buf] += 1

        self.data_distribution = []
        
        for latitude in range(-90, 91):
            for longitude in range(-180, 181):
                if self.augmented_data[latitude][longitude] != 0:
                    self.augmented_data[latitude][longitude] /= self.neighbor_points[latitude][longitude]
                    self.data_distribution.append(self.augmented_data[latitude][longitude])
                    self.only_averaged_values.append(self.augmented_data[latitude][longitude])
                    only_averages_lons.append(longitude)
                    self.only_averages_lats.append(latitude)

        mag_latitudes_augmented = np.append(mag_latitudes, self.only_averages_lats, 0)
        mag_longitudes_augmented = np.append(mag_longitudes, only_averages_lons, 0)
        values_augmented = np.append(self.values, self.only_averaged_values, 0)

        self.mag_data = {'mag_longitudes': mag_longitudes, 'mag_latitudes': mag_latitudes, 'values': self.values}
        self.mag_data_augmented = {'mag_longitudes': mag_longitudes_augmented, 'mag_latitudes': mag_latitudes_augmented, 'values': values_augmented}
        
        self.df_mag_data = pd.DataFrame(data=self.mag_data)
        self.df_mag_data = self.df_mag_data.dropna()
        self.df_mag_data = self.df_mag_data.sort_values(by=['mag_longitudes', 'mag_latitudes'], ascending=False, ignore_index=True)

        self.df_mag_data_augmented = pd.DataFrame(data=self.mag_data_augmented)
        self.df_mag_data_augmented = self.df_mag_data_augmented.dropna()
        self.df_mag_data_augmented = self.df_mag_data_augmented.sort_values(by=['mag_longitudes', 'mag_latitudes'], ascending=False, ignore_index=True)
        
        self.average_value = np.mean(self.df_mag_data_augmented['values'])
        cos_lat = list(map(lambda x: np.cos(x*np.pi/180), self.df_mag_data_augmented['mag_latitudes']))
        self.df_mag_data_augmented['cos_lat'] = cos_lat
    
    def data_binarization(self, method='kmeans'):
        '''
        Бинаризация данных.
        На выбор предлагаются два метода:
            1. K-средних
            2. Метод Оцу
        '''
        if method=='kmeans':
            kmeans = KMeans(n_clusters=2).fit(self.df_mag_data_augmented['cos_lat'].to_numpy().reshape(-1, 1))
            data_clusters = kmeans.labels_.tolist()
            self.data_to_segment = self.df_mag_data_augmented['cos_lat'].copy()

            for i in range(0, len(self.df_mag_data_augmented['cos_lat'])):
                self.data_to_segment[i] = data_clusters[i]
        
        elif method=='otsu':
            self.data_to_segment = self.df_mag_data_augmented['cos_lat'].copy()
            thresh = threshold_otsu(self.df_mag_data_augmented['cos_lat'])
            self.data_to_segment = self.df_mag_data_augmented['cos_lat'].to_numpy() < thresh
            self.data_to_segment = self.data_to_segment.astype(int)    
            
        for i in range(0, len(self.data_to_segment), 1):
            latitude = int(self.df_mag_data_augmented['mag_latitudes'][i])
            longitude = int(self.df_mag_data_augmented['mag_longitudes'][i])

            if self.data_to_segment[i] > 0:
                self.data_to_segment[i] = 1
            else:
                self.data_to_segment[i] = -1 

            self.augmented_data[latitude][longitude] = self.data_to_segment[i]
        
        self.df_mag_data_augmented['values_binarized'] = self.data_to_segment
        
            
    def border_estimation(self):
        '''
        1. Оценка границ овала путем наложения конструктивного элемента
           на бинаризованные данные и подсчета точек, равных "1"  и находящихся выше опорной точки
           и равных "-1" и находящихся ниже опорной точки. Если количество точек обоих видов выше
           порогового значения, то точка выбирается как потенциальная граница.
        2. Корректировка полученных границ
        '''
        criterion_up = 12
        criterion_down = 12

        self.line_longitude_plus = []
        self.line_latitude_plus = []

        '''
        Сканирование карты конструктивным элементом
        долготы от 3 до 357 - чтобы не выйти за границы матрицы с аварийным завершением
        широты от 80 до 40 - потому что ниже не стоит искать. Массив чисел большой,
        180*360 = 64800, в поиске три вложенных цикла, что ведет к временной сложности n^3
        '''
        for longitude in range(3, 357):
            temp_latitude = 0
            for latitude in range(80, 40, -1):
                if self.augmented_data[latitude][longitude] > 0:
                    add_flag_1 = 0
                    add_flag_2 = 0

                    for index_x in range(-3, 4):
                        for index_y in range(1, 4):
                            if self.augmented_data[latitude + index_y][longitude + index_x] > 0:
                                add_flag_1 += 1
                            if self.augmented_data[latitude - index_y][longitude + index_x] < 0:
                                add_flag_2 += 1

                        if add_flag_1 >= criterion_up and add_flag_2 >= criterion_down:
                            if temp_latitude == 0:
                                temp_latitude = latitude
                            if temp_latitude != 0:
                                if len(self.line_longitude_plus) > 0:
                                    if abs(temp_latitude - self.line_longitude_plus[-1]) < abs(self.line_longitude_plus[-1] - latitude):
                                        temp_latitude = latitude
            if temp_latitude != 0:
                self.line_longitude_plus.append(longitude)
                self.line_latitude_plus.append(temp_latitude - 1)

        '''
        Спрямление границ.
        number_of_brute_force_points = 8 - количество соседних точек,
        по которым ведется спрямление. Если выборка точек - пот. границ
        по размерам меньше 8 элементов - количество соседних точек уменьшается.
        '''
        number_of_brute_force_points = 8
        number_of_brute_force_points_2 = number_of_brute_force_points
        self.line_longitude_plus_r = []
        self.line_latitude_plus_r = []

        if len(self.line_longitude_plus) > 4:
            self.line_longitude_plus_r.append(self.line_longitude_plus[0])
            self.line_latitude_plus_r.append(self.line_latitude_plus[0])

            '''
            Вычисление разностей по трем измерениям.
            Если найденная разница больше текущей, то присваиваем новое значение разнице.
            '''
            for i in range(1, len(self.line_longitude_plus)):
                if self.line_longitude_plus[i]-self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1] > 0:
                    dx_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.cos((self.line_longitude_plus_r[len(
                         self.line_longitude_plus_r)-1])*np.pi/180)-np.sin((90-self.line_latitude_plus[i])*np.pi/180)*np.cos((self.line_longitude_plus[i])*np.pi/180))

                    dy_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.sin((self.line_longitude_plus_r[len(
                         self.line_longitude_plus_r)-1])*np.pi/180)-np.sin((90-self.line_latitude_plus[i])*np.pi/180)*np.sin((self.line_longitude_plus[i])*np.pi/180))

                    dz_1 = math.fabs(np.sin((self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin((self.line_latitude_plus[i])*np.pi/180))
                    temp_delta = dx_1*dx_1+dy_1*dy_1+dz_1*dz_1

                    for i_2 in range(1, number_of_brute_force_points_2):
                        dx_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.cos((self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
                            (90-self.line_latitude_plus[i+i_2])*np.pi/180)*np.cos((self.line_longitude_plus[i+i_2])*np.pi/180))

                        dy_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.sin((self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
                            (90-self.line_latitude_plus[i+i_2])*np.pi/180)*np.sin((self.line_longitude_plus[i+i_2])*np.pi/180))

                        dz_1 = math.fabs(np.sin((self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin((self.line_latitude_plus[i+i_2])*np.pi/180))

                        if temp_delta > dx_1*dx_1+dy_1*dy_1+dz_1*dz_1:
                            temp_delta = dx_1*dx_1+dy_1*dy_1+dz_1*dz_1

                    for i_2 in range(number_of_brute_force_points_2):
                        dx_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.cos((self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
                            (90-self.line_latitude_plus[i+i_2])*np.pi/180)*np.cos((self.line_longitude_plus[i+i_2])*np.pi/180))

                        dy_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.sin((self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
                            (90-self.line_latitude_plus[i+i_2])*np.pi/180)*np.sin((self.line_longitude_plus[i+i_2])*np.pi/180))

                        dz_1 = math.fabs(np.sin((self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin((self.line_latitude_plus[i+i_2])*np.pi/180))

                        if temp_delta == dx_1*dx_1+dy_1*dy_1+dz_1*dz_1:
                            self.line_longitude_plus_r.append(self.line_longitude_plus[i+i_2])
                            self.line_latitude_plus_r.append(self.line_latitude_plus[i+i_2])

                if i >= len(self.line_longitude_plus)-number_of_brute_force_points_2:
                    number_of_brute_force_points_2 -= 1

            self.line_longitude_plus_r.append(self.line_longitude_plus[len(self.line_longitude_plus)-1])
            self.line_latitude_plus_r.append(self.line_latitude_plus[len(self.line_longitude_plus)-1])
            self.line_longitude_plus_r.append(self.line_longitude_plus[0]+360)
            self.line_latitude_plus_r.append(self.line_latitude_plus[0])

        self.line_longitude_plus_r_2 = []
        self.line_latitude_plus_r_2 = []
        number_of_brute_force_points_2 = number_of_brute_force_points
        if len(self.line_longitude_plus) > 4:
            self.line_longitude_plus_r_2.append(self.line_longitude_plus[0])
            self.line_latitude_plus_r_2.append(self.line_latitude_plus[0])

            for i in range(1, len(self.line_longitude_plus)):
                if  self.line_longitude_plus[i]-self.line_longitude_plus_r_2[len(self.line_longitude_plus_r_2)-1] > 0:
                    dx_1 = math.fabs(self.line_longitude_plus_r_2[len(self.line_longitude_plus_r_2)-1]-self.line_longitude_plus[i])

                    dy_1 = math.fabs(self.line_latitude_plus_r_2[len(self.line_longitude_plus_r_2)-1]-self.line_latitude_plus[i])

                    temp_delta = dx_1*dx_1+dy_1*dy_1
                    for i_2 in range(1, number_of_brute_force_points_2):
                        dx_1 = math.fabs(self.line_longitude_plus_r_2[len(self.line_longitude_plus_r_2)-1]-self.line_longitude_plus[i+i_2])

                        dy_1 = math.fabs(self.line_latitude_plus_r_2[len(self.line_longitude_plus_r_2)-1]-self.line_latitude_plus[i+i_2])

                        if temp_delta > dx_1*dx_1+dy_1*dy_1:
                            temp_delta = dx_1*dx_1+dy_1*dy_1
                    
                    for i_2 in range(number_of_brute_force_points_2):
                        dx_1 = math.fabs(self.line_longitude_plus_r_2[len(self.line_longitude_plus_r_2)-1]-self.line_longitude_plus[i+i_2])

                        dy_1 = math.fabs(self.line_latitude_plus_r_2[len(self.line_longitude_plus_r_2)-1]-self.line_latitude_plus[i+i_2])

                        if temp_delta == dx_1*dx_1+dy_1*dy_1:
                            self.line_longitude_plus_r_2.append(self.line_longitude_plus[i+i_2])
                            self.line_latitude_plus_r_2.append(self.line_latitude_plus[i+i_2])

                if i >= len(self.line_longitude_plus)-number_of_brute_force_points_2:
                    number_of_brute_force_points_2 -= 1

            self.line_longitude_plus_r_2.append(self.line_longitude_plus[len(self.line_longitude_plus)-1])
            self.line_latitude_plus_r_2.append(self.line_latitude_plus[len(self.line_longitude_plus)-1])
            self.line_longitude_plus_r_2.append(self.line_longitude_plus[0]+360)
            self.line_latitude_plus_r_2.append(self.line_latitude_plus[0])

        '''
        Составление сетки координат для проекции.
        '''
        self.x_setka = []
        self.y_setka = []

        self.x_data_r = []
        self.y_data_r = []
        self.data_r_radian = []

        for i in range(len(self.data_to_segment)):
            if self.df_mag_data_augmented['mag_latitudes'][i] >= 0:
                self.x_data_r.append(90*np.sin((90-self.df_mag_data_augmented['mag_latitudes'][i])*np.pi/180)*np.cos((self.df_mag_data_augmented['mag_longitudes'][i])*np.pi/180))    # decimals=0
                self.y_data_r.append(90*np.sin((90-self.df_mag_data_augmented['mag_latitudes'][i])*np.pi/180)*np.sin((self.df_mag_data_augmented['mag_longitudes'][i])*np.pi/180))
                self.data_r_radian.append(self.data_to_segment[i])

        self.x_data_r_plus = []
        self.y_data_r_plus = []
        for i in range(len(self.line_latitude_plus_r) - 1):
                self.x_data_r_plus.append(90*np.sin((90- self.line_latitude_plus_r[i])*np.pi/180)*np.cos((self.line_longitude_plus_r[i])*np.pi/180))     # decimals=0
                self.y_data_r_plus.append(90*np.sin((90- self.line_latitude_plus_r[i])*np.pi/180)*np.sin((self.line_longitude_plus_r[i])*np.pi/180))
                self.x_setka.append(0)
                self.y_setka.append(0)
                self.x_setka.append(80*np.cos((self.line_longitude_plus_r[i])*np.pi/180))
                self.y_setka.append(80*np.sin((self.line_longitude_plus_r[i])*np.pi/180))
                for i_2 in range(1, self.line_longitude_plus_r[i+1]- self.line_longitude_plus_r[i]):
                    dx = self.line_longitude_plus_r[i+1] - self.line_longitude_plus_r[i]
                    dy = self.line_latitude_plus_r[i+1] - self.line_latitude_plus_r[i]

        self.x_data_r_plus_2 = []
        self.y_data_r_plus_2 = []
        for i in range(0, len(self.line_latitude_plus_r_2)-1, 1):
            self.x_data_r_plus_2.append(90*np.sin((90-self.line_latitude_plus_r_2[i])*np.pi/180)*np.cos((self.line_longitude_plus_r_2[i])*np.pi/180))  # decimals=0
            self.y_data_r_plus_2.append(90*np.sin((90-self.line_latitude_plus_r_2[i])*np.pi/180)*np.sin((self.line_longitude_plus_r_2[i])*np.pi/180))
            
            for i_2 in range(1, self.line_longitude_plus_r_2[i+1]-self.line_longitude_plus_r_2[i]):
                dx = self.line_longitude_plus_r_2[i+1] - self.line_longitude_plus_r_2[i]
                dy = self.line_latitude_plus_r_2[i+1] - self.line_latitude_plus_r_2[i]
                self.x_data_r_plus_2.append(90*np.sin((90-self.line_latitude_plus_r_2[i]-i_2/dx*dy)*np.pi/180)*np.cos((self.line_longitude_plus_r_2[i]+i_2)*np.pi/180))  # decimals=0
                self.y_data_r_plus_2.append(90*np.sin((90-self.line_latitude_plus_r_2[i]-i_2/dx*dy)*np.pi/180)*np.sin((self.line_longitude_plus_r_2[i]+i_2)*np.pi/180))
                
    def make_ellipsis(self):
        '''
        Аппроксимация найденной границы эллипсом, проводится методом наименьших квадратов
        путем минимизации расстояний от оцененных потенциальных границ до эллипса.
        '''
        points = []
        if len(self.x_data_r_plus) > 0:
            for i in range(len(self.x_data_r_plus) - 1):
                points.append([0]*2)
            for i in range(len(self.x_data_r_plus) - 1):
                points[i][0] = self.x_data_r_plus[i]
                points[i][1] = self.y_data_r_plus[i]

        a_points = np.array(points)
        ell = EllipseModel()
        ell.estimate(a_points)

        xc, yc, a, b, theta = ell.params
        '''
        Уравнение эллипса
        '''
        self.x_data_r_plus_3 = []
        self.y_data_r_plus_3 = []
        if len(self.x_data_r_plus) > 0:
            for i in range(360):
                self.x_data_r_plus_3.append(xc+3300*((a*np.cos(i)*np.pi/180)*(np.cos(theta)*np.pi/180)-(b*np.sin(i)*np.pi/180)*(np.sin(theta)*np.pi/180)))
                self.y_data_r_plus_3.append(yc+3300*((a*np.cos(i)*np.pi/180)*(np.sin(theta)*np.pi/180)+(b*np.sin(i)*np.pi/180)*(np.cos(theta)*np.pi/180)))  
                
    def plot_and_save(self):
        '''
        Визуализация всех полученных данных.        
        '''

        '''
        Исходные данные
        '''
        plt.figure(num=None, figsize = (22, 8.5), dpi=80)
        plt.rcParams['axes.facecolor'] = 'white'
        
        sc = plt.scatter(
            self.df_mag_data['mag_longitudes'], 
            self.df_mag_data['mag_latitudes'], 
            c=self.df_mag_data['values'], 
            alpha=1, 
            marker=',', 
            s=3, 
            vmin=0.0, 
            cmap=self.cm, 
            vmax=self.average_value*2
        )
        plt.colorbar(sc, shrink=0.8)
        
        initial_path = os.path.join(self.STR_INITIAL_DATA_PATH, self.frame_name)
        plt.savefig(initial_path + '.jpg')
        plt.close()
        
        '''
        Расширенные данные
        '''
        plt.figure(num=None, figsize = (17, 8.5), dpi=80)
        
        sc = plt.scatter(
            self.df_mag_data_augmented['mag_longitudes'], 
            self.df_mag_data_augmented['mag_latitudes'], 
            c=self.df_mag_data_augmented['values'], 
            alpha=1, 
            marker=',', 
            s=20, 
            vmin=0.0, 
            cmap=self.cm, 
            vmax=self.average_value*2
        )
        
        area_expansion_path = os.path.join(self.STR_AREA_EXPANSION_PATH, self.frame_name)
        plt.savefig(area_expansion_path + '.jpg')
        plt.close()
        
        '''
        Бинаризованные данные        
        '''
        plt.figure(num=None, figsize = (17, 8.5), dpi=80)
        
        sc = plt.scatter(
            self.df_mag_data_augmented['mag_longitudes'], 
            self.df_mag_data_augmented['mag_latitudes'], 
            c=self.df_mag_data_augmented['values_binarized'], 
            alpha=1, 
            marker=',', 
            s=3, 
            vmin=0.0, 
            cmap=self.cm, 
            vmax=self.average_value*2
        )
        
        binary_path = os.path.join(self.STR_BINARY_PATH, self.frame_name)
        plt.savefig(binary_path + '.jpg')
        plt.close()
        
        '''
        Эллипс на проекции
        '''
        plt.figure(figsize=(8.5, 8.5))
        
        self.data_v2 = self.data_r_radian.copy()
        for i in range(len(self.data_r_radian)):
            if self.data_r_radian[i] > 0:
                self.data_v2[i] = '#FF0000'
            else:
                self.data_v2[i] = '#0000FF'
        
        plt.scatter(
            self.x_data_r, 
            self.y_data_r, 
            c=self.data_r_radian, 
            alpha=1, 
            s=3, 
            cmap=self.cm, 
            marker="," , 
            vmax=self.average_value*2
        )
        
        plt.plot(
            self.x_data_r_plus, 
            self.y_data_r_plus, 
            color=self.color_line_plus, 
            linewidth=3
        )
        
        plt.scatter(
            self.x_data_r_plus, 
            self.y_data_r_plus, 
            color=self.color_line_plus, 
            s=3
        )
        
        dat = plt.scatter(
            self.x_data_r, 
            self.y_data_r, 
            c=self.data_v2,
            alpha=1, 
            s=8, 
            cmap=self.cm, 
            marker=","
        )
        plt.colorbar(dat)
        
        if len(self.x_data_r_plus) > 0:
            plt.scatter(
                self.x_data_r_plus_3, 
                self.y_data_r_plus_3, 
                color=self.color_line_plus_2, 
                s=6
            )
        
        plt.plot(
            self.x_setka, 
            self.y_setka, 
            color=self.color_line_minus, 
            linewidth=0.1
        )
         
        projection_path = os.path.join(self.STR_PROJECTION_PATH, self.frame_name)
        plt.savefig(projection_path + '.jpg')
        plt.close()
        
        '''
        Визуализация объединения данных на кластеры        
        '''
        plt.figure(num=None, figsize=(8.5, 8.5), dpi=80)

        sc = plt.scatter(
            self.df_mag_data_augmented['mag_latitudes'], 
            self.df_mag_data_augmented['values'], 
            c=self.df_mag_data_augmented['values_binarized'], 
            alpha=1, 
            s=3, 
            cmap=self.cm, 
            marker=","
        )
        
        kmeans_cos_path = os.path.join(self.STR_KMEANS_COS_PATH, self.frame_name)
        plt.savefig(kmeans_cos_path + '.jpg')
        plt.close()
        
        '''
        Распределение данных
        '''
        plt.figure(num=None, figsize=(17.5, 8.5), dpi=80)

        plt.rcParams['axes.facecolor'] = 'white'
        
        self.data_sorted = np.sort(self.df_mag_data_augmented['values'])
        p = []
        p = 1. * np.arange(len(self.df_mag_data_augmented['values'])) / (len(self.df_mag_data_augmented['values']) - 1)
        plt.plot(self.data_sorted, p)
        
        self.expected_value_number = 0
        for i in range(len(self.df_mag_data_augmented['values'])):
            if self.data_sorted[i] < self.average_value:
                self.expected_value_number = i
        
        plt.plot([self.average_value, self.average_value], [0, 1000], 'r')

        plt.hist(self.df_mag_data_augmented['values'], 200)
        
        cdf_path = os.path.join(self.STR_CDF_PATH, self.frame_name)
        plt.savefig(cdf_path + '.jpg')
        plt.close()
        
        '''
        Объединение рисунков
        '''
        img = Image.new('RGB', (3120, 2040), "white")

        img1 = Image.open(initial_path + '.jpg')
        img2 = Image.open(area_expansion_path + '.jpg')
        img7 = Image.open(cdf_path + '.jpg')
        img3 = Image.open(binary_path + '.jpg')
        img6 = Image.open(projection_path + '.jpg')
        
        img.paste(img1, (0, 0))
        img.paste(img7, (1760, 1360))
        img.paste(img2, (1760, 0))
        img.paste(img3, (90, 680))
        img.paste(img6, (440, 1360))
        
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('arial.ttf', size=45)
        font_2 = ImageFont.truetype('arial.ttf', size=36)
        
        draw.text((650, 0), 'исходные данные', (0, 0, 0), font=font_2)
        draw.text((2300, 0), 'расширение области', (0, 0, 0), font=font_2)
        draw.text((650, 680), 'бинаризованные данные ', (0, 0, 0), font=font_2)
        draw.text((650, 1360), 'проекция ', (0, 0, 0), font=font_2)
        draw.text((1400, 1980), self.frame_time_curr, (0, 0, 0), font=font)
        
        all_path = os.path.join(self.STR_ALL_PATH, self.frame_name)
        
        img.save(all_path + '.jpg')
            
    def start_pipeline(self, file_path):
        self.read_file_data(file_path)
        self.coordinates_transition_and_augmentation()
        self.data_binarization()
        self.border_estimation()
        self.make_ellipsis()
        self.plot_and_save()

if __name__ == '__main__':

    STD_SIMURG_FOLDER = os.path.join(os.getcwd(), 'data')
    STR_SIMURG_FILE_NAME = os.path.join(STD_SIMURG_FOLDER, 'roti_2007_107_-85_86_N_-176_179_E_d9a4.h5')
    SIMURG_FILE_PATH = os.path.join(os.getcwd(), STR_SIMURG_FILE_NAME)

    est = ellipsis_estimator()
    test_time_list = est.read_file_and_times(SIMURG_FILE_PATH)
    est.start_pipeline(test_time_list[215])