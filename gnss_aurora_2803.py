import os
import time
import math
import shutil
from multiprocessing import Lock, Pool

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from PIL import Image, ImageFont, ImageDraw

from sklearn.cluster import KMeans
from skimage.measure import EllipseModel


STR_CURRENT_PATH = os.path.dirname(__file__) + '\\'
STR_MAIN_DATA_FOLDER = 'temp'
STR_MAIN_DATA_PATH = STR_CURRENT_PATH + STR_MAIN_DATA_FOLDER

STR_INITIAL_DATA = 'initial_data\\'
STR_AREA_EXPANSION = 'area_expansion\\'
STR_CDF = 'cdf\\'
STR_BINARY_DATA = 'binary_data\\'
STR_PROJECTION = 'projection\\'
STR_ALL = 'all\\'
STR_KMEANS_COS = 'kmeans_cos\\'
STR_SIMURG_FILE_NAME = 'roti_2007_107_-85_86_N_-176_179_E_d9a4.h5'
STR_MAGLATLON_FILE_NAME = '2010magLATLon_map.dat'
simurg_file_path = STR_CURRENT_PATH + STR_SIMURG_FILE_NAME
magLATLon_file_path = STR_CURRENT_PATH + STR_MAGLATLON_FILE_NAME

data_folders = [
	STR_INITIAL_DATA,
	STR_AREA_EXPANSION,
	STR_CDF,
	STR_BINARY_DATA,
	STR_PROJECTION,
	STR_ALL,
	STR_KMEANS_COS
]


class init(object):
	start_time = time.time()
	threads = 1
	lock = Lock()
	cm = plt.cm.get_cmap('bwr')
	color_line_plus = '#333333'
	color_line_minus = '#00FF00'
	color_line_plus_2 = '#FFFF00'
	data_calendar = '2007-04-17' + ' '


def create_folder(pth):
	if not os.path.exists(pth):
		os.makedirs(pth)


class stream_computing_class(init):
	def parallel_computing(self, frame_index):
		magLat = []
		magLon = []

		magLATLon = open(magLATLon_file_path, 'r')

		for index in range(1, 181 * 361 + 1):
			temporar = magLATLon.readline().split(' ')
			
			if index % 361 == 360: ## значения при долготе -180 совпадают со значениями при долготе 180. Чтобы не дублировать данные для этой точки, надо пропускать повторения
				continue
			
			coords = []
			for i in range(0, len(temporar)):
				if temporar[i] != '':
					coords.append(temporar[i])
			
			try:
				magLat.append(coords[2]) ## заменил coords[0] на coords[2]
			except:
				pass
			
			try:
				magLon.append(coords[3]) ## заменил coords[1] на coords[3]
			except:
				pass
		
		## размер magLat и magLon равен 181 * 360
		
		self.lat = []
		self.lon = []
		self.data = []

		self.hour_index = frame_index // 60
		self.min_index = (frame_index % 60)
		self.frame_number = frame_index - 1080
		self.time_hour = str(self.hour_index)
		self.time_min = str(self.min_index)

		if self.hour_index < 10:
			self.time_hour = '0'+ self.time_hour

		if self.min_index < 10:
			self.time_min = '0'+ self.time_min

		self.date_time = init.data_calendar + self.time_hour + ':' + self.time_min + ':00.000000'

		file = h5py.File(simurg_file_path, 'r')
		##file = h5py.File(simurg_file_path + '_', 'r')

		self.lat.extend(file['data'][self.date_time]['lat'])
		self.lon.extend(file['data'][self.date_time]['lon'])
		self.data.extend(file['data'][self.date_time]['vals'])
		# self.lat.extend(file['data']['lat'])
		# self.lon.extend(file['data']['lon'])
		# self.data.extend(file['data']['vals'])

		## all_data = []
		## all_data.extend(file['data'][self.date_time])
		## all_sz = len(all_data)
		## with h5py.File(simurg_file_path + '_', 'w') as f:
			## dset = f.create_dataset('data', (all_sz,), dtype = [('lat', '<f8'), ('lon', '<f8'), ('vals', '<f8')])
			## dset[:] = all_data[:]
        
		self.latr = np.around(self.lat, decimals=0) ## округлённые значения широты
		self.lonr = np.around(self.lon, decimals=0) ## округлённые значения долготы

		# изменение системы координат
		for i in range(len(self.lat)):
			#если округлённые lat и lon не являются NaN
			if not math.isnan(self.latr[i]) and not math.isnan(self.lonr[i]):
				self.latr[i] = magLat[round(self.latr[i] + 90) * 360 + (round(self.lonr[i]) + 180) % 360] ## теперь широта и долгота в геомагнитных координатах, причём,
				self.lonr[i] = magLon[round(self.latr[i] + 90) * 360 + (round(self.lonr[i]) + 180) % 360] ## судя по файлу 2010magLATLon_map.dat, широта от -90 до 90, а долгота от 0 до 359

		self.latr_temp = np.around(self.latr, decimals=0) ## округлённая геомагнитная широта
		self.data_average = []
		self.data_average_r = []
		self.data_quantity = []
		self.data_temp = []

		for latitude in range(-90, 91):
			self.data_average.append([0]*360)
			self.data_average_r.append([0]*360)
			self.data_quantity.append([0]*360)
			self.data_temp.append([0]*360)
		## в итоге data_average, data_average_r, data_quantity, data_temp - это нулеве матрицы размера 181 * 360
		## print(len(self.data_average), len(self.data_average[0]))

		summ = 0.
		self.expected_value = 0.

		for i in range(len(self.data)):
			if not math.isnan(self.latr[i]) and not math.isnan(self.lonr[i]) and not math.isnan(self.data[i]):
				latitude = round(self.latr[i]) # + 90 ## значение latitude уже от -90 до 90
				longitude = round(self.lonr[i]) - 180 ## надо отнять 180, чтобы значение longitude было от -180 до 179
				self.data_average[latitude][longitude] += self.data[i]
				self.data_quantity[latitude][longitude] += 1
                
		for latitude in range(-90, 91):
			for longitude in range(-180, 180):
				if self.data_quantity[latitude][longitude] != 0:
					self.data_average[latitude][longitude] = self.data_average[latitude][longitude] / self.data_quantity[latitude][longitude] ## среднее значение
					self.data_temp[latitude][longitude] = self.data_average[latitude][longitude]

		self.initial_data = [] ## в этих списках будут храниться
		self.initial_latr = [] ## данные только для пикселей
		self.initial_lonr = [] ## с ненулевым средним значением
		summ = 0

		for latitude in range(-90, 91):
			for longitude in range(-180, 180):
				if self.data_average[latitude][longitude] != 0:
					self.initial_data.append(self.data_average[latitude][longitude])
					self.initial_latr.append(latitude)
					self.initial_lonr.append(longitude)

		for i in range(len(self.initial_data)):
			latitude = self.initial_latr[i] 
			longitude = self.initial_lonr[i]
			if self.initial_lonr[i] > 60: ## зачем?
				self.initial_lonr[i] -= 360 ## после этого значения initial_lonr будут от -299 до 60.
				## Зачем что-то отнимать? Наверное, для сдвига на каких-то графиках

		# расширение области
		self.data = []
		self.data_kmeans = []
		self.latr = []
		self.lonr = []
		summ = 0
		for latitude in range(-90, 91):
			for longitude in range(-180, 180):
				if self.data_average[latitude][longitude] != 0:
					self.data.append(self.data_average[latitude][longitude]) ## записываем средние значения интенсивности
					self.data_kmeans.append(self.data_average[latitude][longitude])
					self.latr.append(latitude)
					self.lonr.append(longitude)
					summ += self.data_average[latitude][longitude]
					self.expected_value += 1 ## на данном этапе в expected_value хранится счётчик

		self.expected_value = summ / self.expected_value ## теперь это среднее значение
		self.data_r = self.data.copy()

		## cos_lat = list(map(lambda x: np.cos(x * np.pi / 180) * 0.0, self.latr)) ## умножение на 0.0 даёт в результате 0. зачем?
		cos_lat = list(map(lambda x: np.cos(x * np.pi / 180), self.latr))
		all_data = {'data': self.data, 'x': self.lonr, 'y': cos_lat}
		df = pd.DataFrame(all_data, columns=['data', 'y'])

		kmeans = KMeans(n_clusters=2).fit(df) ## бинаризация проводится путём кластеризации на 2 множества
		self.data_r = kmeans.labels_.tolist()

		# бинаризация
		for i in range(len(self.data)):
			if not math.isnan(self.latr[i]) and not math.isnan(self.lonr[i]) and not math.isnan(self.data[i]):
				latitude = self.latr[i] ## от -90 до 90
				longitude = self.lonr[i] ## от -180 до 179
				## self.latr_temp[i] = abs(self.latr[i]) ## не используется
				self.data_average_r[latitude][longitude] = self.data_r[i]
				self.data[i] = self.data_average[latitude][longitude] ## что меняется?
			if self.lonr[i] >= 60: ## зачем это условие?
				self.lonr[i] = self.lonr[i] - 360
		# инверсия бинаризация для kmeans

		for i in range(len(self.data)):
			if self.data_r[i] > 0:
				self.data_r[i] = -1
			else:
				self.data_r[i] = 1
			latitude = self.latr[i]
			longitude = self.lonr[i]
			self.data_average_r[latitude][longitude] = self.data_r[i]

		# бинаризованные данные

		self.data_r_2 = []
		self.latr_2 = []
		self.lonr_2 = []

		for latitude in range(-90, 89):
			for longitude in range(-180, 180):
				if (self.data_average_r[latitude][longitude] != 0):
					if (self.data_average_r[latitude + 1][longitude] == 0) and (self.data_average_r[latitude + 2][longitude] * self.data_average_r[latitude][longitude] == 1):
						self.data_average_r[latitude + 1][longitude] = self.data_average_r[latitude][longitude] * 0.999
						self.data_r_2.append(self.data_average_r[latitude+1][longitude])
						self.latr_2.append(latitude + 1)
						if longitude >= 60:
							self.lonr_2.append(longitude - 360)
						else:
							self.lonr_2.append(longitude)
						## self.lonr_2.append(longitude)
		
		for latitude in range(-90, 91):
			for longitude in range(-180, 178):
				if (self.data_average_r[latitude][longitude] != 0):
					if (self.data_average_r[latitude][longitude + 1] == 0) and (self.data_average_r[latitude][longitude + 2] * self.data_average_r[latitude][longitude] == 1):
						self.data_average_r[latitude][longitude + 1] = self.data_average_r[latitude][longitude] * 0.999
						self.data_r_2.append(self.data_average_r[latitude][longitude + 1])
						self.latr_2.append(latitude)
						if longitude >= 60:
							self.lonr_2.append(longitude - 360 + 1)
						if longitude < 60:
							self.lonr_2.append(longitude + 1)
						## self.lonr_2.append(longitude + 1)

		for latitude in range(-90, 90):
			for longitude in range(-180, 179):
				if (self.data_average_r[latitude][longitude] != 0):
					if (self.data_average_r[latitude + 1][longitude] == 0) and (self.data_average_r[latitude + 1][longitude + 1] * self.data_average_r[latitude][longitude] == 1):
						self.data_average_r[latitude + 1][longitude] = self.data_average_r[latitude][longitude] * 0.999
						self.data_r_2.append(self.data_average_r[latitude + 1][longitude])
						self.latr_2.append(latitude + 1)
						if longitude >= 60:
							self.lonr_2.append(longitude - 360)
						if longitude < 60:
							self.lonr_2.append(longitude)
						## self.lonr_2.append(longitude)
						
					if (self.data_average_r[latitude][longitude + 1] == 0) and (self.data_average_r[latitude + 1][longitude + 1] * self.data_average_r[latitude][longitude] == 1):
						self.data_average_r[latitude][longitude + 1] = self.data_average_r[latitude][longitude] * 0.999
						self.data_r_2.append(self.data_average_r[latitude][longitude + 1])
						self.latr_2.append(latitude)
						if longitude >= 60:
							self.lonr_2.append(longitude - 360 + 1)
						if longitude < 60:
							self.lonr_2.append(longitude + 1)
						## self.lonr_2.append(longitude + 1)
						
				if (self.data_average_r[latitude + 1][longitude] != 0):
					if (self.data_average_r[latitude][longitude] == 0) and (self.data_average_r[latitude + 1][longitude] * self.data_average_r[latitude][longitude + 1] == 1):
						self.data_average_r[latitude][longitude] = self.data_average_r[latitude + 1][longitude] * 0.999
						self.data_r_2.append(self.data_average_r[latitude][longitude])
						self.latr_2.append(latitude)
						if longitude >= 60:
							self.lonr_2.append(longitude - 360)
						if longitude < 60:
							self.lonr_2.append(longitude)
						## self.lonr_2.append(longitude)
						
					if (self.data_average_r[latitude + 1][longitude + 1] == 0) and (self.data_average_r[latitude + 1][longitude] * self.data_average_r[latitude][longitude + 1] == 1):
						self.data_average_r[latitude + 1][longitude + 1] = self.data_average_r[latitude + 1][longitude] * 0.999
						self.data_r_2.append(self.data_average_r[latitude + 1][longitude + 1])
						self.latr_2.append(latitude + 1)
						if longitude >= 60:
							self.lonr_2.append(longitude - 360 + 1)
						if longitude < 60:
							self.lonr_2.append(longitude + 1)
						## self.lonr_2.append(longitude + 1)

		self.data_r_3 = []
		self.latr_3 = []
		self.lonr_3 = []
		for latitude in range(-90, 90):
			for longitude in range(-180, 179):
				if self.data_average_r[latitude][longitude] > 0:
					if (self.data_average_r[latitude + 1][longitude] <= 0) or (self.data_average_r[latitude - 1][longitude] <= 0) or (self.data_average_r[latitude][longitude + 1] <= 0) or (self.data_average_r[latitude][longitude - 1] <= 0) or (self.data_average_r[latitude + 1][longitude + 1] <= 0) or (self.data_average_r[latitude - 1][longitude - 1] <= 0) or (self.data_average_r[latitude - 1][longitude + 1] <= 0) or (self.data_average_r[latitude + 1][longitude - 1] <= 0):
						self.data_r_3.append(100)
						self.latr_3.append(latitude)
						if longitude >= 60:
							self.lonr_3.append(longitude - 360)
						if longitude < 60:
							self.lonr_3.append(longitude)
						## self.lonr_3.append(longitude)
						
		for latitude in range(-90, 90):
			for longitude in range(-180, 179):
				if self.data_average_r[latitude][longitude] < 0:
					if (self.data_average_r[latitude + 1][longitude] >= 0) or (self.data_average_r[latitude - 1][longitude] >= 0) or (self.data_average_r[latitude][longitude + 1] >= 0) or (self.data_average_r[latitude][longitude - 1] >= 0) or (self.data_average_r[latitude + 1][longitude + 1] >= 0) or (self.data_average_r[latitude - 1][longitude - 1] >= 0) or (self.data_average_r[latitude - 1][longitude + 1] >= 0) or (self.data_average_r[latitude + 1][longitude - 1] >= 0):
						self.data_r_3.append(-100)
						self.latr_3.append(latitude)
						if longitude >= 60:
							self.lonr_3.append(longitude - 360)
						if longitude < 60:
							self.lonr_3.append(longitude)
						## self.lonr_3.append(longitude)

		criterion_up = 12
		criterion_down = 12
		self.line_longitude_plus = []
		self.line_latitude_plus = []
		for longitude in range(61, 357):
			temp_latitude = 0
			for latitude in range(80, 40, -1):
				if self.data_average_r[latitude][longitude] > 0:
					add_flag_1 = 0
					add_flag_2 = 0
					for index_x in range(-3, 4):
						for index_y in range(1, 4):
							if self.data_average_r[latitude + index_y][longitude + index_x] > 0:
								add_flag_1 += 1
							if self.data_average_r[latitude - index_y][longitude + index_x] < 0:
								add_flag_2 += 1
						if add_flag_1 >= criterion_up and add_flag_2 >= criterion_down:
							if temp_latitude == 0:
								temp_latitude = latitude
							if temp_latitude != 0:
								if len(self.line_latitude_plus) > 0:
									if abs(temp_latitude - self.line_latitude_plus[-1]) < abs(self.line_latitude_plus[-1] - latitude):
										temp_latitude = latitude
			if temp_latitude != 0:
				## self.line_longitude_plus.append(longitude - 360)
				self.line_longitude_plus.append(longitude)
				self.line_latitude_plus.append(temp_latitude - 1)
		for longitude in range(3, 61):
			for latitude in range(80, 40, -1):
				if self.data_average_r[latitude][longitude] > 0:
					add_flag_1 = 0
					add_flag_2 = 0
					for index_x in range(-3, 4):
						for index_y in range(1, 4):
							if self.data_average_r[latitude + index_y][longitude + index_x] > 0:
								add_flag_1 += 1
							if self.data_average_r[latitude - index_y][longitude + index_x] < 0:
								add_flag_2 += 1
						if add_flag_1 >= criterion_up and add_flag_2 >= criterion_down:
							if temp_latitude == 0:
								temp_latitude = latitude
							if temp_latitude != 0:
								if len(self.line_latitude_plus) > 0:
									if abs(temp_latitude - self.line_latitude_plus[-1]) < abs(self.line_latitude_plus[-1] - latitude):
										temp_latitude = latitude
			if temp_latitude != 0:
				self.line_longitude_plus.append(longitude)
				self.line_latitude_plus.append(temp_latitude - 1)

		# поиск окружности

		number_of_brute_force_points = 8
		number_of_brute_force_points_2 = number_of_brute_force_points
		self.line_longitude_plus_r = []
		self.line_latitude_plus_r = []

		if len(self.line_longitude_plus) > 4:
			self.line_longitude_plus_r.append(self.line_longitude_plus[0])
			self.line_latitude_plus_r.append(self.line_latitude_plus[0])
			for i in range(1, len(self.line_longitude_plus)):
				if self.line_longitude_plus[i]-self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1] > 0:
					dx_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.cos((self.line_longitude_plus_r[len(
						self.line_longitude_plus_r)-1])*np.pi/180)-np.sin((90-self.line_latitude_plus[i])*np.pi/180)*np.cos((self.line_longitude_plus[i])*np.pi/180))
					dy_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.sin((self.line_longitude_plus_r[len(
						self.line_longitude_plus_r)-1])*np.pi/180)-np.sin((90-self.line_latitude_plus[i])*np.pi/180)*np.sin((self.line_longitude_plus[i])*np.pi/180))
					dz_1 = math.fabs(np.sin((self.line_latitude_plus_r[len(
						self.line_longitude_plus_r)-1])*np.pi/180)-np.sin((self.line_latitude_plus[i])*np.pi/180))
					temp_delta = dx_1*dx_1+dy_1*dy_1+dz_1*dz_1
					for i_2 in range(1, number_of_brute_force_points_2):
						dx_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.cos((self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
							(90-self.line_latitude_plus[i+i_2])*np.pi/180)*np.cos((self.line_longitude_plus[i+i_2])*np.pi/180))
						dy_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.sin((self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
							(90-self.line_latitude_plus[i+i_2])*np.pi/180)*np.sin((self.line_longitude_plus[i+i_2])*np.pi/180))
						dz_1 = math.fabs(np.sin((self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
							(self.line_latitude_plus[i+i_2])*np.pi/180))
						if temp_delta > dx_1*dx_1+dy_1*dy_1+dz_1*dz_1:
							temp_delta = dx_1*dx_1+dy_1*dy_1+dz_1*dz_1
					for i_2 in range(number_of_brute_force_points_2):
						dx_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.cos((self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
							(90-self.line_latitude_plus[i+i_2])*np.pi/180)*np.cos((self.line_longitude_plus[i+i_2])*np.pi/180))
						dy_1 = math.fabs(np.sin((90-self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)*np.sin((self.line_longitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
							(90-self.line_latitude_plus[i+i_2])*np.pi/180)*np.sin((self.line_longitude_plus[i+i_2])*np.pi/180))
						dz_1 = math.fabs(np.sin((self.line_latitude_plus_r[len(self.line_longitude_plus_r)-1])*np.pi/180)-np.sin(
							(self.line_latitude_plus[i+i_2])*np.pi/180))
						if temp_delta == dx_1*dx_1+dy_1*dy_1+dz_1*dz_1:
							self.line_longitude_plus_r.append(
								self.line_longitude_plus[i+i_2])
							self.line_latitude_plus_r.append(
								self.line_latitude_plus[i+i_2])
				if i >= len(self.line_longitude_plus)-number_of_brute_force_points:
					number_of_brute_force_points_2 = number_of_brute_force_points_2-1
			self.line_longitude_plus_r.append(
				self.line_longitude_plus[len(self.line_longitude_plus)-1])
			self.line_latitude_plus_r.append(
				self.line_latitude_plus[len(self.line_longitude_plus)-1])
			self.line_longitude_plus_r.append(self.line_longitude_plus[0]+360)
			self.line_latitude_plus_r.append(self.line_latitude_plus[0])

		self.line_longitude_plus_r_2 = []
		self.line_latitude_plus_r_2 = []
		number_of_brute_force_points_2 = number_of_brute_force_points
		if len(self.line_longitude_plus) > 4:
			self.line_longitude_plus_r_2.append(self.line_longitude_plus[0])
			self.line_latitude_plus_r_2.append(self.line_latitude_plus[0])
			for i in range(1, len(self.line_longitude_plus)):
				if self.line_longitude_plus[i]-self.line_longitude_plus_r_2[len(self.line_longitude_plus_r_2)-1] > 0:
					dx_1 = math.fabs(self.line_longitude_plus_r_2[len(
						self.line_longitude_plus_r_2)-1]-self.line_longitude_plus[i])
					dy_1 = math.fabs(self.line_latitude_plus_r_2[len(
						self.line_longitude_plus_r_2)-1]-self.line_latitude_plus[i])
					temp_delta = dx_1*dx_1+dy_1*dy_1
					for i_2 in range(1, number_of_brute_force_points_2):
						dx_1 = math.fabs(self.line_longitude_plus_r_2[len(
							self.line_longitude_plus_r_2)-1]-self.line_longitude_plus[i+i_2])
						dy_1 = math.fabs(self.line_latitude_plus_r_2[len(
							self.line_longitude_plus_r_2)-1]-self.line_latitude_plus[i+i_2])
						if temp_delta > dx_1*dx_1+dy_1*dy_1:
							temp_delta = dx_1*dx_1+dy_1*dy_1
					for i_2 in range(number_of_brute_force_points_2):
						dx_1 = math.fabs(self.line_longitude_plus_r_2[len(
							self.line_longitude_plus_r_2)-1]-self.line_longitude_plus[i+i_2])
						dy_1 = math.fabs(self.line_latitude_plus_r_2[len(
							self.line_longitude_plus_r_2)-1]-self.line_latitude_plus[i+i_2])
						if temp_delta == dx_1*dx_1+dy_1*dy_1:
							self.line_longitude_plus_r_2.append(
								self.line_longitude_plus[i+i_2])
							self.line_latitude_plus_r_2.append(
								self.line_latitude_plus[i+i_2])
				if i >= len(self.line_longitude_plus)-number_of_brute_force_points:
					number_of_brute_force_points_2 = number_of_brute_force_points_2-1
			self.line_longitude_plus_r_2.append(
				self.line_longitude_plus[len(self.line_longitude_plus)-1])
			self.line_latitude_plus_r_2.append(
				self.line_latitude_plus[len(self.line_longitude_plus)-1])
			self.line_longitude_plus_r_2.append(
				self.line_longitude_plus[0]+360)
			self.line_latitude_plus_r_2.append(self.line_latitude_plus[0])

		self.x_setka = []
		self.y_setka = []

		self.x_data_r = []
		self.y_data_r = []
		self.data_r_radian = []

		for i in range(len(self.initial_data)):
			if self.initial_latr[i] >= 0:
				self.x_data_r.append(90*np.sin((90-self.initial_latr[i])*np.pi/180)*np.cos(
					(self.initial_lonr[i])*np.pi/180))	# decimals=0
				self.y_data_r.append(90*np.sin((90-self.initial_latr[i])*np.pi/180)*np.sin(
					(self.initial_lonr[i])*np.pi/180))
				self.data_r_radian.append(self.data_r[i])

		file = open(STR_MAIN_DATA_PATH + '\\' + STR_PROJECTION +
					str(self.frame_number)+'.txt', 'w')

		self.x_data_r_plus = []
		self.y_data_r_plus = []
		for i in range(len(self.line_latitude_plus_r) - 1):
			self.x_data_r_plus.append(90*np.sin((90-self.line_latitude_plus_r[i])*np.pi/180)*np.cos(
				(self.line_longitude_plus_r[i])*np.pi/180))	 # decimals=0
			self.y_data_r_plus.append(90*np.sin((90-self.line_latitude_plus_r[i])*np.pi/180)*np.sin(
				(self.line_longitude_plus_r[i])*np.pi/180))
			file.write(str(self.x_data_r_plus[i]) +
					   ';  '+str(self.y_data_r_plus[i])+';\n')
			self.x_setka.append(0)
			self.y_setka.append(0)
			self.x_setka.append(
				80*np.cos((self.line_longitude_plus_r[i])*np.pi/180))
			self.y_setka.append(
				80*np.sin((self.line_longitude_plus_r[i])*np.pi/180))
			for i_2 in range(1, self.line_longitude_plus_r[i+1]-self.line_longitude_plus_r[i]):
				dx = self.line_longitude_plus_r[i+1] - \
					self.line_longitude_plus_r[i]
				dy = self.line_latitude_plus_r[i+1] - \
					self.line_latitude_plus_r[i]
		self.x_data_r_plus_2 = []
		self.y_data_r_plus_2 = []
		for i in range(0, len(self.line_latitude_plus_r_2)-1, 1):
			self.x_data_r_plus_2.append(90*np.sin((90-self.line_latitude_plus_r_2[i])*np.pi/180)*np.cos(
				(self.line_longitude_plus_r_2[i])*np.pi/180))  # decimals=0
			self.y_data_r_plus_2.append(90*np.sin((90-self.line_latitude_plus_r_2[i])*np.pi/180)*np.sin(
				(self.line_longitude_plus_r_2[i])*np.pi/180))
			for i_2 in range(1, self.line_longitude_plus_r_2[i+1]-self.line_longitude_plus_r_2[i]):
				dx = self.line_longitude_plus_r_2[i+1] - \
					self.line_longitude_plus_r_2[i]
				dy = self.line_latitude_plus_r_2[i+1] - \
					self.line_latitude_plus_r_2[i]
				self.x_data_r_plus_2.append(90*np.sin((90-self.line_latitude_plus_r_2[i]-i_2/dx*dy)*np.pi/180)*np.cos(
					(self.line_longitude_plus_r_2[i]+i_2)*np.pi/180))  # decimals=0
				self.y_data_r_plus_2.append(90*np.sin((90-self.line_latitude_plus_r_2[i]-i_2/dx*dy)*np.pi/180)*np.sin(
					(self.line_longitude_plus_r_2[i]+i_2)*np.pi/180))

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

		self.x_data_r_plus_3 = []
		self.y_data_r_plus_3 = []
		# построение эллипса
		if len(self.x_data_r_plus) > 0:
			for i in range(360):
				self.x_data_r_plus_3.append(xc+3300*((a*np.cos(i)*np.pi/180)*(np.cos(
					theta)*np.pi/180)-(b*np.sin(i)*np.pi/180)*(np.sin(theta)*np.pi/180)))
				self.y_data_r_plus_3.append(yc+3300*((a*np.cos(i)*np.pi/180)*(np.sin(
					theta)*np.pi/180)+(b*np.sin(i)*np.pi/180)*(np.cos(theta)*np.pi/180)))
		figure(num=None, figsize=(22, 8.5), dpi=80)
		plt.rcParams['axes.facecolor'] = 'white'
		plt.axis('scaled')
		plt.ylim(-90, 90)
		plt.xlim(-300, 60)
		sc = plt.scatter(self.initial_lonr, self.initial_latr, c=self.initial_data, alpha=1,
						 s=3, cmap=init.cm, marker=",", vmin=0.0, vmax=self.expected_value*2)
		plt.colorbar(sc, shrink=0.8)
		plt.savefig(STR_MAIN_DATA_PATH + '\\' +
					STR_INITIAL_DATA + str(self.frame_number))
		plt.close()

		figure(num=None, figsize=(17, 8.5), dpi=80)
		plt.ylim(-90, 90)
		plt.xlim(-300, 60)
		sc = plt.scatter(self.lonr, self.latr, c=self.data, alpha=1, s=20,
						 cmap=init.cm, marker=",", vmin=0.0, vmax=self.expected_value*2)
		plt.savefig(STR_MAIN_DATA_PATH + '\\' +
					STR_AREA_EXPANSION + str(self.frame_number))
		plt.close()

		figure(num=None, figsize=(17, 8.5), dpi=80)
		plt.ylim(-90, 90)
		plt.xlim(-300, 60)
		sc = plt.scatter(self.lonr, self.latr, c=self.data_r,
						 alpha=1, s=3, cmap=init.cm, marker=",")

		plt.savefig(STR_MAIN_DATA_PATH + '\\' +
					STR_BINARY_DATA + str(self.frame_number))

		self.data_v2 = self.data_r_radian.copy()
		for i in range(len(self.data_r_radian)):
			if self.data_r_radian[i] > 0:
				self.data_v2[i] = '#FF0000'
			else:
				self.data_v2[i] = '#0000FF'

		figure(num=None, figsize=(8.5, 8.5), dpi=80)
		plt.ylim(-90, 90)
		plt.xlim(-90, 90)
		plt.scatter(self.x_data_r, self.y_data_r, c=self.data_r_radian,alpha=1,s=3,cmap=init.cm, marker="," )
		plt.scatter(self.x_data_r, self.y_data_r, c=self.data_r_radian,alpha=1,s=3,cmap=init.cm, marker="," ,vmax=self.expected_value*2)
		# plt.scatter(self.x_center_r_plus, self.y_center_r_plus, c=init.color_line_plus,alpha=1,s=100,cmap=init.cm )			 #Откуда x_center_r_plus????? y_center_r_plus?????????
		# plt.plot(self.x_data_r_plus_3,self.y_data_r_plus_3,color = init.color_line_plus_2,linewidth = 3) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		plt.plot(self.x_data_r_plus, self.y_data_r_plus,
				 color=init.color_line_plus, linewidth=3)

		plt.scatter(self.x_data_r_plus, self.y_data_r_plus,
					color=init.color_line_plus, s=3)

		dat = plt.scatter(self.x_data_r, self.y_data_r, c=self.data_v2,
						  alpha=1, s=8, cmap=init.cm, marker=",")
		plt.colorbar(dat)
		# plt.scatter(rndlat,rndlon,color = 'black',s = 6,marker=",")
		# plt.scatter(self.x_data_r_plus,self.y_data_r_plus,color = 'red',s=8)
		if len(self.x_data_r_plus) > 0:
			sc = plt.scatter(self.x_data_r_plus_3, self.y_data_r_plus_3,
							 color=init.color_line_plus_2, s=6)
			# sc=plt.scatter(self.x_data_r_plus_3,self.y_data_r_plus_3,color = init.color_line_plus_2,s=3)

		# plt.plot(self.x_data_r_minus,self.y_data_r_minus,color = init.color_line_minus,linewidth = 3)

		plt.plot(self.x_setka, self.y_setka, color=init.color_line_minus, linewidth=0.1)

		plt.savefig(STR_MAIN_DATA_PATH + '\\' + STR_PROJECTION + str(self.frame_number))
		plt.close()

		figure(num=None, figsize=(8.5, 8.5), dpi=80)
		plt.ylim(0, 1)
		plt.xlim(-90, 90)

		sc = plt.scatter(self.latr, self.data_kmeans, c=self.data_r, alpha=1, s=3, cmap=init.cm, marker=",")
		plt.savefig(STR_MAIN_DATA_PATH + '\\' + STR_KMEANS_COS + str(self.frame_number))

		plt.close()
		figure(num=None, figsize=(17.5, 8.5), dpi=80)

		plt.rcParams['axes.facecolor'] = 'white'
		self.data_sorted = np.sort(self.data)
		p = []
		p = 1. * np.arange(len(self.data)) / (len(self.data) - 1)
		plt.plot(self.data_sorted, p)
		self.expected_value_number = 0
		for i in range(len(self.data)):
			if self.data_sorted[i] < self.expected_value:
				self.expected_value_number = i
		p[self.expected_value_number]
		plt.plot([self.expected_value, self.expected_value], [0, 1000], 'r')

		plt.hist(self.data, 200)
		plt.savefig(STR_MAIN_DATA_PATH + '\\' + STR_CDF + str(self.frame_number))
		plt.close()

		img = Image.new('RGB', (3120, 2040), "white")

		img1 = Image.open(STR_MAIN_DATA_PATH + '\\' + STR_INITIAL_DATA +
						  str(self.frame_number)+'.png')
		img2 = Image.open(STR_MAIN_DATA_PATH + '\\' +
						  STR_AREA_EXPANSION + str(self.frame_number)+'.png')
		img7 = Image.open(STR_MAIN_DATA_PATH + '\\' + STR_CDF +
						  str(self.frame_number)+'.png')
		img3 = Image.open(STR_MAIN_DATA_PATH + '\\' + STR_BINARY_DATA +
						  str(self.frame_number)+'.png')
		img6 = Image.open(STR_MAIN_DATA_PATH + '\\' + STR_PROJECTION +
						  str(self.frame_number)+'.png')
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
		draw.text((1400, 1980), init.data_calendar +
				  self.time_hour+':'+self.time_min+':00', (0, 0, 0), font=font)
		img.save(STR_MAIN_DATA_PATH + '\\' + STR_ALL +
				 str(self.frame_number)+'.png')
		return (self.frame_number)


if __name__ == '__main__':
	if os.path.exists(STR_MAIN_DATA_PATH):
		shutil.rmtree(STR_MAIN_DATA_PATH)
		time.sleep(0.1)

	if not os.path.exists(STR_MAIN_DATA_PATH):
		init()
		
		for data_folder in data_folders:
			create_folder(STR_MAIN_DATA_PATH + '\\' + data_folder)

		pool_core = Pool(init.threads)
		stream_computing = stream_computing_class()
		for x in pool_core.imap_unordered(stream_computing.parallel_computing, range(1080+0, 1080+1, 1)):
			print(f"{x} frame is completed (lead time: {int(time.time() - init.start_time)})")
		# x = stream_computing.parallel_computing(1080) # comment if not debug
		# print(f"{x} frame is completed (lead time: {int(time.time() - init.start_time)})") # comment if not debug

		pool_core.terminate()

		print(f"{init.threads} threads used {time.time() - init.start_time} seconds")
