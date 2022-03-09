#TODO: формат файла hdf5 - прочитать с помощью библиотеки h5py

import os

#библиотеки для работы с данными
import h5py
import numpy as np

#библиотеки для визуализации данных
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

test_file = 'dtec_20_60_2017_148_-90_90_N_-180_180_E_6c18.h5'

lats = np.array([])
lons = np.array([])
ampltd_tec_vals = np.array([])

with h5py.File(os.path.dirname(__file__) + '\\' + test_file, 'r') as hdf_file:
    hdf_base_items = list(hdf_file.items()) #структура файла hdf5 с наименованиями групп и кол-вом элементом в них
    #print(hdf_base_items)
    data = hdf_file.get('data')
    #print(data)
    data_items = list(data.items()) #структура группы data - дата/время - координаты со значениями (key - value)
    ##print(data_items[0])

    example_dataset = np.array(data.get('2017-05-28 03:30:00.000000')) #значения на координатах в заданную дату/время

    # for key in data.keys():
    for value in data['2017-05-28 03:30:00.000000']:
        lats = np.append(lats, value[0])
        lons = np.append(lons, value[1])
        ampltd_tec_vals = np.append(ampltd_tec_vals, value[2])

    ##print(example_dataset)
    #data = hdf_base_items.get('data')

    ###### выгрузил в файл данные за определенную дату/время
    # datetime_records = open('datetime_records.txt', 'w')
    # for dkey in example_dataset:
    #     datetime_records.write(str(dkey) + '\n')
    # datetime_records.close()
    ###### данные имеют вид: широта, долгота, амплитудное значение TEC

#TODO: отобразить данные на карте с помощью библиотек matplotlib и cartopy

# fig = plt.figure(figsize=(10, 5))
# ax = plt.axes(projection=ccrs.PlateCarree())
#
# plt.title("2017-05-28 03:30:00.000000", fontsize=18, fontweight='bold')
# gl = ax.gridlines(draw_labels=True, dms=True, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
#
# ax.coastlines()
# plt.scatter(lons, lats, c=ampltd_tec_vals, cmap='jet', marker=',', lw=0.5, s=2, transform=ccrs.PlateCarree())
# plt.colorbar(label='TECu/min', pad=0.05)
# plt.show()
#
# fig.savefig('2017-05-28_03-30-00-000000_TEC.png')

#TODO: добавить блок с изображением северного полушария и в полярных координатах
fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection=ccrs.Orthographic(0, 90))
ax.coastlines()
ax.gridlines(draw_labels=True, dms=True, color='gray', alpha=0.5, linestyle='--')
#plt.scatter(lons, lats, c=ampltd_tec_vals, cmap='jet', marker=',', lw=0.5, s=2, transform=ccrs.PlateCarree())
plt.show()

fig.savefig('polar_coords.png')
