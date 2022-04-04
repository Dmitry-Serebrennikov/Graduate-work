#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Название: aurora.py
# Описание: Программа для предварительной обработки данных ROTI
#
# Установка необходимых пакетов и библиотек:
# sudo apt update
# sudo apt install libgeos++-dev libgeos-3.8.0 libgeos-c1v5 libgeos-dev libgeos-doc
# sudo apt install proj-bin libproj-dev
# pip install shapely
# pip install pyshp
# pip install pyproj
# pip install cartopy
# pip install aacgmv2
#

import os
import sys
import h5py
import aacgmv2
import matplotlib
import numpy as np
import datetime as dt
import cartopy.crs as ccrs
from argparse import ArgumentParser

# высота ионосферы для пересчета географических координат в геомагнитные, км
IONOSPHERIC_HEIGHT = 400

def preprocess(file_name, hour_and_minute=''):
    """
    Обработка данных из HDF5-файла
    """

    # сбрасываем флаг использования всех моментов времени из файла
    all_time_steps = False

    # разделяем строку на часы и минуты
    try:
        hour = int(hour_and_minute.split(':')[0])
        minute = int(hour_and_minute.split(':')[1])
    except:
        all_time_steps = True

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
            for date in dates:

                # преобразуем дату в объект datetime
                time = dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

                # если используем все моменты времени или текущий момент совпадает с заданным
                if all_time_steps or (time.hour == hour and time.minute == minute):

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
                    mlats, mlons, _ = aacgmv2.convert_latlon_arr(lats, lons, IONOSPHERIC_HEIGHT, time,
                                                                 method_code='G2A')

                    # определяем минимальное значение
                    roti_min = int(rotis.min())
                    # определяем максимальное значение
                    roti_max = int(rotis.max())
                    # рассчитываем распределение значений ROTI
                    y, X = np.histogram(rotis, range=(roti_min, roti_max), density=True)
                    # удаляем последний элемент массива
                    X = X[:-1]
                    # определяем ширину интервала
                    width = X[1] - X[0]

                    # подключаем пакет matplotlib
                    import matplotlib.pyplot as plt
                    matplotlib.use('Agg')
                    # отображаем распределение значений ROTI
                    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), dpi=120, facecolor='w', edgecolor='k')
                    ax.bar(X, y, width=width * 0.9, align='center', color='brown', alpha=0.8, zorder=100)
                    ax.set_title('Гистограмма распределения ROTI, %s UTC' % time.strftime('%Y-%m-%d %H:%M'),
                                 fontsize=11)
                    ax.set_xlabel('ROTI, TECu/мин')
                    ax.set_ylabel('Плотность вероятности')
                    ax.grid(dashes=(5, 2, 1, 2), zorder=0)
                    plt.savefig('hist-%s.png' % time.strftime('%Y-%m-%dT%H-%M-%S'), dpi=120)

                    # отображаем данные ROTI на карте в геомагнитной системе координат
                    fig = plt.figure(figsize=(9, 9))
                    ax = plt.axes(projection=ccrs.Orthographic(0, 90))
                    ax.set_title('Карта значений ROTI, %s UTC' % time.strftime('%Y-%m-%d %H:%M'))
                    ax.gridlines(draw_labels=True, dms=True, color='gray', alpha=0.5, linestyle='--')
                    ax.scatter(mlons, mlats, c=rotis, cmap='jet', marker=',', lw=0.5, s=2, transform=ccrs.PlateCarree())
                    plt.savefig('map-%s.png' % time.strftime('%Y-%m-%dT%H-%M-%S'), dpi=120)

                    # отключаем пакет matplotlib
                    sys.modules.pop('matplotlib.pyplot')


#preprocess('dtec_20_60_2017_148_-90_90_N_-180_180_E_6c18.h5', '11:00')
def print_help():
    """
    Отображение справочной информации
    """

    print('usage: python3 aurora.py [-h] [-i [INPUT]] [-t [TIME]]')
    print()
    print('optional arguments:')
    print('  -h, --help            show this help message and exit')
    print('  -i [INPUT], --input [INPUT]')
    print('                        input HDF5 file name')
    print('  -t [HH:MM], --time [HH:MM]')
    print('                        UTC time in format HH:MM')

    return


if __name__ == '__main__':

    try:

        # вывод справочной информации
        if len(sys.argv) == 2 and ('-h' in sys.argv[1] or '--help' in sys.argv[1] or '/?' in sys.argv[1]):
            print_help()
            sys.exit()

        # разбор параметров командной строки
        parser = ArgumentParser()
        parser.add_argument('-i', '--input', nargs='?', help='input HDF5 file name')
        parser.add_argument('-t', '--time', nargs='?', help='UTC time in format HH:MM')
        args = parser.parse_args()

        # если указан ключ --input и HDF5-файл существует
        if args.input and os.path.isfile(args.input):
            # задаем имя HDF5-файла
            file_name = args.input
        else:
            # если не задано имя HDF5-файла, то выводим справочную информацию и завершаем работу
            print_help()
            sys.exit()

        # если при вызове указаны час и минута, UTC
        if args.time:
            hour_and_minute = args.time
        else:
            hour_and_minute = ''

        # запускаем обработку данных
        preprocess(file_name, hour_and_minute)

    except KeyboardInterrupt:

        sys.exit()
