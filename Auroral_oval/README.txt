Установка необходимых библиотек:
pip install -r requirements.txt

Примеры запуска:
python aurora.py -i data/hdf5/out.h5 -m otsu -t 09:30:00
python aurora.py -i data/Synthetic_oval/Out1/01_MapROTI.dat -m kmeans

-h / --help   -  для вызова вспомогательного сообщения
-i / --input  -  путь к входному файлу
-m / --method -  применяемый метод (otsu / kmeans / median / quantile)
-t / --time   - анализируемое время в формате HH:MM:SS

Если не указан метод определения порогового значения, используется метод otsu
Если не указан интересуемый момент времени, обрабатывается весь файл