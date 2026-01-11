# spectrumlab

Библиотека для работы с атомно-эмиссионными (АЭС) и атомно-абсорбционными (ААС) спектрами, полученными с использованием линейных детекторов излучения.


### ENV
Преременные окружения алгоритма поиска пиков в спектре:
- `DRAFT_BLINKS_N_COUNTS_MIN=10` - минимальное количество отсчетов пика;
- `DRAFT_BLINKS_N_COUNTS_MAX=100` - максимальное количество отсчетов пика;
- `DRAFT_BLINKS_EXCEPT_CLIPPED_PEAK=True` - исключить пики с "зашкаленными" отсчетами;
- `DRAFT_BLINKS_EXCEPT_WIDE_PEAK=False` - исключить пики с шириной больше `DRAFT_BLINKS_WIDTH_MAX`;
- `DRAFT_BLINKS_EXCEPT_SLOPED_PEAK=True` - исключить пики с наклоном больше `DRAFT_BLINKS_SLOPE_MAX`;
- `DRAFT_BLINKS_EXCEPT_EDGES=False` - исключить крайние отсчеты пика;
- `DRAFT_BLINKS_AMPLITUDE_MIN=0` - минимальная амплатуда пика;
- `DRAFT_BLINKS_WIDTH_MAX=3.5` - максимальная ширина пика;
- `DRAFT_BLINKS_SLOPE_MAX=.25` - максимальный уровень наклона пика;
- `DRAFT_BLINKS_NOISE_LEVEL=10` - уровень амплитуды пика относительно шума;

Преременные окружения алгоритма вычисления формы контура пика в спектре:
- `RETRIEVE_SHAPE_DEFAULT=2;0;.1` - форма контура пика по умолчанию;
- `RETRIEVE_SHAPE_MIN_WIDTH` - минимальная ширина формы контура пика;
- `RETRIEVE_SHAPE_MAX_WIDTH` - максимальная ширина формы контура пика;
- `RETRIEVE_SHAPE_MAX_ASYMMETRY` - максимальная асимметрия формы контура пика;
- `RETRIEVE_SHAPE_ERROR_MAX=.001` - максимальное отклонение пика от формы;
- `RETRIEVE_SHAPE_ERROR_MEAN=.0001` - среднее отклонение пика от формы;
- `RETRIEVE_SHAPE_N_PEAKS_FILTRATED_BY_WIDTH=None` - фильтрация пиков по ширине;
- `RETRIEVE_SHAPE_N_PEAKS_MIN=10` - минимальное количество пиков;
