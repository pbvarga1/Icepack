import os
import copy
from functools import lru_cache
from datetime import datetime, timedelta
from contextlib import closing, contextmanager
from dateutil.relativedelta import relativedelta

import netCDF4
import numpy as np
from scipy.interpolate import griddata

from icepack.constants import MIN_LAT


class CNRMCMIP5:

    def __init__(self, filepath, scale=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._filepath = filepath
        name = os.path.basename(self._filepath)
        self._key = name.split('_')[0]
        self._scale = scale

        with self.open_dataset() as dataset:
            arctic_mask, lons, lats = self._get_arctic_lons_lats(dataset)
            self._dataset_lons, self._dataset_lats = lons, lats
            self._dates, time_slice = self._get_dates_time_slice(dataset)
            if dataset.variables[self.key].ndim == 4:
                data_slice = np.s_[time_slice, 0, arctic_mask, ...]
            else:
                data_slice = np.s_[time_slice, arctic_mask, ...]
            self._dataset_data = dataset.variables[self.key][data_slice]
        self._dataset_data = self._dataset_data * self._scale

        self._data = None
        self._grid_data = None
        self._lats = None
        self._lons = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.key})'

    def _get_arctic_lons_lats(self, dataset):
        lats = dataset.variables['lat'][:]
        lons = dataset.variables['lon'][:]
        arctic_mask = np.array(lats >= self.min_lat)
        if lats.ndim == 2:
            arctic_mask = np.any(arctic_mask, axis=1)
            arctic_lons = lons[arctic_mask, ...].copy()
            arctic_lats = lats[arctic_mask, ...].copy()
        else:
            arctic_lons, arctic_lats = np.meshgrid(
                lons,
                lats[arctic_mask],
            )
        return arctic_mask, arctic_lons, arctic_lats

    def _get_dates_time_slice(self, dataset):
        time = dataset.variables['time'][:]
        start_index = stop_index = None
        dates = []
        for index, days_since_epoch in enumerate(time):
            date = self.epoch + timedelta(days=days_since_epoch)
            dates.append(date)
            if start_index is None and date >= self.start_date:
                start_index = index
                continue
            if stop_index is None and date >= self.stop_date:
                stop_index = index
                break
        else:
            stop_index = index
        time_slice = np.s_[start_index:stop_index + 1]
        return dates, time_slice

    @property
    def filepath(self):
        return str(self._filepath)

    @property
    def by_month(self):
        return 'mon' in self.filepath

    @property
    def by_day(self):
        return 'day' in self.filepath

    @contextmanager
    def open_dataset(self):
        with closing(netCDF4.MFDataset(self._filepath)) as dataset:
            yield dataset

    @property
    def epoch(self):
        return datetime(2006, 1, 1)

    @property
    def start_date(self):
        return datetime(2016, 1, 1, 0, 0, 0)

    @property
    def stop_date(self):
        return datetime(2026, 1, 1, 0, 0, 0)

    @property
    @lru_cache(maxsize=1)
    def total_years(self):
        return relativedelta(self.stop_date, self.start_date).years

    @property
    def min_lat(self):
        return MIN_LAT

    @property
    def key(self):
        return self._key

    @property
    def dates(self):
        return copy.deepcopy(self._dates)

    @lru_cache(maxsize=1)
    def get_hours_from_start(self):
        start_date = self.start_date
        diffs = [(date - start_date).total_seconds() for date in self.dates]
        hours_from_start = np.array(diffs) // 3600
        hours_from_start = hours_from_start.astype(int).tolist()
        return hours_from_start

    @property
    def dataset_times(self):
        return np.array([(d - self.epoch).days for d in self.dates])

    @property
    def dataset_lons(self):
        return np.array(self._dataset_lons.copy())

    @property
    def dataset_lats(self):
        return np.array(self._dataset_lats.copy())

    @property
    def dataset_data(self):
        return np.ma.array(self._dataset_data.copy())

    @property
    def dataset_shape(self):
        return self._dataset_data.shape

    @property
    def dataset_size(self):
        return self._dataset_lons.size

    @property
    def data(self):
        if self._data is None:
            return self.dataset_data
        else:
            return np.ma.array(self._data.copy())

    @property
    def mask(self):
        mask = np.ma.getmaskarray(self.data[0])
        for data in self.data[1:]:
            mask = mask | np.ma.getmaskarray(data)
        return mask

    @property
    def lons(self):
        if self._lons is None:
            return self._dataset_lons.copy()
        else:
            return self._lons.copy()

    @property
    def lats(self):
        if self._lats is None:
            return self._dataset_lats.copy()
        else:
            return self._lats.copy()

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._lons.size

    def set_grid_data(self, lats, lons):
        same_lats = np.array_equal(lats, self.dataset_lats)
        same_lons = np.array_equal(lons, self.dataset_lons)
        if same_lats and same_lons:
            self._data = self.dataset_data
            self._lons = self.dataset_lons
            self._lats = self.dataset_lats
            return

        datas = []
        points = self._dataset_lats.flatten(), self._dataset_lons.flatten()
        for datain in self.dataset_data:
            data = griddata(
                points=points,
                values=datain.flatten(),
                xi=(lats, lons),
                method='nearest'
            )
            datas.append(data)
        self._data = np.ma.vstack(
            [np.ma.expand_dims(d, axis=0) for d in datas]
        )
        self._lons = lons
        self._lats = lats

    def get_data(self, lat_idx, lon_idx, interp_to_hours):
        data = self.data[:, lat_idx, lon_idx]
        if interp_to_hours:
            datap = data
            xp = self.get_hours_from_start()
            x = np.arange(1, self.total_years * 365 * 24 + 1)
            data = np.interp(x, xp, datap)
        return data
