from contextlib import closing
from collections import namedtuple

import netCDF4
import numpy as np
import pandas as pd

from icepack.cnrmcmip5 import CNRMCMIP5

ForcingSet = namedtuple('ForcingSet', ('atm', 'ocn', 'bgc'))


class IcePackData:

    def __init__(self, rsds_path, rlds_path, uas_path, vas_path, tas_path,
                 huss_path, pr_path, tos_path, sos_path, mlotst_path, uo_path,
                 vo_path, si_path, no3_path, **kwargs):
        self._rsds_path = rsds_path
        self._rlds_path = rlds_path
        self._uas_path = uas_path
        self._vas_path = vas_path
        self._tas_path = tas_path
        self._huss_path = huss_path
        self._pr_path = pr_path
        self._tos_path = tos_path
        self._sos_path = sos_path
        self._mlotst_path = mlotst_path
        self._uo_path = uo_path
        self._vo_path = vo_path
        self._si_path = si_path
        self._no3_path = no3_path
        print('loading rsds')
        self._rsds = CNRMCMIP5(self._rsds_path)
        print('loading rlds')
        self._rlds = CNRMCMIP5(self._rlds_path)
        print('loading uas')
        self._uas = CNRMCMIP5(self._uas_path)
        print('loading vas')
        self._vas = CNRMCMIP5(self._vas_path)
        print('loading tas')
        self._tas = CNRMCMIP5(self._tas_path)
        print('loading huss')
        self._huss = CNRMCMIP5(self._huss_path)
        print('loading pr')
        self._pr = CNRMCMIP5(self._pr_path)
        print('loading tos')
        self._tos = CNRMCMIP5(self._tos_path)
        print('loading sos')
        self._sos = CNRMCMIP5(self._sos_path)
        print('loading mlotst')
        self._mlotst = CNRMCMIP5(self._mlotst_path)
        print('loading uo')
        self._uo = CNRMCMIP5(self._uo_path)
        print('loading vo')
        self._vo = CNRMCMIP5(self._vo_path)
        print('loading si')
        self._si = CNRMCMIP5(self._si_path, scale=1000)
        print('loading no3')
        self._no3 = CNRMCMIP5(self._no3_path, scale=1000)

        lowest_res_forcing = self._get_lowest_res_forcing()
        self._lats = lowest_res_forcing.dataset_lats
        self._lons = lowest_res_forcing.dataset_lons
        self._mask = np.zeros(self._lats.shape).astype(bool)

        print('interpolating to uniform data')
        lowest_res_forcing = self._set_uniform_grid_data()

    @property
    def atm(self):
        return [
            self._rsds,
            self._rlds,
            self._uas,
            self._vas,
            self._tas,
            self._huss,
            self._pr,
        ]

    @property
    def atm_names(self):
        return [
            'rsds',
            'rlds',
            'uas',
            'vas',
            'tas',
            'huss',
            'pr',
        ]

    @property
    def ocn(self):
        return [
            self._tos,
            self._sos,
            self._mlotst,
            self._uo,
            self._vo,
        ]

    @property
    def ocn_names(self):
        return [
            'tos',
            'sos',
            'mlotst',
            'uo',
            'vo',
        ]

    @property
    def bgc(self):
        return [
            self._si,
            self._no3,
        ]

    @property
    def bgc_names(self):
        return [
            'si',
            'no3',
        ]

    @property
    def all_forcing(self):
        return self.atm + self.ocn + self.bgc

    @property
    def lats(self):
        return self._lats.copy()

    @property
    def lons(self):
        return self._lons.copy()

    @property
    def mask(self):
        return self._mask.copy()

    @property
    def shape(self):
        return self._lats.shape

    def _get_lowest_res_forcing(self):
        lowest_res_forcing = None
        for forcing in self.all_forcing:
            if lowest_res_forcing is None:
                lowest_res_forcing = forcing
                continue
            if forcing.dataset_size < lowest_res_forcing.dataset_size:
                lowest_res_forcing = forcing
        return lowest_res_forcing

    def _set_uniform_grid_data(self):
        for forcing in self.all_forcing:
            forcing.set_grid_data(self._lats, self._lons)
            self._mask = self._mask | forcing.mask

    @property
    def iter_lat_lon(self):
        lats = self.lats
        lons = self.lons
        for latidx, lonidx in np.ndindex(lats.shape):
            if not self._mask[latidx, lonidx]:
                lat = lats[latidx, lonidx]
                lon = lons[latidx, lonidx]
                yield latidx, lonidx, lat, lon

    def get_forcing_df(self, latidx, lonidx, names, forcings, interp_to_hours):
        return pd.DataFrame(
            {
                name: forcing.get_data(latidx, lonidx, interp_to_hours)
                for name, forcing in zip(names, forcings)
            }
        )

    def get_forcing_set(self, latidx, lonidx):
        atm = self.get_forcing_df(
            latidx=latidx,
            lonidx=lonidx,
            names=self.atm_names,
            forcings=self.atm,
            interp_to_hours=True,
        )
        ocn = self.get_forcing_df(
            latidx=latidx,
            lonidx=lonidx,
            names=self.ocn_names,
            forcings=self.ocn,
            interp_to_hours=True,
        )
        bgc = self.get_forcing_df(
            latidx=latidx,
            lonidx=lonidx,
            names=self.bgc_names,
            forcings=self.bgc,
            interp_to_hours=False,
        )
        return ForcingSet(atm, ocn, bgc)

    @property
    def iter_forcing(self):
        for latidx, lonidx, lat, lon in self.iter_lat_lon:
            atm = self.get_forcing_df(
                latidx=latidx,
                lonidx=lonidx,
                names=self.atm_names,
                forcings=self.atm,
                interp_to_hours=True,
            )
            ocn = self.get_forcing_df(
                latidx=latidx,
                lonidx=lonidx,
                names=self.ocn_names,
                forcings=self.ocn,
                interp_to_hours=True,
            )
            bgc = self.get_forcing_df(
                latidx=latidx,
                lonidx=lonidx,
                names=self.bgc_names,
                forcings=self.bgc,
                interp_to_hours=False,
            )
            yield latidx, lonidx, lat, lon, ForcingSet(atm, ocn, bgc)

    def create_dataset(self, filepath):
        with closing(netCDF4.Dataset(filepath, mode='w')) as dataset:
            dtype = np.float32
            atm_time = dataset.createDimension('atm_time', self._rsds.shape[0])
            ocn_time = dataset.createDimension('ocn_time', self._uo.shape[0])
            bgc_time = dataset.createDimension('bgc_time', self._no3.shape[0])
            lat, lon = self.lats.shape
            lat = dataset.createDimension('lat', (lat,))
            lon = dataset.createDimension('lon', (lon,))
            print('creating lats')
            lats = dataset.createVariable('lats', dtype, (lat, lon))
            lats[:] = self.lats
            print('creating lons')
            lons = dataset.createVariable('lons', dtype, (lat, lon))
            lons[:] = self.lons
            for name, atm in zip(self.atm_names, self.atms):
                var = dataset.createVariable(name, dtype, (atm_time, lat, lon))
                var[:] = atm.data
            for name, ocn in zip(self.ocn_names, self.ocns):
                var = dataset.createVariable(name, dtype, (ocn_time, lat, lon))
                var[:] = ocn.data
            for name, bgc in zip(self.bgc_names, self.bgcs):
                var = dataset.createVariable(name, dtype, (bgc_time, lat, lon))
                var[:] = bgc.data
