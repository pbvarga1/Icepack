import re
from dataclasses import dataclass

import h5py
import numpy as np

RE_AIR_TEMP = re.compile(r'air temperature \(C\)\s+=\s+(-?\d+\.\d+)')
RE_ICE_AREA = re.compile(r'area fraction\s+=\s+(-?\d+\.\d+)')
RE_MELT_MOND = re.compile(r'pond height \(m\)\s+=\s+(-?\d+\.\d+)')
RE_ICE_THICKNESS = re.compile(r'avg ice thickness \(m\)\s+=\s+(-?\d+\.\d+)')
RE_SURFACE_TEMP = re.compile(r'surface temperature\(C\)\s+=\s+(-?\d+\.\d+)')
RE_SNOW_DEPTH = re.compile(r'avg snow depth \(m\)\s+=\s+(-?\d+\.\d+)')
RE_SALINITY = re.compile(r'avg salinity \(ppt\)\s+=\s+(-?\d+\.\d+)')
RE_BRINE = re.compile(r'avg brine thickness \(m\)=\s+(-?\d+\.\d+)')
RE_TOP_MELT = re.compile(r'top melt \(m\)\s+=\s+(-?\d+\.\d+)')
RE_BOTTOM_MELT = re.compile(r'bottom melt \(m\)\s+=\s+(-?\d+\.\d+)')
RE_LATERAL_MELT = re.compile(r'lateral melt \(m\)\s+=\s+(-?\d+\.\d+)')
RE_BRINE_HEIGHT = re.compile(r'\s+hbrine, \(m\)\s+=\s+(-?\d+\.\d+)')
RE_DARCY = re.compile(r'darcy speed \(\+down m/s\)=\s+(-?\d+\.\d+)')
RE_RAIN = re.compile(r'rainfall \(m\)\s+=\s+(-?\d+\.\d+)')
RE_FCONDBOT = re.compile(r'bot  conductive flux\s+=\s+(-?\d+\.\d+)')
RE_SPOND = re.compile(r'pond salinity \(ppt\)\s+=\s+(-?\d+\.\d+)')
RE_SICE = re.compile(
    r'\s+Sice bulk S \(ppt\)\s+-+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'
    r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)'
)
RE_TICE = re.compile(
    r'\s+Sice Temperature \(C\)\s+-+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'
    r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)'
)
RE_OCN = re.compile(r'ocean height\s+\(m\)\s+=\s+(-?\d+\.\d+)')
RE_ICE_MASS = re.compile(r'ice mass\s+\(kg m-2\)\s+=\s+(-?\d+\.\d+)')
RE_PERMEABILITY = re.compile(r'ice permeability \(m2\)\s+=\s+(-?\d+\.\d+)')
RE_POND_ALBEDO = re.compile(r'pond albedo\s+=\s+(-?\d+\.\d+)')
RE_TIME = re.compile(r'istep\d+:\s+\d+\s+idate:\s+(\d+)\s+sec:\s+(\d+)')
RE_PUMP = re.compile(r'pump amount \(m\)\s+=\s+(-?\d+\.\d+)')
RE_SUBL_COND = re.compile(r'subl/cond \(m ice\)\s+=\s+(-?\d+\.\d+)')
RE_CONGEL = re.compile(r'congelation \(m\)\s+=\s+(-?\d+\.\d+)')
RE_SNOICE = re.compile(r'snow-ice \(m\)\s+=\s+(-?\d+\.\d+)')
RE_EFFECTIVE_DHI = re.compile(r'effective dhi \(m\)\s+=\s+(-?\d+\.\d+)')
RE_NEW_ICE = re.compile(r'new ice \(m\)\s+=\s+(-?\d+\.\d+)')


def _get_data_from_output(output, regex):
    return np.array(regex.findall(output), dtype=float)


@dataclass(frozen=True)
class OutputData:

    lat: int
    lon: int
    melt_pond: np.ndarray
    ice_thickness: np.ndarray
    surface_temp: np.ndarray
    # snow_depth: np.ndarray
    salinity: np.ndarray
    # brine_thickness: np.ndarray
    top_melt: np.ndarray
    bottom_melt: np.ndarray
    lateral_melt: np.ndarray
    # brine_height: np.ndarray
    darcy: np.ndarray
    # rain: np.ndarray
    ice_area: np.ndarray
    air_temp: np.ndarray
    # fcondbot: np.ndarray
    sice: np.ndarray
    spond: np.array
    # ocean_height: np.ndarray
    # ice_mass: np.ndarray
    permeability: np.ndarray
    pond_albedo: np.ndarray
    times: np.ndarray
    pump_amount: np.ndarray
    subl_cond: np.ndarray
    # congelation: np.ndarray
    # snoice: np.ndarray
    # effective_dhi: np.ndarray
    # new_ice: np.ndarray
    ice_temp: np.ndarray

    @property
    def group_name(self):
        return f'{self.lat}_{self.lon}'

    @property
    def dates(self):
        dates = []
        for time in self.times[1:]:
            yearmonthday, seconds = time
            year = yearmonthday[:4]
            month = yearmonthday[4:6]
            day = yearmonthday[6:]
            date = np.datetime64(f'{year}-{month}-{day}')
            dates.append(date + np.timedelta64(int(seconds), 's'))
        dates = np.array(dates, dtype=np.dtype('datetime64[h]'))
        return dates

    @property
    def total_melt(self):
        return self.top_melt + self.bottom_melt + self.lateral_melt

    @property
    def total_loss(self):
        evap = self.subl_cond.copy()
        evap[evap > 0] = 0
        return self.total_melt + evap

    @property
    def cumulative_loss(self):
        return np.cumsum(self.total_loss)

    # def create_hdf5_group(self, hdf5_path, base):
    #     if base:
    #         group_name = f'{self.group_name}/base'
    #     else:
    #         group_name = f'{self.group_name}/experimental'

    #     with h5py.File(hdf5_path, 'a') as f:
    #         grp = f.create_group(group_name)
    #         grp.create_dataset(name='lat', data=np.array([self.lat]))
    #         grp.create_dataset(name='lon', data=np.array([self.lon]))
    #         grp.create_dataset(name='melt_pond', data=self.melt_pond)
    #         grp.create_dataset(name='ice_thickness', data=self.ice_thickness)
    #         grp.create_dataset(name='surface_temp', data=self.surface_temp)
    #         grp.create_dataset(name='snow_depth', data=self.snow_depth)
    #         grp.create_dataset(name='salinity', data=self.salinity)
    #         grp.create_dataset(
    #             name='brine_thickness',
    #             data=self.brine_thickness,
    #         )
    #         grp.create_dataset(name='top_melt', data=self.top_melt)
    #         grp.create_dataset(name='bottom_melt', data=self.bottom_melt)
    #         grp.create_dataset(name='lateral_melt', data=self.lateral_melt)
    #         grp.create_dataset(name='brine_height', data=self.brine_height)
    #         grp.create_dataset(name='darcy', data=self.darcy)
    #         grp.create_dataset(name='rain', data=self.rain)


def create_output_data(output_path, lat, lon):
    with open(output_path, 'r') as f:
        output = f.read()
    output_data = OutputData(
        lat=lat,
        lon=lon,
        melt_pond=_get_data_from_output(output, RE_MELT_MOND),
        ice_thickness=_get_data_from_output(output, RE_ICE_THICKNESS),
        surface_temp=_get_data_from_output(output, RE_SURFACE_TEMP),
        # snow_depth=_get_data_from_output(output, RE_SNOW_DEPTH),
        salinity=_get_data_from_output(output, RE_SALINITY),
        # brine_thickness=_get_data_from_output(output, RE_BRINE),
        top_melt=_get_data_from_output(output, RE_TOP_MELT),
        bottom_melt=_get_data_from_output(output, RE_BOTTOM_MELT),
        lateral_melt=_get_data_from_output(output, RE_LATERAL_MELT),
        # brine_height=_get_data_from_output(output, RE_BRINE_HEIGHT),
        darcy=_get_data_from_output(output, RE_DARCY),
        # rain=_get_data_from_output(output, RE_RAIN),
        ice_area=_get_data_from_output(output, RE_ICE_AREA),
        air_temp=_get_data_from_output(output, RE_AIR_TEMP),
        # fcondbot=_get_data_from_output(output, RE_FCONDBOT),
        sice=_get_data_from_output(output, RE_SICE),
        spond=_get_data_from_output(output, RE_SPOND),
        # ocean_height=_get_data_from_output(output, RE_OCN),
        # ice_mass=_get_data_from_output(output, RE_ICE_MASS),
        permeability=_get_data_from_output(output, RE_PERMEABILITY),
        pond_albedo=_get_data_from_output(output, RE_POND_ALBEDO),
        times=np.array(RE_TIME.findall(output)),
        pump_amount=_get_data_from_output(output, RE_PUMP),
        subl_cond=_get_data_from_output(output, RE_SUBL_COND),
        # congelation=_get_data_from_output(output, RE_CONGEL),
        # snoice=_get_data_from_output(output, RE_SNOICE),
        # effective_dhi=_get_data_from_output(output, RE_EFFECTIVE_DHI),
        # new_ice=_get_data_from_output(output, RE_NEW_ICE),
        ice_temp=_get_data_from_output(output, RE_TICE),
    )
    return output_data
