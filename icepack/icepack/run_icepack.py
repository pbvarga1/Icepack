import io
import os
import re
import json
import shutil
import asyncio
import argparse
import subprocess
from contextlib import contextmanager

import h5py
import jinja2
import aiofiles
import numpy as np

from icepack.constants import MIN_LAT
from icepack.icepack_data import IcePackData
from icepack.read_diag import create_output_data


def create_case(case, machine, env):
    proc = subprocess.run(
        ['./icepack.setup', '--case', case, '--mach', machine, '--env', env],
        stdout=subprocess.PIPE,
        text=True
    )
    print(proc.stdout)
    match = re.search(
        r'ICE_CASEDIR\s+=\s+(.*?{0})'.format(case),
        proc.stdout
    )
    if not match:
        match = re.search(
            r'ERROR, case {0} already exists'.format(case),
            proc.stdout,
        )
        if match:
            case_dir = os.path.join(os.getcwd(), case)
        else:
            raise RuntimeError('Did not create case')
    else:
        case_dir = match.group(1)
    return case_dir


def create_forcing_data_path(name, data_dir, lat, lon):
    file_name = f'{name}_{lat}_{lon}.txt'
    file_path = os.path.join(data_dir, 'THESIS', file_name)
    return file_path


def create_forcing_data(data_dir, forcing_set, lat, lon, overwrite=False):
    file_names = {}
    for name, forcing in forcing_set._asdict().items():
        file_path = create_forcing_data_path(name, data_dir, lat, lon)
        file_names[name] = file_path
        if overwrite or not os.path.exists(file_path):
            with open(file_path, mode='w', newline='') as f:
                forcing.to_csv(
                    f,
                    sep=' ',
                    header=None,
                    index=False,
                    float_format='%.6f',
                )
    return file_names


async def write_csv_buffer(file_path, csv_buffer):
    async with aiofiles.open(file_path, 'w') as f:
        return await f.write(csv_buffer.getvalue())


async def create_forcing_data_async(data_dir, forcing_set, lat, lon,
                                    overwrite=False):
    futures = []
    for name, forcing in forcing_set._asdict().items():
        file_path = create_forcing_data_path(name, data_dir, lat, lon)
        if overwrite or not os.path.exists(file_path):
            csv_buffer = io.StringIO(newline='')
            forcing.to_csv(
                csv_buffer,
                sep=' ',
                header=None,
                index=False,
                float_format='%.6f',
            )
            futures.append(write_csv_buffer(file_path, csv_buffer))
    await asyncio.gather(*futures)


def create_icepack_in(case_dir, data_dir, atm_data_file, ocn_data_file,
                      bgc_data_file, pump_start, pump_end, pump_amount,
                      pump_repeats, lat, lon, base):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            os.path.abspath(os.path.dirname(__file__))
        ),
        keep_trailing_newline=True,
    )
    template = env.get_template('icepack_in.template')
    if base:
        icepack_in = os.path.join(case_dir, f'icepack_in_base_{lat}_{lon}')
    else:
        icepack_in = os.path.join(case_dir, f'icepack_in_{lat}_{lon}')
    with open(icepack_in, mode='w') as f:
        f.write(
            template.render(
                data_dir=data_dir,
                atm_data_file=atm_data_file,
                ocn_data_file=ocn_data_file,
                bgc_data_file=bgc_data_file,
                pump_start=pump_start,
                pump_end=pump_end,
                pump_amount=pump_amount,
                pump_repeats=pump_repeats,
                lat=lat,
                lon=lon,
            )
        )
    return icepack_in


async def create_icepack_in_async(case_dir, data_dir, atm_data_file,
                                  ocn_data_file, bgc_data_file, pump_start,
                                  pump_end, pump_amount, pump_repeats, lat,
                                  lon, base):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            os.path.abspath(os.path.dirname(__file__))
        ),
        keep_trailing_newline=True,
        enable_async=True
    )
    template = env.get_template('icepack_in.template')
    template_fut = template.render_async(
        data_dir=data_dir,
        atm_data_file=atm_data_file,
        ocn_data_file=ocn_data_file,
        bgc_data_file=bgc_data_file,
        pump_start=pump_start,
        pump_end=pump_end,
        pump_amount=pump_amount,
        pump_repeats=pump_repeats,
        lat=lat,
        lon=lon,
    )
    if base:
        icepack_in = os.path.join(case_dir, f'icepack_in_base_{lat}_{lon}')
    else:
        icepack_in = os.path.join(case_dir, f'icepack_in_{lat}_{lon}')
    async with aiofiles.open(icepack_in, mode='w') as f:
        await f.write(await template_fut)
    return icepack_in


@contextmanager
def work_in_case_dir(case_dir):
    cwd = os.getcwd()
    try:
        os.chdir(case_dir)
        yield
    except Exception:
        raise
    finally:
        os.chdir(cwd)


def build():
    proc = subprocess.run(
        ['./icepack.build'],
        stdout=subprocess.PIPE,
        text=True
    )
    print(proc.stdout)


def submit(case_name, icepack_in=''):
    proc = subprocess.run(
        ['./icepack.run', icepack_in],
        stdout=subprocess.PIPE,
        text=True,
    )
    print(proc.stdout)
    match = re.search(
        pattern=r'ICEPACK rundir is (.*?{0}\/?)'.format(case_name),
        string=proc.stdout,
    )
    if not match:
        raise RuntimeError('Failed!')
    output_dir = match.group(1)
    return output_dir


async def submit_async(case_name, icepack_in=''):
    cmd = f'./icepack.run {os.path.basename(icepack_in)}'
    print(cmd)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    stdout = stdout.decode()
    print(stdout)
    match = re.search(
        pattern=r'ICEPACK rundir is (.*?{0}\/?)'.format(case_name),
        string=stdout,
    )
    if not match:
        raise RuntimeError('Failed!')
    output_dir = match.group(1)
    return output_dir


def create_hdf5_file(case_name, vers=0):
    cwd = os.getcwd()
    name = f'{case_name}_{vers:02d}.hdf5'
    hdf5_path = os.path.join(cwd, name)
    if os.path.exists(hdf5_path):
        hdf5_path = create_hdf5_file(case_name, vers + 1)
    return hdf5_path


def get_icepack_settings(settings, case_dir, lat, lon, forcing_set=None):
    settings = dict(settings)
    if settings['overwrite']:
        file_names = create_forcing_data(
            data_dir=settings['data_dir'],
            forcing_set=forcing_set,
            lat=lat,
            lon=lon,
            overwrite=settings['overwrite'],
        )
    else:
        file_names = {
            'atm': create_forcing_data_path(
                name='atm',
                data_dir=settings['data_dir'],
                lat=lat,
                lon=lon,
            ),
            'ocn': create_forcing_data_path(
                name='ocn',
                data_dir=settings['data_dir'],
                lat=lat,
                lon=lon,
            ),
            'bgc': create_forcing_data_path(
                name='bgc',
                data_dir=settings['data_dir'],
                lat=lat,
                lon=lon,
            ),
        }
    icepack_settings = {
        'case_dir': case_dir,
        'data_dir': settings['data_dir'],
        'atm_data_file': os.path.basename(file_names['atm']),
        'ocn_data_file': os.path.basename(file_names['ocn']),
        'bgc_data_file': os.path.basename(file_names['bgc']),
        'pump_start': 0,
        'pump_end': 0,
        'pump_amount': settings['pump_amount'],
        'pump_repeats': settings['pump_repeats'],
        'lat': lat,
        'lon': lon,
    }
    return icepack_settings


async def run_locations(settings, case_dir, case_name, locations, indices,
                        max_ice_diff, five_years, decade, hdf5_path):
    futures = []
    melt_days = {}
    for lat, lon in locations:
        futures.append(run_icepack_location_base(
            settings=settings,
            case_dir=case_dir,
            case_name=case_name,
            lat=lat,
            lon=lon,
            melt_days=melt_days,
            forcing_set=None,
        ))
    await asyncio.gather(*futures)
    futures = []
    for lat, lon in locations:
        futures.append(run_icepack_location_exp(
            settings=settings,
            case_dir=case_dir,
            case_name=case_name,
            lat=lat,
            lon=lon,
            melt_days=melt_days,
        ))
    outdirs = await asyncio.gather(*futures)
    iter_compare = zip(outdirs, locations, indices)
    for output_dir, (lat, lon), (latidx, lonidx) in iter_compare:
        compare_base_to_exp(
            lat=lat,
            lon=lon,
            latidx=latidx,
            lonidx=lonidx,
            output_dir=output_dir,
            max_ice_diff=max_ice_diff,
            five_years=five_years,
            decade=decade,
            hdf5_path=hdf5_path,
        )


async def run_icepack_location_base(settings, case_dir, case_name, lat, lon,
                                    melt_days, forcing_set=None):
    icepack_settings = get_icepack_settings(
        settings=settings,
        case_dir=case_dir,
        lat=lat,
        lon=lon,
        forcing_set=forcing_set,
    )
    icepack_settings['pump_amount'] = 0
    icepack_in = await create_icepack_in_async(**icepack_settings, base=True)
    output_dir = await submit_async(case_name, icepack_in)
    full_ITD = f'ice_diag.{lat}_{lon}.full_ITD'
    full_ITD_base = f'ice_diag.{lat}_{lon}_base.full_ITD'
    output_path = os.path.join(output_dir, full_ITD)
    base_output_path = os.path.join(output_dir, full_ITD_base)
    shutil.copy(output_path, base_output_path)
    base_output = create_output_data(base_output_path, lat, lon)
    first_melt_day = np.argmax(base_output.melt_pond[:365])
    melt_days[(lat, lon)] = first_melt_day


async def run_icepack_location_exp(settings, case_dir, case_name, lat, lon,
                                   melt_days):
    icepack_settings = get_icepack_settings(
        settings=settings,
        case_dir=case_dir,
        lat=lat,
        lon=lon,
        forcing_set=None,
    )
    first_melt_day = melt_days[(lat, lon)]
    icepack_settings['pump_start'] = first_melt_day + 10
    icepack_settings['pump_end'] = first_melt_day + 10
    icepack_in = await create_icepack_in_async(**icepack_settings, base=False)
    output_dir = await submit_async(case_name, icepack_in)
    return output_dir


def compare_base_to_exp(lat, lon, latidx, lonidx, output_dir, max_ice_diff,
                        five_years, decade, hdf5_path):
    full_ITD = f'ice_diag.{lat}_{lon}.full_ITD'
    full_ITD_base = f'ice_diag.{lat}_{lon}_base.full_ITD'
    base_output_path = os.path.join(output_dir, full_ITD_base)
    output_path = os.path.join(output_dir, full_ITD)
    output = create_output_data(output_path, lat, lon)
    base_output = create_output_data(base_output_path, lat, lon)
    ice_diff = output.ice_thickness - base_output.ice_thickness
    max_ice_diff[latidx, lonidx] = np.max(ice_diff)
    print(f'Max ice diff: {np.max(ice_diff)} m')
    decade[latidx, lonidx] = ice_diff[-1]
    five_years[latidx, lonidx] = ice_diff[365 * 5]
    base_output.create_hdf5_group(
        hdf5_path=hdf5_path,
        base=True,
    )
    output.create_hdf5_group(
        hdf5_path=hdf5_path,
        base=False,
    )


def run(case_name, settings_path):
    print(os.getcwd())
    with open(settings_path, 'r') as stream:
        settings = json.load(stream)
    print(f"Creating case {case_name}")
    case_dir = create_case(
        case=case_name,
        machine=settings['machine'],
        env=settings['env']
    )
    forcing_paths = settings['forcing_paths']
    forcing_data_dir = forcing_paths['data_dir']
    forcing_paths = {
        name: os.path.join(forcing_data_dir, path)
        for name, path in forcing_paths.items()
    }
    del forcing_paths['data_dir']
    print("loading icepack data")
    icepack_data = IcePackData(**forcing_paths)
    # try:
    #     icepack_data.create_dataset(f'{case_name}.nc')
    # except Exception as e:
    #     print(f'Failed to create with exception: {str(e)}')
    hdf5_path = create_hdf5_file(case_name)
    with h5py.File(hdf5_path):
        pass
    max_ice_diff = np.zeros(icepack_data.shape)
    decade = np.zeros(icepack_data.shape)
    five_years = np.zeros(icepack_data.shape)

    with work_in_case_dir(case_dir):
        print("Compiling")
        build()
        if settings['overwrite']:
            icepack_iter = icepack_data.iter_forcing
        else:
            icepack_iter = icepack_data.iter_lat_lon
        for icepack_items in icepack_iter:
            if settings['overwrite']:
                latidx, lonidx, lat, lon, forcing_set = icepack_items
            else:
                latidx, lonidx, lat, lon = icepack_items
            if lat < MIN_LAT:
                continue
            print(lat, lon)
            if settings['overwrite']:
                file_names = create_forcing_data(
                    data_dir=settings['data_dir'],
                    forcing_set=forcing_set,
                    lat=lat,
                    lon=lon,
                    overwrite=settings['overwrite'],
                )
            else:
                file_names = {
                    'atm': create_forcing_data_path(
                        name='atm',
                        data_dir=settings['data_dir'],
                        lat=lat,
                        lon=lon,
                    ),
                    'ocn': create_forcing_data_path(
                        name='ocn',
                        data_dir=settings['data_dir'],
                        lat=lat,
                        lon=lon,
                    ),
                    'bgc': create_forcing_data_path(
                        name='bgc',
                        data_dir=settings['data_dir'],
                        lat=lat,
                        lon=lon,
                    ),
                }
            icepack_settings = {
                'case_dir': case_dir,
                'data_dir': settings['data_dir'],
                'atm_data_file': os.path.basename(file_names['atm']),
                'ocn_data_file': os.path.basename(file_names['ocn']),
                'bgc_data_file': os.path.basename(file_names['bgc']),
                'pump_start': 0,
                'pump_end': 0,
                'pump_amount': settings['pump_amount'],
                'pump_repeats': settings['pump_repeats'],
                'lat': lat,
                'lon': lon,
            }
            base_settings = dict(icepack_settings)
            base_settings['pump_amount'] = 0
            create_icepack_in(**base_settings)
            output_dir = submit(case_name)
            full_ITD = f'ice_diag.{lat}_{lon}.full_ITD'
            base_output_path = os.path.join(output_dir, full_ITD)
            base_output = create_output_data(base_output_path, lat, lon)
            first_melt_day = np.argmax(base_output.melt_pond[:365])
            icepack_settings['pump_start'] = first_melt_day + 10
            icepack_settings['pump_end'] = first_melt_day + 10
            create_icepack_in(**icepack_settings)
            output_dir = submit(case_name)
            output_path = os.path.join(output_dir, full_ITD)
            output = create_output_data(output_path, lat, lon)
            ice_diff = output.ice_thickness - base_output.ice_thickness
            max_ice_diff[latidx, lonidx] = np.max(ice_diff)
            print(f'Max ice diff: {np.max(ice_diff)} m')
            decade[latidx, lonidx] = ice_diff[-1]
            five_years[latidx, lonidx] = ice_diff[365 * 5]
            base_output.create_hdf5_group(
                hdf5_path=hdf5_path,
                base=True,
            )
            output.create_hdf5_group(
                hdf5_path=hdf5_path,
                base=False,
            )

    with h5py.File(hdf5_path, 'a') as f:
        f.create_dataset(name='max', data=max_ice_diff)
        f.create_dataset(name='decade', data=decade)
        f.create_dataset(name='five_years', data=five_years)


async def run_async(case_name, settings_path):
    print(os.getcwd())
    with open(settings_path, 'r') as stream:
        settings = json.load(stream)
    print(f"Creating case {case_name}")
    case_dir = create_case(
        case=case_name,
        machine=settings['machine'],
        env=settings['env']
    )
    forcing_paths = settings['forcing_paths']
    forcing_data_dir = forcing_paths['data_dir']
    forcing_paths = {
        name: os.path.join(forcing_data_dir, path)
        for name, path in forcing_paths.items()
    }
    del forcing_paths['data_dir']
    print("loading icepack data")
    icepack_data = IcePackData(**forcing_paths)
    # try:
    #     icepack_data.create_dataset(f'{case_name}.nc')
    # except Exception as e:
    #     print(f'Failed to create with exception: {str(e)}')
    hdf5_path = create_hdf5_file(case_name)
    with h5py.File(hdf5_path):
        pass
    max_ice_diff = np.zeros(icepack_data.shape)
    decade = np.zeros(icepack_data.shape)
    five_years = np.zeros(icepack_data.shape)

    with work_in_case_dir(case_dir):
        print("Compiling")
        build()
        locations = []
        indices = []
        for latidx, lonidx, lat, lon in icepack_data.iter_lat_lon:
            if lat < MIN_LAT:
                continue
            locations.append((lat, lon))
            indices.append((latidx, lonidx))
            if len(locations) == 10:
                await run_locations(
                    settings=settings,
                    case_dir=case_dir,
                    case_name=case_name,
                    locations=locations,
                    indices=indices,
                    max_ice_diff=max_ice_diff,
                    five_years=five_years,
                    decade=decade,
                    hdf5_path=hdf5_path,
                )
                locations = []
                indices = []

    with h5py.File(hdf5_path, 'a') as f:
        f.create_dataset(name='max', data=max_ice_diff)
        f.create_dataset(name='decade', data=decade)
        f.create_dataset(name='five_years', data=five_years)
        f.create_dataset(name='lats', data=icepack_data.lats)
        f.create_dataset(name='lons', data=icepack_data.lons)
        f.create_dataset(name='mask', data=icepack_data.mask)


async def create_all_forcing_data(case_name, settings_path):
    with open(settings_path, 'r') as stream:
        settings = json.load(stream)
    forcing_paths = settings['forcing_paths']
    forcing_data_dir = forcing_paths['data_dir']
    print(forcing_data_dir)
    forcing_paths = {
        name: os.path.join(forcing_data_dir, path)
        for name, path in forcing_paths.items()
    }
    del forcing_paths['data_dir']
    icepack_data = IcePackData(**forcing_paths)
    futures = []
    for latidx, lonidx, lat, lon, forcing_set in icepack_data.iter_forcing:
        print(lat, lon)
        futures.append(create_forcing_data_async(
            data_dir=settings['data_dir'],
            forcing_set=forcing_set,
            lat=lat,
            lon=lon,
            overwrite=settings['overwrite'],
        ))
        if len(futures) == 10:
            await asyncio.gather(*futures)
            futures = []
    if futures:
        await asyncio.gather(*futures)


async def run_input_files(case_name, input_files):
    print(input_files)
    processes = []
    for icepack_in in input_files:
        print(icepack_in)
        processes.append(submit_async(case_name, icepack_in))
        if len(processes) == 3:
            await asyncio.gather(*processes)
            processes = []
    if len(processes):
        await asyncio.gather(*processes)


def cli():
    parser = argparse.ArgumentParser(
        description="Paramaterize icepack"
    )
    parser.add_argument(
        'case_name',
        type=str,
    )
    parser.add_argument(
        'settings',
        type=str
    )
    parser.add_argument(
        '--forcing_only',
        action='store_true',
        help="Only create the forcing data"
    )
    args = parser.parse_args()
    if args.forcing_only:
        asyncio.run(create_all_forcing_data(
            case_name=args.case_name,
            settings_path=args.settings,
        ))
    else:
        asyncio.run(run_async(
            case_name=args.case_name,
            settings_path=args.settings,
        ))


def cli2():
    parser = argparse.ArgumentParser(
        description="Multiple input files"
    )
    parser.add_argument(
        'case_name',
        type=str,
    )
    parser.add_argument('-i', nargs='+', dest='input_files', type=str)
    args = parser.parse_args()
    asyncio.run(run_input_files(args.case_name, args.input_files))
