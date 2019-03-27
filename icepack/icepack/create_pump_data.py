import argparse

import os

import numpy as np
import pandas as pd


def create_pump_data(times, amounts, total_hours):
    pumps = np.zeros(total_hours)
    if len(amounts) == 1:
        amounts = np.zeros(len(times)) + amounts[0]
    assert len(amounts) == len(times)
    times = np.array(times)
    amounts = np.array(amounts)
    pumps[times] = amounts
    pump_data = pd.DataFrame({'pump_amnt_data': pumps})
    return pump_data


def create_pump_data_file(filepath, times, amounts, total_hours):
    pump_data = create_pump_data(times, amounts, total_hours)
    with open(filepath, 'w') as f:
        pump_data.to_csv(
            f,
            sep=' ',
            header=None,
            index=False,
            float_format='%.6f',
        )
    return


def cli():
    parser = argparse.ArgumentParser(description='Create pumping data')
    parser.add_argument('filepath', type=str, help='Path to new data file')
    parser.add_argument('-t', dest='times', type=int, nargs='+')
    parser.add_argument('-p', dest='amounts', type=float, nargs='+')
    parser.add_argument('-n', dest='total_hours', type=int)
    args = parser.parse_args()
    if os.path.exists(args.filepath):
        raise RuntimeError(f'{args.filepath} already exists')
    create_pump_data_file(
        filepath=os.path.abspath(args.filepath),
        times=args.times,
        amounts=args.amounts,
        total_hours=args.total_hours,
    )
