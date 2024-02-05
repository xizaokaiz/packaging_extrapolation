#!/usr/bin/env python3

from packaging_extrapolation import Extrapolation
import argparse
import sys
import pandas as pd
import os
from packaging_extrapolation import UtilLog


def get_args():
    parser = argparse.ArgumentParser(description="Log file path")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Enter the result file path",
    )

    parser = argparse.ArgumentParser(description="Output file path")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Enter the output file path",
    )

    args = parser.parse_args().__dict__
    return args


def main():
    params = get_args()
    log_path = params['path']
    output_path = params['output']

    folder_path = log_path
    data_df = pd.DataFrame(columns=['mol', 'HF', 'MP2', 'MP4', 'CCSD', 'CCSD(T)'])
    i = 0
    for file_name in os.listdir(folder_path):
        data_df.at[i, 'mol'] = file_name
        source_file_path = os.path.join(folder_path, file_name)
        energy_dict = UtilLog.get_log_values(source_file_path)
        # energy_list = list(energy_dict.values())
        data_df.at[i, 'HF'] = energy_dict.get('HF')
        data_df.at[i, 'MP2'] = energy_dict.get('MP2')
        data_df.at[i, 'MP4'] = energy_dict.get('MP4')
        data_df.at[i, 'CCSD'] = energy_dict.get('CCSD')
        data_df.at[i, 'CCSD(T)'] = energy_dict.get('CCSD(T)')
        i += 1
        print(energy_dict)
    print(data_df)
    data_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    sys.exit(main())
