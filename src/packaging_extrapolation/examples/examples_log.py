from packaging_extrapolation import UtilLog

"""
Extracting energy from many log files.
"""


if __name__ == "__main__":
    input_path = '../data/log'
    output_path = '../data/log_result.csv'
    result = UtilLog.extract_energy(input_path, output_path)
