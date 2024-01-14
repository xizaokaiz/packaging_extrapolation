#!/usr/bin/env python3

from packaging_extrapolation import Extrapolation
import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser(description="Extrapolation to the complete basis set")
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="Chose a extrapolation method",
    )
    parser.add_argument(
        "-xe",
        "--x_energy",
        type=float,
        help="Energy for X",
    )
    parser.add_argument(
        "-ye",
        "--y_energy",
        type=float,
        help="Energy for Y",
    )
    parser.add_argument(
        "-low",
        "--low_cardinal_number",
        type=int,
        help="Cardinal number for X",
    )
    parser.add_argument(
        "-high",
        "--high_cardinal_number",
        type=int,
        help="Cardinal number for Y",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Extrapolation parameter alpha",
    )
    args = parser.parse_args().__dict__
    return args


def main():
    params = get_args()
    model = Extrapolation.FitMethod(
        method=params['method'],
        x_energy=params['x_energy'],
        y_energy=params['y_energy'],
        low_card=params['low_cardinal_number'],
        high_card=params['high_cardinal_number'],
    )
    result = model.get_function(params['alpha'])
    print(f"Extrapolation energy: {result}")


if __name__ == "__main__":
    sys.exit(main())
