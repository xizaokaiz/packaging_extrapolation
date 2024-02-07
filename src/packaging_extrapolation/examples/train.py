from packaging_extrapolation import UtilTools
from packaging_extrapolation.Extrapolation import FitMethod
import pandas as pd
import numpy as np

"""
Calculate more systems.
"""

if __name__ == "__main__":
    # Input file.
    data = pd.read_csv('../data/hf.CSV')

    model = FitMethod()
    method_name = 'Klopper_1986'
    x_energy_list, y_energy_list = data['aug-cc-pvdz'], data['aug-cc-pvtz']
    low_card, high_card, alpha = 2, 3, 4.25
    result = UtilTools.train_alpha(model=model,
                                   method=method_name,
                                   x_energy_list=x_energy_list,
                                   y_energy_list=y_energy_list,
                                   alpha=alpha)
    print(result)
    df = pd.DataFrame()
    df['CBS Energy'] = result
    # Output file.
    df.to_csv('CBS_Energy.csv', index=False)
