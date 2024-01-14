from packaging_extrapolation import UtilTools, Extrapolation
import pandas as pd
"""
外推法实例
"""

if __name__ == '__main__':
    data = pd.read_csv(r'../data/hf.CSV')
    model = Extrapolation.FitMethod()

    method, x_energy_list, y_energy_list, alpha, level = 'Feller_1992', data['aug-cc-pvdz'], data['aug-cc-pvtz'], 1.367, 'dt'
    ext_energy_list = UtilTools.train(model,
                                      method=method,
                                      x_energy_list=x_energy_list,
                                      y_energy_list=y_energy_list,
                                      alpha=alpha, level=level)
    for i in range(len(ext_energy_list)):
        print(f"The extrapolated energy of {data['mol'][i]} is: {ext_energy_list[i]} Eh")
