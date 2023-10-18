from packaging_extrapolation import UtilTools, Extrapolation
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv(r'../data/hf.CSV')
    model = Extrapolation.FitMethod()
    ext_energy_list = UtilTools.train_alpha(model,
                                            method='Feller_1992',
                                            x_energy_list=data['aug-cc-pvdz'],
                                            y_energy_list=data['aug-cc-pvtz'],
                                            alpha=1.367, level='dt')
    for i in range(len(ext_energy_list)):
        print('The extrapolated energy of {} is: {} Eh'.format(data['mol'][i], ext_energy_list[i]))
