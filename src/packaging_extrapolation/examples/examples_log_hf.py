"""
通过两个log文件外推能量
"""
from packaging_extrapolation import UtilLog, Extrapolation, UtilTools

if __name__ == '__main__':
    # 能量类型
    method_type = 'HF'
    file_path_low = '../data/H2O_ccsdt_avdz.log'
    file_path_high = '../data/H2O_ccsdt_avtz.log'

    x_energy = UtilLog.get_log_values(file_path_low, method_type)
    y_energy = UtilLog.get_log_values(file_path_high, method_type)

    model = Extrapolation.FitMethod()
    alpha, level, method = 1.367, 'dt', 'Feller_1992'

    CBS_energy = UtilTools.CBS(model,
                               method=method,
                               x_energy=x_energy,
                               y_energy=y_energy,
                               alpha=alpha,
                               level=level)
    print('The result is %s' % CBS_energy)
