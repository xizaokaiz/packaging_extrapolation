"""
通过两个log文件外推能量
"""
from packaging_extrapolation import UtilLog, Extrapolation, UtilTools

if __name__ == '__main__':
    # 能量类型
    method_type = 'HF'
    # 获取能量
    file_path_low = '../data/H2O_ccsdt_avdz.log'
    file_path_high = '../data/H2O_ccsdt_avtz.log'
    alpha = 1.367  # 拟合参数
    level = 'dt'  # 外推水平

    low_energy = UtilLog.get_log_values(file_path_low)
    high_energy = UtilLog.get_log_values(file_path_high)

    model = Extrapolation.FitMethod()
    ext_energy = UtilTools.train_alpha(model,
                                       method='Feller_1992',
                                       x_energy_list=low_energy,
                                       y_energy_list=high_energy,
                                       alpha=alpha, level=level)
    print('The result is %s' % ext_energy.pop())
