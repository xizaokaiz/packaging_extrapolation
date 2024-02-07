import itertools

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from packaging_extrapolation.Extrapolation import *
from packaging_extrapolation.Extrapolation import FitMethod
import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import least_squares

kcal = 627.51


# 判断是否是列表
def is_list(obj):
    if isinstance(obj, list):
        return list
    else:
        return list(obj)


# 拟合参数
def opt_alpha(loss_model, limit, init_guess):
    result = least_squares(fun=loss_model, x0=init_guess,
                           args=(limit,))
    return result.x[0]


def calc_energy(model, alpha=None):
    # 无alpha
    if alpha is None:
        energy = model.get_function()
    # 有alpha
    else:
        energy = model.get_function(alpha)
    return energy


def calc_MaxPosMAD(y_true, y_pred):
    return np.max(y_true - y_pred) * kcal


def calc_MaxNegMAD(y_true, y_pred):
    return np.min(y_true - y_pred) * kcal


def calc_MSD(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) * kcal


def calc_RMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False) * kcal


def calc_MAD(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) * kcal


def calc_max_MAD(y_true, y_pred):
    return np.max(abs(y_true - y_pred)) * kcal


def calc_min_MAD(y_true, y_pred):
    return np.min(abs(y_true - y_pred)) * kcal


def calc_R2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def calc_avg_alpha(alpha):
    return np.average(alpha)


def calc_max_alpha(alpha):
    return np.max(alpha)


def calc_min_alpha(alpha):
    return np.min(alpha)


def calc_median(alpha):
    return np.median(alpha)


def calc_std(alpha):
    return np.std(alpha)


def calc_var(alpha):
    return np.var(alpha)


def calc_me(energy_list, limit_list):
    return np.sum(limit_list - energy_list) / len(energy_list) * kcal


def print_information(*, mol_list, energy_list, alpha_list, limit_energy, level):
    energy_list = np.array(energy_list)
    limit_energy = np.array(limit_energy)

    print('***************************************')
    print('  ', level, ' average_alpha = {:.5f}         '.format(calc_avg_alpha(alpha_list)))
    # print('  ', level, ' compute_alpha = {:.5f}         '.format(calc_avg_alpha(alpha_list) - calc_std(alpha_list)))
    print('  ', level, ' MAD = {:.3f}                   '.format(calc_MAD(energy_list, limit_energy)))

    max_mad_index = np.argmax(abs(energy_list - limit_energy))

    print('The max MAD mol index is {} {}'.format(max_mad_index, mol_list[max_mad_index]))

    print('  ', level, ' max MAD = {:.3f}                   '.format(calc_max_MAD(energy_list, limit_energy)))
    print('  ', level, ' max Max_Pos_MAD = {:.3f}                   '.format(calc_MaxPosMAD(limit_energy, energy_list)))
    print('  ', level, ' max Max_Neg_MAD = {:.3f}                   '.format(calc_MaxNegMAD(limit_energy, energy_list)))

    print('  ', level, ' RMSD = {:.3f}                   '.format(calc_RMSE(energy_list, limit_energy)))
    print('  ', level, ' ME = {:.3f}                   '.format(calc_me(energy_list, limit_energy)))
    # print('  ', level, ' R2 = {:.3f}                   '.format(calc_R2(energy_list, limit_energy)))

    min_alpha = calc_min_alpha(alpha_list)
    max_alpha = calc_max_alpha(alpha_list)
    print('   Range of alpha : [{:.2f},{:.2f}]      '.format(min_alpha, max_alpha, '.2f'))
    print('   Median of alpha : {:.3f}            '.format(calc_median(alpha_list)))
    print('   alpha 标准差 : {:.3f}'.format(calc_std(alpha_list)))
    print('   alpha 方差 : {:.3f}'.format(calc_var(alpha_list)))
    print('***************************************')


# 计算电子数
def count_ele(mol_list):
    mol_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18}

    count_list = []

    mol_list = is_list(mol_list)

    for i in range(len(mol_list)):
        mol_name = mol_list[i]
        count = 0
        j = 0
        # 遍历分子名
        while j < len(mol_name):
            k = 1
            mol_str = mol_name[j]
            j += 1
            # 判断是否带小写字符
            if j < len(mol_name) and 'z' > mol_name[j] > 'a':
                mol_str += mol_name[j]
                j += 1
            # 判断是否带数字
            if j < len(mol_name) and '9' > mol_name[j] > '0':
                # 记录原子个数
                k = int(mol_name[j])
                j += 1

            # 是否带电
            if j < len(mol_name) and mol_name[j] == '+':
                count -= 1
                j += 1
            if j < len(mol_name) and mol_name[j] == '-':
                count += 1
                j += 1

            # print(mol_str)
            count += mol_dict.get(mol_str) * k
        count_list.append(count)
    return count_list


# 计算价电子数
def count_val_ele(mol_list):
    mol_dict = {'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
                'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8}

    count_list = []

    mol_list = is_list(mol_list)

    for i in range(len(mol_list)):
        mol_name = mol_list[i]
        count = 0
        j = 0
        # 遍历分子名
        while j < len(mol_name):
            k = 1
            mol_str = mol_name[j]
            j += 1
            # 判断是否带小写字符
            if j < len(mol_name) and 'z' > mol_name[j] > 'a':
                mol_str += mol_name[j]
                j += 1
            # 判断是否带数字
            if j < len(mol_name) and '9' > mol_name[j] > '0':
                # 记录原子个数
                k = int(mol_name[j])
                j += 1

            # 是否带电
            if j < len(mol_name) and mol_name[j] == '+':
                # count -= 1
                j += 1
            if j < len(mol_name) and mol_name[j] == '-':
                # count += 1
                j += 1

            # print(mol_str)
            count += mol_dict.get(mol_str) * k
        count_list.append(count)
    return count_list


# 计算原子化能
def calc_atomize_energy(*, mol_data, atom_data, level='d'):
    if level == 'd':
        temp = 'aug-cc-pvdz'
    elif level == 't':
        temp = 'aug-cc-pvtz'
    elif level == 'q':
        temp = 'aug-cc-pvqz'
    elif level == '5':
        temp = 'aug-cc-pv5z'
    elif level == '6':
        temp = 'aug-cc-pv6z'
    else:
        return ValueError('Invalid level,please input d,t,q,5 or 6.')

    atom_dict = get_atom_dict(atom_data, temp)
    energy_list = []
    for i in mol_data.index:
        mol_name = mol_data['mol'][i]
        mol_energy = mol_data[temp][i]
        atomize_energy = get_atomize_energy(mol_name, mol_energy, atom_dict)
        energy_list.append(atomize_energy)
    return energy_list


# 计算单个分子的原子化能
def get_atomize_energy(mol_name, mol_energy, atom_dict):
    atom_energy_sum = 0
    i = 0
    while i < len(mol_name):
        atom = mol_name[i]
        count = 1
        i += 1
        # 判断是否有小写字符
        if i < len(mol_name) and 'z' > mol_name[i] > 'a':
            atom += mol_name[i]
            i += 1
        # 判断是否有数字
        if i < len(mol_name) and '9' > mol_name[i] > '0':
            # 记录个数
            count = int(mol_name[i])
            i += 1
        # print(atom)
        atom_energy_sum += atom_dict.get(atom) * count
    return atom_energy_sum - mol_energy


# 构造原子能量映射
def get_atom_dict(data, temp):
    atom_dict = {}
    for i in data.index:
        atom_name = data['mol'][i]
        atom_energy = data[temp][i]
        atom_dict.update({atom_name: float(atom_energy)})
    return atom_dict


# 判断对象是否为Series
def is_series(obj):
    return isinstance(obj, pd.Series)


# Series转list
def to_list(obj):
    return list(obj.values)


def fun_model(alpha, model, x_energy_list, y_energy_list, temp):
    energy_list = []
    for i in range(len(x_energy_list)):
        x_energy = x_energy_list[i]
        y_energy = y_energy_list[i]
        model.update_energy(x_energy, y_energy)
        energy = model.get_function(alpha)
        energy_list.append(energy)
    return energy_list


def loss_model(alpha, model, x_energy_list, y_energy_list, limit_list, temp):
    energy_list = []
    for i in range(len(x_energy_list)):
        x_energy = x_energy_list[i]
        y_energy = y_energy_list[i]
        model.update_energy(x_energy, y_energy)
        energy = model.get_function(alpha)
        energy_list.append(energy)
    if temp == 'RMSD':
        result = calc_RMSE(limit_list, energy_list)
    elif temp == 'MAD':
        result = calc_MAD(limit_list, energy_list)
    else:
        return ValueError('Invalid assessment of indicators')
    # print(result)
    return result


def train_alpha(*, model, method, x_energy_list, y_energy_list, alpha, low_card, high_card):
    """
    Calculate more systems.
    :param model: Extrapolation model.
    :param method: CBS Extrapolation method name.
    :param x_energy_list: E(X) energy list.
    :param y_energy_list: E(Y) energy list.
    :param alpha: Extrapolation parameter.
    :param low_card: The cardinal number X.
    :param high_card: The cardinal number Y.
    :return: CBS energy list.
    """

    CBS_energy = []
    model.update_card(low_card, high_card)
    model.update_method(method)
    for i in x_energy_list.index:
        x_energy = x_energy_list[i]
        y_energy = y_energy_list[i]

        model.update_energy(x_energy, y_energy)

        energy = model.get_function(alpha)
        CBS_energy.append(energy)
    return CBS_energy

