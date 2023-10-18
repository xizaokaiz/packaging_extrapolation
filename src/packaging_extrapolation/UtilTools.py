import itertools

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from packaging_extrapolation.Extrapolation import *
from sklearn.model_selection import KFold
from scipy.optimize import least_squares

kcal = 627.51


# 拆分,两列数据拆成一列
def split_data(data):
    return data.iloc[:, 0], data.iloc[:, 1]


# 交叉验证，返回的是索引
def k_fold_index(data, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_index_list = []
    test_index_list = []
    for train_index, test_index in kf.split(data):
        train_index_list.append(train_index)
        test_index_list.append(test_index)
    return train_index_list, test_index_list


# 根据索引返回数据集
def train_test_split(X, y, train_index, test_index):
    return X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]


# 交叉验证最低水平函数
def train_dt_k_fold(model, *, X, y, k, method, init_guess=0.001):
    """
        使用k-1的数据参数，拟合第k个数据集的参数
        返回平均精度
    """
    level = 'dt'
    # 评估指标
    train_mad_list = []
    train_rmsd_list = []
    train_max_list = []
    test_mad_list = []
    test_rmsd_list = []
    test_max_list = []
    # 平均参数
    avg_alpha_list = []
    # 获取索引
    train_index_list, test_index_list = k_fold_index(X, k)
    k_index = 0
    for i in range(len(train_index_list)):
        k_index += 1
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_index_list[i], test_index_list[i])
        # 分割x_energy,y_energy
        x_energy_list_train, y_energy_list_train = split_data(X_train)
        x_energy_list_test, y_energy_list_test = split_data(X_test)

        # 训练，获取avg_alpha

        _, alpha_list = train(model, x_energy_list=x_energy_list_train,
                              y_energy_list=y_energy_list_train,
                              limit_list=y_train, method=method, level=level, init_guess=init_guess)
        avg_alpha = calc_avg_alpha(alpha_list)
        avg_alpha = avg_alpha - calc_std(alpha_list)

        # 训练集使用avg_alpha
        energy_list = train_alpha(model, x_energy_list=x_energy_list_train,
                                  y_energy_list=y_energy_list_train,
                                  alpha=avg_alpha, method=method, level=level)

        # 训练集误差评估指标
        train_mad = calc_MAD(energy_list, y_train)
        train_max_mad = calc_max_MAD(energy_list, y_train)
        train_rmsd = calc_RMSE(energy_list, y_train)

        # 验证集使用avg_alpha计算能量
        energy_list = train_alpha(model, x_energy_list=x_energy_list_test, y_energy_list=y_energy_list_test, alpha=avg_alpha,
                                  method=method, level=level)

        # 验证集误差评估指标
        test_mad = calc_MAD(energy_list, y_test)
        test_max_mad = calc_max_MAD(energy_list, y_test)
        test_rmsd = calc_RMSE(energy_list, y_test)

        print('*****************************************')
        print(k_index, '折训练集误差，MAD={:.3f} ，RMSD={:.3f}'.format(train_mad, train_rmsd))
        print(k_index, '折训练集最大误差，MaxMAD={:.3f}'.format(train_max_mad))
        print(k_index, '折验证集误差，MAD={:.3f} ，RMSD={:.3f}'.format(test_mad, test_rmsd))
        print(k_index, '折验证集最大误差，MaxMAD={:.3f}'.format(test_max_mad))
        print(k_index, '折数据集，alpha={:.5f}'.format(avg_alpha))
        print('*****************************************')
        print()

        train_mad_list.append(train_mad)
        train_rmsd_list.append(train_rmsd)
        train_max_list.append(train_max_mad)
        test_mad_list.append(test_mad)
        test_rmsd_list.append(test_rmsd)
        test_max_list.append(test_max_mad)
        avg_alpha_list.append(avg_alpha)

    train_avg_mad = np.average(train_mad_list)
    train_avg_rmsd = np.average(train_rmsd_list)
    train_avg_max = np.average(train_max_list)
    test_avg_mad = np.average(test_mad_list)
    test_avg_rmsd = np.average(test_rmsd_list)
    test_avg_max = np.average(test_max_list)
    avg_alpha = np.average(avg_alpha_list)

    print('平均训练集误差，MAD={:.3f} ，RMSD={:.3f}'.format(train_avg_mad, train_avg_rmsd))
    print('平均训练集最大误差，MaxMAD={:.3f}'.format(train_avg_max))
    print('平均验证集误差，MAD={:.3f} ，RMSD={:.3f}'.format(test_avg_mad, test_avg_rmsd))
    print('平均验证集最大误差，MaxMAD={:.3f}'.format(test_avg_max))
    print('5折平均alpha，alpha={:.5f}'.format(avg_alpha))

    eva_list = [train_avg_mad, train_avg_rmsd,
                test_avg_mad, test_avg_rmsd, avg_alpha]

    # 返回k折平均mad,rmsd
    return eva_list


# 交叉验证训练函数
def train_k_fold(model, *, X, y, k, method, level='dt', init_guess=0.001):
    """
    使用k-1的数据参数，拟合第k个数据集的参数
    返回平均精度
    """
    # 评估指标
    train_mad_list = []
    train_rmsd_list = []
    train_max_list = []
    test_mad_list = []
    test_rmsd_list = []
    test_max_list = []
    # 平均参数
    avg_alpha_list = []
    # 获取索引
    train_index_list, test_index_list = k_fold_index(X, k)
    k_index = 0
    for i in range(len(train_index_list)):
        k_index += 1
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_index_list[i], test_index_list[i])
        # 分割x_energy,y_energy
        x_energy_list_train, y_energy_list_train = split_data(X_train)
        x_energy_list_test, y_energy_list_test = split_data(X_test)

        # 训练，获取avg_alpha
        _, avg_alpha = get_k_fold_alpha(model, x_energy_list_train,
                                        y_energy_list_train, y_train, method, level, init_guess=init_guess)

        # 训练集使用avg_alpha
        energy_list = train_alpha(model, x_energy_list=x_energy_list_train,
                                  y_energy_list=y_energy_list_train,
                                  alpha=avg_alpha, method=method, level=level)

        # 训练集误差评估指标
        train_mad = calc_MAD(energy_list, y_train)
        train_max_mad = calc_max_MAD(energy_list, y_train)
        train_rmsd = calc_RMSE(energy_list, y_train)

        # 验证集使用avg_alpha计算能量
        energy_list = train_alpha(model, x_energy_list=x_energy_list_test, y_energy_list=y_energy_list_test, alpha=avg_alpha,
                                  method=method, level=level)

        # 验证集误差评估指标
        test_mad = calc_MAD(energy_list, y_test)
        test_max_mad = calc_max_MAD(energy_list, y_test)
        test_rmsd = calc_RMSE(energy_list, y_test)

        print('*****************************************')
        print(k_index, '折训练集误差，MAD={:.3f} ，RMSD={:.3f}'.format(train_mad, train_rmsd))
        print(k_index, '折训练集最大误差，MaxMAD={:.3f}'.format(train_max_mad))
        print(k_index, '折验证集误差，MAD={:.3f} ，RMSD={:.3f}'.format(test_mad, test_rmsd))
        print(k_index, '折验证集最大误差，MaxMAD={:.3f}'.format(test_max_mad))
        print(k_index, '折数据集，alpha={:.5f}'.format(avg_alpha))
        print('*****************************************')
        print()

        train_mad_list.append(train_mad)
        train_rmsd_list.append(train_rmsd)
        train_max_list.append(train_max_mad)
        test_mad_list.append(test_mad)
        test_rmsd_list.append(test_rmsd)
        test_max_list.append(test_max_mad)
        avg_alpha_list.append(avg_alpha)

    train_avg_mad = np.average(train_mad_list)
    train_avg_rmsd = np.average(train_rmsd_list)
    train_avg_max = np.average(train_max_list)
    test_avg_mad = np.average(test_mad_list)
    test_avg_rmsd = np.average(test_rmsd_list)
    test_avg_max = np.average(test_max_list)
    avg_alpha = np.average(avg_alpha_list)

    print('平均训练集误差，MAD={:.3f} ，RMSD={:.3f}'.format(train_avg_mad, train_avg_rmsd))
    print('平均训练集最大误差，MaxMAD={:.3f}'.format(train_avg_max))
    print('平均验证集误差，MAD={:.3f} ，RMSD={:.3f}'.format(test_avg_mad, test_avg_rmsd))
    print('平均验证集最大误差，MaxMAD={:.3f}'.format(test_avg_max))
    print('5折平均alpha，alpha={:.5f}'.format(avg_alpha))

    eva_list = [train_avg_mad, train_avg_rmsd,
                test_avg_mad, test_avg_rmsd, avg_alpha]

    # 返回k折平均mad,rmsd
    return eva_list


# 拟合k-1数据集，获取avg_alpha
def get_k_fold_alpha(model, x_energy_list, y_energy_list, limit, method, level, init_guess):
    energy_list, alpha_list = train(model, x_energy_list=x_energy_list,
                                    y_energy_list=y_energy_list,
                                    limit_list=limit, method=method, level=level, init_guess=init_guess)
    avg_alpha = calc_avg_alpha(alpha_list)
    return energy_list, avg_alpha


# 单点外推训练函数
def train_uspe(model, *, x_energy_list, tot_energy_list, alpha=None, limit_list=None, init_guess=0.001, level=2):
    if alpha is None and limit_list is None:
        raise ValueError('Alpha and limit_list must be assigned to one or the other')

    # 默认不拟合
    flag = False
    if limit_list is not None:
        flag = True
    energy_list = []
    alpha_list = []
    model.update_card(level)

    x_energy_list = is_list(x_energy_list)
    tot_energy_list = is_list(tot_energy_list)
    limit_list = is_list(limit_list)
    for i in range(len(x_energy_list)):
        x_energy = x_energy_list[i]
        tot_energy = tot_energy_list[i]
        model.update_energy(x_energy, tot_energy)

        # 拟合
        if flag:
            limit = limit_list[i]
            result = least_squares(model.loss_function, init_guess, args=(limit,))
            alpha = result.x[0]
            alpha_list.append(alpha)

        energy = model.USPE(alpha)
        energy_list.append(energy)
    if flag:
        return energy_list, alpha_list
    return energy_list


# 指定alpha的训练函数
def train_alpha(model, *, method, x_energy_list, y_energy_list, alpha, level='dt'):
    energy_list = []

    x_energy_list = is_list(x_energy_list)
    y_energy_list = is_list(y_energy_list)

    if level == 'dt':
        low_card = 2
        high_card = 3
    elif level == 'tq':
        low_card = 3
        high_card = 4
    elif level == 'q5':
        low_card = 4
        high_card = 5
    elif level == '56':
        low_card = 5
        high_card = 6
    else:
        raise ValueError('Invalid level name')

    model.update_card(low_card, high_card)
    model.update_method(method)
    for i in range(len(x_energy_list)):
        x_energy = x_energy_list[i]
        y_energy = y_energy_list[i]

        model.update_energy(x_energy, y_energy)

        energy = model.get_function(alpha)
        energy_list.append(energy)
    return energy_list


# 训练函数，返回能量和alpha列表
def train(model, *, method, x_energy_list, y_energy_list, limit_list=None, init_guess=0.01, level='dt'):
    # 默认需要拟合参数，默认需要拟合
    flag = True
    if limit_list is None:
        flag = False
    energy_list = []
    alpha_list = []

    x_energy_list = is_list(x_energy_list)
    y_energy_list = is_list(y_energy_list)
    limit_list = is_list(limit_list)

    # 判断基数
    if level == 'dt':
        low_card = 2
        high_card = 3
    elif level == 'tq':
        low_card = 3
        high_card = 4
    elif level == 'q5':
        low_card = 4
        high_card = 5
    else:
        raise ValueError('Invalid level name')
    model.update_method(method)
    model.update_card(low_card, high_card)

    for i in range(len(x_energy_list)):
        x_energy = x_energy_list[i]
        y_energy = y_energy_list[i]

        # 更新能量
        model.update_energy(x_energy, y_energy)

        # 为ture，则拟合
        if flag:
            limit = limit_list[i]
            alpha = opt_alpha(model.loss_function, limit, init_guess)
            alpha_list.append(alpha)
            energy = calc_energy(model, alpha)

        else:
            energy = calc_energy(model)

        energy_list.append(energy)
    if flag:
        return energy_list, alpha_list
    return energy_list


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


# 计算能量
def calc_energy(model, alpha=None):
    # 无alpha
    if alpha is None:
        energy = model.get_function()
    # 有alpha
    else:
        energy = model.get_function(alpha)
    return energy


# 计算最大正偏差
def calc_MaxPosMAD(y_true, y_pred):
    return np.max(y_true - y_pred) * kcal


# 计算最大负偏差
def calc_MaxNegMAD(y_true, y_pred):
    return np.min(y_true - y_pred) * kcal


# 计算MSD
def calc_MSD(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


# 计算RMSD
def calc_RMSE(y_true, y_pred):
    return calc_MSD(y_true, y_pred) ** 0.5 * kcal


# 计算MAD
def calc_MAD(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) * kcal


# 计算max_MAD
def calc_max_MAD(y_true, y_pred):
    return np.max(abs(y_true - y_pred)) * kcal


# 计算min_MAD
def calc_min_MAD(y_true, y_pred):
    return np.min(abs(y_true - y_pred)) * kcal


# 计算R2
def calc_R2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# 计算平均alpha
def calc_avg_alpha(alpha):
    return np.average(alpha)


# 计算最大alpha
def calc_max_alpha(alpha):
    return np.max(alpha)


# 计算最小alpha
def calc_min_alpha(alpha):
    return np.min(alpha)


# 计算中位数
def calc_median(alpha):
    return np.median(alpha)


# 计算标准差
def calc_std(alpha):
    return np.std(alpha)


# 计算方差
def calc_var(alpha):
    return np.var(alpha)


# 计算误差，带正负
def calc_me(energy_list, limit_list):
    return np.sum(limit_list - energy_list) / len(energy_list) * kcal


# 存入评估指标
def input_result(result_df, *, index, energy_list, limit_list, alpha_list=None):
    result_df['RMSD'][index] = calc_RMSE(limit_list, energy_list)
    result_df['MAD'][index] = calc_MAD(limit_list, energy_list)
    result_df['MaxAD'][index] = calc_max_MAD(limit_list, energy_list)
    if alpha_list is not None:
        result_df['avg_alpha'][index] = calc_avg_alpha(alpha_list)
        result_df['min_alpha'][index] = calc_min_alpha(alpha_list)
        result_df['max_alpha'][index] = calc_max_alpha(alpha_list)
    return result_df


# 列表一维化
def flatting(ls):
    return list(itertools.chain.from_iterable(ls))


# 画图：一个外推水平的分子参数分布图
def plot_alpha_value(model, *, mol_list, x_energy_list, y_energy_list, fitting_list,
                     limit_energy, method, level='dt', init_guess=0.001):
    _, alpha_list = train(model, method=method, x_energy_list=x_energy_list,
                          y_energy_list=y_energy_list, limit_list=fitting_list,
                          level=level, init_guess=init_guess)

    avg_alpha = calc_avg_alpha(alpha_list)
    # avg_alpha = round(avg_alpha, 4)
    # alpha_std = calc_std(alpha_list)
    # mid_alpha = calc_median(alpha_list)
    # avg_alpha = avg_alpha - alpha_std
    energy_list = train_alpha(model, method=method, x_energy_list=x_energy_list,
                              y_energy_list=y_energy_list,
                              level=level, alpha=avg_alpha)

    print_information(mol_list=mol_list, energy_list=energy_list,
                      alpha_list=alpha_list,
                      limit_energy=limit_energy, level=level)

    plot_alpha(mol_list=mol_list, alpha_list=alpha_list, level=level)
    return energy_list, alpha_list


def plot_alpha(*, mol_list, alpha_list, level):
    plt.figure(figsize=(10, 6))
    plt.plot(mol_list, alpha_list, '.')
    plt.xticks(rotation=-80)
    plt.xticks(fontsize=6)
    plt.ylabel('alpha value')
    plt.xlabel('species')
    plt.title(level)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()


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


# 穷举
def exhaustion_alpha(model, *, method, x_energy_list, y_energy_list, limit_list, init_alpha, init_step=1000, level='dt'):
    error_df = pd.DataFrame(columns=['alpha', 'MAD', 'MaxMAD'])
    mad_list = []
    max_mad_list = []
    alpha_list = []
    alpha = init_alpha
    for i in range(init_step):
        alpha_list.append(alpha)
        energy_list = train_alpha(model, x_energy_list=x_energy_list, y_energy_list=y_energy_list,
                                  alpha=alpha, level=level, method=method)
        mad = calc_MAD(limit_list,
                       energy_list)
        max_mad = calc_max_MAD(limit_list,
                               energy_list)
        mad_list.append(mad)
        max_mad_list.append(max_mad)
        alpha += 0.001
    error_df['alpha'] = alpha_list
    error_df['MAD'] = mad_list
    error_df['MaxMAD'] = max_mad_list
    min_mad_alpha = error_df['alpha'][np.argmin(error_df['MAD'])]
    min_maxMad_alpha = error_df['alpha'][np.argmin(error_df['MaxMAD'])]
    print('使MAD最小的alpha值为 {}，最小MAD为 {}'.format(min_mad_alpha, np.min(error_df['MAD'])))
    print('使MaxMAD最小的alpha值为：{},最小MaxMAD为 {}'.format(min_maxMad_alpha, np.min(error_df['MaxMAD'])))
    return error_df
