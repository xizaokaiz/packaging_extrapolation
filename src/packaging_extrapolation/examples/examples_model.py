import joblib
import numpy as np
import pandas as pd
from packaging_extrapolation import UtilTools

"""
回归模型实例
"""

if __name__ == '__main__':
    # 加载模型
    lin_model = joblib.load('../model_detail/HF_model/trained_model_1.pkl')

    data = pd.read_csv('../data/hf.CSV')
    # 特征
    X = np.array(data.iloc[:, 1:3]).reshape(-1, 2)
    # 预测
    y_pred = lin_model.predict(X)

    print('Linear model predict : \n {}'.format(y_pred))
    print('MAD :{} kcal/mol'.format(UtilTools.calc_MAD(np.array(data['aug-cc-pv6z']).reshape(-1, 1), y_pred)))
    print('Max MAD :{} kcal/mol'.format(UtilTools.calc_max_MAD(np.array(data['aug-cc-pv6z']).reshape(-1, 1), y_pred)))
