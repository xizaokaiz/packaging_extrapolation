import numpy as np


class Method:

    def __init__(self, low_card, high_card, x_energy, y_energy, method=None):
        self.low_card = low_card
        self.high_card = high_card
        self.method = method
        self.x_energy = x_energy
        self.y_energy = y_energy

    # 更新method标识
    def update_method(self, method):
        self.method = method

    # 更新基数
    def update_card(self, low_card, high_card):
        self.low_card = low_card
        self.high_card = high_card

    # 更新能量
    def update_energy(self, x_energy, y_energy):
        self.x_energy = x_energy
        self.y_energy = y_energy

    # 获取当前方法名
    def method_name(self):
        return self.method



class FitMethod(Method):

    def __init__(self, *, low_card=None, high_card=None, x_energy=None, y_energy=None, method=None):
        super().__init__(low_card, high_card, x_energy, y_energy, method)

    # Klopper_1986
    def Klopper_1986(self, alpha):
        x_eng = self.x_energy
        y_eng = self.y_energy
        exp_x = np.exp(-alpha * np.sqrt(self.low_card))
        exp_y = np.exp(-alpha * np.sqrt(self.high_card))
        return (exp_x * y_eng - exp_y * x_eng) / (exp_x - exp_y)

    # Feller_1992
    def Feller_1992(self, alpha):
        x_eng = self.x_energy
        y_eng = self.y_energy
        exp_x = np.exp(-alpha * self.low_card)
        exp_y = np.exp(-alpha * self.high_card)
        return (exp_x * y_eng - exp_y * x_eng) / (exp_x - exp_y)

    # Martin_1996
    def Martin_1996(self, alpha):
        x_eng = self.x_energy
        y_eng = self.y_energy
        x_alpha = (self.low_card + 1 / 2) ** -alpha
        y_alpha = (self.high_card + 1 / 2) ** -alpha
        return (y_eng * x_alpha - x_eng * y_alpha) / (x_alpha - y_alpha)

    # Truhlar_1998
    def Truhlar_1998(self, alpha):
        x_eng = self.x_energy
        y_eng = self.y_energy
        x_alpha = self.low_card ** -alpha
        y_alpha = self.high_card ** -alpha
        return (x_alpha * y_eng - y_alpha * x_eng) / (x_alpha - y_alpha)

    # Gdanitz_2000
    def Gdanitz_2000(self, s):
        x_s = (self.low_card + s) ** 3
        y_s = (self.high_card + s) ** 3
        return (y_s * self.y_energy - x_s * self.x_energy) / (y_s - x_s)

    # Jensen_2001
    def Jensen_2001(self, alpha):
        x_eng = self.x_energy
        y_eng = self.y_energy
        exp_x = (self.low_card + 1) * np.exp(-alpha * np.sqrt(self.low_card))
        exp_y = (self.high_card + 1) * np.exp(-alpha * np.sqrt(self.high_card))
        return (y_eng * exp_x - x_eng * exp_y) / (exp_x - exp_y)

    # HuhLee_2003
    def HuhLee_2003(self, alpha):
        x_eng = self.x_energy
        y_eng = self.y_energy
        x_alpha = (self.low_card + alpha) ** -3
        y_alpha = (self.high_card + alpha) ** -3
        return (x_alpha * y_eng - y_alpha * x_eng) / (x_alpha - y_alpha)

    # Bakowies_2007
    def Bkw_2007(self, alpha):
        x_eng = self.x_energy
        y_eng = self.y_energy
        x_alpha = (self.low_card + 1) ** -alpha
        y_alpha = (self.high_card + 1) ** -alpha
        return (x_alpha * y_eng - y_alpha * x_eng) / (x_alpha - y_alpha)

    # USTE(x-1,x)
    @staticmethod
    def x_hat(a3, x, a5_0, c, m, alpha):
        a5 = a5_0 + c * a3 ** m
        with np.errstate(divide='ignore', invalid='ignore'):
            return (x + alpha) ** -3 * (1 + a5 / a3 / (x + alpha) ** 2)

    # USTE_X
    def USTE_X(self, a3, method='cc'):
        y2_eng = self.x_energy
        y3_eng = self.y_energy
        alpha = -3 / 8
        m = 1
        a5_0 = 0.1660699
        c = -1.4222512
        if method == 'mpn':
            a5_0 = 0.0960668
            c = -1.582009
        if method == 'MRCI':
            a5_0 = 0.003769
            c = -1.1784771 * 5 / 4
        y2 = self.x_hat(a3, self.low_card, a5_0, c, m, alpha)
        y3 = self.x_hat(a3, self.high_card, a5_0, c, m, alpha)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (y2 * y3_eng - y3 * y2_eng) / (y2 - y3)

    # OAN_C
    def OAN_C(self, s):
        x_eng = self.x_energy
        y_eng = self.y_energy
        # s = 2.091
        return (3 ** 3 * y_eng - s ** 3 * x_eng) / (3 ** 3 - s ** 3)
        # return (self.high_card ** self.high_card * y_eng - s ** self.high_card * x_eng) / (3 ** 3 - s ** 3)

    # Schwenke_2005
    def Schwenke_2005(self, fc):
        x_eng = self.x_energy
        y_eng = self.y_energy
        return (y_eng - x_eng) * fc + x_eng

    # 另外方法1
    def test1(self,la):
        x_eng = self.x_energy
        y_eng = self.y_energy
        return y_eng+la*(1-y_eng/x_eng)

    # 获取函数标识列表
    @staticmethod
    def get_method_str():
        methods = ['Klopper_1986', 'Feller_1992', 'Martin_1996', 'Truhlar_1998',
                   'Jensen_2001', 'Gdanitz_2000', 'HuhLee_2003', 'Schwenke_2005', 'Bkw_2007', 'USTE_X', 'OAN_C']
        return methods

    # 获取函数列表
    def get_method_list(self):
        methods = [self.Klopper_1986, self.Feller_1992, self.Martin_1996,
                   self.Truhlar_1998, self.Gdanitz_2000, self.HuhLee_2003,
                   self.Schwenke_2005, self.Bkw_2007, self.USTE_X, self.OAN_C]
        return methods

    # 根据函数标识获取函数
    def get_function(self, alpha):
        method = self.method
        if method == 'Klopper_1986':
            y_pred = self.Klopper_1986(alpha)
        elif method == 'Feller_1992':
            y_pred = self.Feller_1992(alpha)
        elif method == 'Martin_1996':
            y_pred = self.Martin_1996(alpha)
        elif method == 'Truhlar_1998':
            y_pred = self.Truhlar_1998(alpha)
        elif method == 'Gdanitz_2000':
            y_pred = self.Gdanitz_2000(alpha)
        elif method == 'HuhLee_2003':
            y_pred = self.HuhLee_2003(alpha)
        elif method == 'Jensen_2001':
            y_pred = self.Jensen_2001(alpha)
        elif method == 'Bkw_2007':
            y_pred = self.Bkw_2007(alpha)
        elif method == 'USTE_X':
            y_pred = self.USTE_X(alpha)
        elif method == 'OAN_C':
            y_pred = self.OAN_C(alpha)
        elif method == 'Schwenke_2005':
            y_pred = self.Schwenke_2005(alpha)
        elif method == 'test1':
            y_pred = self.test1(alpha)
        else:
            raise ValueError("Invalid function name")
        return y_pred

    # 损失函数
    def loss_function(self, alpha, limit):
        method = self.method
        if method == 'Klopper_1986':
            y_pred = self.Klopper_1986(alpha)
        elif method == 'Feller_1992':
            y_pred = self.Feller_1992(alpha)
        elif method == 'Martin_1996':
            y_pred = self.Martin_1996(alpha)
        elif method == 'Truhlar_1998':
            y_pred = self.Truhlar_1998(alpha)
        elif method == 'Gdanitz_2000':
            y_pred = self.Gdanitz_2000(alpha)
        elif method == 'Jensen_2001':
            y_pred = self.Jensen_2001(alpha)
        elif method == 'HuhLee_2003':
            y_pred = self.HuhLee_2003(alpha)
        elif method == 'Bkw_2007':
            y_pred = self.Bkw_2007(alpha)
        elif method == 'USTE_X':
            y_pred = self.USTE_X(alpha)
        elif method == 'OAN_C':
            y_pred = self.OAN_C(alpha)
        elif method == 'Schwenke_2005':
            y_pred = self.Schwenke_2005(alpha)
        elif method == 'test1':
            y_pred = self.test1(alpha)
        else:
            raise ValueError("Invalid function name")
        return y_pred - limit


# USPE
class USPE(Method):

    def __init__(self, x=None, x_energy=None, tot_energy=None, alpha=None):
        self.x = x
        self.x_energy = x_energy
        self.tot_energy = tot_energy
        self.alpha = alpha

    def update_card(self, x):
        self.x = x

    def update_alpha(self, alpha):
        self.alpha = alpha

    def update_energy(self, x_energy, tot_energy):
        self.x_energy = x_energy
        self.tot_energy = tot_energy

    # USPE
    def USPE(self, alpha):
        x = self.x
        # if x == 2:
        #     x = 1.91
        # elif x == 3:
        #     x = 2.71
        # elif x == 4:
        #     x = 3.68
        # elif x == 5:
        #     x = 4.71
        # elif x == 6:
        #     x = 5.7
        return self.x_energy + alpha * self.tot_energy / x ** 3

    def loss_function(self, alpha, limit):
        return abs(self.USPE(alpha) - limit)

    @staticmethod
    def get_method_str():
        return ['USPE']
