# Created by Shlyankin Nickolay & Vladimir Michailov & Alena Zahodyakina
import scipy.special as sp
import math
import numpy as np
import numpy.linalg as alg




class Task(object):
    def analytic_decision(self):
        self.answer_analytic = [[math.exp(
            -(2.0 * self.alpha / (self.c * self.l) + (self.k * (self.j0zeros ** 2) / (self.c * (self.R ** 2)))) * t) *
                                 sp.j0(self.j0zeros * r / self.R) + self.Uc for r in self.r] for t in self.t]
        return self.answer_analytic

    def __init__(self, R, l, k, c, alpha, T, Uc, K, I):
        self.j0zeros = sp.jn_zeros(0, 1)[0]  # 2.404825557695773
        self.alpha = alpha
        self.R = R
        self.l = l
        self.k = k
        self.c = c
        self.T = T
        self.Uc = Uc
        self.K = K
        self.I = I
        self.hr = R / I
        self.I = I + 1
        self.r = [self.hr * i for i in range(self.I)]
        self.ht = T / K
        self.K = K + 1
        self.t = [self.ht * k for k in range(self.K)]
        self.psi = [sp.j0(r * self.j0zeros / R) + self.Uc for r in self.r]

    def TDMA(self, a, b, c, f):
        """
        решение системы с трехдиагональной матрицей
        :param a: побочная диагональ под главной [1:n-1]
        :param b: побочная диагональ над главной [0:n-2]
        :param c: главная диагональ [0:n-1]
        :param f: правая часть
        :return: решение системы
        """
        # переводим в float
        # a, b, c, f = map(lambda k_list: map(float, k_list), (a, b, c, f))
        alpha = [-b[0] / c[0]]
        beta = [f[0] / c[0]]
        n = len(f)
        x = [0] * n

        for i in range(1, n - 1):
            alpha.append(-b[i] / (c[i] + a[i - 1] * alpha[i - 1]))
            beta.append((f[i] - a[i - 1] * beta[i - 1]) / (c[i] + a[i - 1] * alpha[i - 1]))

        x[n - 1] = (f[n - 1] - a[n - 2] * beta[n - 2]) / (c[n - 1] + a[n - 2] * alpha[n - 2])
        for i in reversed(range(n - 1)):
            x[i] = alpha[i] * x[i + 1] + beta[i]
        return x

class TaskExplicit(Task):

    def isStable(self):
        return (1 - (2 * self.k * self.ht) / ((self.hr ** 2) * self.c) - (self.alpha * self.ht) / (self.c * self.l)) > 0

    def calculate(self):
        for k in range(self.K):
            local_answer = np.zeros(self.I, np.float64)
            if k == 0:
                for i in range(self.I - 1):
                    local_answer[i] = self.psi[i]
                local_answer[self.I - 1] = self.Uc
            else:
                local_answer[0] = self.answer[k-1][1] * (4 * self.k * self.ht / ((self.hr ** 2) * self.c)) + \
                                  self.answer[k-1][0] * (1 - (4 * self.k * self.ht) / (self.c * (self.hr ** 2)) - (2 * self.alpha * self.ht) / (self.l * self.c)) + \
                                  ((2 * self.alpha * self.ht * self.Uc) / (self.l * self.c))
                for i in range(1, self.I - 1):
                    local_answer[i] = self.answer[k-1][i+1] * (self.k * self.ht / ((self.hr ** 2) * self.c) + self.k * self.ht / (2 * self.hr * self.r[i] * self.c)) + \
                                      self.answer[k-1][i] * (1 - (2 * self.ht * self.k) / ((self.hr ** 2) * self.c) - (2 * self.alpha * self.ht) / (self.l * self.c)) + \
                                      self.answer[k-1][i-1] * (self.k * self.ht / ((self.hr ** 2) * self.c) - self.k * self.ht / (2 * self.hr * self.r[i] * self.c)) + \
                                      (2 * self.alpha * self.ht * self.Uc / (self.l * self.c))
                local_answer[self.I - 1] = self.Uc
            self.answer.append(local_answer)
        return self.answer

    def calculateAbsError(self):
        if self.answer is not None and self.answer_analytic is not None:
            return np.max(np.abs(np.array(self.answer_analytic) - np.array(self.answer)))

    def __init__(self, R, l, k, c, alpha, T, Uc, K, I):
        super(TaskExplicit, self).__init__(R, l, k, c, alpha, T, Uc, K, I)
        self.name = "Явная схема"
        self.color = '000'
        self.answer_analytic = []
        self.answer = []

class TaskImplicit(Task):
    """docstring"""

    def isStable(self):
        return True

    def calculate(self):
        for k in range(1, self.K):
            f = [self.Fi(i, k) for i in range(1, self.I - 1)]
            f.insert(0, self.F0(k))
            f.append(self.Uc)
            # self.f.append(alg.solve(self.A, f)) # обычное решение системы, для проверки работы прогонки
            self.f.append(self.TDMA(self.a_diag, self.b_diag, self.c_diag, f))
        self.answer = self.f
        return self.f

    def calculateAbsError(self):
        if self.answer is not None and self.answer_analytic is not None:
            return np.max(np.abs(np.array(self.answer_analytic) - np.array(self.answer)))

    def __init__(self, R, l, k, c, alpha, T, Uc, K, I):
        super(TaskImplicit, self).__init__( R, l, k, c, alpha, T, Uc, K, I)
        self.name = "Неявная схема"
        self.color = 'g'
        self.Ci = lambda i: self.c / self.ht + \
                            (2.0 * self.alpha) / self.l + \
                            (2.0 * self.k) / (self.hr ** 2)
        self.Bi = lambda i: self.k / (2.0 * self.hr * self.r[i]) + \
                            self.k / (self.hr ** 2)
        self.Ai = lambda i: self.k / -(2.0 * self.r[i] * self.hr) + \
                            self.k / (self.hr ** 2)

        self.Fi = lambda i, k: (self.c * self.f[k - 1][i]) / self.ht + \
                               (self.alpha * 2.0 * self.Uc) / self.l

        self.C0 = self.c / self.ht + \
                    (4.0 * self.k) / (self.hr ** 2) + \
                    (2.0 * self.alpha) / self.l
        self.B0 = 4.0 * self.k / (self.hr ** 2)
        self.F0 = lambda k: (self.f[k - 1][0] * self.c) / self.ht + \
                            (2.0 * self.alpha * self.Uc) / self.l

        self.f = []
        f = self.psi
        f[len(f) - 1] = self.Uc
        self.f.append(f)

        self.A = [[0 for i in range(self.I)] for j in range(self.I)]
        self.a_diag = []
        self.b_diag = []
        self.c_diag = []
        for i in range(self.I):
            for j in range(i - 1, i + 2):
                if i == 0:
                    if i == j:
                        self.A[i][j] = self.C0
                        self.c_diag.append(self.A[i][j])
                    if j == i + 1:
                        self.A[i][j] = -self.B0
                        self.b_diag.append(self.A[i][j])
                elif i == self.I - 1:
                    if i == j:
                        self.A[i][j] = 1
                        self.c_diag.append(1)
                        self.a_diag.append(0)
                else:
                    if i == j:
                        self.A[i][j] = self.Ci(i)
                        self.c_diag.append(self.A[i][j])
                    if j == i - 1:
                        self.A[i][j] = -self.Ai(i)
                        self.a_diag.append(self.A[i][j])
                    if j == i + 1:
                        self.A[i][j] = -self.Bi(i)
                        self.b_diag.append(self.A[i][j])
        self.answer_analytic = []
        self.answer = []


class TaskCrankNicholson(Task):
    """docstring"""

    def isStable(self):
        return (1 - (2 * self.k * self.ht) / ((self.hr ** 2) * self.c) - (self.alpha * self.ht) / (self.c * self.l)) > 0

    def calculate(self):
        for k in range(1, self.K):
            f = [self.Fi(i, k) for i in range(1, self.I - 1)]
            f.insert(0, self.F0(k))
            f.append(self.Uc)
            # self.f.append(alg.solve(self.A, f))
            self.f.append(self.TDMA(self.a_diag, self.b_diag, self.c_diag, f))
        self.answer = self.f
        return self.f

    def calculateAbsError(self):
        if self.answer is not None and self.answer_analytic is not None:
            return np.max(np.abs(np.array(self.answer_analytic) - np.array(self.answer)))

    def __init__(self, R, l, k, c, alpha, T, Uc, K, I):
        super(TaskCrankNicholson, self).__init__( R, l, k, c, alpha, T, Uc, K, I)
        self.name = "схема Кранка-Николсона"
        self.color = 'r'
        self.Ci = lambda i: self.c / self.ht + \
                            self.alpha / self.l + \
                            self.k / (self.hr ** 2)
        self.C2i = lambda i: self.c / self.ht - \
                             self.alpha / self.l - \
                             self.k / (self.hr ** 2)
        self.Bi = lambda i: self.k / (4.0 * self.hr * self.r[i]) + \
                            self.k / (2.0 * (self.hr ** 2))
        self.Ai = lambda i: self.k / -(4.0 * self.r[i] * self.hr) + \
                            self.k / (2.0 * (self.hr ** 2))

        self.Fi = lambda i, k: self.Bi(i) * self.f[k - 1][i + 1] + \
                               self.C2i(i) * self.f[k - 1][i] + \
                               self.Ai(i) * self.f[k - 1][i - 1] + \
                               self.alpha * 2.0 * self.Uc / self.l

        self.C0 = self.c / self.ht + \
                  2.0 * self.k / (self.hr ** 2) + \
                  self.alpha / self.l
        self.B0 = 2.0 * self.k / (self.hr ** 2)
        self.F0 = lambda k: self.f[k - 1][1] * self.B0 + \
                            self.f[k - 1][0] * (self.c / self.ht -
                                                2.0 * self.k / (self.hr ** 2) -
                                                self.alpha / self.l) \
                            + self.alpha * 2.0 * self.Uc / self.l
        self.f = []
        f = self.psi
        f[len(f) - 1] = self.Uc
        self.f.append(f)

        self.A = [[0 for i in range(self.I)] for j in range(self.I)]
        self.a_diag = []
        self.b_diag = []
        self.c_diag = []
        for i in range(self.I):
            for j in range(i - 1, i + 2):
                if i == 0:
                    if i == j:
                        self.A[i][j] = self.C0
                        self.c_diag.append(self.A[i][j])
                    if j == i + 1:
                        self.A[i][j] = -self.B0
                        self.b_diag.append(self.A[i][j])
                elif i == self.I - 1:
                    if i == j:
                        self.A[i][j] = 1
                        self.c_diag.append(1)
                        self.a_diag.append(0)
                else:
                    if i == j:
                        self.A[i][j] = self.Ci(i)
                        self.c_diag.append(self.A[i][j])
                    if j == i - 1:
                        self.A[i][j] = -self.Ai(i)
                        self.a_diag.append(self.A[i][j])
                    if j == i + 1:
                        self.A[i][j] = -self.Bi(i)
                        self.b_diag.append(self.A[i][j])
        self.answer_analytic = []
        self.answer = []
