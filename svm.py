from __future__ import annotations

import math

import scipy.optimize
import numpy.random
import matplotlib.pyplot

from typing import Union, List
from operator import add, mul
from math import sqrt
from functools import total_ordering
from random import sample

@total_ordering
class Scalar:
    def __init__(self: Scalar, val: float):
        self.val = float(val)

    def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
        if isinstance(other, Scalar):
            return Scalar(self.val * other.val)
        elif isinstance(other, Vector):
            return Vector(*[i * self.val for i in iter(other)])
        else:
            raise TypeError("{wrongType} should be either Scalar or Vector".format(wrongType=type(other)))

    def __add__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val + other.val)

    def __sub__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(other.val - self.val)

    def __truediv__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val / other.val)

    def __rtruediv__(self: Scalar, other: Vector) -> Vector:
        return Vector(*[i / self.val for i in iter(other)])

    def __repr__(self: Scalar) -> str:
        return "Scalar(%r)" % self.val

    def sign(self: Scalar) -> int:
        return -1 if self.val < 0 else 1 if self.val > 0 else 0

    def __float__(self: Scalar) -> float:
        return self.val

    def __lt__(self: Scalar, other: Scalar):
        return self.val < other.val

    def __eq__(self: Scalar, other: Scalar):
        return self.val == other.val


@total_ordering
class Vector:
    def __init__(self: Vector, *entries: List[float]):
        self.entries = entries

    @staticmethod
    def zero(size: int) -> Vector:
        return Vector(*[0 for i in range(size)])

    def __add__(self: Vector, other: Vector) -> Vector:
        return Vector(*list(map(add, self.entries, other.entries)))

    def __sub__(self: Vector, other: Vector) -> Vector:
        return self + Scalar(-1) * other

    def __mul__(self: Vector, other: Vector) -> Scalar:
        return Scalar(sum(list(map(mul, self.entries, other.entries))))

    def magnitude(self: Vector) -> Scalar:
        return Scalar(sqrt(sum([i**2 for i in iter(self)])))

    def unit(self: Vector) -> Vector:
        return self / self.magnitude()

    def __len__(self: Vector) -> int:
        return len(self.entries)

    def __repr__(self: Vector) -> str:
        return "Vector%s" % repr(self.entries)

    def __iter__(self: Vector):
        return iter(self.entries)

    def __lt__(self: Vector, other: Vector):
        return self.magnitude() < other.magnitude()

    def __eq__(self: Vector, other: Vector):
        return self.magnitude() == other.magnitude()


class Perceptron:
    def __init__(self: Perceptron, d: int):
        self.D = d

        # Weights
        self.w = Vector.zero(d)
        # Bias
        self.b = Scalar(0)

        # Cached weights and bias
        self.u = Vector.zero(d)
        self.B = Scalar(0)

    def clear(self: Perceptron):
        self.w = Vector.zero(self.D)
        self.b = Scalar(0)
        self.u = Vector.zero(self.D)
        self.B = Scalar(0)

    def train(self: Perceptron, x: List[Vector], y: List[Scalar],  iters: int):
        for it in range(iters):
            for i in range(len(x)):
                a = (self.w * x[i]) + self.b
                if (a * y[i]).sign() <= 0:
                    self.w += y[i] * x[i]
                    self.b += y[i]

    def averaged(self: Perceptron, x: List[Vector], y: List[Scalar],  iters: int):
        c = 1
        for it in range(iters):
            for i in range(len(x)):
                a = (self.w * x[i]) + self.b
                if (a * y[i]).sign() <= 0:
                    self.w += y[i] * x[i]
                    self.b += y[i]
                    self.u += y[i] * Scalar(c) * x[i]
                    self.B += y[i] * Scalar(c)
                c += 1
        self.w -= Scalar(1 / c) * self.u
        self.b -= Scalar(1 / c) * self.B

    def test(self: Perceptron, x: List[Vector], y: List[Scalar]):
        sum = 0
        for i in range(len(x)):
            sum += 1 if ((self.w * x[i] + self.b) * y[i]).sign() > 0 else 0
        return sum / len(x)


# Draw decision boundary, given vector w and bias
#  the decision boundary is the hyperplane perpendicular to weights vector (w),
#  shifted by bias (b) along w in opposite direction than w [ Daumé 4.3 ]
def draw_decision_boundary(w, b, col):
    matplotlib.pyplot.axline(xy1=(0 - math.cos(math.atan2(w[1], w[0])) * b,
                                  0 - math.sin(math.atan2(w[1], w[0])) * b),
                             xy2=(-(w[1]) - math.cos(math.atan2(w[1], w[0])) * b,
                                  (w[0]) - math.sin(math.atan2(w[1], w[0])) * b), color=col)

# Task #1
def f(x):
    return x**2


def df(x):
    return 2*x


withJac = scipy.optimize.minimize(f, numpy.random.randint(-1000, 1000), jac=df)

woJac = scipy.optimize.minimize(f, numpy.random.randint(-1000, 1000), jac=False)

# scipy.optimize.minimize() always performs better with function providing gradient vector
print("scipy.optimize.minimize() performs better %s gradient vector function" % ("WITH" if withJac.get('nit') < woJac.get('nit') else "WITHOUT" ))


# Task #2
def hinge_loss_surrogate(y_gold, y_pred):
    return numpy.max([.0, 1.0 - y_gold * y_pred])


def pNorm(w, p):
    wSum = .0
    for i in range(len(w)):
        wSum += math.fabs(math.pow(w[i], p))
    return pow(wSum, 1.0 / p)


# Added xs and ys as an input arguments, to avoid using global variables
def svm_loss(wb, C, D, txs, tys):
    loss_sum = .0
    for i in range(len(tys)):
        loss_sum += hinge_loss_surrogate(tys[i], numpy.dot(wb[:D], txs[i]) + wb[D])
    return pNorm(wb[:D], 2) / 2 + C * loss_sum


# Task #3
def svm(D, txs, tys):
    svmwb = (scipy.optimize.minimize(svm_loss, numpy.array([.0, .0, .0]), args=(.25, D, txs, tys), jac=False))['x']
    matplotlib.pyplot.scatter(svmwb[0], svmwb[1], marker='o', color='green')
    draw_decision_boundary(svmwb[:2], svmwb[2], 'green')


# Task #4
def gradient_hinge_loss_surrogate(y_gold, y_pred):
    if hinge_loss_surrogate(y_gold, y_pred) == .0:
        return .0
    else:
        return -y_pred * y_gold


def gradient_svm_loss(wb, C, D, txs, tys):
    loss_sum = wb[:D]
    tys_sum = .0
    for i in range(len(tys)):
        loss_sum += gradient_hinge_loss_surrogate(tys[i], numpy.dot(wb[:D], txs[i]) + wb[D])
        tys_sum += gradient_hinge_loss_surrogate(tys[i], numpy.dot(wb[:D], txs[i]) + wb[D])
    return numpy.concatenate((wb[:D] + C * loss_sum, numpy.array([-tys_sum])))


def svm_grad(D, txs, tys, use_gradient):
    conv = [.01, -.01, .01]
    if use_gradient:
        for i in range(100):
            conv = (scipy.optimize.minimize(svm_loss, numpy.array(conv), args=(.05, D, txs, tys), jac=gradient_svm_loss))['jac']
        svmwb = (scipy.optimize.minimize(svm_loss, numpy.array(conv), args=(.05, D, txs, tys), jac=gradient_svm_loss))['x']
    else:
        svmwb = (scipy.optimize.minimize(svm_loss, numpy.array(conv), args=(.05, D, txs, tys), jac=False))['x']
    matplotlib.pyplot.scatter(svmwb[0], svmwb[1], marker='o', color='blue')
    draw_decision_boundary(svmwb[:2], svmwb[2], 'blue')


# Task #5
samples_count = 20

# Create two isolated clusters of points in 2 dimensions
x_plus = numpy.random.normal(loc=[1, 1], scale=0.5, size=(samples_count, 2))
x_minus = numpy.random.normal(loc=[-1, -1], scale=0.5, size=(samples_count, 2))

# Graph the hyperplane found by training an averaged perceptron
per = Perceptron(2)


# Function to convert numpy ndarrays to list of vectors
def ndarrToListOf2dVectors(a: numpy.ndarray) -> List[Vector]:
    return [Vector(*i) for i in a.tolist()]


# Convert numpy ndarrays to list of vectors, so that our pure-python perceptron could handle em
# We place em into single list containing positive ones, followed by negative ones
xs = ndarrToListOf2dVectors(x_plus) + ndarrToListOf2dVectors(x_minus)

# Mark list of answers for perceptron. If we change sign of answers in list
# then we will find trained perceptron weights vector being pointed opposite side
ys = [Scalar(.25)] * samples_count + [Scalar(-.25)] * samples_count

# Train averaged perceptron in 10 buckets of 25 trainings, shuffling training data before each bucket
for i in range(10):
    ys, xs = [list(t) for t in zip(*sample(list(zip(ys, xs)), samples_count * 2))]
    per.averaged(xs, ys, 25)

# Graph data for averaged perceptron
# Mark perceptron weights vector direction with red dot
matplotlib.pyplot.scatter(per.w.entries[0], per.w.entries[1], marker='o', color='red')
draw_decision_boundary(per.w.entries, per.b.val, 'red')


# Graph data for support vector machine without gradient
svm(2, numpy.concatenate((x_plus, x_minus)), numpy.array([.25 for i in range(samples_count)] + [-.25 for i in range(samples_count)]))

# Graph data for support vector machine with gradient
svm_grad(2, numpy.concatenate((x_plus, x_minus)), numpy.array([.25 for i in range(samples_count)] + [-.25 for i in range(samples_count)]), True)

# Plot input clusters of points
matplotlib.pyplot.scatter(x_plus[:, 0], x_plus[:, 1], marker='+', color='blue')
matplotlib.pyplot.scatter(x_minus[:, 0], x_minus[:, 1], marker='x', color='red')

matplotlib.pyplot.savefig("svm-svm-perceptron.pdf")

matplotlib.pyplot.show()
