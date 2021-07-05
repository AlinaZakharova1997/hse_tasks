from __future__ import annotations
from typing import Union, List
from operator import add, mul
from math import sqrt
from random import randint
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


samplesCount = 500
trainSamples = samplesCount//10

per = Perceptron(2)

# Task #3
print("Task #3: expect output to be mostly around 1")
for i in range(100):
    v = Vector(randint(-100, 100), randint(-100, 100))
    xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(samplesCount)]
    ys = [v * x * Scalar(randint(-1, 9))for x in xs]

    # Make a 90-10 test-train split
    trainXs = xs[:trainSamples]
    trainYs = ys[:trainSamples]
    testXs = xs[trainSamples:]
    testYs = ys[trainSamples:]

    per.train(trainXs, trainYs, 10)

    # You should get that w is some multiple of v, and the performance should be very good.
    print("%f" % 0 if v.entries[0] == 0 or v.entries[1] == 0 else
          ((per.w.entries[0] / v.entries[0]) / (per.w.entries[1] / v.entries[1])))

    per.clear()

# Task #4
print("Task #4: expect success rate output to be around .5")
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(samplesCount)]
ys = [Scalar(1) if x.entries[0] * x.entries[1] < 0 else Scalar(-1) for x in xs]

# You should get some relatively random w, and the performance should be terrible.
per.train(xs[:trainSamples], ys[:trainSamples], 100)

# should print around 0.5 success rate
print(per.test(xs[trainSamples:], ys[trainSamples:]))

per.clear()

# Task #5
print("Task #5:")
v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(samplesCount)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]

# Sort the training data from task 3 by y
ys, xs = [list(t) for t in zip(*sorted(zip(ys, xs)))]

trainXs = xs[:trainSamples]
trainYs = ys[:trainSamples]
testXs = xs[trainSamples:]
testYs = ys[trainSamples:]

# Graph the performance on both train and test sets versus epochs
# for perceptron trained on no permutation
print("Graph the performance w/o permutation:")
for i in range(100):
    per.train(trainXs, trainYs, 1)
    print("%f %f" % (per.test(testXs, testYs), per.test(trainXs, trainYs)))

per.clear()

# Graph the performance on both train and test sets versus epochs
# for perceptron trained on random permutation at the beginning
ys, xs = [list(t) for t in zip(*sample(list(zip(ys, xs)), samplesCount))]

trainXs = xs[:trainSamples]
trainYs = ys[:trainSamples]
testXs = xs[trainSamples:]
testYs = ys[trainSamples:]

print("Graph the performance on random permutation at the beginning:")
for i in range(100):
    per.train(trainXs, trainYs, 1)
    print("%f %f" % (per.test(testXs, testYs), per.test(trainXs, trainYs)))

per.clear()

# Graph the performance on both train and test sets versus epochs
# for perceptron trained on random permutation at each epoch
print("Graph the performance on random permutation at each epoch:")
for i in range(100):
    ys, xs = [list(t) for t in zip(*sample(list(zip(ys, xs)), samplesCount))]

    trainXs = xs[:trainSamples]
    trainYs = ys[:trainSamples]
    testXs = xs[trainSamples:]
    testYs = ys[trainSamples:]

    per.train(trainXs, trainYs, 1)

    print("%f %f" % (per.test(testXs, testYs), per.test(trainXs, trainYs)))

per.clear()


# Task #6
print("Graph the performance of averaged perceptron on random permutation at each epoch:")
for i in range(100):
    ys, xs = [list(t) for t in zip(*sample(list(zip(ys, xs)), samplesCount))]

    trainXs = xs[:trainSamples]
    trainYs = ys[:trainSamples]
    testXs = xs[trainSamples:]
    testYs = ys[trainSamples:]

    per.averaged(trainXs, trainYs, 1)

    print("%f %f" % (per.test(testXs, testYs), per.test(trainXs, trainYs)))

per.clear()
