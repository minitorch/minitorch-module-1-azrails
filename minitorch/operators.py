"""Collection of the core mathematical operators used throughout the code base."""

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$
import math
from typing import Callable, Generator, Iterable


# TODO: Implement for Task 0.1.
def add(l: float, r: float) -> float:
    """Adds two numbers"""
    return l + r


def id(l: float) -> float:
    """Returns the input unchanged"""
    return float(l)


def mul(l: float, r: float) -> float:
    """Multiplies two numbers"""
    return l * r


def neg(l: float) -> float:
    """Negates a number"""
    return -float(l)


def lt(l: float, r: float) -> bool:
    """Checks if one number is less than another"""
    return l < r


def eq(l: float, r: float) -> bool:
    """Checks if two numbers are equal"""
    return l == r


def max(l: float, r: float) -> float:
    """Returns the larger of two numbers"""
    return l if l > r else r


def exp(l: float) -> float:
    """Calculates the exponential function"""
    return math.exp(l)


def sigmoid(l: float) -> float:
    """Calculates the sigmoid function"""
    return 1 / (1 + exp(-l)) if l >= 0 else exp(l) / (1 + exp(l))


def inv(l: float) -> float:
    """Calculates the reciprocal"""
    return 1.0 / l


def log(l: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(l)


def log_back(l: float, r: float) -> float:
    """Computes the derivative of log times a second arg"""
    return inv(l) * r


def inv_back(l: float, r: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -inv(l) ** 2 * r


def relu(l: float) -> float:
    """Applies the ReLU activation function"""
    return 0.0 if lt(l, 0) or eq(l, 0) else l


def relu_back(l: float, r: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return 0 if l <= 0 else r


def is_close(l: float, r: float) -> float:
    """Check that numbers is similar"""
    res = l - r
    if res < 0:
        return neg(res) < 1e-2
    return res < 1e-2


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(
    fn: Callable[[float], float], lst: Iterable[float]
) -> Generator[float, None, None]:
    """Apply function to all elements"""
    for elem in lst:
        yield fn(elem)


def zipWith(
    fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]
) -> Generator[float, None, None]:
    """Apply function to all pairs"""
    for el1, el2 in zip(lst1, lst2):
        yield fn(el1, el2)


def reduce(
    fn: Callable[[float, float], float], lst: Iterable[float], initializer: float = 0.0
) -> float:
    """Apply cumalative function"""
    res = initializer
    for el in lst:
        res = fn(el, res)
    return res


def negList(lst: list[float]) -> list[float]:
    """Apply negates to all numbers"""
    return list(map(neg, lst))


def addLists(lst1: list[float], lst2: list[float]) -> list[float]:
    """Sum two wectors"""
    return list(zipWith(add, lst1, lst2))


def sum(lst: list[float]) -> float:
    """Sum all values in vector"""
    return reduce(add, lst, 0)


def prod(lst: list[float]) -> float:
    """Prod all values in vector"""
    return reduce(mul, lst, 1.0)
