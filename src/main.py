from typing import Tuple, Optional, NoReturn, Literal
from collections.abc import Callable
from dataclasses import dataclass
import math
import random


# ======
# convinience definition
# ======

Function = Callable[['Vec2'], float]


def trace(f):
    VERBOSE = False
    DEBUG = False
    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        if not DEBUG:
            return res
        else:
            if VERBOSE:
                params = ', '.join(map(str, args))
                keywords = "" if len(kwargs) == 0 else str(kwargs)
                print(
                    f"{f.__name__}({params}, {keywords}) =\n {res}"
                )
            else:
                print(f"{f.__name__}() = {res}")

            return res
    return wrapper


def panic(msg: str) -> NoReturn:
    """Raise RuntimeError with given message"""
    raise RuntimeError(msg)


def DEBUG(*args, **kwargs) -> None:
    """Just a fancy keyword for a print()"""
    print(*args, **kwargs)


@dataclass(frozen=True, eq=True)
class Vec2:
    pair: Tuple[float, float]

    def add(self, other: 'Vec2') -> 'Vec2':
        """x + y"""
        x1, x2 = self.pair
        y1, y2 = other.pair
        return point(x1 + y1, x2 + y2)

    def sub(self, other: 'Vec2') -> 'Vec2':
        """x - y"""
        return self + other * -1

    def times(self, k: float) -> 'Vec2':
        """k * x"""
        x1, x2 = self.pair
        return point(x1 * k, x2 * k)

    def __add__(self, other: 'Vec2') -> 'Vec2':
        return self.add(other)

    def __sub__(self, other: 'Vec2') -> 'Vec2':
        return self.sub(other)

    def __mul__(self, k: float) -> 'Vec2':
        return self.times(k)

    def __repr__(self):
        x, y = self.pair
        return f"({x:.5f}, {y:.5f})"

    def norm(self) -> float:
        """||x||"""
        x1, x2 = self.pair
        return math.sqrt(x1 ** 2 + x2 ** 2)


def point(x1: float, x2: float) -> Vec2:
    """Helper function to conver x1, x2 to Vec2"""
    return Vec2((x1, x2))


# =======
# constants
# =======
X0 = point(0, 0)
K = 0.1  # swen step coefficient
STEP_NUM = 1_000  # swen max step number
GOLDEN_EPSILON = 0.1  # precision of golden_ratio
EPSILON = 0.0000001  # presicion of optimization
RADIUS = 0.1  # radius of random search
N = 20  # number of random tries in random search


def lstep(xi: Vec2, si: Vec2, k: float) -> float:
    """Returns step for Swen alorithm"""
    return k * xi.norm() / si.norm()


def choose_direction(f: Function, x0: Vec2, s: Vec2, d: float) -> Optional[float]:
    """Direction for Swen algorithm

    Calculate function in [x0 - step, x + step]
    Returns sign() of direction needed to minimize a function
    """

    delta = lstep(x0, s, d)

    left = f(x0 - s * delta)
    exact = f(x0)
    right = f(x0 + s * delta)

    if left > exact and exact > right:
        return 1.0
    elif left < exact and exact < right:
        return -1.0
    else:  # if left == right
        return None


@trace
def swen_interval(f: Function, x0: Vec2, s: Vec2, d: float) -> Tuple[float, float, float]:
    """Base interval for use with golden-ratio or DSK-Powell

    Returns (left, approx_min, right)"""
    step = lstep(x0, s, d)
    direction = choose_direction(f, x0, s, d)
    if direction is None:
        return (-step, 0, step)

    k = 0
    way = [
        (0.0, f(x0))
    ]
    while k < STEP_NUM:
        previous_point, fp = way[-1]

        next_point = previous_point + direction * step * (2 ** k)
        fn = f(x0 + s * next_point)
        if fn > fp:
            middle_point = (next_point + previous_point) / 2
            fm = f(x0 + s * middle_point)

            if fm < fp:
                return (previous_point, middle_point, next_point)
            else:
                if len(way) < 2:
                    panic("tried to get 2th point from the end, "
                          "while way in swen_alogrithm has only one value")
                next_previos_point, _ = way[-2]
                return (next_previos_point, previous_point, middle_point)
        else:
            k += 1
            way.append((next_point, fn))
    panic(f"more than {STEP_NUM=} steps in swen algorithm")


@trace
def golden_ratio(
        f: Function,
        interval: Tuple[float, float],
        x: Vec2,
        s: Vec2,
        eps: float,
) -> float:
    """Find extremum of `f` on `interval` by searching with `s` direction"""
    left, right = interval

    while abs(left - right) > eps:
        length = abs(left - right)
        x1 = left + 0.382 * length
        x2 = left + 0.618 * length

        f1 = f(x + s * x1)
        f2 = f(x + s * x2)

        if f1 < f2:
            left, right = left, x1
        else:
            left, right = x1, right
    # we are adding `eps` to avoid zero paths
    return left + eps


@trace
def random_try(f: Function, x: Vec2, r: float, n: int) -> Vec2:
    """Return direction to search for next minimum

    Tries points in the radius `r` around `x` and find minimum of `f` for these tries"""
    acc = 0
    tries = []

    # @trace
    def try_point(dot):
        si = point(math.sin(dot), math.cos(dot))
        xi = x + si * r
        fi = f(xi)
        tries.append((si, fi))
        return (si, fi)

    while acc < math.pi:
        acc += math.pi / n
        try_point(acc + random.uniform(0, math.pi/n))
    s, _ = min(tries, key=lambda pair: pair[1])  # type: ignore
    return s


@trace
def satisfactory_result(delta_point: Vec2, delta_f: float, eps: float) -> bool:
    return delta_point.norm() < eps and delta_f < eps


@trace
def minimize_with_random_search(
        f: Function,
        x0: Vec2,
        *,
        way_search: Literal["golden_ratio", "dsk"],
        d_swen: float = K,
        golden_eps: float = GOLDEN_EPSILON,
        n: int = N,
        radius: float = RADIUS,
        eps: float = EPSILON,
) -> Vec2:
    """Minimize function `f` with method of random search and constant step"""
    search = True
    xi = x0
    fi = f(x0)
    while search:
        s = random_try(f, xi, radius, n)
        interval = swen_interval(f, xi, s, d_swen)
        lmbd = 0.0
        if way_search == "golden_ratio":
            left, _, right = interval
            lmbd = golden_ratio(f, (left, right), xi, s, golden_eps)
        xn = xi + s * lmbd
        fn = f(xn)
        if satisfactory_result(xn - xi, fn - fi, eps):
            search = False
        else:
            xi = xn
            fi = fn
    return xi


class Rozenbrok:
    counter: int

    def __init__(self):
        self.counter = 0

    def __repr__(self):
        return f"f-{self.counter}"

    #@trace
    def __call__(self, point: Vec2) -> float:
        #import time
        #time.sleep(0.05)

        self.counter += 1
        A = 1
        B = 100
        x, y = point.pair
        return (A - x) ** 2 + B * (y - x ** 2) ** 2


f = Rozenbrok()
extremum = minimize_with_random_search(
    f,
    point(0.0, 0.0),
    way_search="golden_ratio",
    d_swen=0.1,
    golden_eps=0.01,
    n=3,
    radius=0.1,
    eps=0.001,
)

minimum = f(extremum)
log10_times = math.log(f.counter, 10)
print(f"{extremum=}\n"
      f"{minimum=}\n"
      f"{log10_times=}")
