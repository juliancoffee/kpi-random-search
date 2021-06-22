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


def debug(*args, **kwargs) -> None:
    """Just a fancy keyword for a print()"""
    print(*args, **kwargs)


def float_to_exp(x: float) -> str:
    digits = 0
    if x > 1:
        while x > 1:
            x /= 10
            digits += 1
    else:
        while x < 1:
            x *= 10
            digits -= 1
    return f"{x:.3} * 10^{digits}"


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
    # we are adding noise to avoid zero paths
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

    # @trace
    def __call__(self, point: Vec2) -> float:
        #import time
        # time.sleep(0.05)

        self.counter += 1
        A = 1
        B = 100
        x, y = point.pair
        return (A - x) ** 2 + B * (y - x ** 2) ** 2


# === Tests ===
def test_starts():
    d_swen = 0.01
    way_search = "golden_ratio"
    golden_eps = 0.01
    n = 3
    radius = 1e-05
    eps = 0.001
    print("==== Testing how function calculation varies depending on starting point ====")
    print(
        f"{d_swen=}\n"
        f"{way_search=}\n"
        f"{golden_eps=}\n"
        f"{n=}\n"
        f"{radius=}\n"
        f"{eps=}\n"
    )
    tests = 10
    results = []
    times = []
    for i in range(tests):
        f = Rozenbrok()
        start = point(1.0, 1.0) * 10 * i
        print(
            f"Starting at {start}"
        )
        extremum = minimize_with_random_search(
            f,
            start,
            way_search=way_search,
            d_swen=d_swen,
            golden_eps=golden_eps,
            n=n,
            radius=radius,
            eps=eps,
        )

        minimum = f(extremum)
        count = float_to_exp(f.counter)
        times.append(f.counter)
        results.append(minimum)
        print(f"min = {float_to_exp(minimum)} with {count}")
    print("==== Results =====")
    min_res = float_to_exp(min(results))
    max_res = float_to_exp(max(results))
    avg_res = float_to_exp(sum(results) / tests)
    print(
        f"Function results.     min: {min_res}, avg: {avg_res}, max: {max_res}")
    min_time = float_to_exp(min(times))
    max_time = float_to_exp(max(times))
    avg_time = float_to_exp(sum(times) / tests)
    print(
        f"Function calculation. min: {min_time}, avg: {avg_time}, max: {max_time}")
    print(
        ">\n"
        "Calculation time doesn't seem to correlate with starting point"
    )


def test_golden_epsilon():
    start = point(50, 50)
    d_swen = 0.01
    way_search = "golden_ratio"
    n = 3
    radius = 1e-05
    eps = 0.001
    print("")
    print("== Testing how function calculation varies depending on `e` in golden_ratio ==")
    print(
        f"{start=}\n"
        f"{d_swen=}\n"
        f"{way_search=}\n"
        f"{n=}\n"
        f"{radius=}\n"
        f"{eps=}\n"
    )
    epsilons = [0.2, 0.1, 0.05, 0.02, 0.01, 0.001]
    tests = len(epsilons)
    for golden_eps in epsilons:
        print(
            f"Using {golden_eps=}"
        )
        results = []
        times = []
        for _ in range(5):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                golden_eps=golden_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        print("==== Results =====")
        min_res = float_to_exp(min(results))
        max_res = float_to_exp(max(results))
        avg_res = float_to_exp(sum(results) / tests)
        print(
            f"Function results.     min: {min_res}, avg: {avg_res}, max: {max_res}")
        min_time = float_to_exp(min(times))
        max_time = float_to_exp(max(times))
        avg_time = float_to_exp(sum(times) / tests)
        print(
            f"Function calculation. min: {min_time}, avg: {avg_time}, max: {max_time}")
    print(
        ">\n"
        "Function minimum is more precise with lesser epsilon for golden ratio.\n"
        "(it comes with more of function calculation)\n"
        "Buf when epsilon for golden_ratio coming close to random method epsilon\n"
        "method starts to produce false-positives"
    )


def test_swen_precision():
    start = point(50, 50)
    way_search = "golden_ratio"
    golden_eps = 0.01
    n = 3
    radius = 1e-05
    eps = 0.001
    print("")
    print(
        "== "
        "Testing how function calculation varies depending on "
        "`d` in swen_algorithm =="
    )
    print(
        f"{start=}\n"
        f"{golden_eps=}\n"
        f"{way_search=}\n"
        f"{n=}\n"
        f"{radius=}\n"
        f"{eps=}\n"
    )
    ds = [0.05, 0.01, 0.001, 0.0001]
    tests = len(ds)
    for d_swen in ds:
        print(
            f"Using {d_swen=}"
        )
        results = []
        times = []
        for _ in range(5):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                golden_eps=golden_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        print("==== Results =====")
        min_res = float_to_exp(min(results))
        max_res = float_to_exp(max(results))
        avg_res = float_to_exp(sum(results) / tests)
        print(
            f"Function results.     min: {min_res}, avg: {avg_res}, max: {max_res}")
        min_time = float_to_exp(min(times))
        max_time = float_to_exp(max(times))
        avg_time = float_to_exp(sum(times) / tests)
        print(
            f"Function calculation. min: {min_time}, avg: {avg_time}, max: {max_time}")
    print(
        ">\n"
        "Using lesser `d` for step calculation in swen interval "
        "improves both speed and precision of method\n"
        "But as in case with precision of golden_ratio it produces false-positives "
        "when `d` is lesser than `eps` for random search method"
    )

def test_radius():
    start = point(50, 50)
    d_swen = 0.01
    way_search = "golden_ratio"
    golden_eps = 0.01
    n = 5
    eps = 0.001
    print("")
    print(
        "== "
        "Testing how function calculation varies depending on "
        "`radius` of random search =="
    )
    print(
        f"{d_swen=}\n"
        f"{start=}\n"
        f"{golden_eps=}\n"
        f"{way_search=}\n"
        f"{n=}\n"
        f"{eps=}\n"
    )
    rs = [1.0, 0.5, 0.2, 0.1, 0.01, 0.001, 0.000_01, 0.000_000_1]
    tests = len(rs)
    for radius in rs:
        print(
            f"Using {radius=}"
        )
        results = []
        times = []
        for _ in range(10):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                golden_eps=golden_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        print("==== Results =====")
        min_res = float_to_exp(min(results))
        max_res = float_to_exp(max(results))
        avg_res = float_to_exp(sum(results) / tests)
        print(
            f"Function results.     min: {min_res}, avg: {avg_res}, max: {max_res}")
        min_time = float_to_exp(min(times))
        max_time = float_to_exp(max(times))
        avg_time = float_to_exp(sum(times) / tests)
        print(
            f"Function calculation. min: {min_time}, avg: {avg_time}, max: {max_time}")
    print(
        ">\n"
        "Lesser radius produces both better speeds and precision, but after some point "
        " it's getting to worse values"
    )

def test_tries():
    start = point(50, 50)
    d_swen = 0.01
    way_search = "golden_ratio"
    golden_eps = 0.01
    radius = 1e-05
    eps = 0.001
    print("")
    print(
        "== "
        "Testing how function calculation varies depending on "
        "number of tries in random search =="
    )
    print(
        f"{d_swen=}\n"
        f"{start=}\n"
        f"{way_search=}\n"
        f"{golden_eps=}\n"
        f"{radius=}\n"
        f"{eps=}\n"
    )
    ns = [3, 5, 7, 8, 9, 10, 12, 15]
    tests = len(ns)
    for n in ns:
        print(
            f"Using {n=}"
        )
        results = []
        times = []
        for _ in range(10):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                golden_eps=golden_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        print("==== Results =====")
        min_res = float_to_exp(min(results))
        max_res = float_to_exp(max(results))
        avg_res = float_to_exp(sum(results) / tests)
        print(
            f"Function results.     min: {min_res}, avg: {avg_res}, max: {max_res}")
        min_time = float_to_exp(min(times))
        max_time = float_to_exp(max(times))
        avg_time = float_to_exp(sum(times) / tests)
        print(
            f"Function calculation. min: {min_time}, avg: {avg_time}, max: {max_time}")
    print(
        ">\n"
        "More tries produces more function calculation and doesn't seem to be correlacted "
        "with precision of method"
    )

def test_epsilon():
    start = point(50, 50)
    d_swen = 0.01
    way_search = "golden_ratio"
    golden_eps = 0.01
    radius = 1e-05
    n = 9
    print("")
    print(
        "== "
        "Testing how function calculation varies depending on "
        "epsilon in random search =="
    )
    print(
        f"{d_swen=}\n"
        f"{start=}\n"
        f"{way_search=}\n"
        f"{golden_eps=}\n"
        f"{radius=}\n"
        f"{n=}\n"
    )
    epsilons = [1.0, 0.5, 0.1, 0.01, 0.001, 0.000_1, 0.000_01]
    tests = len(epsilons)
    for eps in epsilons:
        print(
            f"Using {eps=}"
        )
        results = []
        times = []
        for _ in range(10):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                golden_eps=golden_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        print("==== Results =====")
        min_res = float_to_exp(min(results))
        max_res = float_to_exp(max(results))
        avg_res = float_to_exp(sum(results) / tests)
        print(
            f"Function results.     min: {min_res}, avg: {avg_res}, max: {max_res}")
        min_time = float_to_exp(min(times))
        max_time = float_to_exp(max(times))
        avg_time = float_to_exp(sum(times) / tests)
        print(
            f"Function calculation. min: {min_time}, avg: {avg_time}, max: {max_time}")
    print(
        ">\n"
        "eps = 0.001 seems to give best result in term of calculation times and presicion"
    )

#test_starts()
#test_golden_epsilon()
#test_swen_precision()
#test_radius()
#test_tries()
#test_epsilon()
