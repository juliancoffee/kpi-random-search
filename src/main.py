from typing import Tuple, Optional, NoReturn, Sequence
from collections.abc import Callable
from dataclasses import dataclass
import math
import random
import functools


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


def avg(iterable: Sequence[float]) -> float:
    """Returns mean of sequence of `-inf` if empty"""
    if len(iterable) == 0:
        return float("-inf")
    return sum(iterable) / len(iterable)


def moda(xs: list[float]) -> float:
    """Returns average value with removing anomaly big values"""
    data = [min(xs)]
    values = sorted(xs)[1:]
    for x in values:
        if abs(x - avg(data)) < 100:
            data.append(x)
    return avg(data)


def show_results(results: list[float], times: list[int]):
    print("==== Results =====")
    min_res = min(results)
    max_res = max(results)
    avg_res = avg(results)
    moda_res = moda(results)
    print(
        f"Function results.\n"
        f"min: {min_res:.2e}, moda: {moda_res:.2e}, avg: {avg_res:.2e}, max: {max_res:.2e}")
    min_time = min(times)
    max_time = max(times)
    print(
        f"Function calculation. [{min_time:.2e} : {max_time:.2e}]")


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
DSK_EPSILON = 0.01  # precision of dsk_powell
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
    """Returns way length to go to minimize a function `f`

    Finds extremum of `f` on `interval` by searching with `s` direction"""
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
    # add noise to avoid zero paths
    return left + eps


@trace
def dsk_powell(
        f: Function,
        interval: Tuple[float, float, float],
        x0: Vec2,
        s: Vec2,
        eps: float,
) -> float:
    """Returns way length to go to minimize a function `f`

    Finds extremum of `f` on `interval` by quadratic approximation"""

    def final(x2: float, x_star: float, f2: float, f_star: float):
        first = abs(x2 - x_star)
        second = abs(f2 - f_star)
        return first < eps and second < eps

    x1, x2, x3 = interval
    while True:
        # compute functions
        f1 = f(x0 + s * x1)
        f2 = f(x0 + s * x2)
        f3 = f(x0 + s * x3)

        a1 = (f2 - f1) / (x2 - x1)
        a2 = (1 / (x3 - x1)) * \
            ((f3 - f2) / (x3 - x2) - ((f2 - f1) / (x2 - x1)))

        x_star = (x1 + x2) / 2 - a1 / (2 * a2)
        f_star = f(x0 + s * x_star)

        if final(x2, x_star, f2, f_star):
            return x_star
        else:
            x_min, _ = min(
                (x1, f1),
                (x2, f2),
                (x3, f3),
                (x_star, f_star),
                key=lambda pair: pair[1])
            left = filter(lambda x: x < x_min and x != x_min,
                          (x1, x2, x3, x_star))
            right = filter(lambda x: x > x_min and x != x_min,
                           (x1, x2, x3, x_star))
            x1 = max(left)
            x2 = x_min
            x3 = min(right)


@trace
def random_try(f: Function, x: Vec2, r: float, n: int) -> Vec2:
    """Return direction to search for next minimum

    Tries points in the radius `r` around `x` and find minimum of `f` for these tries"""
    acc = 0.0
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
def satisfactory_result(
    next_point: Vec2,
    point: Vec2,
    fk1: float,
    fk: float,
    eps: float,
) -> bool:
    delta_x = (next_point - point).norm()  # / point.norm()
    delta_f = abs((fk1 - fk))  # / abs(fk)
    return delta_x < eps and delta_f < eps


@trace
def minimize_with_random_search(
        f: Function,
        x0: Vec2,
        *,
        way_search: str,  # Literal["golden_ratio", "dsk"]
        d_swen: float = K,
        dsk_eps: float = DSK_EPSILON,
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
        elif way_search == "dsk":
            lmbd = dsk_powell(f, interval, xi, s, dsk_eps)
        xn = xi + s * lmbd
        fn = f(xn)
        if satisfactory_result(xn, xi, fn, fi, eps):
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
    @functools.lru_cache
    def calculate(self, point: Vec2) -> float:
        self.counter += 1
        A = 1
        B = 100
        x, y = point.pair
        return (A - x) ** 2 + B * (y - x ** 2) ** 2

    def __call__(self, point: Vec2) -> float:
        #import time
        # time.sleep(0.05)

        return self.calculate(point)


# === Tests ===
def test_way_search():
    # test data
    start = point(50, 50)
    d_swen = 0.01
    n = 9
    radius = 0.001
    # placeholders, filled later
    dsk_eps = 0.0
    golden_eps = 0.0
    eps = 0.0
    print(
        "\n== "
        "Testing how function calculation varies depending on "
        "method for way seach (golden_ratio or dsk_powell) =="
    )
    print(
        f"{d_swen=}\n"
        f"{start=}\n"
        f"{radius=}\n"
        f"{n=}"
    )
    methods = ["golden_ratio", "dsk"]
    for way_search in methods:
        if way_search == "golden_ratio":
            golden_eps = 0.01
            eps = 0.001
            print(f"\nUsing GoldenRatio\n"
                  f"{golden_eps=}\n"
                  f"{eps=}")
        elif way_search == "dsk":
            eps = 1e-09
            dsk_eps = 0.000_1
            print(f"\nUsing DSK-Powell\n"
                  f"{dsk_eps=}\n"
                  f"{eps=}")
        results = []
        times = []
        tests = 10
        for _ in range(tests):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                golden_eps=golden_eps,
                dsk_eps=dsk_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        show_results(results, times)
    print(
        ">\n"
        "Using DSK-Powell method for optimal path"
        " allows us to get much more precise values"
    )


def test_starts():
    input("press any key to continue")
    d_swen = 0.001
    way_search = "dsk"
    dsk_eps = 0.000_1
    n = 10
    radius = 0.001
    eps = 1e-09
    print("==== Testing how function calculation varies depending on starting point ====")
    print(
        f"{d_swen=}\n"
        f"{way_search=}\n"
        f"{dsk_eps=}\n"
        f"{n=}\n"
        f"{radius=}\n"
        f"{eps=}\n"
    )
    tests = 20
    results = []
    times = []
    for i in range(tests):
        f = Rozenbrok()
        # add epsilon to avoid zero division
        start = point(1.0, 1.0) * (10 * (i - 10) + eps)
        print(
            f"Starting at {start}"
        )
        extremum = minimize_with_random_search(
            f,
            start,
            way_search=way_search,
            d_swen=d_swen,
            dsk_eps=dsk_eps,
            n=n,
            radius=radius,
            eps=eps,
        )

        minimum = f(extremum)
        count = f.counter
        times.append(f.counter)
        results.append(minimum)
        print(f"min = {minimum:.2e} with {count=:.2e}")
    show_results(results, times)
    print(
        ">\n"
        "Calculation time doesn't seem to correlate with starting point"
    )


def test_golden_epsilon():
    input("press any key to continue")
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
    for golden_eps in epsilons:
        print(
            f"Using {golden_eps=}"
        )
        results = []
        times = []
        tests = 5
        for _ in range(tests):
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
        show_results(results, times)
    print(
        ">\n"
        "Function minimum is more precise with lesser epsilon for golden ratio.\n"
        "(it comes with more of function calculation)\n"
        "Buf when epsilon for golden_ratio coming close to random method epsilon\n"
        "method starts to produce false-positives"
    )


def test_swen_precision():
    input("press any key to continue")
    start = point(50, 50)
    way_search = "dsk"
    dsk_eps = 0.000_1
    n = 10
    radius = 0.001
    eps = 1e-09
    print(
        "\n== "
        "Testing how function calculation varies depending on "
        "`d` in swen_algorithm =="
    )
    print(
        f"{start=}\n"
        f"{dsk_eps=}\n"
        f"{way_search=}\n"
        f"{n=}\n"
        f"{radius=}\n"
        f"{eps=}\n"
    )
    ds = [0.05, 0.01, 0.001, 0.0001, eps * 0.001]
    for d_swen in ds:
        print(
            f"Using {d_swen=}"
        )
        results = []
        times = []
        tests = 5
        for _ in range(tests):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                dsk_eps=dsk_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        show_results(results, times)
    print(
        ">\n"
        "Using lesser `d` for step calculation in swen interval "
        "improves both speed and precision of method\n"
        "But as in case with precision of golden_ratio it produces false-positives "
        "when `d` is lesser than `eps` for random search method"
    )


def test_radius():
    input("press any key to continue")
    start = point(50, 50)
    d_swen = 1e-05
    way_search = "dsk"
    dsk_eps = 0.000_1
    n = 10
    eps = 1e-09
    print("")
    print(
        "== "
        "Testing how function calculation varies depending on "
        "`radius` of random search =="
    )
    print(
        f"{d_swen=}\n"
        f"{start=}\n"
        f"{dsk_eps=}\n"
        f"{way_search=}\n"
        f"{n=}\n"
        f"{eps=}\n"
    )
    rs = [1.0, 0.5, 0.2, 0.1, 0.01, 0.001, 0.000_01, 0.000_000_1]
    for radius in rs:
        print(
            f"Using {radius=}"
        )
        results = []
        times = []
        tests = 10
        for _ in range(tests):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                dsk_eps=dsk_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        show_results(results, times)
    print(
        ">\n"
        "Different radius doestn't seem to correlate with precision much"
    )


def test_tries():
    input("press any key to continue")
    start = point(50, 50)
    d_swen = 1e-05
    way_search = "dsk"
    dsk_eps = 0.000_1
    radius = 0.001
    eps = 1e-09
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
        f"{dsk_eps=}\n"
        f"{radius=}\n"
        f"{eps=}\n"
    )
    ns = [3, 5, 7, 8, 9, 10, 12, 15, 20]
    for n in ns:
        print(
            f"Using {n=}"
        )
        results = []
        times = []
        tests = 10
        for _ in range(tests):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                dsk_eps=dsk_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        show_results(results, times)
    print(
        ">\n"
        "More tries produces slightly more precise values"
    )


def test_epsilon():
    input("press any key to continue")
    start = point(50, 50)
    d_swen = 1e-05
    way_search = "dsk"
    dsk_eps = 0.000_1
    n = 10
    radius = 0.001
    print(
        "\n== "
        "Testing how function calculation varies depending on "
        "epsilon in random search =="
    )
    print(
        f"{d_swen=}\n"
        f"{start=}\n"
        f"{way_search=}\n"
        f"{dsk_eps=}\n"
        f"{radius=}\n"
        f"{n=}\n"
    )
    epsilons = [0.000_1, 1e-06, 1e-09, 1e-12]
    for eps in epsilons:
        print(
            f"Using {eps=}"
        )
        results = []
        times = []
        tests = 10
        for _ in range(tests):
            f = Rozenbrok()
            extremum = minimize_with_random_search(
                f,
                start,
                way_search=way_search,
                d_swen=d_swen,
                dsk_eps=dsk_eps,
                n=n,
                radius=radius,
                eps=eps,
            )

            minimum = f(extremum)
            times.append(f.counter)
            results.append(minimum)
        show_results(results, times)
    print(
        ">\n"
        "Lesser epsilon gives more precise calculation, but requires more time"
    )

test_way_search()
test_starts()
test_golden_epsilon()
test_swen_precision()
test_radius()
test_tries()
test_epsilon()
