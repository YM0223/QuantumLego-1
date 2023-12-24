"""Copied stuff from tensor_enum.py."""

import functools
import itertools
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from numpy.polynomial import Polynomial


def genbstring(n: int) -> Iterator[npt.NDArray[np.int64]]:
    """Generate all binary strings of length n."""
    for b in itertools.product([0, 1], repeat=n):
        yield np.asarray(b, dtype=np.int64)


def _is_1d(arr: npt.NDArray) -> None:
    if arr.ndim != 1:
        msg = "Array must be 1D."
        raise ValueError(msg)


def _is_2d(arr: npt.NDArray) -> None:
    if arr.ndim != 2:  # noqa: PLR2004
        msg = "Array must be 2D."
        raise ValueError(msg)


def stabilizer_wt(stab: npt.NDArray) -> int:
    """Return the weight of a stabilizer."""
    _is_1d(stab)
    n = stab.size // 2
    ret = (stab[:n] + stab[n:] > 0).sum()
    return int(ret)


def xz_wt(stab: npt.NDArray) -> tuple[int, int]:
    """Return the X and Z weights of a stabilizer."""
    _is_1d(stab)
    n = stab.size // 2
    retw, retz = (stab[:n] > 0).sum(), (stab[n:] > 0).sum()
    return int(retw), int(retz)


def scalar_enum(h: npt.NDArray) -> npt.NDArray[np.int64]:
    """Return the scalar enumeration of a stabilizer code."""
    _is_2d(h)
    rows, cols = h.shape
    n = cols // 2
    enum = np.zeros(n + 1, dtype=np.int64)
    for v in genbstring(rows):
        enum[stabilizer_wt((v.T @ h) % 2)] += 1
    return enum


def double_enum(h: npt.NDArray) -> npt.NDArray[np.int64]:
    """Return the double enumeration of a stabilizer code."""
    rows, cols = h.shape
    n = cols // 2
    enum = np.zeros((n + 1, n + 1), dtype=np.int64)
    for v in genbstring(rows):
        w, z = xz_wt((v.T @ h) % 2)  # xz_wt の戻り値を変数に分解
        enum[w, z] += 1  # アンパック演算子を使用せずにインデックスを指定
    return enum


#@functools.cache
@functools.lru_cache(maxsize=None)
def _macwilliams_impl(r_size: int) -> npt.NDArray[np.float64]:
    # tab[i, j] = x**i coefficient of (1 - x)**j * (1 + 3 * x) ** (r_size - 1 - j)
    tab = np.zeros((r_size, r_size), dtype=np.float64)
    a = Polynomial([1, -1])
    b = Polynomial([1, 3])
    for i_co in range(r_size):
        f = a**i_co * b ** (r_size - 1 - i_co)
        assert isinstance(f, Polynomial)
        tab[:, i_co] = f.coef
    return tab

#@functools.cache
@functools.lru_cache(maxsize=None)
def _macwillamsdouble_impl(r_size: int) -> npt.NDArray:
    # tab[i, j, k, l] = x**i * y**j coefficient of
    # (1 - x)**k * (1 + x) ** (r_size - 1 - k) *
    # (1 - y)**l * (1 + y) ** (r_size - 1 - l)
    pretab = np.zeros((r_size, r_size), dtype=np.float64)
    a = Polynomial([1, -1])
    b = Polynomial([1, 1])
    for i_co in range(r_size):
        f = a**i_co * b ** (r_size - 1 - i_co)
        assert isinstance(f, Polynomial)
        pretab[:, i_co] = f.coef
    tab = np.einsum("ij,kl->ikjl", pretab, pretab)
    assert isinstance(tab, np.ndarray)
    return tab


def macwilliams(a: npt.NDArray, r_size: int) -> npt.NDArray:
    """Return the MacWilliams transform of a polynomial a."""
    tab = _macwilliams_impl(r_size)
    res = tab @ a
    # Is it correct? Not by a.sum()?
    return res / res[0]


def macwilliams_double(a: npt.NDArray, r_size: int) -> npt.NDArray:
    """Return the double MacWilliams transform of a polynomial a."""
    tab = _macwillamsdouble_impl(r_size)
    # WARNING: ji, not ij
    res = np.einsum("ijkl,kl->ji", tab, a)
    res /= a.sum()
    assert isinstance(res, np.ndarray)
    return res
