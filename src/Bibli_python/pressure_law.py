import numpy as np
from math import sqrt, log

__all__ = [
    "Sm_2015_Rashenko",
    "Sm_1997_Datchi",
    "T_Ruby_Sm_1997_Datchi",
    "Ruby_2020_Shen",
    "Ruby_1986_Mao",
    "SrFCl",
    "Rhodamine_6G_2024_Dembele",
    "P_Birch",
    "Birch_M",
    "Birch_M_save",
]


def Sm_2015_Rashenko(peakMax, lamb0=685.41, sigmalambda=None):
    """Pressure law from Rashenko 2015 for Samarium."""
    deltalambda = peakMax - lamb0
    P = 4.20 * deltalambda * (1 + 0.020 * deltalambda) / (1 + 0.036 * deltalambda)
    if sigmalambda is None:
        return P
    coe1, coe2, coe3 = 4.20, 0.02, 0.036
    sigmaP = sigmalambda * (((coe1 * (1 + (coe2 * deltalambda)) + (coe1 * coe2 * deltalambda)) * (1 + (coe3 * deltalambda)) - coe3 * deltalambda * coe1 * (1 + (coe2 * deltalambda))) / ((1 + (coe3 * deltalambda)) ** 2))
    return P, sigmaP


def Sm_1997_Datchi(peakMax, lamb0=685.41, sigmalambda=None):
    deltalambda = peakMax - lamb0
    P = 4.032 * deltalambda * (1 + deltalambda * 9.29e-3) / (1 + deltalambda * 2.32e-2)
    return P


def T_Ruby_Sm_1997_Datchi(peakMaxR, peakMaxS, lamb0S=685.41, lamb0R=694.3, sigmalambda=None):
    deltaR = peakMaxR - lamb0R
    deltaS = peakMaxS - lamb0S
    T = 300 + 137 * (deltaR - 1.443 * deltaS)
    if sigmalambda is None:
        return T
    sigmaT = 137 * np.sqrt(sigmalambda[0] ** 2 + (1.443 * sigmalambda[1]) ** 2)
    return T, sigmaT


def Ruby_2020_Shen(peakMax, lamb0=694.3, sigmalambda=None):
    deltalambda = peakMax - lamb0
    P = 1870 * deltalambda / lamb0 * (1 + 5.63 * (deltalambda / lamb0))
    if sigmalambda is None:
        return P
    sigmaP = sigmalambda * (1870 / lamb0 * (1 + 2 * 5.63 * (deltalambda / lamb0)))
    return P, sigmaP


def Ruby_1986_Mao(peakMax, lamb0=694.24, sigmalambda=None, hydro=True):
    deltalambda = peakMax - lamb0
    B = 5 if hydro else 7.665
    P = 1904 / B * ((1 + deltalambda / lamb0) ** B - 1)
    if sigmalambda is None:
        return P
    sigmaP = sigmalambda * 1904 / lamb0 * (1 + deltalambda / lamb0) ** (B - 1)
    return P, sigmaP


def SrFCl(peakMax, lamb0=690.1, sigmalambda=None):
    x = (peakMax - lamb0) / lamb0
    coe1, coe2 = 620.6, -4.92
    P = coe1 * x * (1 + coe2 * x)
    if sigmalambda is None:
        return P
    sigmaP = sigmalambda * coe1 * (1 + coe2 * x) / lamb0
    return P, sigmaP


def Rhodamine_6G_2024_Dembele(peakMax, lamb0=551.3916, sigmalambda=None):
    # Calibration coefficients to be determined properly
    a = 0
    b = 0.04166
    deltalambda = peakMax - lamb0
    P = a * (deltalambda) ** 2 + b * deltalambda
    if sigmalambda is None:
        return P
    sigmaP = a * sigmalambda * deltalambda * 2 + b * sigmalambda
    return P, sigmaP


def P_Birch(V, V0, K0, K0P, alphaKt=0, T=300):
    return (3 / 2) * K0 * ((V0 / V) ** (7 / 3) - (V0 / V) ** (5 / 3)) * (1 + (3 / 4) * (K0P - 4) * ((V0 / V) ** (2 / 3) - 1)) + alphaKt * (T - 300)


def Birch_M(V, V0, K0, K0P):
    return (3 / 2) * K0 * ((V0 / V) ** (7 / 3) - (V0 / V) ** (5 / 3)) * (1 + (3 / 4) * (K0P - 4) * ((V0 / V) ** (2 / 3) - 1))


def Birch_M_save(V, V0, K0, K0P):
    if np.isscalar(V):
        if V == 0:
            return 0
        return Birch_M(V, V0, K0, K0P)
    if isinstance(V, np.ndarray):
        if (V == 0).any():
            raise ValueError("V ne peut pas contenir de zéros pour éviter la division par zéro")
        return Birch_M(V, V0, K0, K0P)
    raise ValueError("V doit être un scalaire ou un tableau NumPy")
