from __future__ import annotations

import copy
import os
import random
import re
import time
import warnings
from math import log, sqrt
from multiprocessing import cpu_count
from typing import Any, Iterable, Optional

import dill
import fabio
import numpy as np
import pandas as pd
import pyFAI
from PIL import Image
from deap import algorithms, base, creator, tools
from lmfit.models import (
    GaussianModel,
    MoffatModel,
    Pearson4Model,
    PseudoVoigtModel,
    SplitLorentzianModel,
)
from pathos.multiprocessing import ProcessingPool as Pool
from pynverse import inversefunc
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, peak_widths
from scipy.special import beta, gamma
from scipy.spatial.distance import cdist


# ============================================================
# A) Constantes + utilitaires généraux
# ============================================================

# --- DEAP: création sûre (évite l'erreur si relancé/importé plusieurs fois)
def _ensure_deap_creator():
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

_ensure_deap_creator()

# --- I/O objets (CEDd)
def SAVE_CEDd(obj, bit_try: bool = False) -> None:
    """Sauvegarde obj via dill dans obj.CEDd_path. Option: retire data_drx avant dump."""
    if not obj:
        return
    if bit_try:
        try:
            if hasattr(obj, "data_drx"):
                obj.data_drx = None
            dill.dump(obj, open(obj.CEDd_path, "wb"))
        except Exception as e:
            warnings.warn(f"SAVE_CEDd impossible: {e}")
        return
    dill.dump(obj, open(obj.CEDd_path, "wb"))

def LOAD_CEDd(CEDd_path: str, bit_try: bool = False) -> Optional[Any]:
    """Charge un objet dill et remet CEDd_path."""
    if not CEDd_path:
        return None
    if bit_try:
        try:
            obj = dill.load(open(CEDd_path, "rb"))
            obj.CEDd_path = CEDd_path
            return obj
        except Exception as e:
            warnings.warn(f"LOAD_CEDd impossible: {e}")
            return None
    obj = dill.load(open(CEDd_path, "rb"))
    obj.CEDd_path = CEDd_path
    return obj

# --- Fichiers
def Load_last(folder: str, extend: Optional[str] = None, file: bool = True):
    """Retourne (latest_path, latest_name) selon date de modif."""
    if file:
        if extend is not None:
            names = [f for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder, f)) and extend in f]
        else:
            names = [f for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder, f))]
    else:
        names = [f for f in os.listdir(folder)]

    if not names:
        return None, None

    names.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)))
    latest_name = names[-1]
    latest_path = os.path.join(folder, latest_name)
    return latest_path, latest_name

def extraire_numero(fichier: str) -> float:
    """Ex: ..._123.npy -> 123 (sinon inf)."""
    m = re.search(r'_(\d+)\.npy$', fichier)
    return int(m.group(1)) if m else float("inf")

# --- Somme de fonctions (utile pour modèles composés)
def Gen_sum_F(list_F):
    """
    Construit une fonction sum_F(x, *params) qui somme chaque f(x, params_f).
    NB: suppose que chaque f a signature f(x, ...).
    """
    def sum_F(x, *params):
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=float)
        idx = 0
        for f in list_F:
            n = f.__code__.co_argcount - 1
            out += f(x, *params[idx:idx+n])
            idx += n
        return out
    return sum_F

# --- Helpers fit
# ============================================================
# B) Calibration DRX + intégration 2D -> 1D (pyFAI) + Dialog Qt
# ============================================================

ENERGY_CONSTANT_KEV_M = 1.239841984e-6  # si déjà défini dans A, tu peux supprimer ici


def Integrate_DRX(
    img2d: np.ndarray,
    mask: Optional[np.ndarray],
    ai: pyFAI.AzimuthalIntegrator,
    theta_range: Optional[Iterable[float]] = None,
    pby2theta: float = 50,
    nb_point_default: int = 9000,
):
    """
    Intégration radiale pyFAI en 2θ (deg).
    - img2d : array 2D
    - mask  : 0=masqué, 1=gardé (pyFAI attend typiquement 1=masque, mais ici tu passes un binaire;
              si besoin, inverse selon ton convention)
    """
    if img2d is None or ai is None:
        raise ValueError("Integrate_DRX: img2d et ai sont requis.")

    try:
        if theta_range is not None:
            t0, t1 = float(theta_range[0]), float(theta_range[1])
            nb_point = max(10, int(abs(t1 - t0) * float(pby2theta)))
        else:
            nb_point = int(nb_point_default)

        tth, intens = ai.integrate1d(img2d, nb_point, mask=mask, unit="2th_deg")

        if theta_range is not None:
            lo, hi = min(t0, t1), max(t0, t1)
            m = (tth >= lo) & (tth <= hi)
            tth, intens = tth[m], intens[m]

        return tth, intens

    except Exception as e:
        raise RuntimeError(f"Erreur intégration: {e}") from e


class Calib_DRX:
    """
    Charge un .mask + un .poni, calcule (optionnellement) l'énergie depuis ai.wavelength,
    et ouvre un dialog pour changer calibration/borne theta.
    """
    def __init__(
        self,
        file_mask: Optional[str] = None,
        file_poni: Optional[str] = None,
        theta_range: Iterable[float] = (0.0, 40.0),
        energy: Optional[float] = None,
    ):
        self.file_mask = file_mask
        self.file_poni = file_poni
        self.theta_range = list(theta_range)
        self.energy = energy

        self.mask = None
        self.ai = None

        self.Load_calib()

    def Load_calib(self, file_mask: Optional[str] = None, file_poni: Optional[str] = None) -> None:
        if file_mask is not None:
            self.file_mask = file_mask
        if file_poni is not None:
            self.file_poni = file_poni

        if self.file_mask:
            if not os.path.isfile(self.file_mask):
                raise FileNotFoundError(f"Masque introuvable: {self.file_mask}")
            mask = np.array(Image.open(self.file_mask))
            self.mask = (mask > 0).astype(np.uint8)

        if self.file_poni:
            if not os.path.isfile(self.file_poni):
                raise FileNotFoundError(f"PONI introuvable: {self.file_poni}")
            self.ai = pyFAI.load(self.file_poni)

        if self.ai is not None:
            wl = getattr(self.ai, "wavelength", None)
            if wl:
                self.energy = ENERGY_CONSTANT_KEV_M / wl

    def Change_calib(self, file_img: str, energy: Optional[float] = None) -> Optional[float]:
        if energy is not None:
            self.energy = energy

        from .calibration_ui import CalibDialog, QDialog

        if QDialog is None:
            raise ImportError("Qt/PyQtGraph indisponible: CalibDialog ne peut pas être instancié.")

        dialog = CalibDialog(
            file_img=file_img,
            mask=self.mask,
            ai=self.ai,
            theta_range=self.theta_range,
            folder_start=os.path.dirname(self.file_poni) if self.file_poni else None,
            energy=self.energy,
        )
        if dialog.exec_() == QDialog.Accepted:
            self.file_mask = dialog.file_mask
            self.file_poni = dialog.file_poni
            self.mask = dialog.mask
            self.ai = dialog.ai
            self.theta_range = dialog.theta_range
            self.energy = dialog.energy
            return self.energy
        return None

class CalibDialog:  # pragma: no cover - proxy for compatibility
    def __new__(cls, *args, **kwargs):
        from .calibration_ui import CalibDialog as _CalibDialog

        return _CalibDialog(*args, **kwargs)

# ============================================================
# C) Fonctions de forme de pic + classe Pics (lmfit)
# ============================================================

# -----------------------------
# Formes analytiques (optionnel)
# (utilisées surtout pour l'estimation bruit/MC via curve_fit)
# -----------------------------
def PseudoVoigt(x, center, ampH, sigma, fraction):
    """Pseudo-Voigt normalisé via hauteur ampH (approx de ton code)."""
    amp = ampH / (
        ((1 - fraction)) / (sigma * sqrt(np.pi / log(2))) + (fraction) / (np.pi * sigma)
    )
    sigma_g = sigma / sqrt(2 * log(2))
    return (
        (1 - fraction)
        * amp
        / (sigma_g * sqrt(2 * np.pi))
        * np.exp(-((x - center) ** 2) / (2 * sigma_g**2))
        + fraction * amp / np.pi * (sigma / ((x - center) ** 2 + sigma**2))
    )

def Moffat(x, center, ampH, sigma, beta_):
    amp = ampH * beta_
    return amp * (((x - center) / sigma) ** 2 + 1) ** (-beta_)

def SplitLorentzian(x, center, ampH, sigma, sigma_r):
    amp = ampH * 2 / (np.pi * (sigma + sigma_r))
    sig = np.where(x < center, sigma, sigma_r)
    return amp * (sig**2) / ((x - center) ** 2 + sig**2)

def PearsonIV(x, center, ampH, sigma, m, skew):
    """
    Pearson type IV (version identique à ton code).
    Attention : ce n'est PAS la formule interne de lmfit.Pearson4Model.
    """
    center = center + sigma * skew / (2 * m)
    norm = (1 + (skew / (2 * m)) ** 2) ** (-m) * np.exp(-skew * np.arctan(-skew / (2 * m)))
    return (ampH / norm) * (1 + ((x - center) / sigma) ** 2) ** (-m) * np.exp(
        -skew * np.arctan((x - center) / sigma)
    )

def Gaussian(x, center, ampH, sigma):
    return ampH * np.exp(-((x - center) ** 2) / (2 * sigma**2))


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


# -----------------------------
# Classe Pics
# -----------------------------
class Pics:
    """
    Décrit un pic (paramètres + modèle lmfit).
    - Paramètres stockés sous forme: [valeur, [min, max]]
    - f_model : fonction analytique correspondante (si besoin pour MC)
    - model   : modèle lmfit utilisé pour le fit
    """

    def __init__(
        self,
        name: str = "",
        ctr: float = 0.0,
        ampH: float = 1.0,
        coef_spe=(0.5,),
        sigma: float = 0.15,
        inter: float = 3.0,
        model_fit: str = "PseudoVoigt",
        Delta_ctr: float = 0.5,
        amp: Optional[float] = None,
    ):
        self.name = str(name).replace("-", "_")
        self.model_fit = str(model_fit)
        self.inter = float(inter)

        inter_min = max(1 - self.inter, 0)

        self.ctr = [float(ctr), [float(ctr - Delta_ctr), float(ctr + Delta_ctr)]]
        self.sigma = [float(sigma), [float(sigma * inter_min), float(sigma * (1 + self.inter))]]
        self.ampH = [float(ampH), [float(ampH * inter_min), float(ampH * (1 + self.inter))]]

        coef_spe = np.atleast_1d(coef_spe).astype(float).tolist()
        self.coef_spe = [[c, [c * inter_min, c * (1 + self.inter)]] for c in coef_spe]

        self.best_fit = None
        self.help = "Pics: définition de modèle de pic"
        self.name_coef_spe = []
        self.model = None
        self.f_model = None
        self.f_amp = None

        # init selon modèle
        self._init_model(Delta_ctr=Delta_ctr)

        # amplitude intégrée (lmfit utilise 'amplitude', toi tu règles plutôt via height)
        if amp is None:
            amp = self.f_amp(self.ampH[0], [c[0] for c in self.coef_spe], self.sigma[0])
        self.amp = [float(amp), [float(amp * inter_min), float(amp * (1 + self.inter))]]

        self._set_param_hints()

    # ---- conversions height -> amplitude (intégrée) (ton choix)
    @staticmethod
    def _amp_pseudovoigt(ampH, coef, sigma):
        frac = float(coef[0]) if coef else 0.5
        return ampH / (((1 - frac)) / (sigma * sqrt(np.pi / log(2))) + (frac) / (np.pi * sigma))

    @staticmethod
    def _amp_gaussian(ampH, coef, sigma):
        return ampH * sigma * sqrt(2 * np.pi)

    @staticmethod
    def _amp_moffat(ampH, coef, sigma):
        beta_ = float(coef[0]) if coef else 1.0
        return ampH * beta_

    @staticmethod
    def _amp_splitlorentz(ampH, coef, sigma):
        sig_r = float(coef[0]) if coef else sigma
        return ampH / (2 * np.pi * (sigma + sig_r))

    @staticmethod
    def _amp_pearsoniv(ampH, coef, sigma):
        # coef=[m, skew]
        m = float(coef[0]) if len(coef) >= 1 else 1.1
        skew = float(coef[1]) if len(coef) >= 2 else 0.0
        normalization = (abs(gamma(m + 1j * skew / 2) / gamma(m)) ** 2) / (sigma * beta(m - 0.5, 0.5))
        return ampH / (
            normalization * (1 + (skew / (2 * m)) ** 2) ** (-m) * np.exp(-skew * np.arctan(-skew / (2 * m)))
        )

    def _init_model(self, Delta_ctr=0.5):
        """Initialise lmfit model + f_model + contraintes spécifiques."""
        inter_min = max(1 - self.inter, 0)

        if self.model_fit == "PseudoVoigt":
            self.f_amp = self._amp_pseudovoigt
            self.name_coef_spe = ["fraction"]
            self.model = PseudoVoigtModel(prefix=self.name)
            self.f_model = PseudoVoigt

            # clamp fraction dans [0,1]
            frac0 = float(self.coef_spe[0][0]) if self.coef_spe else 0.5
            frac0 = max(0.0, min(1.0, frac0))
            self.coef_spe = [[frac0, [max(0.0, frac0 * inter_min), min(1.0, frac0 * (1 + self.inter))]]]

        elif self.model_fit == "Moffat":
            self.f_amp = self._amp_moffat
            self.name_coef_spe = ["beta"]
            self.model = MoffatModel(prefix=self.name)
            self.f_model = Moffat

        elif self.model_fit == "SplitLorentzian":
            self.f_amp = self._amp_splitlorentz
            self.name_coef_spe = ["sigma_r"]
            self.model = SplitLorentzianModel(prefix=self.name)
            self.f_model = SplitLorentzian

        elif self.model_fit == "Gaussian":
            self.f_amp = self._amp_gaussian
            self.name_coef_spe = []
            self.model = GaussianModel(prefix=self.name)
            self.f_model = Gaussian
            self.coef_spe = []

        elif self.model_fit == "PearsonIV":
            self.f_amp = self._amp_pearsoniv
            self.name_coef_spe = ["expon", "skew"]
            self.model = Pearson4Model(prefix=self.name)
            self.f_model = PearsonIV

            # forcer 2 coefs : m (expon) et skew
            if len(self.coef_spe) < 2:
                m0 = float(self.coef_spe[0][0]) if self.coef_spe else 1.1
                self.coef_spe = [
                    [max(0.505, m0), [max(0.501, m0 * inter_min), max(0.51, m0 * (1 + self.inter))]],
                    [0.0, [-1.0, 1.0]],
                ]
            else:
                m0 = max(0.505, float(self.coef_spe[0][0]))
                skew0 = float(self.coef_spe[1][0])
                self.coef_spe[0][0] = m0
                self.coef_spe[0][1] = [max(0.501, m0 * inter_min), max(0.51, m0 * (1 + self.inter))]
                # bornes skew : larges (tu peux resserrer si tu veux)
                self.coef_spe[1][0] = skew0
                self.coef_spe[1][1] = [skew0 * (1 - self.inter) - 1.0, skew0 * (1 + self.inter) + 1.0]

        else:
            raise ValueError(f"model_fit inconnu: {self.model_fit}")

        # refresh ctr bounds (au cas où)
        c = float(self.ctr[0])
        self.ctr[1] = [float(c - Delta_ctr), float(c + Delta_ctr)]

    def _set_param_hints(self):
        """Applique les hints lmfit (amplitude, sigma, center, + coef_spe)."""
        # coef spé
        for i, name_spe in enumerate(self.name_coef_spe):
            self.model.set_param_hint(
                self.name + name_spe,
                value=float(self.coef_spe[i][0]),
                min=float(self.coef_spe[i][1][0]),
                max=float(self.coef_spe[i][1][1]),
            )

        # amplitude (intégrée)
        self.model.set_param_hint(
            self.name + "amplitude",
            value=float(self.amp[0]),
            min=float(self.amp[1][0]),
            max=float(self.amp[1][1]),
        )

        # sigma
        self.model.set_param_hint(
            self.name + "sigma",
            value=float(self.sigma[0]),
            min=float(self.sigma[1][0]),
            max=float(self.sigma[1][1]),
        )

        # center (lmfit)
        self.model.set_param_hint(
            self.name + "center",
            value=float(self.ctr[0]),
            min=float(self.ctr[1][0]),
            max=float(self.ctr[1][1]),
        )

    def Update(
        self,
        ctr=None,
        ampH=None,
        coef_spe=None,
        sigma=None,
        inter=None,
        model_fit=None,
        Delta_ctr=0.4,
        amp=None,
    ):
        """Met à jour paramètres + remet les hints."""
        if inter is not None:
            self.inter = float(inter)

        inter_min = max(1 - self.inter, 0)

        if ctr is not None:
            ctr = float(ctr)
            self.ctr = [ctr, [ctr - float(Delta_ctr), ctr + float(Delta_ctr)]]

        if sigma is not None:
            sigma = float(sigma)
            self.sigma = [sigma, [sigma * inter_min, sigma * (1 + self.inter)]]
        else:
            s = float(self.sigma[0])
            self.sigma = [s, [s * inter_min, s * (1 + self.inter)]]

        if coef_spe is not None:
            vals = np.atleast_1d(coef_spe).astype(float).tolist()
            self.coef_spe = [[v, [v * inter_min, v * (1 + self.inter)]] for v in vals]
        else:
            self.coef_spe = [[c[0], [c[0] * inter_min, c[0] * (1 + self.inter)]] for c in self.coef_spe]

        if model_fit is not None and str(model_fit) != self.model_fit:
            self.model_fit = str(model_fit)
            self._init_model(Delta_ctr=float(Delta_ctr))

        # amplitude intégrée
        if amp is not None:
            amp_val = float(amp)
        else:
            if ampH is not None:
                self.ampH = [float(ampH), [float(ampH * inter_min), float(ampH * (1 + self.inter))]]
            amp_val = float(self.f_amp(self.ampH[0], [c[0] for c in self.coef_spe], self.sigma[0]))

        self.amp = [amp_val, [amp_val * inter_min, amp_val * (1 + self.inter)]]
        self._set_param_hints()

    def Out_ctr(self) -> float:
        """Retourne la position centrale actuelle (2θ)."""
        return float(self.ctr[0])

    def Out_model(self, out=None):
        """
        Lit les params du fit lmfit et met à jour ctr/ampH/sigma/coef_spe.
        Note: lmfit retourne 'amplitude' et pas forcément 'height' pour tous les modèles.
        """
        if out is None:
            return None

        # sigma
        sig = out.params[self.name + "sigma"].value
        sig_err = out.params[self.name + "sigma"].stderr or 0.0
        self.sigma = [float(sig), [float(sig - sig_err), float(sig + sig_err)]]

        # center/position (robuste)
        if self.name + "center" in out.params:
            ctr_param = out.params[self.name + "center"]
        elif self.name + "position" in out.params:
            ctr_param = out.params[self.name + "position"]
        else:
            raise KeyError(f"Paramètre center/position introuvable pour {self.name}")

        ctr = ctr_param.value
        ctr_err = ctr_param.stderr or 0.0

        self.ctr = [float(ctr), [float(ctr - ctr_err), float(ctr + ctr_err)]]

        # amplitude (intégrée)
        amp = out.params[self.name + "amplitude"].value
        amp_err = out.params[self.name + "amplitude"].stderr or 0.0
        self.amp = [float(amp), [float(amp - amp_err), float(amp + amp_err)]]

        # coef_spe
        coef_vals = []
        for name_spe in self.name_coef_spe:
            v = out.params[self.name + name_spe].value
            e = out.params[self.name + name_spe].stderr or 0.0
            coef_vals.append([float(v), [float(v - e), float(v + e)]])
        self.coef_spe = coef_vals

        # recalcul ampH si possible (inverse approx via f_amp)
        # ici on garde ampH tel quel (ton code le sort depuis out.params['height'] parfois)
        self._set_param_hints()
        return [self.ctr[0], self.amp[0], self.sigma[0], np.array([c[0] for c in self.coef_spe])]
    


    # --------------------------------------------------------------
    # Estimation σ(center) dominée par le bruit (même logique que ton code)
    # --------------------------------------------------------------
    def estimate_sigma_center_from_noise(self, x, y, zone_baseline=None, N_MC=0, mode="auto", verbose=False):
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) < 8:
            return None

        center = float(self.ctr[0])
        sigma = float(self.sigma[0])
        ampH = float(self.ampH[0])
        coef = [c[0] for c in self.coef_spe] if self.coef_spe else []

        # bruit hors pic
        if zone_baseline is None:
            mask = (x < center - 3 * sigma) | (x > center + 3 * sigma)
        else:
            xmin, xmax = zone_baseline
            mask = (x > xmin) & (x < xmax)
        sigma_n = float(np.std(y[mask])) if np.any(mask) else float(np.std(y))

        # choix mode
        if mode == "auto":
            mode = "poisson" if np.mean(y) < 1000 else "gauss"
        poisson = mode.lower().startswith("p")

        # estimation analytique
        delta_x = float(np.mean(np.diff(x)))
        FWHM = 2.3548 * sigma
        M = max(1.0, FWHM / max(delta_x, 1e-12))

        if poisson:
            # approx "nombre de photons" via modèle
            N_phot = None
            if callable(self.f_model):
                try:
                    p0 = [center, ampH, sigma] + list(coef)
                    y_model = self.f_model(x, *p0)
                    N_phot = float(np.sum(np.clip(y_model, 0, None)))
                except Exception:
                    N_phot = None
            if N_phot is None:
                N_phot = float(max(abs(ampH) * FWHM / max(delta_x, 1e-12), 1e-12))
            SNR = sqrt(N_phot)
        else:
            SNR = abs(ampH) / max(sigma_n, 1e-12)

        alpha = 0.60
        sigma_ctr = alpha * FWHM / (SNR * sqrt(M))

        # Monte Carlo optionnel
        if N_MC > 0 and callable(self.f_model):
            try:
                p0 = [center, ampH, sigma] + list(coef)
                y_model = self.f_model(x, *p0)
                ctr_MC = []
                for _ in range(int(N_MC)):
                    if poisson:
                        y_noisy = np.random.poisson(np.clip(y_model, 0, None))
                    else:
                        y_noisy = y_model + np.random.normal(0, sigma_n, size=len(y_model))
                    try:
                        popt, _ = curve_fit(self.f_model, x, y_noisy, p0=p0, maxfev=800)
                        ctr_MC.append(float(popt[0]))
                    except Exception:
                        continue
                if len(ctr_MC) > 5:
                    sigma_ctr = float(np.std(ctr_MC))
                    if verbose:
                        print(f"MC({self.name}) σ_center={sigma_ctr:.4g} ({len(ctr_MC)} essais, mode={mode})")
            except Exception as e:
                if verbose:
                    print(f"{self.name}: erreur MC ({e})")

        self.sigma_ctr_stat = float(sigma_ctr)
        return {
            "sigma_ctr": float(sigma_ctr),
            "sigma_n": float(sigma_n),
            "SNR": float(SNR),
            "FWHM": float(FWHM),
            "method": f"{'MC-' if N_MC > 0 else ''}{mode}",
        }


# ============================================================
# DRX only: Element (fusion Gauge + Element)
# ============================================================


class Element:
    """
    Classe unique DRX :
    - remplace l'ancien couple Gauge + Element
    - ne contient PLUS aucune logique Ruby/Sm/etc.
    - gère: pics, modèle lmfit, calcul a/b/c, V, P, incertitudes
    """

    def __init__(self, Element_ref: Element_Bibli, name: str):
        # --- "ancien Gauge" minimal ---
        self.name = str(name)
        self.name_spe = "DRX"

        self.pics = []
        self.model = None
        self.bit_model = False

        self.bit_fit = False
        self.fit = "Fit Non effectué"
        self.indexX = None

        self.study = pd.DataFrame()
        self.study_add = pd.DataFrame()

        # --- spécifique Element ---
        self.Element_ref = Element_ref
        self.E = Element_ref.E
        self.maille = Element_ref.symmetrie

        # paramètres maille / eos
        self.a = Element_ref.A
        self.b = Element_ref.B
        self.c = Element_ref.C

        self.sigma_a = 0.0
        self.sigma_b = 0.0
        self.sigma_c = 0.0

        self.alpha = getattr(Element_ref, "ALPHA", None)
        self.beta = getattr(Element_ref, "BETA", None)
        self.gamma = getattr(Element_ref, "GAMMA", None)

        self.rca = getattr(Element_ref, "rCA", None)
        self.rba = getattr(Element_ref, "rBA", None)

        self.V = getattr(Element_ref, "V", None)
        self.sigma_V = 0.0

        self.P_start = getattr(Element_ref, "P_start", 0.0)
        self.P = self.P_start
        self.sigma_P = 0.0

        self.T = getattr(Element_ref, "T", 300.0)
        self.sigma_T = 0.0

        # structures internes (ancien Element)
        self.nb_pic = Element_ref.save_var.count(True) if getattr(Element_ref, "save_var", None) is not None else 0
        self.pic_ref = []
        self.deltaP0i = []
        self.l_dhkl = []

        self.init_ref()

    # ------------------------------------------------------------
    # "ancien Gauge" : utilitaires minimaux
    # ------------------------------------------------------------
    def Clear(self, c=None) -> None:
        self.study.loc[:, :] = c
        self.study_add = pd.DataFrame()
        self.bit_model = False
        self.model = None
        self.bit_fit = False
        self.fit = "Fit Non effectué"

    def Update_model(self) -> None:
        self.model = None
        for p in self.pics:
            self.model = p.model if self.model is None else (self.model + p.model)
        self.bit_model = True

    # ------------------------------------------------------------
    # ancien Element : init pics théoriques
    # ------------------------------------------------------------
    def init_ref(self, verbose: bool = False) -> None:
        self.pic_ref = []
        self.pics = []
        self.deltaP0i = []
        self.model = None

        if self.Element_ref.thetas_PV == []:
            self.Element_ref.Eos_Pdhkl(P=0)

        text_print = ""
        i = 0
        n_p = 0
        self.nb_pic = self.Element_ref.save_var.count(True)

        # premier pic = référence
        while n_p < self.nb_pic:
            if self.Element_ref.save_var[i] is True:
                name_dhkl = "D" + "".join(map(str, self.Element_ref.thetas_PV[i][2:]))
                theta2 = self.Element_ref.thetas_PV[i][0]  # 2θ en degrés

                self.pic_ref.append([name_dhkl, theta2])
                text_print += " ; " + name_dhkl

                new_pic = Pics(
                    name=f"{self.name}_p{name_dhkl}",
                    ctr=theta2,
                    model_fit="PearsonIV",
                    coef_spe=[1.1, 0],
                )
                self.pics.append(new_pic)

                if n_p == 0:
                    self.model = new_pic.model
                    self.lamb_fit = theta2
                    self.deltaP0i = [[0, 1]]
                else:
                    self.model = self.model + new_pic.model
                    inten = int(self.Element_ref.Dhkl["I"][i]) / 100.0
                    self.deltaP0i.append([theta2 - self.lamb_fit, inten])

                n_p += 1
            i += 1

        self.bit_model = True
        if verbose:
            print("Dhkl ref refresh :", text_print)

    # ------------------------------------------------------------
    # DRX : minimisation sur d(hkl)
    # ------------------------------------------------------------
    def minimisation(self, verbose: bool = False) -> None:
        if self.a is None or self.b is None or self.c is None:
            raise ValueError("Element.minimisation: a/b/c init manquants")
        if self.E is None:
            raise ValueError("Element.minimisation: énergie E manquante")

        # --- préparation d_exp et poids ---
        amps = [p.ampH[0] for p in self.pics]
        amp_max = max(amps) if amps else 1.0

        Dhkl_exp, weights, sigma_d_list = [], [], []

        lam = (1239.8 / self.E) * 1e-9  # m

        for p in self.pics:
            two_theta = np.deg2rad(p.Out_ctr())  # 2θ en rad
            theta = two_theta / 2.0

            d_exp = (lam / (2 * np.sin(theta))) * 1e10  # Å
            Dhkl_exp.append(d_exp)

            # incertitude sur 2θ (si sigma_ctr_total existe sinon borne)
            if hasattr(p, "sigma_ctr_total"):
                sigma_2theta = np.deg2rad(float(p.sigma_ctr_total))
            else:
                low, high = p.ctr[1]
                sigma_2theta = np.deg2rad((high - low) / 2.0) if high > low else 1e-4

            sigma_theta = sigma_2theta / 2.0
            dd_dtheta = -lam * np.cos(theta) / (2 * (np.sin(theta) ** 2)) * 1e10
            sigma_d = max(abs(dd_dtheta) * sigma_theta, 1e-4)

            amp_rel = p.ampH[0] / amp_max if amp_max > 0 else 1.0
            weights.append(amp_rel / (sigma_d ** 2))
            sigma_d_list.append(sigma_d)

        Dhkl_exp = np.asarray(Dhkl_exp)
        weights = np.asarray(weights)
        sigma_d_list = np.asarray(sigma_d_list)

        # --- d_calc modèles ---
        def d_tetragonal(h, k, l, a, c):
            return a * np.sqrt(1.0 / ((h**2 + k**2) + (a * l / c) ** 2))

        def d_hexagonal(h, k, l, a, c):
            return np.sqrt(1.0 / ((4 / 3 * (h**2 + k**2 + h * k) / a**2) + (l**2 / c**2)))

        def d_cubic(h, k, l, a):
            return a / np.sqrt(h**2 + k**2 + l**2)

        def d_orthorhombic(h, k, l, a, b, c):
            return 1.0 / np.sqrt((h / a) ** 2 + (k / b) ** 2 + (l / c) ** 2)

        def d_rhombohedral(h, k, l, a, alpha_deg):
            alpha_rad = np.deg2rad(alpha_deg)
            num = 1 - 3 * np.cos(alpha_rad) ** 2 + 2 * np.cos(alpha_rad) ** 3
            den = (
                (h**2 + k**2 + l**2) * np.sin(alpha_rad) ** 2
                - 2 * (np.cos(alpha_rad) - np.cos(alpha_rad) ** 2) * (k * l + l * h + h * k)
            )
            return a * np.sqrt(num / den)

        def get_hkl(p):
            # p[0] = "D123" ou autre
            m = re.match(r"D([-+]?\d+)([-+]?\d+)([-+]?\d+)", p[0])
            if not m:
                raise ValueError(f"Format dhkl invalide: {p[0]}")
            return int(m.group(1)), int(m.group(2)), int(m.group(3))

        def get_uncertainties(result):
            if hasattr(result, "hess_inv"):
                try:
                    cov = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else result.hess_inv
                    return np.sqrt(np.diag(cov))
                except Exception:
                    return None
            return None

        lattice_map = {
            "TETRAGONAL": (lambda h, k, l, a, c: d_tetragonal(h, k, l, a, c), [self.a, self.c]),
            "HEXAGONAL": (lambda h, k, l, a, c: d_hexagonal(h, k, l, a, c), [self.a, self.c]),
            "CUBIC": (lambda h, k, l, a: d_cubic(h, k, l, a), [self.a]),
            "ORTHORHOMBIC": (lambda h, k, l, a, b, c: d_orthorhombic(h, k, l, a, b, c), [self.a, self.b, self.c]),
            "RHOMBOHEDRAL": (lambda h, k, l, a: d_rhombohedral(h, k, l, a, self.alpha), [self.a]),
        }

        chosen = None
        for key in lattice_map:
            if key in self.maille:
                chosen = key
                break
        if chosen is None:
            raise ValueError(f"Symétrie non gérée: {self.maille}")

        d_func, param0 = lattice_map[chosen]

        def error(params):
            d_calc = np.array([d_func(*get_hkl(p), *params) for p in self.pic_ref])
            diff = Dhkl_exp - d_calc
            return np.sum(weights * diff**2)

        min_res = minimize(error, param0, method="BFGS")
        errs = get_uncertainties(min_res)

        # maj paramètres selon maille
        if chosen == "CUBIC":
            self.a = self.b = self.c = round(min_res.x[0], 3)
            self.rca = 1.0
            if errs is not None:
                self.sigma_a = self.sigma_b = self.sigma_c = float(errs[0])

        elif chosen in ("TETRAGONAL", "HEXAGONAL"):
            self.a = self.b = round(min_res.x[0], 3)
            self.c = round(min_res.x[1], 3)
            self.rca = self.c / self.a if self.a else None
            if errs is not None and len(errs) >= 2:
                self.sigma_a = self.sigma_b = float(errs[0])
                self.sigma_c = float(errs[1])

        elif chosen == "ORTHORHOMBIC":
            self.a = round(min_res.x[0], 3)
            self.b = round(min_res.x[1], 3)
            self.c = round(min_res.x[2], 3)
            self.rba = self.b / self.a if self.a else None
            self.rca = self.c / self.a if self.a else None
            if errs is not None and len(errs) >= 3:
                self.sigma_a, self.sigma_b, self.sigma_c = map(float, errs[:3])

        elif chosen == "RHOMBOHEDRAL":
            self.a = self.b = self.c = round(min_res.x[0], 3)
            self.rca = 1.0
            if errs is not None:
                self.sigma_a = self.sigma_b = self.sigma_c = float(errs[0])

        if verbose:
            print(f"MINIMISATION DONE ({chosen}): a={self.a}, b={self.b}, c={self.c}")

    # ------------------------------------------------------------
    # DRX : V puis P
    # ------------------------------------------------------------
    def calcul_V(self, verbose: bool = False) -> None:
        if not self.a or not self.b or not self.c:
            raise ValueError("Element.calcul_V: a/b/c invalides")

        alpha_rad = np.deg2rad(self.alpha) if self.alpha is not None else None

        if "CUBIC" in self.maille:
            self.V = round(self.a**3, 3)
        elif "TETRAGONAL" in self.maille:
            self.V = round(self.a**2 * self.c, 3)
        elif "HEXAGONAL" in self.maille:
            self.V = round((np.sqrt(3) / 2) * self.a**2 * self.c, 3)
        elif "ORTHORHOMBIC" in self.maille:
            self.V = round(self.a * self.b * self.c, 3)
        elif "RHOMBOHEDRAL" in self.maille:
            if alpha_rad is None:
                raise ValueError("Element.calcul_V: alpha manquant pour rhombo")
            self.V = round(self.a**3 * np.sqrt(1 - 3*np.cos(alpha_rad)**2 + 2*np.cos(alpha_rad)**3), 3)
        else:
            raise ValueError(f"Symétrie non gérée pour V: {self.maille}")

        if verbose:
            print(f"calcul_V: V={self.V}")

    def calcul_P(self, V0c: Optional[float] = None, T: float = 298.0, verbose: bool = False) -> None:
        if V0c is None:
            V0c = self.Element_ref.V0
        if self.V is None:
            raise ValueError("Element.calcul_P: V non calculé")

        self.T = float(T)
        Pt = 0.0
        if getattr(self.Element_ref, "ALPHAKT", None) is not None:
            Pt = float(self.Element_ref.ALPHAKT) * (self.T - 298.0)

        eta = (V0c / self.V) ** (1 / 3)

        P_bm = (3/2) * self.Element_ref.K0 * (eta**7 - eta**5) * (1 + (3/4) * (self.Element_ref.K0P - 4) * (eta**2 - 1))
        self.P = round(float(P_bm + Pt), 3)

        if verbose:
            print(f"calcul_P: P={self.P} (GPa)")

    def CALCUL(self, mini: bool = True, verbose: bool = False) -> None:
        if mini:
            self.minimisation(verbose=verbose)
        else:
            # si tu veux garder calcul_abc non-minimisation, tu peux le recoller ici
            raise NotImplementedError("CALCUL(mini=False) : recoller calcul_abc si tu en as besoin")

        self.calcul_V(verbose=verbose)
        self.calcul_P(verbose=verbose)

    # ------------------------------------------------------------
    # API simple
    # ------------------------------------------------------------
    def Calcul(self, mini: bool = True, verbose: bool = False) -> None:
        """
        Compat avec ton ancien pipeline qui appelait Gauge.Calcul().
        Ici: c'est 100% DRX.
        """
        self.CALCUL(mini=mini, verbose=verbose)

        # dataframe de sortie (cohérent avec ton ancien code)
        row = np.array([[self.a, self.b, self.c, self.rca, self.V, self.P]], dtype=object)
        self.study = pd.DataFrame(
            row,
            columns=[f"a_{self.name}", f"b_{self.name}", f"c_{self.name}", f"c/a_{self.name}", f"V_{self.name}", f"P_{self.name}"],
        )

class SpectreDRX:
    """Spectre DRX 1D (2θ/intensité) avec baseline optionnelle."""

    def __init__(
        self,
        tth: Iterable[float],
        intens: Iterable[float],
        E: Optional[float] = None,
        baseline: Optional[Iterable[float]] = None,
    ):
        self.wnb = np.asarray(tth, dtype=float)
        self.spec = np.asarray(intens, dtype=float)
        if self.wnb.shape != self.spec.shape:
            raise ValueError("SpectreDRX: tth et intens doivent avoir la même taille.")

        if baseline is None:
            self.blfit = np.zeros_like(self.spec)
        else:
            self.blfit = np.asarray(baseline, dtype=float)
            if self.blfit.shape != self.spec.shape:
                raise ValueError("SpectreDRX: baseline doit avoir la même taille que intens.")

        self.y_corr = self.spec - self.blfit
        self.E = E

        self.Gauges = []
        self.bit_fit = False
        self.study = pd.DataFrame()
        self.lambda_error = round((self.wnb[-1] - self.wnb[0]) * 0.5 / len(self.wnb), 4)

    def Calcul_study(self, mini: bool = True, verbose: bool = False) -> None:
        """Calcule a/b/c/V/P pour toutes les jauges DRX associées."""
        if not self.Gauges:
            raise ValueError("SpectreDRX.Calcul_study: aucune jauge DRX associée.")
        for gauge in self.Gauges:
            gauge.Calcul(mini=mini, verbose=verbose)
        self.study = pd.concat([x.study for x in self.Gauges], axis=1)
        self.bit_fit = True

    def get_local_signal_for_pic(self, pic_target: Pics):
        """
        Reconstruit un signal local pour un pic en retirant les autres composantes.
        """
        x_all = np.asarray(self.wnb)
        y_all = np.asarray(self.y_corr)

        y_fit_total = np.zeros_like(y_all)
        for gauge in self.Gauges:
            for p in gauge.pics:
                params_f = p.model.make_params()
                y_fit_total += p.model.eval(params_f, x=x_all)

        y_fit_target = pic_target.model.eval(pic_target.model.make_params(), x=x_all)

        y_local = y_all - (y_fit_total - y_fit_target)

        center, sigma = float(pic_target.ctr[0]), float(pic_target.sigma[0])
        mask = (x_all > center - 6 * sigma) & (x_all < center + 6 * sigma)
        if not np.any(mask):
            mask = slice(None)

        return x_all[mask], y_local[mask]

    def estimate_all_sigma_noise(self, N_MC: int = 0) -> None:
        """Applique estimate_sigma_center_from_noise() à tous les pics du spectre."""
        for gauge in self.Gauges:
            for p in gauge.pics:
                x_loc, y_loc = self.get_local_signal_for_pic(p)
                if x_loc is None or len(x_loc) < 10:
                    continue
                try:
                    res = p.estimate_sigma_center_from_noise(x_loc, y_loc, N_MC=N_MC)
                    if res:
                        p.sigma_ctr_stat = res["sigma_ctr"]
                except Exception as e:
                    warnings.warn(f"Estimation bruit échouée pour {p.name}: {e}")


Spectre = SpectreDRX


class Element_Bibli:
    """Lecture des fiches éléments (Dhkl, EOS, maille)."""

    def __init__(self, file=None, E: Optional[float] = None):
        self.file = file
        self.name=None
        self.K0 = None
        self.K0P = None
        self.ALPHAKT=None
        self.V0 = None
        self.V = None
        self.A,self.B,self.C =None,None,None
        self.ALPHA,self.BETA,self.GAMMA =None,None,None
        self.symmetrie = None
        self.Dhkl=None
        self.E=E
        self.name_dhkl=None
        self.save_var=None
        self.rCA=None
        self.rBA=None
        self.EoS=None
        self.T=300
        self.T_range=None
        self.P_range=[-10,1000]
        self.Vmin=0.8
        self.Z_max=None
        self.P_start=0
        self.thetas_PV=[]
        self.domaine=[]
        if self.file is not None:
            self.Extract()

    def EoS_VP(self, P: float) -> float:
        if self.V0 is None or self.K0 is None or self.K0P is None:
            raise ValueError("Element_Bibli.EoS_VP: paramètres EOS manquants.")
        self.V = inversefunc(
            lambda x: Birch_M(x, self.V0, self.K0, self.K0P),
            y_values=P,
            domain=[0.1, None],
        )
        return float(self.V)

    def EoS_PV(self, V: float) -> float:
        if self.V0 is None or self.K0 is None or self.K0P is None:
            raise ValueError("Element_Bibli.EoS_PV: paramètres EOS manquants.")
        self.P_start = Birch_M(V, self.V0, self.K0, self.K0P)
        return float(self.P_start)

    def Extract(self):
        if self.file is None:
            raise ValueError("Element_Bibli.Extract: aucun fichier source.")

        M = []
        self.name_dhkl = []

        for i in range(len(self.file[0])):
            l_name = self.file[0][i]
            value = str(self.file[1][i]).strip()

            # --- Cas COMMENT ---
            if "COMMENT" in l_name:
                ligne = value
                name = ligne.split('/')[0]
                name = (
                    name.split(',')[0]
                    .strip()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                )
                self.name = name

                # extraire les plages de T et P
                match_temp = re.search(r"/T\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]", ligne)
                if match_temp:
                    self.T_range = (float(match_temp.group(1)), float(match_temp.group(2)))

                match_press = re.search(r"/P\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]", ligne)
                if match_press:
                    self.P_range = (float(match_press.group(1)), float(match_press.group(2)))

                match_Z = re.search(r"/Z=\s*(\d+)", ligne)
                if match_Z:
                    self.Z_max = int(match_Z.group(1))

            # --- Cas DIHKL ---
            elif "DIHKL" in l_name:
                # ici split marche parfaitement car ton fichier est tabulé
                parts = value.split()
                if len(parts) == 5:
                    dhkl, I, h, k, l = map(float, parts)
                    M.append([dhkl, I, h, k, l])
                    self.name_dhkl.append((h,k,l))

            # --- Cas K0, K0P, ALPHA... etc ---
            elif "K0" in l_name and "T" not in l_name:
                if "P" in l_name:
                    self.K0P = float(value)
                else:
                    self.K0 = float(value)

            elif "ALPHAT" == l_name:
                self.ALPHAKT = float(value)
            elif "ALPHA" == l_name:
                self.ALPHA = float(value)
            elif "BETA" == l_name:
                self.BETA = float(value)
            elif "GAMMA" == l_name:
                self.GAMMA = float(value)
            elif "A" == l_name:
                self.A = float(value)
            elif "B" == l_name:
                self.B = float(value)
                self.rBA = self.B / self.A
            elif "C" == l_name:
                self.C = float(value)
                self.rCA = self.C / self.A
            elif "V0" in l_name or "VOLUME" in l_name:
                self.V0 = float(value)
            elif "Vmin" in l_name:
                self.Vmin = float(value)

        # Créer DataFrame DIHKL
        self.Dhkl = pd.DataFrame(M, columns=["Dhkl", "I", "h", "k", "l"])

        # Récupérer la symétrie
        self.symmetrie = self.file[1][int(np.where(self.file[0] == "SYMMETRY")[0])]

        # Si V0 pas défini → calcul selon la symétrie
        if self.V0 is None:
            if "CUBIC" in self.symmetrie:
                self.V0 = round(self.A**3, 3)
                self.ALPHA = self.BETA = self.GAMMA = 90
            elif "ORTHORHOMBIC" in self.symmetrie:
                self.V0 = round(self.A * self.B * self.C, 3)
                self.ALPHA = self.BETA = self.GAMMA = 90
            elif "TETRAGONAL" in self.symmetrie:
                self.V0 = round(self.A**2 * self.C, 3)
                self.ALPHA = self.BETA = self.GAMMA = 90
            elif "HEXAGONAL" in self.symmetrie:
                self.V0 = round(np.sqrt(3) / 2 * self.A**2 * self.C, 3)
                self.ALPHA = self.BETA = 90
                self.GAMMA = 120
            elif "RHOMBOHEDRAL" in self.symmetrie:
                alpha_rad = np.deg2rad(self.ALPHA)
                self.V0 = round(
                    self.A**3
                    * np.sqrt(1 - np.cos(alpha_rad) ** 2)
                    / np.cos(alpha_rad),
                    3,
                )

        # Construire l'EoS
        try:
            P = Birch_M_save(
                np.linspace(self.V0 * 1.05, self.V0 * self.Vmin, 1000),
                self.V0,
                self.K0,
                self.K0P,
            )
            f = inversefunc(
                lambda x: Birch_M_save(x, self.V0, self.K0, self.K0P),
                domain=[self.V0 * self.Vmin, self.V0 * 1.05],
            )
            self.EoS = [P, f(P)]
        except Exception:
            self.Vmin = 0.8
            P = Birch_M_save(
                np.linspace(self.V0 * 1.05, self.V0 * self.Vmin, 1000),
                self.V0,
                self.K0,
                self.K0P,
            )
            f = inversefunc(
                lambda x: Birch_M_save(x, self.V0, self.K0, self.K0P),
                domain=[self.V0 * self.Vmin, self.V0 * 1.05],
            )
            self.EoS = [P, f(P)]
            warnings.warn("Vmin trop bas → reset à 0.8*V0")


    def E_theta(self, l: int, E: Optional[float] = None) -> float:
        if E is not None:
            self.E = E
        if self.E is None:
            raise ValueError("Element_Bibli.E_theta: énergie non définie.")
        X = 360 / np.pi * np.arcsin((1239.8 / self.E) * 1e-9 / (self.Dhkl["Dhkl"][l] * 2e-10))
        return float(X)
    
    def Eos_Pdhkl(self, P: float, extract: bool = False):
        V = self.EoS_VP(P)
        thetas_PV = []
        if self.E is None:
            raise ValueError("Energie non définie, veuillez la définir avant de calculer les angles")
        if  "CUBIC" in self.symmetrie: 
            a=V**(1/3)
        elif "ORTHORHOMBIC" in self.symmetrie:
            a=(V/(self.rCA*self.rBA))**(1/3)
            b=a*self.rBA
            c=a*self.rCA
        elif "TETRAGONAL" in self.symmetrie:
            a=(V/self.rCA)**(1/3)
            c=a*self.rCA
        elif "HEXAGONAL" in self.symmetrie:
            a=(V/(np.sqrt(3)/2*self.rCA))**(1/3)
            c=a*self.rCA
        elif "RHOMBOHEDRAL"  in self.symmetrie:
            a = (V/np.sqrt(1 - 3*np.cos(self.ALPHA*2*np.pi/360)**2+2*np.cos(self.ALPHA*2*np.pi/360)**3))**(1/3)

        elif "MONOCLINIC" in self.symmetrie:
            a=(V/(self.rBA*self.rCA*np.sin(self.BETA*2*np.pi/360)))**(1/3)
            b=self.rBA*a
            c=self.rCA*a
        else:
            raise ValueError("MAILLE != (cubic,tetra,hexa,mono,rhombo)")
        
        for i in range(min(30,len(self.Dhkl))):
            h,k,l=int(self.Dhkl["h"][i]),int(self.Dhkl["k"][i]),int(self.Dhkl["l"][i])
            if  "CUBIC" in self.symmetrie:
                dhkl= a/np.sqrt(h**2+k**2+l**2)
            elif "ORTHORHOMBIC" in self.symmetrie:
                dhkl= 1/np.sqrt((h/a)**2+(k/b)**2+(l/c)**2)
            elif "TETRAGONAL" in self.symmetrie:
                dhkl= a/np.sqrt(h**2+k**2+((a/c)*l)**2)
            elif "HEXAGONAL" in self.symmetrie:
                dhkl= 1/np.sqrt((4/3)*(h**2+k**2+h*k)/a**2+(l/c)**2)
            elif "RHOMBOHEDRAL" in self.symmetrie:
                dhkl=a*np.sqrt((1-3*np.cos(self.ALPHA*2*np.pi/360)**2+2*np.cos(self.ALPHA*2*np.pi/360)**3)/(((h**2+k**2+l**2)*np.sin(self.ALPHA*2*np.pi/360)**2-(np.cos(self.ALPHA*2*np.pi/360)-np.cos(self.ALPHA*2*np.pi/360)**2)*2*(k*l+l*h+h*k))))
            elif "MONOCLINIC" in self.symmetrie:
                dhkl=np.sqrt((1-np.cos(self.BETA*2*np.pi/360))/((h**2/a**2)+(k**2/b**2)*np.sin(self.BETA*2*np.pi/360)**2+(l**2/c**2)-(2*l*h/(c*a))*np.cos(self.BETA*2*np.pi/360)))
            
            x=round(360/np.pi*np.arcsin((1239.8/self.E)*1e-9/(dhkl*2e-10)),3)
            thetas_PV.append((x,self.Dhkl["I"][i],h,k,l))
        if extract is True:
            return thetas_PV 
        else:
            self.thetas_PV=thetas_PV #[pic for pic,_,_,_,_ in thetas_PV]      
    
    def _debug(self):
        self.name_dhkl=[]
        for i in range(len(self.Dhkl)):
            self.name_dhkl.append((int(self.Dhkl.h[i]),int(self.Dhkl.k[i]),int(self.Dhkl.l[i])))


class DRX:
    """Outils DRX: bibliothèque, détection de pics, GA d'identification."""

    def __init__(self, folder: Optional[Iterable[str]] = None, E: Optional[float] = None, Borne=None):
        self.Bibli_elements={}
        self.E=E
        self.Borne=Borne or [5, 30]
        self.detected_peaks=[]
        
        gen,NGEN ,self.limite_elements=None,None,None
        
        #self.element_colors = {}
        if folder:

            if isinstance(folder, list):
                self.list_file = folder  # on garde la liste
                for file in folder:
                    if os.path.isfile(file):  # vérifie que le chemin est valide
                        try:
                            df = pd.read_csv(file, sep=':', header=None, skipfooter=0, engine='python')
                            new_element = Element_Bibli(file=df, E=self.E)
                            self.Bibli_elements[new_element.name] = new_element
                        except Exception as e:
                            warnings.warn(f"Erreur lecture fichier {file} : {e}")
            else:
                self.list_file = []
                if os.path.isdir(folder):
                    for file_name in os.listdir(folder):
                        file_path = os.path.join(folder, file_name)
                        if os.path.isfile(file_path):
                            try:
                                df = pd.read_csv(file_path, sep=':', header=None, skipfooter=0, engine='python')
                                new_element = Element_Bibli(file=df, E=self.E)
                                self.Bibli_elements[new_element.name] = new_element
                                self.list_file.append(file_path)
                            except Exception as e:
                                warnings.warn(f"Erreur lecture fichier {file_path} : {e}")
        else:
            self.list_file = []
            pass
        

    def Extract_Bibli(self, RUN):
        Bibli_elements={}
        element_colors = {}
        l_c=["r","g","b","o","m"]
        list_name=[]
        for n_spec,s in enumerate(RUN.Spectra):
            for j in s.Gauges:
                name=j.name
                if name not in list_name:
                    list_name.append(name)
                    Bibli_elements[name]=Element_Bibli(file=j.Element_ref.file)         
                    for n_theta,t in enumerate(j.Element_ref.thetas_PV):
                        if j.Element_ref.save_var[n_theta] is True:
                            pass
                    element_colors[name]=l_c[len(Bibli_elements)-1]
        return Bibli_elements , element_colors , list_name
    
    def F_Find_peaks(
        self,
        x: Iterable[float],
        y: Iterable[float],
        height: float,
        distance: float,
        prominence: float,
        width: float,
        number_peak_max: int,
        width_min: float = 1,
        width_step: float = 1,
    ):
        """
        Détection de pics avec repli sur width si aucun pic trouvé.
        
        Parameters :
            width_min : largeur minimale à atteindre pour ne pas boucler infiniment
            width_step : valeur à soustraire à width à chaque tentative
        """
        y=np.asarray(y)
        current_width = width
        index_peaks = []

        while current_width >= width_min:
            index_peaks, result = find_peaks(
                y,
                height=height,
                distance=distance,
                prominence=prominence,
                width=current_width
            )
            if len(index_peaks) > 0:
                break  # au moins un pic trouvé → on sort

            # Réduction de la largeur pour essayer encore
            current_width -= width_step

        # Si toujours rien trouvé
        if len(index_peaks) == 0:
            return np.array([]), np.array([]), {}

        # --- ESTIMATION DU BRUIT (hors pics, exclusion ±3σ autour de chaque pic) ---
        results_half = peak_widths(y, index_peaks, rel_height=0.5)
        left_ips, right_ips = results_half[2], results_half[3]

        mask_bruit = np.ones_like(y, dtype=bool)
        for left, right in zip(left_ips, right_ips):
            fwhm = right - left
            sigma = fwhm / 2.355
            centre = (left + right) / 2
            start = int(np.floor(centre - 3 * sigma))
            end = int(np.ceil(centre + 3 * sigma))
            start = max(0, start)
            end = min(len(y) - 1, end)
            mask_bruit[start:end+1] = False

        if np.any(mask_bruit):
            niveau_bruit = np.std(y[mask_bruit])
        else:
            niveau_bruit = np.std(y)  # fallback

        # --- FILTRAGE DES PICS SELON LE BRUIT HORS-PICS ---
        prominences = result["prominences"]
        heights = result["peak_heights"]
        mask = (prominences >= 1.5 * niveau_bruit) | (heights >= 1.5 * niveau_bruit)
        index_peaks = index_peaks[mask]
        for key in result:
            result[key] = result[key][mask]

        if len(index_peaks) > 0:
            zipped_lists = list(zip(result["prominences"], index_peaks))
            zipped_lists.sort()
            _, index_peaks = zip(*zipped_lists)
            index_peaks = np.array(index_peaks[-number_peak_max:])
        else:
            index_peaks = np.array([])


        return index_peaks, x[index_peaks], result

        
    def F_Find_compo(
        self,
        detected_peaks,
        NGEN: int = 100,
        MUTPB: float = 0.2,
        CXPB: float = 0.5,
        POPINIT: int = 100,
        pressure_range=None,
        max_ecart_pressure: float = 2,
        theta2_range=None,
        max_elements: int = 3,
        tolerance: float = 0.1,
        indiv_start=None,
        bibli_element_perso=None,
        print_process: bool = False,
        parallel: bool = False,
    ):

        if detected_peaks is None:
            return None, [], []
        
        start_total = time.perf_counter()   # chrono global

        warnings.filterwarnings("ignore", message="Results obtained with less than")

        bibliotheque = bibli_element_perso if bibli_element_perso else self.Bibli_elements
        pressure_range = pressure_range or [0, 100]
        theta2_range = theta2_range or [0, 90]
        pression_min, pression_max = pressure_range
        theta2_inf, theta2_sup = min(a for a, b in theta2_range), max(b for a, b in theta2_range)
        detected_peaks = np.array([pt for pt in detected_peaks if any(a <= pt <= b for a, b in theta2_range)])
        limite_ecart_pression = max_ecart_pressure / 5

        def extract_best_indiv(best_ind):
            # ---- reconstruire tous les pics théoriques de l'individu ----
            theoretical_subset = []
            element_pics = {}  # pour stocker les pics de chaque élément
            for element, p in best_ind:
                th_pics = bibliotheque[element].Eos_Pdhkl(p, extract=True)
                th_valid = [(pic, inten, *rest) for pic, inten, *rest in th_pics
                            if theta2_inf < pic < theta2_sup]
                element_pics[element] = (p, th_valid)
                theoretical_subset.extend([pic for pic, *_ in th_valid])

            if not theoretical_subset:
                return [], []

            # dictionnaire valeur -> indices (gère doublons)
            th_index_map = {}
            for idx, val in enumerate(theoretical_subset):
                th_index_map.setdefault(val, []).append(idx)

            # ---- associer pics détectés ↔ théoriques ----
            distances = cdist(np.array(detected_peaks).reshape(-1, 1),
                            np.array(theoretical_subset).reshape(-1, 1),
                            metric="euclidean")

            indices_dp2th = np.argmin(distances, axis=1)  # meilleur match th pour chaque dp
            indices_th2dp = np.argmin(distances, axis=0)  # meilleur match dp pour chaque th

            valid_th = {th_idx for dp_idx, th_idx in enumerate(indices_dp2th)
                        if indices_th2dp[th_idx] == dp_idx}

            # ---- reconstruire l'individu final ----
            reconstructed_indiv, Gauges = [], []
            for element, (p, pics) in element_pics.items():
                element_ref = copy.deepcopy(self.Bibli_elements[element])
                element_ref.save_var = []

                valid_pics_for_element = [pic for pic in pics
                                        if any(idx in valid_th for idx in th_index_map.get(pic[0], []))]

                if valid_pics_for_element:
                    reconstructed_indiv.append((element, p, valid_pics_for_element))
                    element_ref.Eos_Pdhkl(P=p)

                    # marquer save_var
                    for th in element_ref.thetas_PV:
                        th_val = th[0]
                        if th_val in th_index_map:
                            matched = any(idx in valid_th for idx in th_index_map[th_val])
                            element_ref.save_var.append(matched)
                        else:
                            element_ref.save_var.append(False)

                    # créer Gauge
                    new_element = Element(element_ref, name=element)
                    new_element.P = p
                    for i, ppic in enumerate(valid_pics_for_element):
                        new_element.pics[i].ctr[0] = ppic[0]
                    new_element.Update_model()
                    Gauges.append(new_element)

            return reconstructed_indiv, Gauges

        def clamp_pressure_interval(global_min, global_max, p_range):
            # Cast en float et sécurise l'ordre
            try:
                pmin, pmax = float(p_range[0]), float(p_range[1])
                if pmin > pmax:
                    pmin, pmax = pmax, pmin
            except Exception:
                return float(global_min), float(global_max)

            if pmin >=global_max and global_min >= pmax:
                return pmin ,pmax
   
            return  max(float(global_min), pmin), min(float(global_max), pmax)

        # ---- fitness optimisé ----
        def fitness(individual):
            if not individual:
                return (np.exp(20),)

            theoretical_positions, theoretical_weights, pressions = [], [], []
            for elem, p in individual:
                pics = [th for th in bibliotheque[elem].Eos_Pdhkl(p, extract=True)
                        if theta2_inf < th[0] < theta2_sup]
                if pics:
                    th_pos, th_int = zip(*[(pic, intensity) for pic, intensity, *_ in pics])
                    theoretical_positions.extend(th_pos)
                    theoretical_weights.extend(th_int)
                pressions.append(p)

            if not theoretical_positions:
                return (np.exp(20),)

            coef_element = len(pressions) ** 1.25
            weights = np.array(theoretical_weights, dtype=float)
            weights /= weights.max() if weights.size else 1.0

            # distances DP ↔ TH
            dps = detected_peaks.reshape(-1, 1)
            ths = np.array(theoretical_positions, dtype=float).reshape(1, -1)
            distances = np.abs(dps - ths)

            indices_th = distances.argmin(axis=1)
            indices_dp = distances.argmin(axis=0)
            mask = np.arange(len(indices_th))
            valid = mask[indices_dp[indices_th] == mask]

            d_valid = distances[valid, indices_th[valid]]
            score_distance = np.expm1(d_valid / tolerance).sum()

            # score diff nb pics
            nb_valide = valid.size
            score_diff = np.expm1(abs(nb_valide - len(detected_peaks)) * 3)

            # score pression
            dp = max(pressions) - min(pressions)
            mean_p = np.mean(np.abs(pressions))
            ecart_p = dp / mean_p if mean_p else 0
            score_pression = np.expm1(ecart_p / limite_ecart_pression)

            # pénalité pics non appariés
            th_indices_matches = set(indices_th[valid])
            penalite = sum(1.5 * weights[idx] for idx in range(len(weights)) if idx not in th_indices_matches)

            score_total = (score_distance + score_diff + score_pression + penalite) * coef_element
            return (score_total,)

        # ---- init plus léger ----
        def init_individu():
            if indiv_start is not None:
                indiv = [(name, float(p)) for name, p in indiv_start]
            else:
                taille = np.random.randint(1, min(len(bibliotheque), max_elements) + 1)
                indiv = [(elem, float(np.random.uniform(*bibliotheque[elem].P_range)))
                                for elem in np.random.choice(list(bibliotheque.keys()), size=taille, replace=False)]
            return creator.Individual(indiv)

        def adaptive_mutation(individu, gen, NGEN):
            """
            Mutation adaptative :
            - exploration forte au début (gros déplacements, ajouts/suppressions fréquents)
            - exploitation fine vers la fin (petites variations de pression)
            """
            if len(individu) == 0:
                elem = np.random.choice(list(bibliotheque.keys()))
                emin, emax = clamp_pressure_interval(pression_min, pression_max, bibliotheque[elem].P_range)
                p = round(np.random.uniform(emin, emax), 1)
                return creator.Individual([(elem, p)]),

            progress = gen / NGEN
            exploration = 1 - progress  # proche de 1 au début, proche de 0 à la fin
            des = np.random.rand()

            # --- modifier la pression ---
            if des <= 0.7 * (0.5 + exploration):
                i = np.random.randint(len(individu))
                elem, pression = individu[i]
                emin, emax = clamp_pressure_interval(pression_min, pression_max, bibliotheque[elem].P_range)
                if emin < emax:
                    delta = (emax - emin) * (0.05 + 0.25 * exploration)  # diminue avec le temps
                    low = max(emin, pression - delta)
                    high = min(emax, pression + delta)
                    new_p = round(np.random.uniform(low, high), 1)
                    individu[i] = (elem, new_p)
                return individu,

            # --- ajouter un élément (plus fréquent au début) ---
            elif des <= 0.85 * (0.5 + exploration):
                dispo = [e for e in bibliotheque if e not in [el for el, _ in individu]]
                if dispo:
                    elem = np.random.choice(dispo)
                    emin, emax = clamp_pressure_interval(pression_min, pression_max, bibliotheque[elem].P_range)
                    if emin < emax:
                        new_p = round(np.random.uniform(emin, emax), 1)
                        individu.append((elem, new_p))
                return individu,

            # --- supprimer un élément (seulement si >1 et surtout au début) ---
            else:
                if len(individu) > 1 and np.random.rand() < exploration:
                    individu.pop(np.random.randint(len(individu)))
                return individu,

        
        def make_mutation_with_gen(gen, NGEN):
            def _mutation(ind):
                return adaptive_mutation(ind, gen, NGEN)
            return _mutation


        def custom_cxTwoPoint(ind1, ind2):
            """
            Crossover à 2 points avec sécurité :
            - échange de segments
            - suppression des doublons
            - clamp + normalisation
            """
            size = min(len(ind1), len(ind2))
            if size < 2:
                return ind1, ind2

            cxpoint1 = random.randint(1, size - 1)
            cxpoint2 = random.randint(cxpoint1, size - 1)
            if cxpoint1 == cxpoint2:
                return ind1, ind2

            # Segments
            segment1 = copy.deepcopy(ind1[cxpoint1:cxpoint2])
            segment2 = copy.deepcopy(ind2[cxpoint1:cxpoint2])

            # Correction doublons / cohérence
            add_in1 = [ele for ele in segment2 if ele[0] not in [x[0] for x in ind1]]
            add_in2 = [ele for ele in segment1 if ele[0] not in [x[0] for x in ind2]]

            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = add_in1, add_in2

            return ind1, ind2

        # ---- DEAP setup ----
        _ensure_deap_creator()
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, init_individu)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", custom_cxTwoPoint)
        toolbox.register("mutate", adaptive_mutation, gen=0, NGEN=NGEN)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", fitness)

            # ---- choix parallélisation ou non ----
        if parallel:
            n_jobs = max(1, int(cpu_count() * 0.7))
            pool = Pool(processes=n_jobs)
            toolbox.register("map", pool.map)
        else:
            toolbox.register("map", map)  # version séquentielle

        # ---- évolution ----
        population = toolbox.population(n=POPINIT)
        fits = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fits):
            ind.fitness.values = fit

        best_ind = tools.selBest(population, k=1)[0]
        score, n_convergence = best_ind.fitness.values[0], 0

        # dictionnaire global pour le cache
        fitness_cache = {}

        def evaluate_with_cache(ind):
            """Évalue la fitness avec cache pour éviter les recalculs inutiles."""
            key = tuple(sorted((str(e), round(float(p), 3)) for e, p in ind))
            if key in fitness_cache:
                return fitness_cache[key]
            else:
                fit = toolbox.evaluate(ind)
                fitness_cache[key] = fit
                return fit

        def maintain_diversity(pop, init_func, fitness_key=lambda ind: ind.fitness.values[0],min_div=0.5):
            """
            Maintient la diversité dans la population :
            - Si diversité < min_div, on remplace une fraction des pires individus
            - En priorité les doublons, sinon les moins bons
            """
            # Clé pour identifier un individu (élément+pression arrondis)
            keys = [tuple(sorted((str(e), round(float(p), 3)) for e, p in ind)) for ind in pop]
            unique_count = len(set(keys))
            diversity = unique_count / len(pop)

            replaced = 0
            if diversity < min_div:  
                n_replace = int(len(pop) * (min_div-diversity)*1.5)

                # tri des indices par fitness (pire en premier)
                sorted_idx = sorted(range(len(pop)), key=lambda i: fitness_key(pop[i]), reverse=True)

                # on marque les doublons
                seen = set()
                dup_idx = []
                for i, k in enumerate(keys):
                    if k in seen:
                        dup_idx.append(i)  # doublon
                    else:
                        seen.add(k)

                # on cible d’abord les doublons, puis les pires
                target_idx = dup_idx + sorted_idx
                target_idx = target_idx[:n_replace]

                for idx in target_idx:
                    new_ind = init_func()
                    if len(new_ind) == 0:  # sécurité anti-individu vide
                        elem = np.random.choice(list(bibliotheque.keys()))
                        emin, emax = bibliotheque[elem].P_range
                        p = round(np.random.uniform(emin, emax), 3)
                        new_ind = creator.Individual([(elem, p)])
                    pop[idx] = new_ind
                    replaced += 1

            return pop, diversity, replaced

        
        for gen in range(NGEN):
            start_gen = time.perf_counter()

            # 👉 mettre à jour la mutation avec la génération courante
            toolbox.unregister("mutate")
            toolbox.register("mutate", make_mutation_with_gen(gen, NGEN))

            offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

            pop_old=len({tuple(sorted(ind)) for ind in population})

            # évaluation avec cache
            fits = [evaluate_with_cache(ind) for ind in offspring]
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            
            # élite + sélection
            elite = tools.selBest(population, k=1)
            population = toolbox.select(offspring, k=len(population) - len(elite)) + elite

            # diversité forcée si trop faible
            population, diversity, replaced = maintain_diversity(population, toolbox.individual)

            # meilleur individu
            best_ind = tools.selBest(population, k=1)[0]

            
            if best_ind.fitness.values[0] == score:
                n_convergence += 1
            else:
                n_convergence = 0
            score = best_ind.fitness.values[0]

            end_gen = time.perf_counter()
            if print_process and gen%10==0:
                print(f"Génération {gen}: Score={score:.4f}, "
                    f"Individu={best_ind}, "
                    f"Durée={end_gen - start_gen:.3f} s,"
                    f"Diversité={int(len({tuple(sorted(ind)) for ind in population})/len(population)*100)}%<--{int(pop_old/len(population)*100)}%")


            if score < len(best_ind)*2.2 or n_convergence >= max(NGEN*0.1, 20):
                break

        end_total = time.perf_counter()

        if print_process:
            mode = "Parallèle" if parallel else "Séquentiel"
            print(f"⚡ {mode} terminé en {end_total - start_total:.3f} s "
                f"({gen+1} générations) \n core = {score}, Individu = {[(e,p) for e,p in best_ind]}")

        if parallel:
            pool.close(); pool.join()

        reconstructed_indiv, Gauges = extract_best_indiv(best_ind)
        return best_ind, reconstructed_indiv, Gauges
    
    def Auto_dif(self, CEDX):
        best_ind=None
        for i,X in enumerate(CEDX.Spectra):
            _ , detected_peaks,_= self.F_Find_peaks(X.wnb,X.y_corr,height=max(X.spec)*0.03,distance=1,prominence=max(X.y_corr)*0.025,width=32,number_peak_max=15)
            if i==0:
                best_ind,_,Gauges =self.F_Find_compo(detected_peaks,NGEN = 300,MUTPB = 0.4,CXPB =0.4,POPINIT=150,pressure_range=[0,20],max_ecart_pressure=1,max_elements=3,tolerance=0.1,print_process=True)
            else:
                best_ind,_,Gauges =self.F_Find_compo(detected_peaks,NGEN = 300,MUTPB = 0.5,CXPB =0.5,POPINIT=150,pressure_range=[last_pressure*0.9-2,last_pressure*1.1+2],max_ecart_pressure=1,
                                    max_elements=3,tolerance=0.1,indiv_start=best_ind,print_process=True)
            CEDX.Spectra[i].Gauges=Gauges
            CEDX.Spectra[i].bit_fit=True
            CEDX.Spectra[i].Calcul_study(mini=True)
            last_pressure=np.mean([g.P for g in CEDX.Spectra[i].Gauges])
            CEDX.Summary=pd.concat([CEDX.Summary,pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),CEDX.Spectra[i].study],ignore_index=False,axis=1)],ignore_index=True)
        return CEDX

    def set_E(self, E: float) -> None:
        self.E=E
        for elem in self.Bibli_elements:
            self.Bibli_elements[elem].E=E




class CED_DRX:
    """Construction de spectres DRX depuis images + synthèse."""

    def __init__(
        self,
        data_drx,
        calib: Calib_DRX,
        E: Optional[float],
        data_oscillo=None,
        time_index=None,
        flip_axis: Optional[int] = None,
        drx: Optional[DRX] = None,
    ):
        if data_oscillo is not None or time_index is not None:
            raise ValueError("CED_DRX: la logique oscillo/spectro n'est pas supportée en mode DRX-only.")

        self.Spectra = []
        self.Summary = pd.DataFrame()
        self.data_drx = data_drx
        self.calib = calib
        self.note = ""
        self.sigma_dist = 0
        self.ClassDRX = drx or DRX(E=E)

        if isinstance(data_drx, list):
            fichiers_tries = sorted(data_drx, key=extraire_numero)
        elif os.path.isfile(data_drx):
            data = fabio.open(data_drx)
            n_frames = data.nframes
            for i in range(n_frames):
                frame = data.getframe(i).data
                if flip_axis is not None:
                    frame = np.flip(frame) if flip_axis == 2 else np.flip(frame, axis=flip_axis)
                tth, intens = Integrate_DRX(
                    frame, self.calib.mask, self.calib.ai, theta_range=self.calib.theta_range
                )
                self.Spectra.append(SpectreDRX(tth, intens, E=E))
            return
        elif os.path.isdir(data_drx):
            fichiers_tries = [os.path.join(data_drx, f) for f in sorted(os.listdir(data_drx), key=extraire_numero)]
        else:
            raise ValueError(f"data_drx doit être un fichier, un dossier, ou une liste, pas {type(data_drx)}")

        for f in fichiers_tries:
            data = fabio.open(f).data
            if flip_axis is not None:
                data = np.flip(data) if flip_axis == 2 else np.flip(data, axis=flip_axis)
            tth, intens = Integrate_DRX(
                data, self.calib.mask, self.calib.ai, theta_range=self.calib.theta_range
            )
            self.Spectra.append(SpectreDRX(tth, intens, E=E))

    def fit_selected_spectra(
        self,
        index_start: int = 0,
        index_stop: Optional[int] = None,
        best_ind=None,
        NGEN: int = 300,
        MUTPB: float = 0.4,
        CXPB: float = 0.4,
        POPINIT: int = 150,
        pressure_range=None,
        max_ecart_pressure: float = 1,
        max_elements: int = 3,
        tolerance: float = 0.1,
        print_process: bool = True,
        custom_peak_params=None,
    ) -> None:
        """
        Applique le fitting automatique aux spectres entre index_start et index_stop (inclus).
        
        Parameters:
            index_start : int
                Index de début (inclus).
            index_stop : int or None
                Index de fin (inclus). Si None, jusqu'au dernier spectre.
            best_ind : Individual or None
                Individu de référence pour guider les fits suivants.
            custom_peak_params : dict or None
                Dictionnaire contenant les paramètres de détection de pics.

        Modifie :
            self.Summary et les objets Spectre concernés (ajoute Gauges, bit_fit, study).
        """

        if index_stop is None:
            index_stop = len(self.Spectra) - 1

        self.Summary = pd.DataFrame()
        last_pressure = None

        for i in range(index_start, index_stop + 1):
            X = self.Spectra[i]

            # Détection des pics
            peak_args = {
                "height": max(X.spec) * 0.03,
                "distance": 1,
                "prominence": max(X.y_corr) * 0.025,
                "width": 32,
                "number_peak_max": 15
            }
            if custom_peak_params:
                peak_args.update(custom_peak_params)

            _, detected_peaks, _ = self.ClassDRX.F_Find_peaks(
                X.wnb, X.y_corr, **peak_args
            )

            # Fit des compositions
            if i == index_start or best_ind is None:
                best_ind, _, Gauges = self.ClassDRX.F_Find_compo(
                    detected_peaks,
                    NGEN=NGEN,
                    MUTPB=MUTPB,
                    CXPB=CXPB,
                    POPINIT=POPINIT,
                    pressure_range=pressure_range or [0, 20],
                    max_ecart_pressure=max_ecart_pressure,
                    max_elements=max_elements,
                    tolerance=tolerance,
                    print_process=print_process
                )
            else:
                best_ind, _, Gauges = self.ClassDRX.F_Find_compo(
                    detected_peaks,
                    NGEN=NGEN,
                    MUTPB=MUTPB,
                    CXPB=CXPB,
                    POPINIT=POPINIT,
                    pressure_range=[last_pressure * 0.9 - 2, last_pressure * 1.1 + 2],
                    max_ecart_pressure=max_ecart_pressure,
                    max_elements=max_elements,
                    tolerance=tolerance,
                    indiv_start=best_ind,
                    print_process=print_process
                )

            self.Spectra[i].Gauges = Gauges
            self.Spectra[i].bit_fit = True
            self.Spectra[i].Calcul_study(mini=True)

            last_pressure = np.mean([g.P for g in Gauges])
            spec_df = pd.concat([pd.DataFrame({"n°Spec": [int(i)]}), self.Spectra[i].study], axis=1)
            self.Summary = pd.concat([self.Summary, spec_df], ignore_index=True)

    def Corr_Summary(self, num_spec=None, N_MC: int = 0, verbose: bool = False) -> None:
        """
        Recalcule le Summary complet ou d’un seul spectre,
        en tenant compte des incertitudes σ(center) issues du bruit local.
        Aucun refit n’est effectué.
        """

        self.Summary = pd.DataFrame()
        
        if num_spec is None:
            specs = range(len(self.Spectra))
        else:
            specs = [num_spec]

        if verbose:
            print(f"\n🔁 Correction Summary ({len(specs)} spectres à traiter)...")

        for i in specs:
            spec = self.Spectra[i]

            if hasattr(spec, "Gauges") and spec.Gauges:
                # 1) bruit local
                try:
                    spec.estimate_all_sigma_noise(N_MC)
                except Exception as e:
                    warnings.warn(f"Estimation bruit échouée sur spectre {i}: {e}")

                # 1bis) ajout incertitude distance
                for G in spec.Gauges:
                    for p in G.pics:
                        if hasattr(p, "sigma_ctr_stat"):
                            tth_deg = float(p.ctr[0])
                            sigma_geom = self.sigma_2theta_from_distance(tth_deg)
                            p.sigma_ctr_geom = sigma_geom
                            p.sigma_ctr_total = float(
                                np.sqrt(p.sigma_ctr_stat **2 + sigma_geom**2)
                            )
                        else:
                            tth_deg = float(p.ctr[0])
                            sigma_geom = self.sigma_2theta_from_distance(tth_deg)
                            p.sigma_ctr_geom = sigma_geom
                            p.sigma_ctr_total = float(sigma_geom)

                # 2) recalcul des jauges en utilisant ces sigmas
                for G in spec.Gauges:
                    try:
                        G.Calcul(mini=True, verbose=verbose)
                    except Exception as e:
                        warnings.warn(f"Gauge {G.name}: erreur Calcul() ({e})")

                # 3️⃣ Mets à jour le DataFrame du spectre
                try:
                    spec.study = pd.concat([x.study for x in spec.Gauges], axis=1)
                    spec.bit_fit = True
                except Exception as e:
                    warnings.warn(f"Fusion study spectre {i}: {e}")
                    continue
            else:
                if verbose:
                    print(f"⚠️ Spectre {i}: pas de Gauges, ignoré.")

            # 4️⃣ Ajoute au résumé global
            spec_df = pd.concat(
                [pd.DataFrame({"n°Spec": [int(i)]}), spec.study],
                axis=1
            )
            self.Summary = pd.concat([self.Summary, spec_df], ignore_index=True)

            if verbose:
                print(f"✅ Spectre {i+1}/{len(self.Spectra)} mis à jour.")

        # 5️⃣ Applique les éventuelles corrections manuelles depuis self.note
        if hasattr(self, "note") and isinstance(self.note, str):
            corrections = re.findall(r"\[(.*?)\]\s*->\s*([-+]?\d*\.?\d+)", self.note)
            for col, val in corrections:
                if col in self.Summary.columns:
                    try:
                        val = float(val)
                        self.Summary[col] = self.Summary[col] + val
                    except Exception as e:
                        warnings.warn(f"Impossible d'appliquer correction sur {col}: {e}")
            if verbose:
                print(f"📋 Corrections appliquées : {self.note}")

        if verbose:
            print("\n✅ Correction Summary terminée.\n")

    def sigma_2theta_from_distance(self, tth_deg: float) -> float:
        """
        Calcule σ(2θ) en degrés due à l’incertitude sur la distance détecteur.
        """
        if self.calib is None or self.calib.ai is None or getattr(self.calib.ai, "dist", None) is None:
            raise ValueError("sigma_2theta_from_distance: calibration ou distance détecteur manquante.")

        tth_rad = np.deg2rad(float(tth_deg))  # convertit 2θ → rad

        # d(2θ)/dD = -0.5 * sin(2*2θ) / D
        d2tth_dD = -0.5 * np.sin(2 * tth_rad) / self.calib.ai.dist

        sigma_2tth_rad = abs(d2tth_dD) * self.sigma_dist
        return float(np.rad2deg(sigma_2tth_rad))  # renvoie en degrés


if __name__ == "__main__":
    print("Module DRX chargé. Utilisez Calib_DRX, DRX, CED_DRX et SpectreDRX.")
