import numpy as np
from math import sqrt, log
from lmfit.models import (
    PseudoVoigtModel,
    MoffatModel,
    SplitLorentzianModel,
    Pearson4Model,
    GaussianModel,
)
from scipy.special import gamma, beta
from scipy.optimize import curve_fit


"""------------------------------------- LOI FIT -------------------------------------"""      
def PseudoVoigt(x,center,ampH,sigma,fraction):
    amp=ampH/(((1-fraction))/(sigma*sqrt(np.pi/log(2)))+(fraction)/(np.pi*sigma))
    sigma_g=(sigma/(sqrt(2*log(2))))
    return (1-fraction)*amp/(sigma_g*sqrt(2*np.pi))*np.exp(-(x-center)**2/(2*sigma_g**2))+fraction*amp/np.pi*(sigma/((x-center)**2+sigma**2))

def Moffat(x,center,ampH,sigma,beta):
    amp=ampH*beta
    return amp*(((x-center)/sigma)**2+1)**(-beta)

def SplitLorenzian(x, center, ampH, sigma, sigma_r):
    amp = ampH * 2 / (np.pi * (sigma + sigma_r))
    sig = np.where(x < center, sigma, sigma_r)
    y = amp * (sig**2) / ((x - center)**2 + sig**2)
    return y

def PearsonIV(x, center,ampH, sigma, m, skew):
    center =center + sigma*skew/(2*m)
    return ampH / ((1 + (skew/(2*m))**2)**-m * np.exp(-skew * np.arctan(-skew/(2*m)))) * (1 + ((x - center) / sigma)**2)**-m * np.exp(-skew * np.arctan((x - center) / sigma))

 #  amp= ampH / (normalization*(1 + (skew/(2*m))**2)**-m * np.exp(-skew * np.arctan(-skew/(2*m))))   #amp * normalization * (1 + ((x - center) / sigma)**2)**-m * np.exp(-skew * np.arctan((x - center) / sigma))

def Gaussian(x,center,ampH,sigma):
    return ampH*np.exp(-(x-center)**2/(2*sigma**2))

class Pics:
    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, name='', ctr=0, ampH=1, coef_spe=[0.5], sigma=0.15,
                 inter=3, model_fit="PseudoVoigt", Delta_ctr=0.5, amp=None,weight_sigma=None):
        self.name = name.replace("-", "_")
        self.ctr = [ctr, [ctr-Delta_ctr, ctr+Delta_ctr]]
        self.model_fit = model_fit
        self.inter = inter
        inter_min = max(1-inter, 0)

        self.sigma = [sigma, [sigma*inter_min, sigma*(1+inter)]]
        self.ampH = [ampH, [ampH*inter_min, ampH*(1+inter)]]
        self.best_fit = None
        self.help = "Pics: définition de modèle de pics de luminescence"
        coef_spe = np.array(coef_spe)
        self.coef_spe = [[c, [c*inter_min, c*(1+inter)]] for c in coef_spe]

        # init modèle
        self._init_model(model_fit, coef_spe, ctr, Delta_ctr, inter_min)

        # amplitude intégrée
        if amp is None:
            amp = self.f_amp(self.ampH[0], [float(c[0]) for c in self.coef_spe], sigma)
        self.amp = [amp, [amp*inter_min, amp*(1+inter)]]

        # param hints
        self._set_param_hints()

    # ------------------------------------------------------------------
    # Fonctions d’amplitude
    # ------------------------------------------------------------------
    def Amp_PsD(self, ampH, coef_spe, sigma):
        return ampH / (((1-coef_spe[0]))/(sigma*sqrt(np.pi/log(2))) +
                       (coef_spe[0])/(np.pi*sigma))

    def Amp_Gaussian(self, ampH, coef_spe, sigma):
        return ampH * sigma * sqrt(2*np.pi)

    def Amp_Moffat(self, ampH, coef_spe, sigma):
        return ampH * coef_spe[0]

    def Amp_SplitL(self, ampH, coef_spe, sigma):
        return ampH / (2*np.pi*(sigma+coef_spe[0]))

    def Amp_PearsonIV(self, ampH, coef_spe, sigma):
        m, skew = coef_spe[0], coef_spe[1]
        normalization = np.abs(gamma(m + 1j*skew/2)/gamma(m))**2 / (sigma*beta(m-0.5, 0.5))
        return ampH / (normalization*(1 + (skew/(2*m))**2)**-m *
                       np.exp(-skew*np.arctan(-skew/(2*m))))

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------
    def _init_model(self, model_fit, coef_spe, ctr, Delta_ctr, inter_min):
        """Initialise modèle selon type"""
        if model_fit == "PseudoVoigt":
            self.f_amp = self.Amp_PsD
            self.name_coef_spe = ["fraction"]
            self.model = PseudoVoigtModel(prefix=self.name)
            self.f_model = PseudoVoigt
            self.coef_spe = [[max(min(1, coef_spe[0]), 0),
                              [coef_spe[0]*inter_min,
                               max(0.1, min(1, coef_spe[0]*(1+self.inter)))]]]

        elif model_fit == "Moffat":
            self.f_amp = self.Amp_Moffat
            self.name_coef_spe = ["beta"]
            self.model = MoffatModel(prefix=self.name)
            self.f_model = Moffat

        elif model_fit == "SplitLorentzian":
            self.f_amp = self.Amp_SplitL
            self.name_coef_spe = ["sigma_r"]
            self.model = SplitLorentzianModel(prefix=self.name)
            self.f_model = SplitLorenzian

        elif model_fit == "Gaussian":
            self.f_amp = self.Amp_Gaussian
            self.name_coef_spe = []
            self.model = GaussianModel(prefix=self.name)
            self.f_model = Gaussian

        elif model_fit == "PearsonIV":
            self.f_amp = self.Amp_PearsonIV
            self.name_coef_spe = ["expon", "skew"]
            self.model = Pearson4Model(prefix=self.name)
            self.f_model = PearsonIV
            if len(coef_spe) < 2:
                coef_spe = [coef_spe[0], coef_spe[0]]
            lc = [coef_spe[1]*(1-self.inter)-0.1, coef_spe[1]*(1+self.inter)+0.1]
            self.coef_spe = [
                [max(0.505, coef_spe[0]),
                 [max(0.501, coef_spe[0]*inter_min),
                  max(0.51, coef_spe[0]*(1+self.inter))]],
                [coef_spe[1], [min(lc), max(lc)]]
            ]

    def _set_param_hints(self):
        """Met à jour les paramètres lmfit"""
        for i, name_spe in enumerate(self.name_coef_spe):
            self.model.set_param_hint(self.name+name_spe,
                                      value=self.coef_spe[i][0],
                                      min=self.coef_spe[i][1][0],
                                      max=self.coef_spe[i][1][1])
        self.model.set_param_hint(self.name+'amplitude',
                                  value=self.amp[0], min=self.amp[1][0], max=self.amp[1][1])
        self.model.set_param_hint(self.name+'sigma',
                                  value=self.sigma[0], min=self.sigma[1][0], max=self.sigma[1][1])
        
        if self.model_fit == "PearsonIV":
            shift = self.sigma[0] * self.coef_spe[1][0] / (2*self.coef_spe[0][0])
            ctr = self.ctr[0] + shift
            ctr_min = self.ctr[1][0] + shift
            ctr_max = self.ctr[1][1] + shift
        else:
            ctr,ctr_min,ctr_max=self.ctr[0],self.ctr[1][0],self.ctr[1][1]
        self.model.set_param_hint(self.name+'center',
                                  value=ctr, min=ctr_min, max=ctr_max)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def Update(self, ctr=None, ampH=None, coef_spe=None, sigma=None,
           inter=None, model_fit=None, Delta_ctr=None, amp=None, move=False):
        """Met à jour les paramètres du pic"""
        if Delta_ctr is None:
            Delta_ctr = 0.4
        if inter is not None:
            self.inter = float(inter)
        inter_min = max(1 - self.inter, 0)

        # --- maj ctr/sigma si fournis
        if ctr is not None:
            self.ctr = [ctr, [ctr - Delta_ctr, ctr + Delta_ctr]]

        if sigma is not None:
            self.sigma = [sigma, [sigma * inter_min, sigma * (1 + self.inter)]]
        else:
            # rafraîchir les bornes si inter a changé
            s = self.sigma[0]
            self.sigma = [s, [s * inter_min, s * (1 + self.inter)]]

        # --- maj coef_spe
        if coef_spe is not None:
            vals = np.atleast_1d(coef_spe).astype(float).tolist()
            self.coef_spe = [[v, [v * inter_min, v * (1 + self.inter)]] for v in vals]
        else:
            # rafraîchir bornes existantes avec inter courant
            self.coef_spe = [[c[0], [c[0] * inter_min, c[0] * (1 + self.inter)]] for c in self.coef_spe]

        # --- si modèle change
        if model_fit is not None and model_fit != self.model_fit:
            self.model_fit = model_fit
            self._init_model(model_fit, [c[0] for c in self.coef_spe],
                            ctr or self.ctr[0], Delta_ctr, inter_min)
        else:
            # appliquer règles spécifiques même sans changement de modèle
            if self.model_fit == "PseudoVoigt" and len(self.coef_spe) >= 1:
                frac = max(0.0, min(1.0, float(self.coef_spe[0][0])))
                self.coef_spe[0][0] = frac
                self.coef_spe[0][1] = [frac * inter_min,
                                    max(0.1, min(1.0, frac * (1 + self.inter)))]

            elif self.model_fit == "PearsonIV":
                if len(self.coef_spe) < 2:
                    # forcer 2 paramètres : m et skew
                    m0 = 0.9
                    skew0 = 0.0
                    self.coef_spe = [[m0, [m0 * inter_min, m0 * (1 + self.inter)]],
                                    [skew0, [skew0 * inter_min, skew0 * (1 + self.inter)]]]

                m0 = max(0.501, float(self.coef_spe[0][0]))
                skew0 = float(self.coef_spe[1][0])
                lc = [skew0 * (1 - self.inter) - 1.0,
                    skew0 * (1 + self.inter) + 1.0]

                self.coef_spe[0][0] = m0
                self.coef_spe[0][1] = [max(0.501, m0 * inter_min),
                                    max(0.51, m0 * (1 + self.inter))]
                self.coef_spe[1][1] = [min(lc), max(lc)]
                """
                if not move:
                    ctr_phys = self.ctr[0] + self.sigma[0] * skew0 / (2 * m0)
                    self.ctr = [ctr_phys, [ctr_phys - Delta_ctr, ctr_phys + Delta_ctr]]
                """
            elif self.model_fit == "SplitLorentzian" and len(self.coef_spe) >= 1:
                sig_r = max(1e-12, float(self.coef_spe[0][0]))
                self.coef_spe[0][0] = sig_r
                self.coef_spe[0][1] = [sig_r * inter_min, sig_r * (1 + self.inter)]

            # (Gaussian / Moffat : pas de contraintes spécifiques en plus)

        # --- amplitude recalculée
        if ampH is not None and amp is None:
            self.ampH = [ampH, [ampH * inter_min, ampH * (1 + self.inter)]]
            amp_val = self.f_amp(self.ampH[0], [c[0] for c in self.coef_spe], self.sigma[0])
        elif amp is not None:
            amp_val = amp
        else:
            amp_val = self.f_amp(self.ampH[0], [c[0] for c in self.coef_spe], self.sigma[0])

        self.amp = [amp_val, [amp_val * inter_min, amp_val * (1 + self.inter)]]

        # --- mise à jour des hints lmfit
        self._set_param_hints()


    def Out_model(self, out=None, l_params=None, l_sigma=None):
        """Mise à jour des paramètres (fit, liste ou init)"""
        if out is not None:
            sigma = round(out.params[self.name+'sigma'].value, 3)
            sigma_sigma = round(out.params[self.name+'sigma'].stderr or 0, 3)
            self.sigma = [sigma, [sigma-sigma_sigma, sigma+sigma_sigma]]

            ampH = round(out.params[self.name+'height'].value, 3)
            sigma_ampH = round(out.params[self.name+'height'].stderr or 0, 3)
            self.ampH = [ampH, [ampH-sigma_ampH, ampH+sigma_ampH]] 

            coef_spe = np.array([round(out.params[self.name+name_spe].value, 3)
                                 for name_spe in self.name_coef_spe])

            
            if self.model_fit == 'PearsonIV':
                ctr = round(out.params[self.name+'position'].value, 3)
                sigma_ctr = round(out.params[self.name+'position'].stderr or 0, 3)
            else:
                ctr = round(out.params[self.name+'center'].value, 3)
                sigma_ctr = round(out.params[self.name+'center'].stderr or 0, 3)
            
            self.ctr = [ctr, [ctr-sigma_ctr, ctr+sigma_ctr]]


            return [ctr, ampH, sigma, coef_spe]

        elif l_params is not None:
            ctr = round(l_params[0], 3)
            ampH = round(l_params[1], 3)
            sigma = round(l_params[2], 3)
            coef_spe = np.array([round(p, 3) for p in l_params[3]])

            if self.model_fit == 'PearsonIV':
                shift=sigma*coef_spe[1]/(2*coef_spe[0])
                ctr=ctr+shift

            if l_sigma is not None:
                sigma_ctr = round(l_sigma[0], 3)
                sigma_ampH = round(l_sigma[1], 3)
                sigma_sigma = round(l_sigma[2], 3)

                self.ctr = [ctr, [ctr-sigma_ctr, ctr+sigma_ctr]]
                self.ampH = [ampH, [ampH-sigma_ampH, ampH+sigma_ampH]]
                self.sigma = [sigma, [sigma-sigma_sigma, sigma+sigma_sigma]]

                coef_spe = np.array([
                    [round(coef, 3), [round(coef-sig, 3), round(coef+sig, 3)]]
                    for coef, sig in zip(l_params[4], l_sigma[4])
                ])

            return [ctr, ampH, sigma, coef_spe]

        else:
            params = self.model.make_params()
            
            sigma = round(params[self.name+'sigma'].value, 3)
            sigma_sigma=round(params[self.name+'sigma'].stderr or 1e-4,3)

            ampH = round(params[self.name+'height'].value, 3)
            sigma_ampH=round(params[self.name+'height'].stderr or 1e-4,3)

            coef_spe = np.array([round(params[self.name+name_spe].value, 3)
                                 for name_spe in self.name_coef_spe])
            sigma_coef_spe=np.array([ round(params[self.name+name_spe].stderr or 1e-4,3) for name_spe in self.name_coef_spe])
            
            if self.model_fit == 'PearsonIV':
                ctr = round(params[self.name+'position'].value, 3)
                sigma_ctr=round(params[self.name+'position'].stderr or 1e-4,3)
            else:
                ctr = round(params[self.name+'center'].value, 3)
                sigma_ctr=round(params[self.name+'center'].stderr or 1e-4,3)

            self.ctr = [ctr, [ctr-sigma_ctr, ctr+sigma_ctr]]
            self.sigma = [sigma, [sigma-sigma_sigma, sigma+sigma_sigma]]
            self.ampH = [ampH, [ampH-sigma_ampH, ampH+sigma_ampH]]
            self.coef_spe = [ [coef,[coef-sigma,coef+sigma]] for coef,sigma in zip(coef_spe,sigma_coef_spe)]

            return [ctr, ampH, sigma, coef_spe], params

    def Out_ctr(self):
        #if self.model_fit == "PearsonIV":
        #return self.ctr[0] - self.sigma[0]*self.coef_spe[1][0]/(2*(self.coef_spe[0][0]))
        return self.ctr[0]
    
        # --------------------------------------------------------------
    # Estimation σ(center) dominée par le bruit (unité = même que x)
    # --------------------------------------------------------------
    def estimate_sigma_center_from_noise(self, x, y, zone_baseline=None, N_MC=0, mode="auto",verbose=False):
        """
        Estime σ(center) en tenant compte du bruit (Poisson ou Gaussien).

        Parameters
        ----------
        x, y : array-like
            Données du pic.
        zone_baseline : tuple(float, float) ou None
            Zone pour estimer le bruit hors pic.
        N_MC : int
            Nombre de tirages Monte Carlo (0 = pas de simulation).
        mode : str
            "poisson", "gauss", ou "auto"
            - "poisson" : bruit de comptage photonique (JUNGFRAU, PILATUS, EIGER)
            - "gauss"   : bruit additif constant (CCD, CMOS)
            - "auto"    : choix automatique selon intensité moyenne du signal

        Returns
        -------
        dict : {
            'sigma_ctr', 'sigma_n', 'SNR', 'FWHM', 'method'
        }
        """


        x, y = np.asarray(x), np.asarray(y)
        if len(x) < 8:
            return None

        # --- paramètres du pic ---
        center = float(self.ctr[0])
        sigma  = float(self.sigma[0])
        ampH   = float(self.ampH[0])
        coef   = [c[0] for c in getattr(self, "coef_spe", [])]

        # --- 1️⃣ bruit hors pic ---
        if zone_baseline is None:
            mask = (x < center - 3 * sigma) | (x > center + 3 * sigma)
        else:
            xmin, xmax = zone_baseline
            mask = (x > xmin) & (x < xmax)

        sigma_n = float(np.std(y[mask])) if np.any(mask) else float(np.std(y))

        # --- 2️⃣ choix du mode ---
        if mode == "auto":
            # heuristique : si max(y) < 1000 → Poisson, sinon Gauss
            mode = "poisson" if np.mean(y) < 1000 else "gauss"

        poisson = (mode.lower().startswith("p"))

        # --- 3️⃣ estimation analytique ---
        delta_x = float(np.mean(np.diff(x)))
        FWHM = 2.3548 * sigma
        M = max(1.0, FWHM / max(delta_x, 1e-12))

            # 3a) estimation du SNR
        if poisson:
            # On essaie d'utiliser le modèle pour estimer le nombre total de photons dans le pic
            N_phot = None
            if hasattr(self, "f_model") and callable(self.f_model):
                try:
                    p0 = [center, ampH, sigma] + list(coef)
                    y_model = self.f_model(x, *p0)
                    # On suppose que y_model représente des comptes (ou proportionnel)
                    N_phot = float(np.sum(np.clip(y_model, 0, None)))
                except Exception:
                    N_phot = None

            if N_phot is None:
                # fallback : aire ~ ampH * FWHM / delta_x (ordre de grandeur)
                N_phot = float(max(abs(ampH) * FWHM / max(delta_x, 1e-12), 1e-12))

            SNR = sqrt(N_phot)

        else:
            # bruit additif constant : SNR = amplitude / bruit RMS
            SNR = abs(ampH) / max(sigma_n, 1e-12)

        alpha = 0.60  # constante empirique CRLB
        sigma_ctr = alpha * FWHM / (SNR * sqrt(M))

        # --- 4️⃣ Monte Carlo optionnel ---
        if N_MC > 0 and hasattr(self, "f_model") and callable(self.f_model):
            try:
                p0 = [center, ampH, sigma] + list(coef)
                y_model = self.f_model(x, *p0)
                ctr_MC = []

                for _ in range(N_MC):
                    if poisson:
                        y_noisy = np.random.poisson(np.clip(y_model, 0, None))
                    else:
                        y_noisy = y_model + np.random.normal(0, sigma_n, size=len(y))

                    try:
                        popt, _ = curve_fit(self.f_model, x, y_noisy, p0=p0, maxfev=800)
                        ctr_MC.append(popt[0])
                    except Exception:
                        continue

                if len(ctr_MC) > 5:
                    sigma_ctr = float(np.std(ctr_MC))
                    if verbose:
                        print(f"⚙️ MC({self.name}): σ_center = {sigma_ctr:.5g} ({len(ctr_MC)} essais, mode={mode})")

            except Exception as e:
                print(f"⚠️ {self.name}: erreur Monte Carlo ({e})")

        # --- 5️⃣ sortie ---
        self.sigma_ctr_stat  = float(sigma_ctr)
        return {
            "sigma_ctr": float(sigma_ctr),
            "sigma_n": float(sigma_n),
            "SNR": float(SNR),
            "FWHM": float(FWHM),
            "method": f"{'MC-' if N_MC > 0 else ''}{mode}"
        }


    def Help(self, request=None):
        if request is None:
            print("Choisissez un modèle pour obtenir des informations détaillées :")
            print("Disponible : 'PseudoVoigt', 'Moffat', 'SplitLorentzian', 'Gaussian', 'PearsonIV'.")
        elif request == 'param':
            if self.model_fit == "PseudoVoigt":
                print("\nModèle : PseudoVoigt")
                print("Description : Modèle hybride entre un Lorentzien et un Gaussien.")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic, écart type (défini par 'sigma').")
                print("  - fraction : Fraction entre le Lorentzien et le Gaussien (0 < fraction < 1).")
                print("  Exemple : PseudoVoigt(x, ctr, ampH, sigma, fraction)")

            elif self.model_fit == "Moffat":
                print("\nModèle : Moffat")
                print("Description : Modèle de pic Moffat, utilisé pour des pics avec une décroissance plus lente qu'un Lorentzien.")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic, écart type (défini par 'sigma').")
                print("  - beta : Paramètre de forme qui détermine l'étalement du pic (beta > 0).")
                print("  Exemple : Moffat(x, ctr, ampH, sigma, beta)")

            elif self.model_fit == "SplitLorentzian":
                print("\nModèle : SplitLorentzian")
                print("Description : Modèle Lorentzien fractionné avec deux valeurs de largeur (sigma et sigma_r).")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic gauche (défini par 'sigma').")
                print("  - sigma_r : Largeur du pic droite (défini par 'sigma_r').")
                print("  Exemple : SplitLorentzian(x, ctr, ampH, sigma, sigma_r)")

            elif self.model_fit == "Gaussian":
                print("\nModèle : Gaussian")
                print("Description : Modèle Gaussien classique.")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic, écart type (défini par 'sigma').")
                print("  Exemple : Gaussian(x, ctr, ampH, sigma)")

            elif self.model_fit == "PearsonIV":
                print("\nModèle : Pearson IV")
                print("Description : Modèle Pearson IV, adapté pour des pics plus asymétriques avec une forme flexible.")
                print("Paramètres :")
                print("  - ctr : Centre du pic (défini par 'center').")
                print("  - ampH : Hauteur du pic (définie par 'height').")
                print("  - sigma : Largeur du pic, écart type (défini par 'sigma').")
                print("  - m : Paramètre de forme de Pearson IV, contrôle la symétrie du pic (m > 0).")
                print("  - skew : Asymétrie du pic, peut être positive ou négative.")
                print("  Exemple : PearsonIV(x, ctr, ampH, sigma, m, skew)")

        elif request == 'test':
            print("OK")
