import sys
import fabio
from matplotlib.widgets import Slider, Button
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5 import QtCore
import os
from lmfit.models import PseudoVoigtModel , MoffatModel , SplitLorentzianModel , Pearson4Model, GaussianModel
from scipy.signal import savgol_filter , find_peaks, peak_widths 
import peakutils as pk
import pandas as pd
import numpy as np
from PIL import Image
import dill
from tqdm import tqdm
from math import sqrt , log
import lecroyscope
import copy
import re
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import gamma , beta
from pynverse import inversefunc
import random
import time 
#import cv2
import warnings
from deap import base, creator, tools, algorithms
from scipy.spatial.distance import cdist
import pyqtgraph as pg
from pybaselines import Baseline
from .pressure_law import *
from .gauge import Gauge, Element


from  .Calibration import Integrate_DRX
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool
def _ensure_deap_creator():
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)


_ensure_deap_creator()

#from scipy.optimize import minimize
#from scipy.special import voigt_profile
""" ------------------------------------- FONCTION GESTION DE FICHIER -------------------------------------"""
def SAVE_CEDd(file_CEDd,bit_try=False):
    if file_CEDd:
        if bit_try==True:
            try:
                #if hasattr(file_CEDd, 'data_drx'):
                #file_CEDd.data_drx = None
                dill.dump( file_CEDd, open( file_CEDd.CEDd_path, "wb" ) )
            except Exception as e:
                print("ERROR : ",e," in SAVE_CEDd")
        else:
            dill.dump( file_CEDd, open( file_CEDd.CEDd_path, "wb" ) )
            
def LOAD_CEDd(CEDd_path,bit_try=False):
    if CEDd_path:
        if bit_try==True:
            try:
                CEDd = dill.load( open( CEDd_path, "rb" ) )
                CEDd.CEDd_path=CEDd_path
                return CEDd
            except Exception as e:
                print("ERROR : ",e," in LOAD_CEDd")
        else:
            CEDd = dill.load( open( CEDd_path, "rb" ) )
            CEDd.CEDd_path=CEDd_path
            return CEDd

def Load_last(Folder,extend=None,file=True):
    if file ==True:
        if extend != None :
            file_names = [f for f in os.listdir(Folder) if os.path.isfile(os.path.join(Folder, f)) and extend in f]
        else:
            file_names = [f for f in os.listdir(Folder) if os.path.isfile(os.path.join(Folder, f))]
    else:
        file_names = [f for f in os.listdir(Folder)]
    if file_names:
        file_names.sort(key=lambda f: os.path.getmtime(os.path.join(Folder, f)))
        latest_file_name = file_names[-1]
        latest_file_path = os.path.join(Folder, latest_file_name)
    return latest_file_path, latest_file_name

def extraire_numero(fichier):
    match = re.search(r'_(\d+)\.npy$', fichier)
    return int(match.group(1)) if match else float('inf')

def Gen_sum_F(list_F):
    def sum_F(x,*params):
        params_split=[]
        index=0
        for f in list_F:
            num_params= f.__code__.co_argcount-1 #Nb parma de la foncion
            params_split.append(params[index:index+num_params])
            index+=num_params
        result = np.array([0 for _ in x])#np.zeros((1,len(x)))
        for f, f_params in zip(list_F,params_split):
            result=result+f(x,*f_params)
        return result
    return sum_F

def _first_or_none(val):
    try:
        import numpy as np
        if val is None:
            return None
        return float(np.atleast_1d(val)[0])
    except Exception:
        return None

def _fit_signature(G, x_len: int):
    sig = []
    for p in getattr(G, "pics", []):
        c = _first_or_none(getattr(p, "ctr", None))
        a = _first_or_none(getattr(p, "amp", None)) if hasattr(p, "amp") else None
        s = _first_or_none(getattr(p, "sig", None)) if hasattr(p, "sig") else None
        sig.append((p.name, c, a, s))
    # encode aussi la longueur de la fenêtre X
    sig.append(("len", int(x_len)))
    return tuple(sig)
"""------------- CLASSE JAUGE SPECTRO -------------------------------------"""      


""" CLASSE JAUGE """
# p1 le pique principale de mesure 
# nb_pic : le nombre de pic
# lamb0 la lambda0 de l'ambitante
# name = nom à trouver
# color_print = [ couleur principale , couleur des pique]

class Element_Bibli:
    def __init__(self, file=None,E=None):
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
        self.T=298
        self.T_range=None
        self.P_range=[-10,1000]
        self.Vmin=0.8
        self.Z_max=None
        self.fu=None
        self.P_start=0
        self.thetas_PV=[]
        self.domaine=[]
        try:
            self.Extract()
        except Exception as e:
            print(e)

    def EoS_VP(self, P_total, T=None):
        """Retourne V à partir de P_total et T, via BM isotherme."""
        if T is None:
            T = self.T
        P_iso = self.P_isotherm_from_total(P_total, T)

        try:
            self.V = inversefunc(
                (lambda x: Birch_M(x, self.V0, self.K0, self.K0P)),
                y_values=P_iso,
                domain=[self.V0 * self.Vmin, self.V0 * 1.05],
            )
        except Exception as e:
            print(e)
        return self.V

    def EoS_PV(self, V, T=None):
        """Retourne P_total à partir de V et T."""
        if T is None:
            T = self.T
        try:
            P_iso = Birch_M(V, self.V0, self.K0, self.K0P)
            self.P_start = self.P_total_from_isotherm(P_iso, T)
        except Exception as e:
            print(e)
        return self.P_start

    def Extract(self):
        if self.file is None:
            return

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

                match_fu = re.search(r"/fu=\s*(\d+)", ligne)
                if match_fu:
                    self.fu = int(match_fu.group(1))

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
        self.symmetrie = self.file[1][np.flatnonzero(self.file[0] == "SYMMETRY")[0]]

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
            print("Vmin too low → reset to 0.8*V0")

    def E_theta(self,l,E=None):
        if E is not None:
            self.E=E
        X= 360/np.pi*np.arcsin((1239.8/self.E)*1e-9/(self.Dhkl["Dhkl"][l]*2e-10))
        return X
    
    def Eos_Pdhkl(self, P_total, T=None, extract=False):
        V = self.EoS_VP(P_total, T=T)
        thetas_PV=[]
        if self.E is None:
            return print("Energie non définie, veuillez la définir avant de calculer les angles")
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
            return print("MAILLE != (cubic,tetra,hexa,mono,rhombo) , A CODER")
        
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

    def P_thermal(self, T):
        """Terme de pression thermique Pt = ALPHAKT*(T-T0=298)."""
        if self.ALPHAKT is None:
            return 0.0
        return float(self.ALPHAKT) * (float(T) - 298)

    def P_isotherm_from_total(self, P_total, T):
        """Retire le terme thermique pour obtenir la pression isotherme BM."""
        return float(P_total) - self.P_thermal(T)

    def P_total_from_isotherm(self, P_iso, T):
        """Ajoute le terme thermique."""
        return float(P_iso) + self.P_thermal(T)



class DRX():
    def __init__(self,folder=None,E=None,Borne=[5,30]):
        self.Bibli_elements={}
        self.E=E
        self.Borne=Borne
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
                            print(f"Erreur lecture fichier {file} : {e}")
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
                                print(f"Erreur lecture fichier {file_path} : {e}")
        else:
            self.list_file = []
            pass
        

    def Extract_Bibli(self,RUN):
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
                        if (j.Element_ref.save_var[n_theta[0]]) is True :
                            print(name,t)
                    element_colors[name]=l_c[len(Bibli_elements)-1]
        return Bibli_elements , element_colors , list_name
    
    def F_Find_peaks(self, x, y, height, distance, prominence, width, number_peak_max, width_min=1, width_step=1):
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

        
    def F_Find_compo(self, detected_peaks, NGEN=100, MUTPB=0.2, CXPB=0.5,
                     POPINIT=100, pressure_range=[0,100], max_ecart_pressure=2,
                     theta2_range=[0,90], max_elements=3, tolerance=0.1,
                     indiv_start=None, bibli_element_perso=None, print_process=False,parallel=False):

        if detected_peaks is None:
            return None, [], []
        
        start_total = time.perf_counter()   # chrono global

        warnings.filterwarnings("ignore", message="Results obtained with less than")

        bibliotheque = bibli_element_perso if bibli_element_perso else self.Bibli_elements
        pression_min, pression_max = pressure_range
        theta2_inf, theta2_sup = min(a for a, b in theta2_range), max(b for a, b in theta2_range)
        detected_peaks = np.array([pt for pt in detected_peaks if any(a <= pt <= b for a, b in theta2_range)])
        limite_ecart_pression = max_ecart_pressure / 5
        
        def element_root(name: str) -> str:
            try:
                return str(name).split("_", 1)[0]
            except Exception:
                return str(name)


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
            if prev_roots:
                cand_roots = {element_root(elem) for elem, _ in individual}
                missing_roots = prev_roots - cand_roots     # racines perdues
                new_roots = cand_roots - prev_roots         # nouvelles racines

                # facteur de relâchement si peu de pics observés
                # (moins il y a de pics, plus on autorise des disparitions)
                n_dp = int(len(detected_peaks))
                relax = min(n_dp*1.5,10)
    

                # poids : à ajuster
                w_missing = 2.0   # pénalité racine perdue (forte)
                w_new     = 1.0   # pénalité racine nouvelle (plus faible)

                score_prior_roots = relax * (w_missing * len(missing_roots) + w_new * len(new_roots))

                # bonus léger si on conserve au moins une racine (stabilise Sn + H2O vs “Sn seul”)
                keep_count = len(prev_roots & cand_roots)
                score_prior_roots -= relax * 0.3 * keep_count

                # optionnel : favoriser le même nombre d'éléments que précédemment (très léger)
                score_prior_roots += relax * 0.2 * abs(len(cand_roots) - len(prev_roots))

                score_total += score_prior_roots

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

        prev_names = []
        prev_roots = set()
        if indiv_start is not None:
            prev_names = [str(n) for n, _ in indiv_start]
            prev_roots = {element_root(n) for n in prev_names}

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
    
    def Auto_dif(self,CEDX):
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

    def set_E(self,E):
        self.E=E
        for elem in self.Bibli_elements:
            self.Bibli_elements[elem].E=E
 
    def F_Find_compoOLD(self, detected_peaks, NGEN=100, MUTPB=0.2, CXPB=0.5, POPINIT=100,pressure_range=[0,100], max_ecart_pressure=2,theta2_range=[0,90], max_elements=3, tolerance=0.1,indiv_start=None, bibli_element_perso=None,print_process=False):
        if detected_peaks is None:
            return None, [], []
        
        
        indiv_start=indiv_start
        warnings.filterwarnings("ignore", message="Results obtained with less than")

        bibliotheque = bibli_element_perso if bibli_element_perso else self.Bibli_elements
        print([(element, bibliotheque[element].P_range) for element in bibliotheque.keys()])
        pression_min, pression_max = pressure_range
        theta2_inf, theta2_sup = min(a for a, b in theta2_range), max(b for a, b in theta2_range)
        detected_peaks = np.array([pt for pt in detected_peaks if any(a <= pt <= b for a, b in theta2_range)])
        limite_ecart_pression = max_ecart_pressure / 5

        # ---- utilitaire ----
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

        # ---- fitness ----
        def fitness(individual):
            theoretical_positions, theoretical_weights, pressions = [], [], []
            for _, p, pics in individual:
                for pic, intensity, *_ in pics:
                    theoretical_positions.append(pic)
                    theoretical_weights.append(intensity)
                pressions.append(p)

            if not theoretical_positions:
                return (np.exp(20),)

            coef_element = len(pressions) ** 1.25
            weights = np.array(theoretical_weights) / max(theoretical_weights, default=1)

            distances = cdist(np.array(detected_peaks).reshape(-1,1),
                            np.array(theoretical_positions).reshape(-1,1))

            indices_th = np.argmin(distances, axis=1)
            indices_dp = np.argmin(distances, axis=0)

            score_distance, indices_valides = 0, []
            for i, th_index in enumerate(indices_th):
                if indices_dp[th_index] == i:
                    indices_valides.append((i, th_index))
                    d = distances[i, th_index]
                    score_distance += (np.exp(d / tolerance) - 1)

            nb_valide = len(indices_valides)
            score_diff = np.exp(abs(nb_valide - len(detected_peaks)) * 3) - 1

            dp = max(pressions) - min(pressions)
            ecart_p = (dp / np.mean(np.abs(pressions))) if np.mean(np.abs(pressions)) else 0
            score_pression = np.exp(ecart_p / limite_ecart_pression) - 1

            th_indices_matches = {th for _, th in indices_valides}
            penalite = 0
            pos_dict = {pos: idx for idx, pos in enumerate(theoretical_positions)}

            for _, _, pics in individual:
                if not pics:
                    continue
                pics_sorted = sorted(pics, key=lambda x: x[1])
                last_match = max((i for i,(pos,_,*_) in enumerate(pics_sorted)
                                if pos in pos_dict and pos_dict[pos] in th_indices_matches), default=-1)
                for i in range(last_match+1, len(pics_sorted)):
                    pos, intensite, *_ = pics_sorted[i]
                    if pos in pos_dict and pos_dict[pos] not in th_indices_matches:
                        penalite += 1.5 * weights[pos_dict[pos]]

            score_total = (score_distance + score_diff + score_pression + penalite) * coef_element
            return (score_total,)

        # ---- init ----
        def init_individu():
            """
            Initialise un individu en respectant toujours la contrainte :
            pression ∈ [pression_min, pression_max] ∩ P_range_element
            """
            if indiv_start is not None:
                sous_ensemble = []
                pressures = []
                for name, p_val, _ in indiv_start:
                    sous_ensemble.append(name)
                    pressures.append(p_val)

                # Si on a des pressions de départ, on prend leur moyenne comme centre
                if pressures:
                    p_mean = round(np.mean(pressures), 1)
                else:
                    p_mean = round(np.random.uniform(pression_min, pression_max), 1)
            else:
                taille = np.random.randint(1, min(len(bibliotheque), max_elements) + 1)
                sous_ensemble = np.random.choice(list(bibliotheque.keys()), size=taille, replace=False)
                p_mean = round(np.random.uniform(pression_min, pression_max), 1)

            individu = []
            for elem in sous_ensemble:
                # Intersection plage globale et P_range locale
                emin, emax = clamp_pressure_interval(pression_min, pression_max, bibliotheque[elem].P_range)
                if emin >= emax:  # intersection vide → on saute cet élément
                    continue

                # Si p_mean est dans la plage, on l’utilise, sinon on tire au hasard dans l’intersection
                if emin <= p_mean <= emax:
                    pression = p_mean
                else:
                    pression = round(np.random.uniform(emin, emax), 1)

                pics = [th for th in bibliotheque[elem].Eos_Pdhkl(pression, extract=True)
                        if theta2_inf < th[0] < theta2_sup]
                individu.append((elem, pression, pics))

            return creator.Individual(individu)

        # ---- mutations ----
                # ---- mutations ----
        def mutation(individu, rapide=False):
            """
            Mutation robuste :
            - modifie la pression (reste dans intersection globale ∩ locale)
            - ou ajoute un élément
            - ou supprime un élément
            """
            des = np.random.rand()

            # --- modifier la pression ---
            if (not rapide and des <= 0.7) or (rapide and des <= 0.5):
                for i, (element, pression, _) in enumerate(individu):
                    emin, emax = clamp_pressure_interval(pression_min, pression_max, bibliotheque[element].P_range)
                    if emin >= emax:
                        new_p = (emin + emax) / 2
                    else:
                        # Mutation locale : petit déplacement autour de la pression actuelle
                        delta = (emax - emin) * (0.1 if rapide else 0.25)
                        low = max(emin, pression - delta)
                        high = min(emax, pression + delta)
                        if low > high:  # sécurité
                            low, high = emin, emax
                        new_p = round(np.random.uniform(low, high), 1)

                    pics = [th for th in bibliotheque[element].Eos_Pdhkl(new_p, extract=True)
                            if theta2_inf < th[0] < theta2_sup]
                    individu[i] = (element, new_p, pics)
                return individu,

            # --- ajouter un élément ---
            elif (not rapide and des > 0.85) or (rapide and 0.5 < des <= 0.7):
                dispo = [e for e in bibliotheque if e not in [el[0] for el in individu]]
                if dispo:
                    elem = np.random.choice(dispo)
                    emin, emax = clamp_pressure_interval(pression_min, pression_max, bibliotheque[elem].P_range)
                    if emin < emax:
                        new_p = round(np.random.uniform(emin, emax), 1)
                        pics = [th for th in bibliotheque[elem].Eos_Pdhkl(new_p, extract=True)
                                if theta2_inf < th[0] < theta2_sup]
                        individu.append((elem, new_p, pics))
                return individu,

            # --- supprimer un élément ---
            elif (not rapide and 0.7 < des <= 0.85) or (rapide and 0.7 < des <= 0.9):
                if len(individu) > 1:
                    individu.pop(np.random.randint(len(individu)))
                return individu,

            # Aucun changement
            return individu,

        def muter_individu(individu, gen=0):
            """
            Wrapper mutation DEAP : renvoie toujours un individu clampé et normalisé.
            """
            rapide = (np.random.rand() > gen / NGEN)
            individu, = mutation(individu, rapide=rapide)  # dépaquetage
            return individu,

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

        def extract_best_indiv(best_ind):
            theoretical_subset = []
            # Générer les pics théoriques à partir de la combinaison
            for element,p,_ in best_ind:
                theoretical_subset.extend([pic for pic,_,_,_,_ in bibliotheque[element].Eos_Pdhkl(p, extract=True) if (pic > theta2_inf and pic < theta2_sup)])  # Ajouter les pics à comparer
        
            # Calculer la matrice de distance entre chaque pic détecté et chaque pic théorique
            distances = cdist(np.array(detected_peaks).reshape(-1, 1), np.array(theoretical_subset).reshape(-1, 1), metric='euclidean')
            # Trouver pour chaque pic détecté l'indice du pic théorique le plus proche
            indices_associés_th ,indices_associés_dp  = np.argmin(distances, axis=1), np.argmin(distances, axis=0)

            # Vérifier si les indices sont cohérents (miroir)
            indices_valides = []
            for i, th_index in enumerate(indices_associés_th):
                if indices_associés_dp[th_index] == i:  # Vérification miroir
                    indices_valides.append(th_index)
            
            # Extraire les pics valides
            valid_pics = set(indices_valides)

            # Reconstruire l'individu avec uniquement les pics valides
            reconstructed_indiv = []
            Gauges=[]
            for element, p, pics in best_ind:
                element_ref=copy.deepcopy(self.Bibli_elements[element])
                
                element_ref.save_var=[]
                valid_pics_for_element=[]
                for pic in pics:
                    if theoretical_subset.index(pic[0]) in valid_pics:
                        valid_pics_for_element.append(pic)

                if valid_pics_for_element != []:
                    reconstructed_indiv.append((element, p, valid_pics_for_element))
                    element_ref.Eos_Pdhkl(P=p)
                    
                    for i,pic in enumerate(element_ref.thetas_PV):
                        if pic[0] in theoretical_subset:
                            element_ref.save_var.append(True if theoretical_subset.index(pic[0]) in valid_pics  else False)
                        else:
                            element_ref.save_var.append(False)
                    new_element = Element(element_ref,name=element)
                    new_element.P=p
                    for i,p in enumerate(valid_pics_for_element):
                        new_element.pics[i].ctr[0]=p[0]
                    new_element.Update_model()
                    Gauges.append(new_element)
    
            return reconstructed_indiv,Gauges
        
        # ---- toolbox ----
        _ensure_deap_creator()

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, init_individu)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", custom_cxTwoPoint)
        toolbox.register("mutate", muter_individu)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", fitness)

        # ---- évolution ----
        population = toolbox.population(n=POPINIT)
        # Évaluer la population initiale
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        best_ind = tools.selBest(population, k=1)[0]
        score = best_ind.fitness.values[0]
        n_convergence = 0

        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

            # mutation custom + clamp
            offspring = [muter_individu(ind, gen)[0] for ind in offspring]


            for fit, ind in zip(map(toolbox.evaluate, offspring), offspring):
                ind.fitness.values = fit

            # élitisme
            elite = tools.selBest(population, k=max(1, int(POPINIT*0.05)))
            population = toolbox.select(offspring, k=len(population) - len(elite)) + elite

            best_ind = tools.selBest(population, k=1)[0]

            if best_ind.fitness.values[0] == score:
                n_convergence += 1
            else:
                n_convergence = 0
            score = best_ind.fitness.values[0]

            if print_process and gen % 10 == 0:
                print(f"Génération {gen}: Score = {score}, Individu = {[(e,p) for e,p,_ in best_ind]}")

            if score < len(best_ind)*2.2 or n_convergence >= max(NGEN*0.1, 20):
                break
        # ---- extraction finale ----
        best_ind = tools.selBest(population, k=1)[0]
        reconstructed_indiv, Gauges = extract_best_indiv(best_ind)
        if print_process:
            print(f"Résultat: Génération {gen}, Score = {score}, Individu = {[(e,p) for e,p,_ in best_ind]}")

        return best_ind, reconstructed_indiv, Gauges


def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

"""------------------------------------- CLASSE ELEMENT SPECTRE -------------------------------------"""      
class Spectre:
    def __init__(self,wnb,spec,Gauges=[],type_filtre="svg",param_f=[9,2],deg_baseline=0,E=None): #lambda0_s=None,lambda0_r=None,lambda0_SrFCl = None,Temperture=False,Model="psdV",pic
        self.wnb=np.array(wnb)
        self.spec=spec
        self.spec_brut=spec
        self.param_f=param_f
        self.deg_baseline=deg_baseline
        self.type_filtre=type_filtre
        self.y_filtre,self.blfit=None,None
        self.x_corr=wnb
        self.Data_treatement(print_data=False)
        self.E=E
        self.X=None
        self.Y=None
        self.dY=None
        self.bit_model=False
        self.model=None
        self.fit="Fit Non effectué"
        self.bit_fit=False
        self.lamb_fit=None
        self.indexX=None
        #FIT PIC
        self.Gauges=Gauges
        self.lambda_error=round((self.wnb[-1]-self.wnb[0])*0.5/len(self.wnb),4)
        #SYNTHESE
        self.study=pd.DataFrame()
        self.help="Spectre: etude de spectre"

    def light_copy(self):
        """Return a lightweight clone with duplicated numerical buffers."""

        clone = copy.copy(self)
        for attr, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(clone, attr, value.copy())
            elif isinstance(value, pd.DataFrame):
                setattr(clone, attr, value.copy(deep=True))
        if hasattr(self, "Gauges"):
            clone.Gauges = list(self.Gauges)
        return clone

    def Corr(self,list_lamb0):
        for i in range(len(self.Gauges)):
            if list_lamb0[i] !=None :
                self.Gauges.lamb0=list_lamb0[i]
            self.Gauges[i].Calcul(input_spe=self.Gauges,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)

    def Print(self, ax=None, ax2=None):
        figure = ax is not None
        if not figure:
            fig, (ax, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 4),
                                        gridspec_kw={'height_ratios': [0.85, 0.15]})

        # Traces brutes
        ax.plot(self.wnb, self.blfit, '-.', c='g', markersize=1)
        ax.plot(self.wnb, self.spec, '-', color='lightgray', markersize=4)
        ax.plot(self.wnb, self.y_corr + self.blfit, '.', color='black', markersize=3)
        lc=["r","g","b","orange"]
        for G in self.Gauges:
            if not getattr(G, "bit_fit", False):
                continue
            c=None
            titre_fiti = f"{G.name}:$\\lambda_0=$" + str(G.lamb0)

            if G.indexX is None:
                G.indexX = self.indexX

            # Fenêtre X et baseline bf
            X = self.wnb[G.indexX] if G.indexX is not None else self.wnb
            bf = self.blfit[G.indexX] if G.indexX is not None else self.blfit

            # Doit-on recalculer ?
            curr_sig = _fit_signature(G, len(X))
            prev_sig = getattr(G, "_last_fit_sig", None)
            need_recompute = (getattr(G, "Y", None) is None) or (prev_sig != curr_sig)

            if need_recompute:
                G.X = X                        # on fige la fenêtre courante
                G.Y = np.zeros_like(bf, float) # modèle SANS baseline
                for p in G.pics:
                    params_f = p.model.make_params()
                    y_p = p.model.eval(params_f, x=G.X)
                    G.Y += y_p
                G._last_fit_sig = curr_sig
            else:
                # Si tu veux TOUJOURS utiliser la fenêtre X courante à l'affichage
                # (même si le fit n'a pas changé), décommente la ligne suivante :
                G.X = X
                pass
            if G.color_print[1] is not None and lc !=[]:
                c=lc.pop(0)

            for i, p in enumerate(G.pics):
                params_f = p.model.make_params()
                y_p = p.model.eval(params_f, x=G.X)  # modèle seul
                y_fill = y_p + bf                    # pour remplissage au-dessus de la baseline

                if G.color_print[1] is not None:
                    titre_pic = rf" ${p.name}^{(G.name[0])}= {round(getattr(p,'ctr',[0])[0],3)}$"
                    ax.fill_between(G.X, y_fill, bf, where=y_fill > bf,
                                    alpha=0.3, label=titre_pic, color=G.color_print[1][i])
                elif c is not None:
                    ax.fill_between(G.X, y_fill, bf, where=y_fill > bf,color=c,alpha=0.25)
                else:
                    ax.fill_between(G.X, y_fill, bf, where=y_fill > bf,color=c,alpha=0.25)


            # Courbe du fit global (modèle + baseline)
            y_model_plus_bf = G.Y + bf
            if getattr(G, "color_print", [None, None])[0] is not None:
                ax.plot(G.X, y_model_plus_bf, '--', label=titre_fiti, markersize=1, c=G.color_print[0])
            else:
                ax.plot(G.X, y_model_plus_bf, '--', label=titre_fiti, markersize=1)

            # Résidus cohérents : (data sans baseline) - (modèle sans baseline)
            data_wo_bf = self.y_corr[G.indexX] if G.indexX is not None else self.y_corr
            G.dY = data_wo_bf - G.Y

            if ax2 is not None:
                denom = np.max(np.abs(G.dY)) if np.any(G.dY) else 1.0
                denom = denom if denom > 0 else 1.0
                if getattr(G, "color_print", [None, None])[0] is not None:
                    ax2.plot(G.X, G.dY / denom, '-', c=G.color_print[0])
                else:
                    ax2.plot(G.X, G.dY / denom, '-')

        # Mise en forme
        ax.minorticks_on()
        ax.tick_params(which='major', length=10, width=1.5, direction='in')
        ax.tick_params(which='minor', length=5, width=1.5, direction='in')
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        ax.set_title(f'$Spectre,\\Delta\\lambda=$' + str(self.lambda_error))
        ax.set_ylabel('Amplitude (U.A.)')
        ax.set_xlim([min(self.wnb), max(self.wnb)])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        if ax2 is not None:
            ax2.axhline(0, color="k", ls='-.')
            ax2.minorticks_on()
            ax2.tick_params(which='major', length=10, width=1.5, direction='in')
            ax2.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax2.tick_params(which='both', bottom=True, top=True, left=True, right=True)
            ax2.set_xlabel(r'$\lambda$ (nm)')
            ax2.set_ylabel(r'$(Data-Fit)/max$ (U.A.)')
            ax2.set_xlim([min(self.wnb), max(self.wnb)])
            ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        else:
            ax.set_xlabel(r'$\lambda$ (nm)')

        ax.legend(loc="best")
        if not figure:
            plt.show()
        else:
            return ax

    def FIT_One_Jauge(self,num_jauge=0,peakMax0=None,wnb_range=3,coef_spe=None,sigma=None,inter=None,model_fit=None,manuel=False,model_jauge=None,Delta_ctr=None):
        G=self.Gauges[num_jauge]
        y_sub=self.y_corr
        for other_G in self.Gauges:
            if other_G.name != G.name and G.state =="Y":
                param=G.model.make_params()
                y_sub=y_sub - G.model.eval(param,x=self.wnb)
        
        if (peakMax0 is not None) and (model_jauge is None):
            G=self.Gauges[num_jauge]
            peakMax=peakMax0


        elif model_jauge is not None:
            G = model_jauge[num_jauge]
            wnb_range_model=(self.wnb[G.indexX][-1]-self.wnb[G.indexX][0])/2
            if  ("Lw" and "Hg") not in G.name_spe:
                if wnb_range_model < self.lambda_error*10:
                    wnb_range=self.lambda_error*10
                elif wnb_range_model <= wnb_range:
                    wnb_range=wnb_range_model+self.lambda_error
            peakMax=G.lamb_fit
        else:
            G=self.Gauges[num_jauge]
            peakMax=G.lamb0
        
        dpic=[dp[0] for dp in G.deltaP0i]
        
        if ("Lw" and "Hg") in G.name_spe :
            match = re.search("Lw(\d+)", G.name_spe)
            Dwnb_low= float(match.group(1))
            match = re.search("Hg(\d+)", G.name_spe)#float(G.name_spe[1:3]) avec l aps Low
            Dwnb_hight= float(match.group(1)) #float(G.name_spe[4:6]) avce H pas Hight 
            G.indexX=np.where((self.wnb > (peakMax-Dwnb_low)) & (self.wnb < (peakMax+Dwnb_hight)))[0]
            wnb_range=Dwnb_hight+Dwnb_low
        else:
            G.indexX=np.where((self.wnb > peakMax-(wnb_range+abs(min(min(dpic),0)))) & (self.wnb < peakMax+(wnb_range+max(max(dpic),0))))[0]

        x_sub = np.array(self.wnb[G.indexX])
        y_sub=np.array(y_sub[G.indexX])

        if Delta_ctr is None:
            Delta_ctr=wnb_range/10
        
        if manuel == False:
            indexX=np.where((x_sub > peakMax-Delta_ctr*2) & (x_sub < peakMax+Delta_ctr*2))[0]
            x_max=x_sub[indexX]
            y_max=y_sub[indexX]
            peakMax = x_max[np.argmax(y_max)]
            ampMax = np.max(y_max)
            
        else:
            i0=np.argmin(abs(peakMax0-x_sub))
            peakMax =x_sub[i0]
            ampMax=y_sub[i0]

        G.Update_Fit(crt=peakMax,ampH=ampMax,coef_spe=coef_spe,sigma=sigma,inter=inter,model_fit=model_fit,Delta_ctr=Delta_ctr)
        G.fit=G.model.fit(y_sub, x=x_sub)
        G.model = G.fit.model
        G.Y= G.fit.best_fit + self.blfit[G.indexX]
        G.dY= G.fit.best_fit - y_sub
        G.X=x_sub
        G.lamb_fit =G.fit.best_values[G.name+'_p1center']
        G.bit_fit=True
        #print(G.fit.fit_report())
        #if "DRX" in G.name_spe :      
        #G.pic=[[G.fit.best_values[G.name + '_p'+str(i+1)+'center'],i] for i in range(G.nb_pic) ]
        
        for p in G.pics:
            new_param=p.Out_model(out=G.fit)
            p.Update(ctr=float(new_param[0]),ampH=float(new_param[1]),coef_spe=new_param[3],sigma=float(new_param[2]))     
        
        self.Gauges[num_jauge]=G
    
    def FIT(self,wnb_range=2,coef_spe=None,sigma=None,inter=None,model_fit=None,model_jauge=None):
        for i,G in enumerate(self.Gauges):
            if G.state == "Y":
                try:
                    self.FIT_One_Jauge(num_jauge=i,peakMax0=G.lamb_fit,wnb_range=wnb_range,coef_spe=coef_spe,sigma=sigma,inter=inter,model_fit=model_fit,model_jauge=model_jauge)
                except Exception as e:
                    G.state="IN_NOISE"
                    print("error:",e,"in fit of :",G.name)
            G.bit_fit=True
        for G in self.Gauges:
            G.Calcul(input_spe=self.Gauges,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)
        self.bit_fit=True
    
    def Clear_study(self,num_jauge):
        self.Gauges[num_jauge].study.loc[:, :] = None

    def Calcul_study(self,mini=True):
        self.lambda_error=round((self.wnb[0]-self.wnb[-1])*0.5/len(self.wnb),4)
        if not self.Gauges:
            print("Warning: self.Gauges est vide. Aucun calcul effectué.")
            self.study = pd.DataFrame()
            return
        for i in range(len(self.Gauges)):
            if ("DRX" in self.Gauges[i].name_spe) and (self.bit_fit is True): #
                self.Gauges[i].bit_fit =True
            self.Gauges[i].Calcul(input_spe=self.Gauges,mini=mini,lambda_error=self.lambda_error)
        self.study =pd.concat([x.study for x in self.Gauges ],axis=1)  
    
    def Data_treatement(self,deg_baseline=None,type_filtre=None,param_f=None,print_data=False,ax=None,ax2=None,print_data_QT=False,):
        if deg_baseline is not None:
            self.deg_baseline=deg_baseline
        if param_f is not None:
            self.param_f=param_f

        baseline_fitter = Baseline()
        self.blfit,_ = baseline_fitter.snip(self.spec, max_iter=10*(2+self.deg_baseline), filter_order=2, smooth_half_window=5*self.deg_baseline) #retrait du dark pk.baseline(self.spec,poly_order=self.deg_baseline)
        
        """if self.deg_baseline ==0:
            deltaBG=min(self.spec)-self.blfit[0]
            if  deltaBG <0 :
                self.blfit =np.array(self.blfit) + deltaBG*1.05
        """
        
        if type_filtre is not None:
            self.type_filtre=type_filtre

        if "svg" == self.type_filtre: # Appliquer un filtre de Savitzky-Golay pour lisser le spectre
            self.y_filtre = savgol_filter(self.spec,window_length=self.param_f[0],polyorder=self.param_f[1])
        elif "fft" == self.type_filtre:
            # Transformée de Fourier du signal
            spectre_fft = np.fft.fft(self.spec)
            # Fréquences associées
            frequences_fft_brut = np.fft.fftfreq(len(self.spec), d=self.wnb[1]-self.wnb[0])
            # Filtrage en supprimant les basses fréquences
            cutoff_low = self.param_f[0]  # Fréquence de coupure inférieure
            cutoff_high = self.param_f[1] # Fréquence de coupure supérieure
            # Supprimer les fréquences indésirables dans cette plage
            spectre_fft_brut=copy.deepcopy(spectre_fft)
            spectre_fft[(np.abs(frequences_fft_brut) > cutoff_low) & (np.abs(frequences_fft_brut) < cutoff_high)] = 0
            # Transformée de Fourier inverse pour ré
            self.y_filtre = np.real(np.fft.ifft(spectre_fft))
        else:
            self.y_filtre=self.spec
        self.y_corr = self.y_filtre - self.blfit

        if print_data is True:
            if ax == None:
                figure=False
                fig, (ax,ax2) =plt.subplots(ncols=1,nrows=2,figsize=(8,4),gridspec_kw={'height_ratios': [0.7, 0.3]})      
            else:
                figure=True

            if "fft" != self.type_filtre:
                # Transformée de Fourier du signal
                spectre_fft_brut = np.fft.fft(self.spec)
               
                # Fréquences associées
                frequences_fft_brut = np.fft.fftfreq(len(self.spec), d=self.wnb[1]-self.wnb[0])
                if self.indexX is not None:
                    spectre_fft_fit = np.fft.fft(self.y_corr[self.indexX])
                    frequences_fft_fit = np.fft.fftfreq(len(self.indexX), d=self.wnb[1]-self.wnb[0])
                else:
                    spectre_fft_fit = np.fft.fft(self.y_corr)
                    frequences_fft_fit = np.fft.fftfreq(len(self.wnb), d=self.wnb[1]-self.wnb[0])

                ax2.plot(np.abs(frequences_fft_fit), np.abs(spectre_fft_fit),'-.g', label='Data_fit')
            else:
                ax2.fill_between([self.param_f[0],self.param_f[1]],min(np.abs(spectre_fft)),max(np.abs(spectre_fft)) , color="red", alpha=0.2,label="freq filtré")
            
            ax.plot(self.wnb,self.blfit,'-.', c='g', markersize=1,label="Bkg")
            ax.plot(self.wnb,self.spec,'-',color='gray',markersize=4,label='Brut')
            ax.plot(self.wnb,self.y_corr+self.blfit,'-.+',color='black',markersize=3,label='Corr + bkg')
            ax.minorticks_on()
            ax.tick_params(which='major', length=10, width=1.5, direction='in')
            ax.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)
            ax.set_title(f'$Spectre \Delta\lambda=$'+ str(self.lambda_error))
            ax.set_xlabel(f'$\lambda$ (nm)')
            ax.set_ylabel('U.A.')
            ax.set_xlim([min(self.wnb),max(self.wnb)])
            ax.ticklabel_format(axis="y",style="sci",scilimits=(0, 0))
            ax2.plot(np.abs(frequences_fft_brut), np.abs(spectre_fft_brut),'-.k', label='Data_brut')
            
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.tick_params(which='major', length=10, width=1.5, direction='in')
            ax2.tick_params(which='minor', length=5, width=1.5, direction='in')
            ax2.tick_params(which = 'both', bottom=True, top=True, left=True, right=True)
            ax2.set_xlabel(f'$f$ (Hz)')
            ax2.set_ylabel('Amplitude (u.a.)')
            ax2.set_title('FFT')
            ax2.legend(loc="best")
            #ax2.set_xlim([min(np.abs(frequences_fft)),max(np.abs(frequences_fft))])
            if figure is False:
                plt.show() 
            else:
                return ax,ax2

        elif print_data_QT is True:
            if ax is None or ax2 is None:
                    print("Erreur : les plotWidgets PyQtGraph doivent être fournis pour l'affichage.")
                    return

            # Affichage du spectre
            ax.clear()
            ax.plot(self.wnb, self.spec, pen=pg.mkPen('dimgray', width=2), name='Brut')
            ax.plot(self.wnb, self.blfit, pen=pg.mkPen('g', style=QtCore.Qt.DashLine), name='Bkg')
            ax.plot(self.wnb, self.y_corr + self.blfit, pen=pg.mkPen('k', width=1), name='Corr + bkg')
            ax.setLabel('bottom', 'λ (nm)')
            ax.setLabel('left', 'U.A.')
            ax.setTitle(f'Spectre Δλ={self.lambda_error:.2f}')
            ax.setXRange(min(self.wnb), max(self.wnb))
            ax.addLegend()

            # Affichage FFT
            ax2.clear()
            spectre_fft_brut = np.fft.fft(self.spec)
            frequences_fft_brut = np.fft.fftfreq(len(self.spec), d=self.wnb[1]-self.wnb[0])
            ax2.plot(np.abs(frequences_fft_brut), np.abs(spectre_fft_brut), pen=pg.mkPen('k', style=QtCore.Qt.DashLine), name='Data_brut')

            if self.type_filtre != "fft":
                dwnb=self.wnb[1]-self.wnb[0]
                if self.indexX is not None:
                    spectre_fft_fit = np.fft.fft(self.y_corr[self.indexX])
                    frequences_fft_fit = np.fft.fftfreq(len(self.indexX), d=dwnb)
                else:
                    spectre_fft_fit = np.fft.fft(self.y_corr)
                    frequences_fft_fit = np.fft.fftfreq(len(self.wnb), d=self.wnb[1]-self.wnb[0])
                ax2.plot(np.abs(frequences_fft_fit), np.abs(spectre_fft_fit), pen=pg.mkPen('g', style=QtCore.Qt.DashLine), name='Data_fit')
            else:
                # Tracer la zone filtrée
                region = pg.LinearRegionItem([self.param_f[0], self.param_f[1]], brush=pg.mkBrush(255, 0, 0, 50))
                ax2.addItem(region)

            ax2.setLogMode(x=True, y=True)
            ax2.setLabel('bottom', 'f (Hz)')
            ax2.setLabel('left', 'Amplitude (u.a.)')
            ax2.setTitle('FFT')
            ax2.addLegend()

    def get_local_signal_for_pic(self, pic_target):
        """
        Reconstruit X_local, Y_local = spectre local du pic, en retirant toutes les autres contributions.
        Fonctionne que le fit vienne de lmfit ou de curve_fit.
        """
        if not hasattr(self, "wnb") or not hasattr(self, "y_corr"):
            return None, None

        x_all = np.asarray(self.wnb)
        y_all = np.asarray(self.spec)-np.asarray(self.blfit)

        # Reconstituer le modèle total
        try:
            y_fit_total = np.zeros_like(y_all)
            for gauge in self.Gauges:
                for p in gauge.pics:
                    params_f = p.model.make_params()
                    y_fit_total += p.model.eval(params_f, x=x_all)
        except Exception:
            y_fit_total = np.zeros_like(y_all)

        # Contribution du pic courant
        try:
            y_fit_target = pic_target.model.eval(pic_target.model.make_params(), x=x_all)
        except Exception:
            y_fit_target = np.zeros_like(y_all)

        # Signal local : spectre corrigé - (toutes les autres composantes)
        y_local = y_all - (y_fit_total - y_fit_target)

        # Fenêtre locale ±3σ
        center, sigma = float(pic_target.ctr[0]), float(pic_target.sigma[0])
        mask = (x_all > center - 6*sigma) & (x_all < center + 6*sigma)
        if not np.any(mask):
            mask = slice(None)

        return x_all[mask], y_local[mask]

    def estimate_all_sigma_noise(self,N_MC=0):
        """
        Applique estimate_sigma_center_from_noise() à tous les pics du spectre.
        """
        #print("\n=== Estimation du bruit local pour chaque pic ===")
        for gauge in self.Gauges:
            for p in gauge.pics:
                x_loc, y_loc = self.get_local_signal_for_pic(p)
                if x_loc is None or len(x_loc) < 10:
                    continue
                try:
                    res = p.estimate_sigma_center_from_noise(x_loc, y_loc,N_MC=N_MC)
                    if res:
                        p.sigma_ctr_stat  = res["sigma_ctr"]
                        #print(f"  {p.name:20s}  σ(center)={res['sigma_ctr']:.3e}  SNR={res['SNR']:.1f}")
                except Exception as e:
                    print(f"  ⚠️  {p.name} : erreur estimation bruit ({e})")
        #print("=== Fin estimation bruit ===\n")

"""------------------------------------- CLASSE BANC CED DYNAMIQUE -------------------------------------"""

class CEDd:
    def __init__(self,data,Gauges_init,N=None,data_Oscillo=None,folder_Movie=None,time_index="Channel2",fps=None,fit=False,skiprow_spec=43,reload=False,type_filtre="svg",param_f=[9,2],Kinetic=False):
        if reload == False:
            self.Kinetic=Kinetic
            """------------------------------------- SPECTRE -------------------------------------"""
            self.data_Spectres= pd.read_csv(data, sep='\s+',header=None, skipfooter=0,skiprows=skiprow_spec, engine='python')
            self.Spectra=[]
            self.Gauges_init=Gauges_init
            self.N=N
            self.Summary=pd.DataFrame()
            self.list_lamb0 =[J.lamb0 for J in self.Gauges_init]
            self.list_nspec=[]
            """------------------------------------- OSCILLOSCOPE -------------------------------------"""
            self.time_index=time_index
            self.data_Oscillo=data_Oscillo
            if self.data_Oscillo != None:
                self.data_Oscillo = pd.read_csv(self.data_Oscillo, sep='\s+', skipfooter=0, engine='python')
            self.Time_spectrum=None
            """------------------------------------- MOVIE -------------------------------------"""
            self.folder_Movie=folder_Movie
            self.Movie=None
            self.list_Movie=None
            self.time_movie=[]
            self.fps=fps
            self.CEDd_path="not_save"
            self.Gauges_select=[None for _ in range(len(self.Gauges_init))]
            self.initData(fit,time_index,type_filtre,param_f)
          
        else:
            self.Kinetic=Kinetic
            " A TOUT REFAIR POUR CHANGER LES ANCIEN DONNER"
            """SPECTRE"""
            self.data_Spectres= data.data_Spectres
            self.Spectra=data.Spectra

            if type(data.Gauges_init) is not str :
                self.Gauges_init=data.Gauges_init
            else:
                self.Gauges_init=Gauges_init
            self.N=data.N
            self.Summary=data.Summary
            self.list_lamb0 =[J.lamb0 for J in self.Gauges_init]
            self.list_nspec=data.list_nspec
            """OSCILO"""
            self.time_index=data.time_index
            self.data_Oscillo=data.data_Oscillo
            self.Time_spectrum=data.Time_spectrum
            """ MOVIE"""
            self.folder_Movie=folder_Movie
            self.Movie=data.Movie
            self.list_Movie=data.list_Movie
            self.time_movie=data.time_movie
            self.fps=data.fps
            self.initData(fit,time_index,param_f)

        self.help="CEDd: Etude d'experience de CEDd spectro/Movie prochainement dif X A CODER"
    
    def initData(self,fit=False,time_index="Channel2",type_filtre="svg",param_f=[9,2]):
        if self.N ==None:
            self.N=len(np.array(self.data_Spectres.drop(columns=[0]))[1])
        if fit == True:
            new_jauges=copy.deepcopy(self.Gauges_init)
            for i in tqdm(range(0,self.N), desc="Progress"):
                new_spec=Spectre(self.data_Spectres[0],self.data_Spectres.drop(columns=[0])[i+1],new_jauges,param_f=param_f,type_filtre=type_filtre)
                try:
                    new_spec.FIT()
                    stu=new_spec.study
                    new_jauges=copy.deepcopy(new_spec.Gauges)
                except Exception as e :
                    print("ERROR:",e, "\n Spec n°:",str(i))
                    stu=self.Spectra[-2].study
                    new_jauges=copy.deepcopy(self.Spectra[-2].Gauges)
                    new_spec.Gauges=new_jauges
                    new_spec.study=stu
                self.Summary=pd.concat([self.Summary,pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),stu],ignore_index=False,axis=1)],ignore_index=True)                   
                if self.Kinetic == False:
                    self.Spectra.append(new_spec)
                self.list_nspec.append(int(i))
            print("Spectre loading & fit DONE")
        else:
            for i in range(1, self.N+1):
                self.Spectra.append(Spectre(self.data_Spectres[0],self.data_Spectres.drop(columns=[0])[i],self.Gauges_init,param_f=param_f,type_filtre=type_filtre))
                self.list_nspec.append(int(i))
            print("Spectre loading DONE")

        if time_index is not None and self.data_Oscillo is not None:
            self.Temps_Pression(temps=np.array(self.data_Oscillo['Time']), signale_spec=np.array(self.data_Oscillo[time_index]))
            print("time DONE")
            
        if self.folder_Movie != None :
            if os.path.isdir(self.folder_Movie):
                self.list_Movie=[os.path.join(self.folder_Movie, name) for name in os.listdir(self.folder_Movie) ]
                self.Movie=[]
                for i in range(len(self.list_Movie)):
                    if os.path.isfile(self.list_Movie[i]):
                        try:
                            self.Movie.append(Image.open(self.list_Movie[i]))
                            self.time_movie.append(i/self.fps)
                        except IOError:
                            pass
            if os.path.isfile(self.folder_Movie):
                cap = cv2.VideoCapture(self.folder_Movie)
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                num_frames = int( cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.time_movie=[i/self.fps for i in range(num_frames)]

            print("Movie DONE")

    def FIT(self,end=None,start=0,G=None,wnb_range=3,init=False):
        if G != None:
            self.Gauges_init=G
        elif init==True:
            self.Gauges_init=self.Spectra[start].Gauges
        self.list_nspec=[]
        self.Spectra=[]
        self.Summary=pd.DataFrame()
        if end == None:
            end =self.N
        if end <= self.N :
            new_jauges=copy.deepcopy(self.Gauges_init)
            for i in tqdm(range(start,end), desc="Progress"):
                new_jauges=self.Select_Gauges(i,new_jauges)
                new_spec=Spectre(self.data_Spectres[0],self.data_Spectres.drop(columns=[0])[i+1],new_jauges)                
                new_spec.FIT(wnb_range=wnb_range)
                new_jauges=copy.deepcopy(new_spec.Gauges)
                self.Summary=pd.concat([self.Summary,pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),new_spec.study],ignore_index=False,axis=1)],ignore_index=True)
                if self.Kinetic == False:
                    self.Spectra.append(new_spec)
                self.list_nspec.append(int(i))
            print("Fit DONE")
        else:
            print("end > number of spectre !")

    def Select_Gauges(self,x,Gauges):
        for i ,select_G in enumerate(self.Gauges_select):
            if not select_G:
                Gauges[i].state="Y"
                continue
            for intervale in select_G:
                if  intervale[0] <=x  <= intervale[1]:
                    Gauges[i].state="Y"
                    continue
            else:
                Gauges[i].state="IN_NOISE"
        return Gauges
    
    def FIT_Corr(self,end=None,start=0,select_gauge=False,inter=None):
        if end is None:
            end =self.N
        if end <= self.N :
            if self.Kinetic is True:
                new_jauges=copy.deepcopy(self.Gauges_init)
                for i in tqdm(range(start+1,end), desc="Progress"):
                    if select_gauge is True:
                        new_jauges=self.Select_Gauges(i,new_jauges)
                    new_spec=Spectre(self.data_Spectres[0],self.data_Spectres.drop(columns=[0])[i],new_jauges)
                    new_spec.FIT(model_jauge=new_jauges,inter=inter)
                    new_jauges=copy.deepcopy(new_spec.Gauges)
                    self.Summary.iloc[i]=pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),new_spec.study],ignore_index=False,axis=1)
                print("Fit Corr Fast Kinetic DONE")
            else:
                new_jauges=copy.deepcopy(self.Spectra[start].Gauges)
                for i in tqdm(range(start+1,end), desc="Progress"):
                    if select_gauge is True:
                        new_jauges=self.Select_Gauges(i,new_jauges)
                    self.Spectra[i].FIT(model_jauge=new_jauges,inter=inter)
                    new_jauges=copy.deepcopy(self.Spectra[i].Gauges)
                    self.Summary.iloc[i]=pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),self.Spectra[i].study],ignore_index=False,axis=1)
                print("Fit Corr DONE")
        else:
            print("end > number of spectre !")

    def Corr_Summary(self,num_spec=None,All=True,lambda_error=None,):
        if All ==True :
            for i in range(len(self.Spectra)):
                for j in range(len(self.Gauges_init)):
                    self.Spectra[i].Gauges[j].lamb0=self.Gauges_init[j].lamb0
                    self.Spectra[i].Gauges[j].f_P=self.Gauges_init[j].f_P
                    self.Spectra[i].Gauges[j].inv_f_P=self.Gauges_init[j].inv_f_P
                if lambda_error is not None: #
                    self.Spectra[i].lambda_error=lambda_error
                self.Spectra[i].Calcul_study()
                self.Summary.iloc[i]=pd.concat([pd.DataFrame({"n°Spec": [int(i)]}),self.Spectra[i].study],ignore_index=False,axis=1)
        else:
            if num_spec is not None:
                i=np.where(np.array(self.Summary["n°Spec"]) == num_spec)[0][0]
                self.Spectra[i].Calcul_study()
                self.Summary.iloc[i]=pd.concat([pd.DataFrame({"n°Spec": [int(num_spec)]}),self.Spectra[i].study],ignore_index=False,axis=1)
           
    def Corr_Movie(self,folder_Movie=None,fps=None):
            if fps != None and folder_Movie==None:
                self.fps=fps
            self.time_movie=[i/self.fps for i in range(len(self.list_Movie))]
            if folder_Movie!=None:
                self.folder_Movie=folder_Movie
                self.list_Movie=[os.path.join(self.folder_Movie, name) for name in os.listdir(self.folder_Movie) ]
                self.Movie=[]
                for i in range(len(self.list_Movie)):
                    if os.path.isfile(self.list_Movie[i]):
                        try:
                            self.Movie.append(Image.open(self.list_Movie[i]))
                            self.time_movie.append(i/self.fps)
                        except IOError:
                            pass
                print("Movie DONE")

    def Temps_Pression(self,temps=None, signale_spec=None,Y=None,data_time=None):
        if type(data_time) != type(None) and type(Y) != type(None):
            self.data_Oscillo = pd.read_csv(data_time, sep='\s+', skipfooter=0, engine='python')
            temps, signale_spec =np.array(self.data_Oscillo['Time']), np.array(self.data_Oscillo[Y])

        if type(temps) !=type(None) and type(signale_spec) !=type(None):
            marche_spec=max(signale_spec)/2

            taille=len(signale_spec)
            delais=taille/10
            stop = False
            up = 0
            ti = 0
            t = []
            i=0
            while stop == False and i < taille: 
                s = 0
                if signale_spec[i] > marche_spec:
                    ti = i
                    up=0
                    while signale_spec[i] > marche_spec:
                        up += 1
                        i += 1
                        if i == taille: 
                            break
                    t.append((temps[i]+temps[ti])/2 )
                    i += up
                    s = 0

                else:
                    i += 1
                    s += 1
                if s > delais or i == len(signale_spec):
                    stop = True        
            self.Time_spectrum=t

        else:
            print("CALCULE FAILED NO DATA")

    def Print(self,num_spec=0,data=[]):
        if data ==[]:
            self.Spectra[num_spec].Print()
        else:
            for i in range(len(data)):
                index=np.where(np.array(self.Summary[data[i]]) != None)[0]
                plt.plot(self.Summary["n°Spec"][index],self.Summary[data[i]][index],'+-',label=data[i])
            plt.legend()
            plt.show()

    def Play_Movie(self):
        if self.folder_Movie is None:
            return print("No Folder Movie !")
        # Lecture vidéo avec OpenCV
        cap = cv2.VideoCapture(self.folder_Movie)
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Texte pour afficher le temps et le numéro de frame
        info_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color="white", fontsize=12, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))

        # Fonction pour mettre à jour le texte du temps et le numéro de la frame
        def update_info_text(frame_number):
            time_in_seconds = frame_number / fps
            info_text.set_text(f"Temps : {time_in_seconds:.3f} s | Frame : {frame_number}/{num_frames - 1}")

        # Fonction pour lire une image donnée de la vidéo
        def read_frame(frame_number):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            else:
                return None

        # Initialisation de la première image
        current_frame = 0
        img = read_frame(current_frame)

        # Affichage initial
        fig, ax = plt.subplots()
        image = ax.imshow(img,vmax=np.max(img),vmin=np.min(img))

        # Fonction de mise à jour pour slider
        def update(val):
            global current_frame
            current_frame = int(slider.val)
            img = read_frame(current_frame)
            if img is not None:
                image.set_data(img)
                image.set_clim(vmax=np.max(img),vmin=np.min(img))
                update_info_text(current_frame)
            plt.draw()

        # Fonction pour aller à l'image suivante
        def next_frame(event):
            global current_frame
            if current_frame < num_frames - 1:
                current_frame += 1
                slider.set_val(current_frame)

        # Fonction pour aller à l'image précédente
        def prev_frame(event):
            global current_frame
            if current_frame > 0:
                current_frame -= 1
                slider.set_val(current_frame)

        # Barre de défilement pour choisir les images
        slider_ax = plt.axes([0.125, 0.92, 0.7, 0.03])
        slider = Slider(slider_ax, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
        slider.on_changed(update)

        # Bouton suivant
        next_button = plt.axes([0.85, 0.9, 0.08, 0.04])
        next_button = Button(next_button, 'Next =>')
        next_button.on_clicked(next_frame)

        # Bouton précédent
        prev_button = plt.axes([0.70, 0.9, 0.08, 0.04])
        prev_button = Button(prev_button, '<= Prev')
        prev_button.on_clicked(prev_frame)
        plt.show()
        cap.release()

    def Help(self):
        print("sorry")

class CED_DRX:
    def __init__(self,data_drx,calib,E,data_oscillo=None,time_index=None,flip_axis=None,param_f=[9,2],deg_baseline=1):
        self.Spectra=[]
        self.Summary=pd.DataFrame()
        self.time_index=time_index
        self.data_oscillo=data_oscillo
        self.data_drx=data_drx
        self.Time_spectrum=None
        self.calib = calib
        self.note=""
        self.sigma_dist=0
        if isinstance(data_drx, list):
            # Cas : liste de fichiers
            fichiers_tries = sorted(data_drx, key=extraire_numero)

        elif os.path.isfile(data_drx):
            # Cas : fichier unique multi-frames
            data_t = fabio.open(data_drx)
            n_frames = data_t.nframes
            for i in range(n_frames):
                if flip_axis is not None:
                    if flip_axis==2:
                        data=np.flip(data_t.getframe(i).data)
                    else:    
                        data=np.flip(data_t.getframe(i).data,axis=flip_axis)
                else:
                    data=data_t.getframe(i).data
                tth, intens = Integrate_DRX(
                    data, self.calib.mask, self.calib.ai,
                    theta_range=self.calib.theta_range
                )
                self.Spectra.append(Spectre(tth, intens, E=E,param_f=param_f,deg_baseline=deg_baseline))

        elif os.path.isdir(data_drx):
            # Cas : dossier contenant des fichiers
            fichiers_tries = [os.path.join(data_drx, f)
                            for f in sorted(os.listdir(data_drx), key=extraire_numero)]
        else:
            raise ValueError(f"data_drx doit être un fichier, un dossier, ou une liste, pas {type(data_drx)}")

        # Si fichiers_tries défini, on les lit
        if 'fichiers_tries' in locals():
            for f in fichiers_tries:
                if flip_axis is not None:
                    if flip_axis==2:
                        data=np.flip(fabio.open(f).data)
                    else:
                        data=np.flip(fabio.open(f).data,axis=flip_axis)   
                else:
                    data=fabio.open(f).data
                tth, intens = Integrate_DRX(
                    data, self.calib.mask, self.calib.ai,
                    theta_range=self.calib.theta_range
                )
                self.Spectra.append(Spectre(tth, intens, E=E))



        if data_oscillo is None:
            self.data_oscillo = None
            self.Time_spectrum = None
            print("No Oscillo Data")
        else:
            if isinstance(data_oscillo, pd.DataFrame):
                df = data_oscillo.copy()

            elif isinstance(data_oscillo, (str, Path)):
                df = pd.read_csv(str(data_oscillo), sep=r"\s+", skipfooter=0, engine="python")

            elif isinstance(data_oscillo, dict):
                df = pd.DataFrame(data_oscillo)

            else:
                raise TypeError(
                    "data_oscillo doit être None, un chemin (str/Path), un pandas.DataFrame, ou un dict de colonnes."
                )

            if df.shape[1] < 1:
                raise ValueError("data_oscillo ne contient aucune colonne (temps manquant).")

            self.data_oscillo = df

            time_col = df.columns[0]
            temps = df[time_col].to_numpy()

            if time_index is None:
                sig = None
            elif isinstance(time_index, int):
                sig = df.iloc[:, time_index].to_numpy()
            else:
            # time_index censé être un nom de colonne
                if time_index not in df.columns:
                    raise KeyError(f"Colonne '{time_index}' introuvable. Colonnes dispo: {list(df.columns)}")
                sig = df[time_index].to_numpy()
            self.Temps_Pression(temps=temps, signale_spec=sig)
            print("Oscillo DONE nb time:", len(self.Time_spectrum))


    def Print(self,num_spec=0,data=[],Oscilo=False):
        if data ==[] and Oscilo ==False:
            self.Spectra[num_spec].Print()
        elif Oscilo ==True:
            #plt.plot(self.data_oscillo["Time"],savgol_filter(self.data_oscillo["Channel1"],9,2),"darkorange",label="Channel1")
            plt.plot(self.data_oscillo["Time"],savgol_filter(self.data_oscillo["Channel2"],9,2),"darkred",label="Channel2")
            plt.plot(self.data_oscillo["Time"],savgol_filter(self.data_oscillo["Channel3"],9,2),"darkblue",label="Channel3")
            #plt.plot(self.data_oscillo["Time"],savgol_filter(self.data_oscillo["Channel4"],9,2),"darkgreen",label="Channel4")
            for t in self.Time_spectrum:
                # trouver l’index du point X le plus proche de t
                idx = np.argmin(np.abs(self.data_oscillo["Time"] - t))
                               
                # tracer une petite barre verticale (ici de 0 jusqu’à y_val)
                plt.plot(self.data_oscillo["Time"][idx],self.data_oscillo[self.time_index][idx]+0.1,"vk",markersize=15)
            plt.legend()
            plt.show()
        else:
            for i in range(len(data)):
                index=np.where(np.array(self.Summary[data[i]]) != None)[0]
                plt.plot(self.Summary["n°Spec"][index],self.Summary[data[i]][index],'v-',label=data[i])
            plt.legend()
            plt.show()

    def fit_selected_spectra(self, index_start=0,index_stop=None,best_ind=None,NGEN=300, MUTPB=0.4, CXPB=0.4, POPINIT=150, pressure_range=[0, 20], max_ecart_pressure=1, max_elements=3, tolerance=0.1, print_process=True, custom_peak_params=None):
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
                    pressure_range=pressure_range,
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

    def Corr_Summary(self, num_spec=None,N_MC=0,verbose=False):
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
                    print(f"⚠️ Estimation bruit échouée sur spectre {i}: {e}")

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
                        G.Calcul(input_spe=spec.Gauges,
                                lambda_error=spec.lambda_error,
                                verbose=verbose)
                    except Exception as e:
                        print(f"⚠️ Gauge {G.name}: erreur Calcul() ({e})")

                # 3️⃣ Mets à jour le DataFrame du spectre
                try:
                    spec.study = pd.concat([x.study for x in spec.Gauges], axis=1)
                    spec.bit_fit = True
                except Exception as e:
                    print(f"⚠️ Fusion study spectre {i}: {e}")
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
                        print(f"Impossible d'appliquer correction sur {col}: {e}")
            print(f"📋 Corrections appliquées : {self.note}")

        if verbose:
            print("\n✅ Correction Summary terminée.\n")

    def Add_Static(self,file_data,time=""):
        data=pd.read_csv(file_data, sep='\s+',header=None,skiprows=23, engine='python')
        E=self.Spectra[0].E
        dt=self.Time_spectrum[2]-self.Time_spectrum[1]
        if time in ("s","start"):

            self.Spectra.insert(0,Spectre(data[0],data[1],E=E))
            self.Time_spectrum.insert(0,self.Time_spectrum[0]-2.5*dt)
        elif time in ("e","end"):
            self.Spectra.append(Spectre(data[0],data[1],E=E))
            self.Time_spectrum.append(self.Time_spectrum[-1]+2.5*dt)

    def Temps_Pression(self,temps=None, signale_spec=None,Y=None,data_time=None):
        if type(data_time) != type(None):
            self.data_oscillo = pd.read_csv(data_time, sep='\s+', skipfooter=0, engine='python')
        if type(Y) != type(None):
            self.time_index=Y

        temps, signale_spec =np.array(self.data_oscillo['Time']), np.array(self.data_oscillo[self.time_index])- pk.baseline(np.array(self.data_oscillo[self.time_index]),50)

        if type(temps) !=type(None) and type(signale_spec) !=type(None):
            marche_spec=max(signale_spec)/4

            taille=len(signale_spec)-1
            delais=taille/10
            stop = False
            up = 0
            ti = 0
            t = []
            i=0
            while stop == False and i < taille: 
                s = 0
                if signale_spec[i] > marche_spec:
                    ti = i
                    while signale_spec[i] > marche_spec:
                        i += 1
                        if i == taille: 
                            break
                    t.append((temps[i]+temps[ti])/2 )
                    i += 1
                    s = 0

                else:
                    i += 1
                    s += 1
                if s > delais or i == len(signale_spec) or len(t)== len(self.Spectra):
                    stop = True        
            self.Time_spectrum=t

        else:
            print("CALCULE FAILED NO DATA")

    def out_data(self):
        pass

    def sigma_2theta_from_distance(self, tth_deg):
        """
        Calcule σ(2θ) en degrés due à l’incertitude sur la distance détecteur.
        """
        if not hasattr(self, "calib") or not hasattr(self, "sigma_dist"):
            return 0.0

        tth_rad = np.deg2rad(float(tth_deg))  # convertit 2θ → rad

        # d(2θ)/dD = -0.5 * sin(2*2θ) / D
        d2tth_dD = -0.5 * np.sin(2 * tth_rad) / self.calib.ai.dist

        sigma_2tth_rad = abs(d2tth_dD) * self.sigma_dist
        return float(np.rad2deg(sigma_2tth_rad))  # renvoie en degrés

""" ------------------------------------- CLASS DE FIT  -------------------------------------"""
  

""" ------------------------------------- CLASS OSCILOSCOPE LECROY-------------------------------------"""
class Oscillo_autosave:
    def __init__(self,IP="100.100.143.2",folder=r"F:\Aquisition_Banc_CEDd\Aquisition_LECROY_Banc_CEDd"):
        self.scope = lecroyscope.Scope(IP)  # IP address of the scope
        self.folder=folder
        print(f"Scope ID: {self.scope.id}")
        print("dossier d'enregistrement"+self.folder)

    def Print(self):
        # Afficher les nouvelles traces sur le graphique
        trace_group = self.scope.read(1, 2, 3, 4)
        for i in range(1, len(trace_group) + 1):
            plt.plot(trace_group[i].x, trace_group[i].y, '.-')
        plt.xlabel('Time')
        plt.ylabel(f'Channel {i}')
        plt.title(f'Trace for Channel {i}')
        plt.grid(True)
        plt.show()


    def save(self,name):
        trace_group = self.scope.read(1, 2, 3, 4)
        time = trace_group.time  # time values are the same for all traces
        df = pd.DataFrame({"Time" :pd.Series(time), 
                   "Channel1" :pd.Series(trace_group[1].y),
                   "Channel2" :pd.Series(trace_group[2].y),
                   "Channel3" :pd.Series(trace_group[3].y),
                   "Channel4" :pd.Series(trace_group[4].y),
                  })
        file_path =os.path.join(self.folder,name)
        if file_path:
            with open(file_path, 'w') as file2write:
                file2write.write(df.to_string())
            print(f"Data saved to {file_path}")



# Obtenir le chemin du fichier actuel (programme en cours d'exécution)
file_path = __file__

# Obtenir la dernière date de modification du fichier
modification_time = os.path.getmtime(file_path)

# Convertir l'horodatage en une date lisible
last_modified_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modification_time))

note_version="CL_FD_Update:"


print("LOADED:",note_version+ last_modified_date)



