import numpy as np
from math import sqrt
from lmfit.models import PseudoVoigtModel, MoffatModel, SplitLorentzianModel, Pearson4Model, GaussianModel
from scipy.special import gamma, beta
from pynverse import inversefunc
import pandas as pd
import re
from scipy.optimize import minimize

from .pic import Pics
from .pressure_law import *


class Gauge:
    def __init__(self,name="",lamb0=None,nb_pic=None,X=None,Y=None,deltaP0i=[],spe=None,f_P=None,name_spe=''):
        self.lamb0=lamb0
        self.nb_pic=nb_pic
        self.deltaP0i=deltaP0i
        self.pics=[]
        self.name=name
        self.X=X
        self.Y=Y
        self.dY=None
        self.spe=spe
        self.f_P=f_P
        self.inv_f_P=None
        self.P=0
        self.T=0
        if f_P is not None:
            self.inv_f_P=inversefunc((lambda x :f_P(x)),domain=[10,1000])
        self.name_spe=name_spe
    
        self.bit_model=False
        self.model=None
        self.color_print=[None,None]
        self.bit_fit=False
        self.lamb_fit=None
        self.indexX=None
        self.fit="Fit Non effectué"
        self.study=pd.DataFrame()
        self.study_add=pd.DataFrame()
        
        self.state="Y" #Y, IN_NOISE , N 1,0,None
        self.Jauge_choice=["Ruby","Sm","SrFCl","Rhodamine6G"]

        self.help="Gauge: G de spectroscopie (Ruby,Samarium,SrFCL) prochainement jauge X A CODER \n #Calib1_251024_16h39_d11_04t11_26 Coefficients ajustés X=x_lamb0 et P=aX**2+bX+c: a = 0, b = 0.04165619359945429,,lambda0 = 551.3915961705188 "

        if "Ruby" in self.name:
            self.Ruby()
        elif ("Sm" in self.name) or  ("Samarium" in self.name):
            self.Sm()
        elif "SrFCl" in self.name:
            self.SrFCl()
        elif "Rhodamine" in self.name:
            self.Rhodamine6G()
        
    def Init_perso(self):
        self.pics=[]
        for i in range(self.nb_pic):
            if "Sig" in self.name_spe:
                match = re.search("Sig(\d+)", self.name_spe)
                sigma= float(match.group(1))
            else:
                sigma=0.25
            if "Mod:" in self.name_spe:
                match = re.search(r"Mod:(\w+)", self.name_spe)
                model_fit= str(match.group(1))
            else:
                model_fit="PseudoVoigt"
            new_pics=Pics(name=self.name + '_p'+str(i+1),ctr=self.lamb0+self.deltaP0i[i][0],sigma=sigma,model_fit=model_fit)
            self.pics.append(new_pics)
            if i == 0:
                self.model = new_pics.model
            else:
                self.model = self.model + new_pics.model
        
        self.bit_model=True

    #JAUGE PREREMPLIS
    def Ruby(self):
        self.lamb0=694.4
        self.nb_pic=2
        self.name="Ruby"
        self.name_spe="Ru"
        self.deltaP0i=[[0,1],[-1.5,0.75]]
        self.color_print=['r',['r','m']]

        if self.f_P == None:
            self.f_P=Ruby_2020_Shen
            self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[690,750])

        self.Init_perso()

    def Calcul_Ruby(self,input_spe):
        if 'RuSmT' in self.name_spe:
            if input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr != None:
                sigmals=input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr
            else :
                sigmals=0
            
            if self.fit.params[self.name+'_p1center'].stderr != None:
                sigmal=self.fit.params[self.name+'_p1center'].stderr
            else :
                sigmal=0
            
            T , sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit,input_spe[self.spe].lamb_fit,lamb0S=input_spe[self.spe].lamb0,lamb0R=self.lamb0,sigmalambda=[sigmal,sigmals])
            self.study_add=pd.DataFrame(np.array([[self.deltaP0i[1][0],T,sigmaT]]),columns=['Deltap12','T','sigmaT'])
        else:
            self.study_add=pd.DataFrame(np.array([self.deltaP0i[1][0]]),columns=['Deltap12'])
                
    def Sm(self,all=False):
        self.lamb0=685.39
        self.nb_pic=1
        self.name="Sm"
        if all ==False:
            self.deltaP0i=[[0,1]]
        else:
            print("to code")

        if self.f_P == None:
            self.f_P= Sm_2015_Rashenko
            self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[683,730])
        self.color_print=['b',['b']]
        self.Init_perso()
    
    def SrFCl(self,all=False): 
        self.lamb0=685.39
        self.nb_pic=1
        self.name="SrFCl"
        if all ==False:
            self.deltaP0i=[[0,1]]
        else:
            print("to code")
        if self.f_P is None:
            self.f_P=SrFCl
            self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[683,730])

        self.color_print=['k',['k']]
        self.Init_perso()

    def Rhodamine6G(self):
        self.lamb0=550
        self.nb_pic=2
        self.name="Rhodamine6G"
        self.deltaP0i=[[0,1],[52,0.5,4]]
        self.color_print=['lime',['lime',"darkgreen"]]
        self.name_spe="Lw50Hg150Sig20Mod:Gaussian"

        if self.f_P == None:
            self.f_P=Rhodamine_6G_2024_Dembele
            self.inv_f_P=inversefunc((lambda x :self.f_P(x)),domain=[530,600])

        self.Init_perso()

    #CALCULE
    def Calcul(self,input_spe=None,mini=True,lambda_error=0,verbose=False):
        if self.bit_fit is False :
            return print("NO FIT of ",self.name,"do it !")
        else:
            if "DRX" in self.name_spe :
                if self.state == "IN_NOISE":
                    self.a,self.b,self.c,self.rca,self.V,self.P=self.Element_ref.A,self.Element_ref.B,self.Element_ref.C,self.Element_ref.rCA,self.Element_ref.V0,0
                else:
                    self.CALCUL(mini=mini,verbose=verbose)
                self.study =pd.concat([pd.DataFrame(np.array([[self.a,self.b,self.c,self.rca,self.V,self.P]]) , columns=['a_'+self.name,'b_'+self.name,'c_'+self.name,'c/a_'+self.name,'V_'+self.name,'P_'+self.name]),self.study_add],axis=1)
                return           
            self.lamb_fit=round(self.pics[0].ctr[0],3)

            sigmal = lambda_error
            if self.state == "Y":
                par = self.name + '_p1center'
                if (self.fit is not None) and (par in self.fit.params) and (self.fit.params[par].stderr is not None):
                    if self.fit.params[par].stderr > lambda_error:
                        sigmal = float(self.fit.params[par].stderr)
                # fallback si stderr absent/faible mais estimation bruit dispo
                if (sigmal == lambda_error) and hasattr(self.pics[0], "sigma_ctr_total") and self.pics[0].sigma_ctr_total :
                    sigmal = float(self.pics[0].sigma_ctr_total )
                self.fwhm=round(self.pics[0].sigma[0],5)
                #self.fwhm=self.fit.best_values[self.name+'_p1sigma']

                for i in range(1,len(self.deltaP0i)):
                    #self.deltaP0i[i][0]= round(self.fit.best_values[self.name+'_p'+str(i+1)+'center'] - self.lamb_fit,3)
                    self.deltaP0i[i][0]= round(self.pics[i].Out_ctr()- self.lamb_fit,5)
                self.P , self.sigmaP = self.f_P(self.lamb_fit,self.lamb0,sigmalambda=sigmal)

            else:
                self.fwhm,self.P,self.sigmaP =0.1 ,0 ,0

            if 'Ru' in self.name_spe:
                self.study_add=pd.DataFrame(np.array([abs(self.deltaP0i[1][0])]),columns=['Deltap12'])
            
                if 'SmT' in self.name_spe :
                    sigmals= lambda_error
                    if self.state == "Y":
                        if input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr != None:
                            if input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr > lambda_error:
                                sigmals=input_spe[self.spe].fit.params[input_spe[self.spe].name+'_p1center'].stderr                        

                        T , sigmaT = T_Ruby_Sm_1997_Datchi(self.lamb_fit,input_spe[self.spe].lamb_fit,lamb0S=input_spe[self.spe].lamb0,lamb0R=self.lamb0,sigmalambda=[sigmal,sigmals])
                    else:
                        T,sigmaT=273,0
                    self.study_add=pd.DataFrame(np.array([[self.deltaP0i[1][0],T,sigmaT]]),columns=['Deltap12','T','sigma_T'])


            self.study =pd.concat([pd.DataFrame(np.array([[self.P,self.sigmaP,self.lamb_fit,self.fwhm,self.state]]) , columns=['P_'+self.name,'sigma_P_'+self.name,'lambda_'+self.name,'fwhm_'+self.name,"State_"+self.name]),self.study_add],axis=1)
    
    def Clear(self,c=None):
        self.study.loc[:, :] = c
        self.bit_model=False
        self.model=None
        self.bit_fit=False
        self.lamb_fit=None
        self.indexX=None
        self.fit="Fit Non effectué"
    
    #MODIFICATIOND E PIQUE
    def Update_Fit(self,crt,ampH,coef_spe=None,sigma=None,inter=None,model_fit=None,Delta_ctr=None):
        for i in range(len(self.pics)):
            if len(self.deltaP0i[i]) >2:
                if sigma is None:
                    sigma= self.pics[0].sigma[0]
                sigma=sigma*self.deltaP0i[i][2]
            self.pics[i].Update(ctr=crt+self.deltaP0i[i][0],ampH=ampH*self.deltaP0i[i][1],coef_spe=coef_spe,sigma=sigma,inter=inter,model_fit=model_fit,Delta_ctr=Delta_ctr)
            if i == 0:
                    self.model = self.pics[i].model
            else:
                self.model = self.model + self.pics[i].model

    def Update_model(self):
        for i in range(len(self.pics)):
            if i == 0:
                    self.model = self.pics[i].model
            else:
                self.model = self.model + self.pics[i].model

class Element(Gauge):
    def __init__(self,Element_ref,name):
        super().__init__()
        self.Element_ref=Element_ref
        self.E=Element_ref.E
        self.nb_pic=self.Element_ref.save_var.count(True)
        self.pic_ref=[]
        self.deltaP0i=[]
        self.a=Element_ref.A
        self.sigma_a=0
        self.b=Element_ref.B
        self.sigma_b=0
        self.c=Element_ref.C
        self.sigma_c=0
        self.alpha=Element_ref.ALPHA
        self.beta=Element_ref.BETA
        self.gamma=Element_ref.GAMMA
        self.rca=Element_ref.rCA
        self.l_a=[]
        self.l_b=[]
        self.l_c=[]
        self.V=Element_ref.V
        self.sigma_V=0
        self.P=Element_ref.P_start
        self.sigma_P=0
        self.P_start=Element_ref.P_start
        self.T=Element_ref.T
        self.sigma_T=0
        self.maille= Element_ref.symmetrie
        self.pic_mini=None
        self.name=name
        self.name_spe="DRX"
        #self.pic=[]
        self.l_dhkl=[] # [0]: dhkl "010" [1]: index [2]: dhkl "2.3"(A) 
        self.init_ref()
    
    def init_ref(self,verbose=False):
        self.pic_ref=[]
        text_print=""
        self.nb_pic=self.Element_ref.save_var.count(True)
        i=0
        n_p=0
        if self.Element_ref.thetas_PV==[]:
            self.Element_ref.Eos_Pdhkl(P=0)
        self.pics=[]
        self.deltaP0i=[]
        while n_p < self.nb_pic:
            if self.Element_ref.save_var[i] is True:
                name_dhkl="D"+"".join(map(str,self.Element_ref.thetas_PV[i][2:])) #"D"+ str(int(self.Element_ref.Dhkl["h"][i]))  +str(int(self.Element_ref.Dhkl["k"][i]))+str(int(self.Element_ref.Dhkl["l"][i]))
                theta=self.Element_ref.thetas_PV[i][0]
                self.pic_ref.append([name_dhkl,theta])
                text_print=text_print+" ; "+name_dhkl 
                new_pics=Pics(name=self.name + f'_p{name_dhkl}',ctr=theta,model_fit="PearsonIV",coef_spe=[1.1,0])
                self.pics.append(new_pics)
                if n_p == 0:
                    self.model = new_pics.model
                    self.lamb_fit =theta
                    self.deltaP0i=[[0,1]]
                else:
                    self.model = self.model + new_pics.model
                    self.deltaP0i.append([theta-self.lamb_fit,int(self.Element_ref.Dhkl["I"][i])/100])
                n_p+=1
            i+=1
        if verbose:
            print(" Dhkl ref refresh :",text_print)
     
    def init_l_dhkl(self):
        self.l_dhkl=[((1239.8/self.E)*1e-9)/(2*np.sin(np.pi*pic.Out_ctr()/360))*1e10 for pic in self.pics]

    def minimisation(self,verbose=False):
        if verbose:
            print("- - - - - - minimisation - - - - - -")
        if self.a is None or self.b is None or self.c is None:
            print("Enter a0 b0 c0 in self.a ,self.b ,self.c")
            return

        #self.init_l_dhkl()

        # --- préparation des valeurs expérimentales et poids ---
        amps = [p.ampH[0] for p in self.pics]
        amp_max = max(amps) if amps else 1.0

        Dhkl_exp, weights ,sigma_d_list= [], [],[]

        lam = (1239.8 / self.E) * 1e-9  # longueur d’onde en m
        amp_max = max([p.ampH[0] for p in self.pics]) if self.pics else 1.0

        for p in self.pics:
            # angle 2θ -> θ en rad
            two_theta = np.deg2rad(p.Out_ctr())  # p.Out_ctr() = 2θ en degrés
            theta = two_theta / 2.0

            # d expérimental (Å)
            d_exp = (lam / (2*np.sin(theta))) * 1e10
            Dhkl_exp.append(d_exp)

            if hasattr(p, "sigma_ctr_total"):
                sigma_2theta = np.deg2rad(float(p.sigma_ctr_total  ))
            else:
                low, high = p.ctr[1]
                sigma_2theta = np.deg2rad((high - low) / 2.0) if high > low else 1e-4

            sigma_theta = sigma_2theta / 2.0

            # propagation : dd/dθ
            dd_dtheta = -lam * np.cos(theta) / (2 * (np.sin(theta)**2)) * 1e10
            sigma_d = abs(dd_dtheta) * sigma_theta
            sigma_d = max(sigma_d, 1e-4)  # plancher de sécurité

            # poids avec amplitude relative
            amp_rel = p.ampH[0] / amp_max if amp_max > 0 else 1.0
            weights.append(amp_rel / (sigma_d**2))
            sigma_d_list.append(sigma_d)

        Dhkl_exp = np.array(Dhkl_exp)
        weights  = np.array(weights)
        sigma_d_list =np.array(sigma_d_list)

        # --- dictionnaire des fonctions de calcul d ---
        def d_tetragonal(h, k, l, a, c):
            return a * np.sqrt(1.0 / ((h**2 + k**2) + (a*l/c)**2))

        def d_hexagonal(h, k, l, a, c):
            return np.sqrt(1.0 / ((4/3*(h**2 + k**2 + h*k)/a**2) + (l**2 / c**2)))

        def d_cubic(h, k, l, a):
            return a * np.sqrt(1.0 / (h**2 + k**2 + l**2))

        def d_orthorhombic(h, k, l, a, b, c):
            return 1.0 / np.sqrt((h/a)**2 + (k/b)**2 + (l/c)**2)

        def d_rhombohedral(h, k, l, a, alpha):
            alpha_rad = alpha * np.pi / 180
            num = 1 - 3*np.cos(alpha_rad)**2 + 2*np.cos(alpha_rad)**3
            den = ((h**2 + k**2 + l**2) * np.sin(alpha_rad)**2
                - 2*(np.cos(alpha_rad) - np.cos(alpha_rad)**2)*(k*l + l*h + h*k))
            return a * np.sqrt(num / den)
        
        def sigma_a_cubic_from_sigma_d(h, k, l, sigma_d):
            H = h*h + k*k + l*l
            return (H**0.5) * sigma_d

        def sigma_a_tetra_from_sigma_d(h, k, sigma_d):
            Hhk = h*h + k*k
            return (Hhk**0.5) * sigma_d

        def sigma_c_tetra_from_sigma_d(l, sigma_d):
            return l * sigma_d

        def sigma_a_hexa_from_sigma_d(h, k, sigma_d):
            Hhk = h*h + h*k + k*k
            return ((4/3)**0.5 * (Hhk**0.5)) * sigma_d

        def sigma_c_hexa_from_sigma_d(l, sigma_d):
            return l * sigma_d


        # --- fonction pour extraire hkl ---
        def get_hkl(p):
            if isinstance(p[0], str):  # cas RHOMBO
                match = re.match(r'D([-+]?\d+)([-+]?\d+)([-+]?\d+)', p[0])
                return int(match.group(1)), int(match.group(2)), int(match.group(3))
            else:
                return int(p[0][1]), int(p[0][2]), int(p[0][3])

        # --- fonction utilitaire pour récupérer les incertitudes ---
        def get_uncertainties(result):
            if hasattr(result, "hess_inv"):
                try:
                    cov = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else result.hess_inv
                    return np.sqrt(np.diag(cov))
                except Exception:
                    return None
            return None

        # --- mapping selon la maille ---
        lattice_map = {
            "TETRAGONAL": (d_tetragonal, [self.a, self.c]),
            "HEXAGONAL": (d_hexagonal, [self.a, self.c]),
            "CUBIC": (d_cubic, [self.a]),
            "ORTHORHOMBIC": (d_orthorhombic, [self.a, self.b, self.c]),
            "RHOMBOHEDRAL": (lambda h,k,l,a: d_rhombohedral(h,k,l,a,self.alpha),
                            [self.a if self.a is not None else self.Element_ref.A])
        }

        # --- choix du modèle ---
        for key, (d_func, param0) in lattice_map.items():
            if key in self.maille:
                # fonction objectif
                def error(params):
                    d_calc_list = [d_func(*get_hkl(p), *params) for p in self.pic_ref]
                    diff = Dhkl_exp - np.array(d_calc_list)
                    return np.sum(weights * diff**2)

                # minimisationSSS
                min_res = minimize(error, param0, method="BFGS")
                errs = get_uncertainties(min_res)
        
                # mise à jour des paramètres
                if key == "TETRAGONAL":
                    self.a = self.b = round(min_res.x[0], 3)
                    self.c = round(min_res.x[1], 3)
                    self.rca = self.c / self.a
                    if errs is not None and np.any(errs < 0.5):
                        self.sigma_a = self.sigma_b = errs[0]
                        self.sigma_c = errs[1]
                    else:
                        # cas a depuis un hk0
                        hkls = [get_hkl(p) for p in self.pic_ref]
                        self.sigma_a=self.sigma_b=self.sigma_c=0
                        for (h,k,l), sd in zip(hkls, sigma_d_list):
                            if l == 0:  # pic hk0
                                self.sigma_a = self.sigma_b = sigma_a_tetra_from_sigma_d(h,k,sd)
                            if h == k == 0:  # pic 00l
                                self.sigma_c = sigma_c_tetra_from_sigma_d(l,sd)
                        

                elif key == "HEXAGONAL":
                    self.a = self.b = round(min_res.x[0], 3)
                    self.c = round(min_res.x[1], 3)
                    self.rca = self.c / self.a
                    if errs is not None and np.any(errs < 0.5):
                        self.sigma_a = self.sigma_b = errs[0]
                        self.sigma_c = errs[1]
                    else:
                        hkls = [get_hkl(p) for p in self.pic_ref]
                        self.sigma_a=self.sigma_b=self.sigma_c=0
                        for (h,k,l), sd in zip(hkls, sigma_d_list):
                            if l == 0:  # hk0
                                self.sigma_a = self.sigma_b = sigma_a_hexa_from_sigma_d(h,k,sd)
                            if h == k == 0:  # 00l
                                self.sigma_c = sigma_c_hexa_from_sigma_d(l,sd)

                elif key == "CUBIC":
                    self.a = self.b = self.c = round(min_res.x[0], 3)
                    self.rca = 1
                    if errs is not None and np.any(errs < 0.5):
                        self.sigma_a = self.sigma_b = self.sigma_c = errs[0]
                    else:
                        h,k,l = get_hkl(self.pic_ref[0])
                        self.sigma_a = self.sigma_b = self.sigma_c = sigma_a_cubic_from_sigma_d(h,k,l, sigma_d_list[0])

                elif key == "ORTHORHOMBIC":
                    self.a = round(min_res.x[0], 3)
                    self.b = round(min_res.x[1], 3)
                    self.c = round(min_res.x[2], 3)
                    self.rba = self.b / self.a
                    self.rca = self.c / self.a
                    if errs is not None:
                        self.sigma_a, self.sigma_b, self.sigma_c = errs

                elif key == "RHOMBOHEDRAL":
                    self.a = self.b = self.c = round(min_res.x[0], 3)
                    self.rca = 1
                    if errs is not None:
                        self.sigma_a = self.sigma_b = self.sigma_c = errs[0]

                # affichage final
                vals = [self.a, self.b, self.c]
                sigmas = [getattr(self, f"sigma_{x}", None) for x in ["a", "b", "c"]]
                txt = "  ".join([
                    f"{par} ± {sig:.4f}" if sig is not None else f"{par}"
                    for par, sig in zip(vals, sigmas)
                ])
                if verbose:
                    print(f"MINIMISATION DONE ({key}): a,b,c = {txt}")
                break

    def calcul_abc(self,verbose=False):
        if verbose:
            print("- - - - - - calcul_abc - - - - - -")
        if type(self.maille)==type(None):
            self.maille= self.Element_ref.symmetrie
        def dhkl2(theta):
            return (((1239.8/self.E)*1e-9)/(2*np.sin(np.pi*theta/360)))**2

        def f1(d2,h,k,l,coefh=1,coefk=1,coefl=1):
            return round(np.sqrt(((coefh*h)**2+(coefk*k)**2+(coefl*l)**2)*d2)*1e10,3)
        
        def f1hexa(d2,h,k):
            return round(np.sqrt((4/3)*((h)**2+h*k+(k)**2)*d2)*1e10,3)
        
        def f2(d2,h,k,l,coefhk,coefa,a): #coefa =4/3  coefhk=1 hexa coefa=1 coefhk=0 tetra 
            return round(np.sqrt(l**2/((1/d2)-((coefa*((h)**2+coefhk*h*k+(k)**2))/((a*1e-10)**2))))*1e10,3)
        
        def f3(d2,h,k,l,coefhk,coefa,c):
            return round(np.sqrt((4/3)*(coefa*((h)**2+coefhk*h*k+(k)**2))*(d2/(1-d2*(l/(c*1e10))**2)))*1e10,3)

        def fromb(d2,h,k,l):
            return round( np.sqrt(d2*((h**2+k**2+l**2)*np.sin(self.alpha*2*np.pi/360)**2-(np.cos(self.alpha*2*np.pi/360)-np.cos(self.alpha*2*np.pi/360)**2)*2*(k*l+l*h+h*k))/(1-3*np.cos(self.alpha*2*np.pi/360)**2+2*np.cos(self.alpha*2*np.pi/360)**3) )*1e10,3)
        
        a,b,c=[],[],[]

        n_pic=len(self.pics)
        n_pic_ref=len(self.pic_ref)
        self.l_a=np.zeros(n_pic)
        self.l_b=np.zeros(n_pic)
        self.l_c=np.zeros(n_pic)


        if "TETRAGONAL" in self.maille:
            done="TETRAGONAL"
            coefh,coefk,coefl,coefa,coefhk=1,1,1,1,0
            
            for i in range(n_pic):
                if i < n_pic_ref and int(self.pic_ref[i][0][3])==0 :
                    a.append(f1(dhkl2(self.pics[i].Out_ctr()),int(self.pic_ref[i][0][1]),int(self.pic_ref[i][0][2]),0,coefh,coefk,0))
                    if verbose:
                        print("a",self.pic_ref[i][0],"=",a[-1])
                    self.l_a[i]=a[-1]
            if a != []:
                self.a=round(sum(a)/len(a),3)
                self.b=self.a
                self.l_b=self.l_a
                for i in range(n_pic):
                    if i < n_pic_ref and int(self.pic_ref[i][0][3]) !=0 : 
                        c.append(f2(dhkl2(self.pics[i].Out_ctr()),int(self.pic_ref[i][0][1]),int(self.pic_ref[i][0][2]),int(self.pic_ref[i][0][3]),coefhk,coefa,self.a))
                        self.l_c[i]=c[-1]
                        if verbose:
                            print("c",self.pic_ref[i][0],"=",c[-1])
                if c!=[]:
                    self.c=round(sum(c)/len(c),3)
                    self.rca=self.c/self.a
                    if verbose:
                        print("Calcul_abc C by A DONE ",done," a_moy:",self.a,"b_moy:",self.b,"c_moy:",self.c, "en A°")
            
            if a is [] and c is []:
                for i in range(n_pic):
                    if i < n_pic_ref and int(self.pic_ref[i][0][1]) ==0 : 
                        c.append(np.sqrt(dhkl2(self.pics[i].Out_ctr()),int(self.pic_ref[i][0][1]),int(self.pic_ref[i][0][2]),int(self.pic_ref[i][0][3])*int(self.pic_ref[i][0][3])))
                        self.l_c[i]=c[-1]
                        if verbose:
                            print("c",self.pic_ref[i][0],"=",c[-1])
                    if c!=[]:
                        self.c=round(sum(c)/len(c),3)

                    for i in range(n_pic):
                        if n_pic_ref and int(self.pic_ref[i][0][1]) != 0 :
                            a.append(f3(dhkl2(self.pics[i].Out_ctr()),int(self.pic_ref[i][0][1]),int(self.pic_ref[i][0][2]),int(self.pic_ref[i][0][3]),coefhk,coefk,self.c))
                            if verbose:
                                print("a",self.pic_ref[i][0],"=",a[-1])
                            self.l_a[i]=a[-1]
                        if a != []:
                            self.a=round(sum(a)/len(a),3)
                            self.b=self.a
                            self.l_b=self.l_a
                            self.rca=self.c/self.a
                            if verbose:
                                print("Calcul_abc A by C DONE ",done," a_moy:",self.a,"b_moy:",self.b,"c_moy:",self.c, "en A°")

                else:
                    if verbose:
                        print(done+"PAS DE PIQUE dhkl pour calculer c")
            else:
                if verbose:
                    print(done+"PAS DE PIQUE dhk0 pour calculer a")
 

        elif "HEXAGONAL" in self.maille:
            done="HEXAGONAL"
            coefa,coefhk=4/3,1
            for i in range(n_pic):
                if i < n_pic_ref and int(self.pic_ref[i][0][3])==0:
                    a.append(f1hexa(dhkl2(self.pics[i].Out_ctr()),int(self.pic_ref[i][0][1]),int(self.pic_ref[i][0][2]))) #,0,coefh,coefk,0))
                    if verbose:
                        print("a",self.pic_ref[i][0],"=",a[-1])
                    self.l_a[i]=a[-1]
            if a != []:
                self.a=round(sum(a)/len(a),3)
                self.b=self.a
                self.l_b=self.l_a
                for i in range(n_pic):
                    if i < n_pic_ref and int(self.pic_ref[i][0][3]) !=0:
                        c.append(f2(dhkl2(self.pics[i].Out_ctr()),int(self.pic_ref[i][0][1]),int(self.pic_ref[i][0][2]),int(self.pic_ref[i][0][3]),coefhk,coefa,self.a))
                        self.l_c[i]=c[-1]
                        if verbose:
                            print("c",self.pic_ref[i][0],"=",c[-1])
                if c!=[]:
                    self.c=round(sum(c)/len(c),3)
                    self.rca=self.c/self.a
                    if verbose:
                        print("Calcul_abc DONE ",done," a_moy:",self.a,"b_moy:",self.b,"c_moy:",self.c, "en A°")
                else:
                    if verbose:
                        print(done+"PAS DE PIQUE dhkl pour calculer c")
            else:
                if verbose:
                    print(done+"PAS DE PIQUE dhk0 pour calculer a")

        
        elif "CUBIC" in self.maille:
            done="CUBIC"
            coefh,coefk,coefl=1,1,1
            
            for i in range(n_pic):
                a.append(f1(dhkl2(self.pics[i].Out_ctr()),int(self.pic_ref[i][0][1]),int(self.pic_ref[i][0][2]),int(self.pic_ref[i][0][3]),coefh,coefk,coefl))
                if verbose:
                    print("a",self.pic_ref[i][0],"=",a[-1])
                self.l_a[i]=a[-1]
            self.a=round(sum(a)/len(a),3)
            self.b=self.a
            self.l_b=self.l_a
            self.c=self.a
            self.rca=1
            self.l_c=self.l_a
            if verbose:
                print("Calcul_abc DONE ",done," a_moy:",self.a,"b_moy:",self.b,"c_moy:",self.c, "en A°")

        elif  "ORTHORHOMBIC" in self.maille:
            done= "ORTHORHOMBIC"
            coefh,coefk,coefl=1,1,1
            
            for i in range(n_pic):
                a.append(f1(dhkl2(self.pics[i].Out_ctr()),int(self.pic_ref[i][0][1]),int(self.pic_ref[i][0][2]),int(self.pic_ref[i][0][3]),coefh,coefk,coefl))
                if verbose:
                    print("a",self.pic_ref[i][0],"=",a[-1])
                self.l_a[i]=a[-1]
            self.a=round(sum(a)/len(a),3)
            self.b=self.a
            self.l_b=self.l_a
            self.c=self.a
            self.rca=1
            self.l_c=self.l_a
            if verbose:
                print("Calcul_abc DONE ",done," a_moy:",self.a,"b_moy:",self.b,"c_moy:",self.c, "en A°")

        elif "RHOMBOHEDRAL" in self.maille:
            done="RHOMBOHEDRAL"           
            for i in range(n_pic):
                match = re.match(r'D([-+]?\d+)([-+]?\d+)([-+]?\d+)', self.pic_ref[i][0])
                a.append(fromb(dhkl2(self.pics[i].Out_ctr()), int(match.group(1)),int(match.group(2)), int(match.group(3))))
                if verbose:
                    print("a",self.pic_ref[i][0],"=",a[-1])
                self.l_a[i]=a[-1]
            self.a=round(sum(a)/len(a),3)
            self.b=self.a
            self.l_b=self.l_a
            self.c=self.a
            self.rca=1
            self.l_c=self.l_a
            if verbose:
                print("Calcul_abc DONE ",done," a_moy:",self.a,"b_moy:",self.b,"c_moy:",self.c, "en A°")

        else:
            if verbose:
                print("MAILLE != (cubic,tetra,hexa,mono,rhombo,mono) , A CODER")

    def calcul_V(self,verbose=False):
        if verbose:
            print("- - - - - - calcul_V - - - - - -")
        if self.a == 0 or self.b == 0 or self.c == 0:
            print("PROBLEME abc non calculé")
            return

        alpha_rad = self.alpha * np.pi / 180 if hasattr(self, "alpha") else None

        # dictionnaire des formules
        formulas = {
            "CUBIC": lambda: self.a**3,
            "TETRAGONAL": lambda: self.a**2 * self.c,
            "HEXAGONAL": lambda: (np.sqrt(3)/2) * self.a**2 * self.c,
            "ORTHORHOMBIC": lambda: self.a * self.b * self.c,
            "RHOMBOHEDRAL": lambda: self.a**3 * np.sqrt(1 - 3*np.cos(alpha_rad)**2 + 2*np.cos(alpha_rad)**3)
        }

        for key, func in formulas.items():
            if key in self.maille:
                self.V = round(func(), 3)

                # initialisation de V0 si absent
                if self.Element_ref.V0 is None:
                    if key == "RHOMBOHEDRAL":
                        alpha0_rad = self.Element_ref.alpha * np.pi / 180
                        self.Element_ref.V0 = self.Element_ref.A**3 * np.sqrt(
                            1 - 3*np.cos(alpha0_rad)**2 + 2*np.cos(alpha0_rad)**3
                        )
                    elif key == "CUBIC":
                        self.Element_ref.V0 = self.Element_ref.A**3
                    elif key == "TETRAGONAL":
                        self.Element_ref.V0 = self.Element_ref.A**2 * self.Element_ref.C
                    elif key == "HEXAGONAL":
                        self.Element_ref.V0 = (np.sqrt(3)/2) * self.Element_ref.A**2 * self.Element_ref.C
                    elif key == "ORTHORHOMBIC":
                        self.Element_ref.V0 = self.Element_ref.A * self.Element_ref.B * self.Element_ref.C

                # propagation des incertitudes (si dispo)
                dV = None
                if hasattr(self, "sigma_a"):
                    if key == "CUBIC":
                        dV = 3 * self.a**2 * self.sigma_a
                    elif key == "TETRAGONAL":
                        dV = np.sqrt((2*self.a*self.c*self.sigma_a)**2 + (self.a**2*self.sigma_c)**2)
                    elif key == "HEXAGONAL":
                        dV = np.sqrt(((np.sqrt(3)*self.a*self.c)*self.sigma_a)**2 +
                                    (((np.sqrt(3)/2)*self.a**2)*self.sigma_c)**2)
                    elif key == "ORTHORHOMBIC":
                        dV = np.sqrt((self.b*self.c*self.sigma_a)**2 +
                                    (self.a*self.c*self.sigma_b)**2 +
                                    (self.a*self.b*self.sigma_c)**2)
                    elif key == "RHOMBOHEDRAL":
                        # approximation linéaire, pas trivial -> on peut laisser TODO
                        pass

                if dV is not None:
                    if verbose:
                        print(f"calcul_V DONE {key}: V = {self.V} ± {dV:.3f}")
                    self.sigma_V = dV
                else:
                    if verbose:
                        print(f"calcul_V DONE {key}: V = {self.V}")

                break
        else:
            print("MAILLE != (cubic,tetra,hexa,ortho,rhombo) , A CODER")
                
    def calcul_P(self, V0c=None, T=298,verbose=False):
        if verbose:
            print("- - - - - - calcul_P - - - - - -")
        if V0c is None:
            V0c = self.Element_ref.V0

        if self.V is None:
            print("PROBLEME V non calculé")
            return

        # correction thermique éventuelle
        self.T = T
        Pt = 0
        if getattr(self.Element_ref, "ALPHAKT", None) is not None:
            Pt = self.Element_ref.ALPHAKT * (self.T - 298)

        # facteur de compression
        eta = (V0c / self.V) ** (1/3)

        # Pression Birch-Murnaghan 3e ordre
        P_bm = (3/2) * self.Element_ref.K0 * (eta**7 - eta**5) * \
            (1 + (3/4) * (self.Element_ref.K0P - 4) * (eta**2 - 1))

        self.P = round(P_bm + Pt, 3)

        # calcul incertitude sur P si sigma_V dispo
        if hasattr(self, "sigma_V"):
            # dérivée ∂P/∂V (approx numérique)
            dV = self.sigma_V
            V_plus = self.V + dV
            V_minus = self.V - dV if self.V > dV else self.V * 0.99

            eta_plus = (V0c / V_plus) ** (1/3)
            P_plus = (3/2) * self.Element_ref.K0 * (eta_plus**7 - eta_plus**5) * \
                    (1 + (3/4) * (self.Element_ref.K0P - 4) * (eta_plus**2 - 1)) + Pt

            eta_minus = (V0c / V_minus) ** (1/3)
            P_minus = (3/2) * self.Element_ref.K0 * (eta_minus**7 - eta_minus**5) * \
                    (1 + (3/4) * (self.Element_ref.K0P - 4) * (eta_minus**2 - 1)) + Pt

            dPdV = (P_plus - P_minus) / (V_plus - V_minus)
            self.sigma_P = abs(dPdV) * self.sigma_V
            if verbose:
                print(f"Calcul_P DONE: P = {self.P} ± {self.sigma_P:.3f} GPa")
        else:
            if verbose:
                print(f"Calcul_P DONE: P = {self.P} GPa")

        if verbose:
            print("Paramètres utilisés: V0=", V0c,
            " V=", self.V,
            " K0=", self.Element_ref.K0,
            " K0P=", self.Element_ref.K0P,
            " alphaKt=", getattr(self.Element_ref, 'ALPHAKT', None),
            " T=", self.T)

    def calcul_T(self, P):
        print("- - - - - - calcul_T - - - - - -")
        if getattr(self.Element_ref, "ALPHAKT", None) is None:
            print("ERROR: alphaKt = None")
            return

        # volume isotherme (à 300 K) pour la pression donnée
        V_p = self.Element_ref.EoS_VP(P)
        dV = self.V - V_p

        # calcul T
        self.T = 298 + (dV / self.Element_ref.V0) / (3 * self.Element_ref.ALPHAKT)

        # calcul incertitude si sigma_V existe
        if hasattr(self, "sigma_V"):
            # dérivée ∂T/∂V
            dTdV = 1.0 / (3 * self.Element_ref.ALPHAKT * self.Element_ref.V0)
            self.sigma_T = abs(dTdV) * self.sigma_V
            print(f"Calcul_T DONE: T = {self.T:.2f} ± {self.sigma_T:.2f} K")
        else:
            print(f"Calcul_T DONE: T = {self.T:.2f} K")

        print("Paramètres utilisés:",
            "P_total=", self.P,
            " V=", self.V,
            " P_T0=", P,
            " V_T0=", V_p,
            " alphaKt=", self.Element_ref.ALPHAKT)

    def CALCUL(self,mini=True,verbose=False):
        if verbose:
            print("START CALCUL "+self.name+" - - ->")
        if mini:
            self.minimisation(verbose=verbose)
        else:
            self.calcul_abc(verbose=verbose)
        self.calcul_V(verbose=verbose)
        self.calcul_P(verbose=verbose)


