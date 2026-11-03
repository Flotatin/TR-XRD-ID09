# Architecture de TR-XRD v2 (`trxrd`)

`trxrd` est la **version 2** du programme de traitement DRX résolu en temps (ID09, DAC dynamique). Elle est
écrite de zéro et vit pour l'instant dans le dossier `trxrd/` du dépôt historique. La **version 1**
(`cedapp` : `main.py`, `src/cedapp`, `src/Bibli_python`) n'est pas modifiée ; elle reste disponible pour relire
les anciennes données ; son dernier état est le commit `6f579f5` de `main` (tag `v1-cedapp`).

Ce document décrit les couches du programme, le format de fichier, l'accès depuis un notebook, la conversion
des anciens fichiers, l'interface graphique prévue et la feuille de route.

---

## 0. Décisions prises

| Sujet | Décision |
|---|---|
| Sauvegarde | Un fichier d'analyse HDF5 **séparé** des données brutes, avec un lien externe vers elles (§5). Les fichiers bruts ne sont jamais modifiés. |
| Versions | **v1 = `cedapp`** (ancienne, figée, commit `6f579f5` de `main`, fichiers `.CEDX`) ; **v2 = `trxrd`** (nouvelle, `2.0.0a1`, fichiers `.trxrd.h5`). |
| Ancien programme | Conservé tel quel pour les anciennes données. Le nouveau programme ira dans un **dépôt séparé** une fois bien avancé (§10). |
| Jauges de spectroscopie (rubis, Sm, SrFCl, rhodamine) | **Abandonnées.** Seules les phases DRX existent. Le convertisseur les ignore en le signalant. |
| Interface graphique | **PySide6** + pyqtgraph : nouveau programme, nouvelle interface (§8). |
| Ajustement des pics | `scipy.optimize.curve_fit`. **lmfit n'est plus utilisé.** Ajustement sur le signal brut moins le fond, sans lissage (§6.2). |
| Profil des pics | **Pearson IV par défaut** : pics asymétriques du synchrotron et de la compression dynamique. |
| Recherche automatique de phases | L'algorithme génétique (deap) est **remplacé** par une recherche déterministe (§6.5) : pas de doublons, même résultat à chaque exécution, et plus juste et plus rapide sur nos essais (§6.6). |
| Détection des pics | Seuils exprimés en nombre de σ du **bruit local**, largeurs en degrés, épaulements détectés (§6.5). |
| Résolution en 2θ | Pas besoin de refaire l'intégration de l'ancien code (601 points). Par défaut, le pas est égal à la taille angulaire du plus petit pixel (§6.1). |
| Axes | Affichage en 2θ (°), **Q (Å⁻¹)** ou d (Å). Le stockage reste en 2θ, l'unité native de l'intégration. |
| Dépendances | Le strict nécessaire, vérifié par un test (§3.2). |

## 1. Pourquoi repartir de zéro

| Problème de l'ancienne version | Conséquence |
|---|---|
| Sauvegarde par `dill.dump(objet_python)` | Pour relire un fichier, il faut **le même code au même chemin de module**, plus PyQt5, lecroyscope, tkinter... Le fichier est illisible hors Python, et l'ouvrir exécute du code. |
| `main.py` de 7 800 lignes et ~320 méthodes | L'interface, les calculs, les entrées/sorties et l'état sont mélangés : toute modification touche tout. |
| Objets métier qui dessinent et lisent des fichiers | Impossible de tester ou de réutiliser un calcul sans Qt ni matplotlib. |
| Code dupliqué (`Bibli_python/`, `cedapp/drx/`, `drx.py`) | On ne sait pas quelle version fait foi. |
| État implicite (`bit_fit`, `study`, `Summary`...) | Les résultats dépendent de l'ordre des clics. |

## 2. Principes

1. **Couches à dépendances descendantes.** Le domaine ne connaît ni Qt, ni HDF5, ni scipy.optimize.
2. **Des données, pas des pickles.** Le domaine est fait de dataclasses simples (tableaux numpy, nombres, chaînes).
3. **Un format ouvert et versionné.** HDF5 structuré à la NeXus, lisible par silx view, h5web, MATLAB ou `h5py`.
4. **Le brut en lecture seule.** On le référence par lien externe, et on copie dans l'analyse ce qu'il faut pour la partager.
5. **Tout calcul fonctionne sans interface.** Le GUI, le notebook et la ligne de commande appellent les mêmes services.

## 3. Les couches

```
          ┌──────────────────┐   ┌──────────────┐   ┌──────────┐
          │ gui (PySide6)    │   │  notebook.py │   │  cli.py  │    ← points d'entrée
          └────────┬─────────┘   └──────┬───────┘   └────┬─────┘
                   │                    │                │
          ┌────────▼────────────────────┼────────────────▼─────┐
          │ services : build_run, fit_spectrum / fit_spectra,   │
          │            Session (état de l'application)          │
          └────────┬──────────────────────────────┬─────────────┘
                   │                              │
   ┌───────────────▼────────┐        ┌────────────▼───────────────┐
   │ io : fichiers          │        │ processing : algorithmes   │
   │ h5store, esrf, jcpds,  │        │ intégration pyFAI, fond,   │
   │ calibration, legacy    │        │ timing, peakfit, cell      │
   └───────────────┬────────┘        └────────────┬───────────────┘
                   └──────────────┬───────────────┘
                          ┌───────▼────────┐
                          │ domain : model │   Run, Spectrum, PhaseFit, Peak, PhaseCard, Calibration...
                          └───────┬────────┘
                          ┌───────▼────────┐
                          │ physics        │   EoS, maille, axes 2θ/Q/d, profils de pics (numpy/scipy)
                          └────────────────┘
```

### 3.1 Qui peut importer quoi (vérifié par `tests/test_architecture.py`)

| Couche | Peut importer | Interdit |
|---|---|---|
| `physics` | numpy, scipy | tout le reste |
| `domain` | `physics`, numpy | h5py, Qt, pyFAI... |
| `io` | `domain`, `physics`, h5py, pandas, fabio | `processing`, `services`, Qt |
| `processing` | `domain`, `physics`, pyFAI, scipy, pybaselines | `io`, `services`, Qt |
| `services` | tout sauf `gui` | Qt |
| `notebook` | `io`, `domain`, `physics`, matplotlib | pyFAI, Qt |
| `gui` (à venir) | `services`, `domain`, `physics`, PySide6, pyqtgraph | h5py, pyFAI, scipy.optimize en direct |

### 3.2 Dépendances

| Usage | Paquets |
|---|---|
| Socle (`pip install ./trxrd`) | numpy, scipy, pandas, h5py, hdf5plugin, fabio, pyFAI, pybaselines |
| `[notebook]` | matplotlib |
| `[gui]` | PySide6, pyqtgraph |
| `[dev]` | pytest |

Ont disparu : lmfit, dill, deap, pathos, lecroyscope, tkinter, PyQt5, peakutils, pynverse, silx et tqdm.
Le test `test_third_party_imports_match_pyproject` échoue si un module importe un paquet non déclaré dans
`trxrd/pyproject.toml`, ou si une dépendance déclarée n'est plus utilisée.

## 4. Arborescence

```
trxrd/
├── pyproject.toml              paquet autonome (futur dépôt)
├── README.md
├── docs/ARCHITECTURE.md        ce document
├── resources/phases/           bibliothèque de fiches .jcpds (Au, Bi, Cr, Cu, Ga, H2O, KCl, Sn, Ta)
├── src/trxrd/
│   ├── domain/model.py         ✅ Run, Spectrum, PhaseFit, Peak, PhaseCard, Calibration, Processing, Source, Oscilloscope
│   ├── physics/
│   │   ├── eos.py              ✅ Birch-Murnaghan, V0(T), inversion P→V sur la branche physique
│   │   ├── lattice.py          ✅ volume, d(hkl) (tenseur métrique général), conversions 2θ ↔ Q ↔ d
│   │   └── profiles.py         ✅ Gauss, pseudo-Voigt, Pearson IV, Moffat, Lorentzienne asymétrique
│   ├── io/
│   │   ├── h5store.py          ✅ save_run / load_run : le format de fichier (§5)
│   │   ├── esrf.py             ✅ piles d'images LIMA (lecture paresseuse), fichier maître BLISS (ch5/ch6)
│   │   ├── calibration.py      ✅ .poni + masque Dioptas → Calibration
│   │   ├── jcpds.py            ✅ fiches de phase
│   │   └── legacy.py           ✅ conversion des .CEDX/.CEDXUp, sans l'ancien code (§7)
│   ├── processing/
│   │   ├── integration.py      ✅ intégration azimutale pyFAI, pas en 2θ automatique
│   │   ├── background.py       ✅ lissage (Savitzky-Golay / FFT) + fond SNIP
│   │   ├── timing.py           ✅ détection des triggers sur l'oscilloscope
│   │   ├── peakfit.py          ✅ ajustement simultané des pics (curve_fit)
│   │   ├── cell.py             ✅ affinement de maille → V → P, avec incertitudes
│   │   ├── peaks.py            ✅ détection des pics (bruit local, épaulements)
│   │   ├── phase_search.py     ✅ quelles phases, à quelles pressions (recherche déterministe)
│   │   ├── geometry.py         ✅ calibration du détecteur sur LaB6 / CeO2 (suivi d'anneau, affinement pyFAI)
│   │   └── rates.py            ✅ pente locale (droite glissante) et pente sur un intervalle
│   ├── services/
│   │   ├── run_builder.py      ✅ scan brut → Run (intégration, fond, oscilloscope, base de temps)
│   │   ├── fitting.py          ✅ ajuster un spectre ou une série (en suivant l'évolution)
│   │   ├── identification.py   ✅ analyse automatique : détection → phases → ajustement → P, suivi des pics
│   │   ├── analysis.py         ✅ réglages de l'utilisateur, analyse sur une copie, pics fixes (GUI, CLI)
│   │   ├── validation.py       ✅ comparaison avec une analyse de référence (`trxrd compare`)
│   │   ├── raw_data.py         ✅ parcours de RAW_DATA (jeux de données, scans du fichier maître), scan brut d'un run
│   │   ├── images.py           ✅ image brute et cake (2θ × azimut) d'un spectre ; aperçu d'un scan brut
│   │   ├── calibration.py      ✅ calibration pas à pas (image, clics, affinement, .poni)
│   │   ├── rates.py            ✅ dP/dt par échantillon ou par phase, dP/dt moyen sur un intervalle
│   │   ├── treatment.py        ✅ traitement des spectres (fond, filtre), détection telle que l'analyse la voit
│   │   └── session.py          ✅ état de l'application pour le GUI (run courant, sélection, événements)
│   ├── notebook.py             ✅ XRDRun : lecture seule pour notebooks
│   ├── cli.py                  ✅ trxrd build | auto | compare | convert | info
│   └── gui/                    ✅ lecteur PySide6 + pyqtgraph (§9) : `trxrd-gui`
│       ├── presenters/         display.py (données affichées), viewer.py, preview.py, analysis.py, treatment.py,
│       │                       calibration.py
│       │                       (presenters), sans Qt
│       ├── views/              plots.py (spectre, cake, grandeurs(t), carte), images.py (échelle, niveaux, couleurs),
│       │                       panels.py, new_run.py, preview.py, analysis.py, treatment.py, calibration.py,
│       │                       main_window.py
│       ├── actions.py          registre unique des QAction (menus, barre d'outils, raccourcis)
│       ├── tasks.py            calculs longs dans un thread, résultat rendu au thread de l'interface
│       └── app.py              assemblage : fenêtre + presenters + actions
└── tests/                      227 tests ; data/ = scan TUTO, calibration, ancien fichier .CEDX d'exemple
```

## 5. Le format de fichier `trxrd-run` (HDF5)

### 5.1 Analyse séparée du brut

- RAW_DATA est en lecture seule à l'ESRF, et le fichier LIMA est la seule copie des images.
- Le fichier d'analyse est léger : 240 Ko pour le scan TUTO de 16 images, contre 27 Mo pour le brut.
- Le lien externe `entry/source/raw_data` pointe vers la pile d'images. silx view affiche donc le brut et
  l'analyse ensemble. Sans le brut, le lien est simplement cassé et le reste se lit normalement.

Nom conseillé : `<dataset>_<scan>.trxrd.h5`, par exemple `water02_0001_scan0027.trxrd.h5`.

### 5.2 Arborescence

```
/                                   NXroot   file_format="trxrd-run", format_version="1.1", creator
└── entry                           NXentry  default → results (ou spectra)
    ├── title, note, created        texte
    ├── ui_state                    JSON libre (zones, préférences d'affichage du GUI)
    ├── source/                     raw_file, raw_dataset, master_file, scan ("27.1")
    │   └── raw_data  ──────────►   lien externe vers la pile d'images brute
    ├── calibration/                energy_ev, wavelength_m, poni (TEXTE du .poni), poni_file,
    │                               mask (uint8 compressé, non nul = masqué), mask_file,
    │                               two_theta_range, points_per_degree, flip_axis, sigma_distance_m
    ├── phases/<nom>/               attributs : symmetry, a, b, c, alpha, beta, gamma, v0, k0, k0p, alpha_t...
    │   ├── jcpds                   texte original de la fiche (traçabilité)
    │   └── reflections/            table d, intensity, h, k, l
    ├── oscilloscope/               NXdata : time, Channel2, Channel3..., trigger_channel
    ├── spectra/                    NXdata signal=intensity, axes=[., two_theta]
    │   ├── two_theta [npt]         (ou [n, npt] si les axes diffèrent, complété par NaN)
    │   ├── intensity [n, npt]
    │   ├── baseline  [n, npt]      fond SNIP (ligne NaN = non calculé)
    │   ├── model     [n, npt]      courbe ajustée totale, sans le fond (NaN = pas d'ajustement)
    │   ├── n_points, time [s], frame, label, fit_range [n, 2]
    │   └── processing/             table : baseline_degree, filter_type, filter_p1, filter_p2,
    │                               baseline_window (°, NaN = automatique ; depuis 1.1)
    ├── fits/
    │   ├── phases/                 une ligne par (spectre, phase) : spectrum, phase, state, a, a_err, b, b_err,
    │   │                           c, c_err, volume, volume_err, pressure, pressure_err, temperature, temperature_err
    │   └── peaks/                  une ligne par pic : fit (ligne de fits/phases), spectrum, phase, name, model,
    │                               center, center_err, height, height_err, width, width_err,
    │                               shape_0, shape_1..., has_hkl, h, k, l, inter, enabled
    └── results/                    NXdata dérivé : pressure_<phase>(time) et ses erreurs
                                    → silx view trace directement P(t) à l'ouverture
```

**Conventions.** Les unités sont données en attribut `units`. Une valeur absente vaut `NaN`. Les tables
sont des groupes de colonnes 1D (l'ordre est dans l'attribut `columns`). La sauvegarde est atomique :
on écrit un fichier `.tmp`, puis on le renomme.

**Versions.** `format_version` suit le schéma majeur.mineur. Un ajout compatible change le mineur. Un
changement incompatible change le majeur, et `load_run` refuse alors le fichier avec un message clair.
- 1.1 (novembre 2026) : colonne `baseline_window` du traitement, la fenêtre du fond SNIP, pour les pics
  larges. Un fichier 1.0 se relit avec une fenêtre automatique, comme avant.

## 6. Traitements

### 6.1 Intégration

L'intégration pyFAI se fait sur la plage `two_theta_range`. Avec `points_per_degree=None` (valeur par
défaut), le pas est la plus petite taille angulaire d'un pixel dans cette plage : la résolution n'est
jamais dégradée, et aucun intervalle n'est plus petit qu'un pixel. Pour la géométrie du TUTO
(Jungfrau 1M à 164 mm, 8–25°), on obtient 0,021°, soit 48 points/degré. La valeur retenue est enregistrée
dans le fichier.

### 6.2 Ajustement des pics : `processing/peakfit.py`

Tous les pics actifs de la fenêtre d'ajustement sont ajustés **ensemble**, avec un seul appel à
`curve_fit` et des bornes : centre ± `center_tolerance`, largeur dans [w/(1+inter), w·(1+inter)], et les
bornes propres à chaque profil. Les erreurs viennent de la covariance, mise à l'échelle par la variance
des résidus.

L'ajustement se fait sur **l'intensité brute moins le fond, sans lissage**. Lisser un pic asymétrique
déplace son maximum ; sur nos essais, le lissage de l'ancien code biaisait P de −0,01 à −0,02 GPa, pour
une incertitude annoncée de 0,002. Lisser corrèle aussi le bruit, et les erreurs sont alors sous-estimées.
Le lissage ne sert donc qu'à la détection et à l'affichage. Le profil par défaut est **Pearson IV**.

Seuls les points proches d'un pic entrent dans l'ajustement : à moins de 0,3° de son centre, ou d'une
largeur à mi-hauteur pour un pic plus large (`FitOptions.local_fwhm`), afin que les queues d'un pic de
liquide ou d'amorphe soient ajustées. Avec 1,5 largeur, des raies cristallines de 0,2 à 0,3° étaient touchées
aussi, et la glace VI de Water05 passait de 0,029 à 0,041 GPa d'écart. Avec 1 largeur, Water05 reste à
0,029 GPa pour les glaces et 0,169 GPa pour Sn.

Les profils (`physics/profiles.py`) partagent la signature `f(x, center, height, width, *shape)`.
`center` est la position du **maximum**, `height` sa valeur. Ce sont les conventions des anciens modèles
lmfit : les pics convertis donnent exactement les mêmes courbes, à 10⁻¹⁴ près.

| Modèle | `width` | `shape` |
|---|---|---|
| Gaussian | écart-type | – |
| PseudoVoigt | demi-largeur à mi-hauteur | fraction (0 = Gauss, 1 = Lorentz) |
| PearsonIV | échelle a | (m, skew) |
| Moffat | échelle | beta |
| SplitLorentzian | demi-largeur à gauche | demi-largeur à droite |

### 6.3 Maille, volume, pression : `processing/cell.py`

1. Pour chaque pic indexé : d = λ / (2 sin θ), et σ_d = d · cot θ · σ_θ. σ(2θ) combine l'erreur
   d'ajustement et l'erreur liée à la distance détecteur (`sigma_distance_m`).
2. Moindres carrés pondérés sur (d_obs − d_calc)/σ_d. Les longueurs libres dépendent du système : a
   (cubique, rhomboédrique) ; a et c (tétragonal, hexagonal) ; a, b et c (orthorhombique, monoclinique). Les
   angles sont ceux de la fiche. **S'il y a moins de pics que de paramètres libres**, on n'ajuste qu'un
   facteur d'échelle isotrope, en gardant les rapports c/a de la fiche.
3. Le volume et son erreur sont propagés depuis la covariance. La covariance n'est jamais plus petite que
   l'erreur de mesure.
4. P vient de l'EoS de la fiche (Birch-Murnaghan, V0(T)), et σ_P = |dP/dV| · σ_V.

### 6.4 Séries de spectres : `services/fitting.py`

- `fit_spectrum(run, i, {"H2O_Ice_VII": 2.0})` place les pics attendus à 2 GPa, ajuste, puis calcule P.
- `fit_spectra(run, range(a, b), {...}, follow=True)` démarre chaque spectre du résultat du précédent. Cela
  suit une compression rapide sans perdre les pics.

Sur des spectres synthétiques, la pression est retrouvée à ±0,05 GPa (`tests/test_fitting.py`).

### 6.5 Analyse automatique : détection → phases → ajustement

**Détection des pics** (`processing/peaks.py`). C'était le point faible de l'ancienne chaîne : si un pic
n'est pas détecté, aucune recherche ne peut le retrouver.

- Le bruit est estimé **localement** : on retire un filtre de Savitzky-Golay court, puis on prend la
  médiane glissante des résidus, avec la correction exacte pour du bruit blanc. Cela suit le bruit de
  Poisson, qui augmente avec l'intensité. Sur des spectres synthétiques, l'estimation tombe à 1 % du vrai σ.
- Un pic est retenu si sa **hauteur au-dessus du niveau local** dépasse `min_snr` σ (5 par défaut). Le seuil
  ne dépend **jamais de l'intensité du pic le plus fort**, si bien qu'un pic faible voisin d'un pic très
  intense est conservé. La prominence, mesurée localement, ne sert qu'à écarter les bosses sur le flanc d'un
  autre pic. Les largeurs sont en degrés.
- Les **épaulements**, ces pics sans maximum propre qu'on observe quand deux phases se chevauchent, sont
  cherchés sur la dérivée seconde.
- Les 0,25° aux bords du diagramme sont ignorés : filtres et fond n'y sont pas fiables.
- **Signal où l'on cherche les maxima** : depuis novembre 2026, à la demande de l'utilisateur, c'est le
  signal **filtré par le filtre du spectre** (onglet Traitement), moins le fond, celui que l'interface
  montre. Avant, c'était un Savitzky-Golay de 7 points, quel que soit le filtre choisi. Le bruit reste
  mesuré sur le signal brut moins le fond ; son effet sur le signal filtré est la norme de la réponse
  impulsionnelle du filtre (`background.noise_gain`), qui vaut pour Savitzky-Golay comme pour la FFT.
  L'ajustement, lui, reste fait sur le signal brut (§5 de PASSATION, point 1). Mesure sur 29 runs de
  janvier 2026 (1 053 spectres, mêmes réglages des deux côtés, sans zone exclue) : 805 spectres aux phases
  de l'analyse manuelle au lieu de 797. Water05_scan0014 gagne 4 spectres, Water01_scan0057 3, et
  Water08_scan0028 en perd 2. Water05 scan 9 reste à 15/16, et Sn y passe de 0,165 à 0,161 GPa. L'ancien
  comportement : `AnalysisSettings(spectrum_filter=False)`, ou `trxrd auto --no-spectrum-filter`.
- **Pics fixes** (`find_static_peaks`) : un pic présent au même 2θ (±0,1°) dans au moins 80 % des spectres
  d'une série est signalé par `trxrd auto`, avec la zone `--exclude` à utiliser (joint, diamant...). Ce
  n'est qu'une suggestion, jamais appliquée seule : dans un run statique, tous les pics sont fixes.
  La suggestion est faite **après** l'analyse et triée par `screen_static_peaks` : un pic n'est pas proposé
  si, dans au moins la moitié des spectres analysés, une phase trouvée y place une raie (I ≥ 5 %) qui ne
  bouge pas elle non plus (< 0,2° sur la série). Quand la pression varie peu, les raies de Sn sont fixes et
  les exclure effaçait Sn (D2O_02_scan0012 : 16/16 → 0/16). Le pic du joint de Water05, traversé par la
  raie (220) de Sn qui bouge, reste proposé.

**Composition** (`processing/phase_search.py`), à la place de l'algorithme génétique :

1. Pour chaque phase, les positions 2θ de toutes ses raies sont tabulées sur une grille fine de pressions
   (0,01 GPa), une seule fois par run et de façon vectorisée. On retient comme candidates les pressions où
   la phase, à elle seule, explique le mieux les pics, en gardant une pression par ensemble de pics
   expliqués différent.
2. **Toutes** les combinaisons de 1 à `max_phases` phases et de leurs pressions candidates sont évaluées,
   chacune **une seule fois**, en un calcul vectorisé. Il n'y a donc ni doublons ni tirage aléatoire : la
   même entrée donne toujours la même réponse.
3. Les meilleures combinaisons sont affinées par moindres carrés (`scipy.optimize.least_squares`) sur les
   positions appariées, puis reclassées.

Le coût à minimiser s'exprime en « pics forts non expliqués » :

| Terme | Signification |
|---|---|
| Σ s_j · min(1, (d_j/tol)²) | distance de chaque pic détecté à la raie la plus proche (1 s'il n'y en a aucune à moins de `tolerance`) ; s_j = min(1, SNR/10) pondère par la significativité ; s_j = 0 pour un pic plus large que `max_fwhm` (0,5°) : une bosse de liquide ou d'amorphe n'explique rien et ne coûte rien (sur Water12, une bosse de 0,75° faisait choisir la glace Ic à la place de la glace VII) |
| 0,5 × Σ intensités des raies fortes absentes | une raie forte attendue mais non vue pénalise, sans exclure (texture, grains) |
| 1 × Σ_phases p_hasard · Σ s_j | **pics qu'une phase expliquerait par hasard** : p_hasard = nombre de raies dans la fenêtre × 2·tol / largeur. Une phase à « forêt de raies » (glace XXI : 72 raies entre 8 et 25°) explique tout par hasard et n'est créditée que de ce qu'elle explique au-delà |
| 0,5 × nombre de phases | une phase n'est ajoutée que si elle explique assez de pics |
| ((P_max − P_min) / 1,5 GPa)² | les phases d'un même échantillon voient des P proches (souple, réglable par `max_pressure_spread`, `None` pour désactiver) |
| 1 × familles disparues, 0 × familles apparues, 0,2 × phases changées | continuité avec le spectre précédent ; la famille est le début du nom, sans tenir compte de la casse (`H2O_Ice_VII` → `h2o`, `KCL_B1` et `KCl_B2` → `kcl`) : un changement de polymorphe (VII → VI → Ih) est peu coûteux, la disparition de l'échantillon l'est. Une famille qui **apparaît** ne coûte rien (`cost_family_appear`) : l'échantillon et le marqueur sont dans la cellule dès le départ, et le coût symétrique empêchait Sn ou la glace VII de revenir après un spectre où ils manquaient |

Les raies faibles d'une fiche peuvent expliquer un pic (texture, anneaux ponctués), mais seules les raies
au-dessus de `min_line_intensity` (5 %) sont exigées. Quand des raies de deux phases se chevauchent, le pic
va à la raie attendue la plus forte (une raie à 100 % de Sn plutôt qu'une raie à 5 % de la glace VI).
`max_pressure_step` limite, si on le souhaite, la variation de P d'un spectre au suivant.

**Zones exclues** (`--exclude`, joint ou diamant) : aucune phase n'y est appariée, et les raies qui y tombent ne
comptent pas comme absentes. Les pics qui s'y trouvent restent toutefois **modélisés comme pics libres**
dans l'ajustement : sans cela, leur queue biaise les raies voisines.

**Ajustement automatique** : seuls les points à moins de 0,3° d'un pic sont ajustés, comme le faisaient les
zones d'ajustement manuelles de la v1. Un pic intense qui n'appartient à aucune phase ne peut donc pas tirer
l'ajustement. Les pics détectés mais non attribués sont ajustés comme pics libres. L'ajustement se fait en
deux temps : d'abord à forme fixe (position, hauteur, largeur), puis avec la forme Pearson IV libre, gardée
seulement si elle améliore l'ajustement. Un pic dont la position est connue à plus de 0,1° près n'entre
pas dans le calcul de la maille.

**Enchaînement** (`services/identification.py`, `trxrd auto`) : pour chaque spectre, détection, puis
composition (avec le spectre précédent comme a priori), puis pics **Pearson IV** placés sur les raies
appariées (position, hauteur et largeur détectées), puis ajustement, maille et P.

**Suivi des pics** (`track_peaks`, repris de la v1). Après le premier spectre, les pics détectés sont
appariés un à un avec ceux du spectre précédent, par affectation optimale
(`scipy.optimize.linear_sum_assignment`) avec un décalage maximal de 0,15°. Si aucun pic significatif
(SNR ≥ 10) n'est apparu ni n'a disparu, si aucune phase n'a perdu tous ses pics, et si aucun pic de SNR ≥ 10
(`track_unexplained_snr`, bosses larges exceptées) ne reste inexpliqué, on garde les mêmes phases et les
mêmes raies, on part des pressions précédentes, et on ajuste directement, sans recherche de phases. Sinon la
recherche complète est relancée. Sans la dernière condition, une solution incomplète (Sn manquant) était
reportée sur des centaines de spectres : une raie de Sn à SNR 160 restait appariée à elle-même, donc
« ni apparue ni disparue ». Les pics faibles peuvent aller et venir sans casser le
suivi : ils sont ajustés comme pics libres. Le suivi se désactive avec `--no-track`.

**Comparaison avec une analyse existante** : `trxrd compare ancien.CEDX --library ...` réanalyse un
fichier de la v1 à l'aveugle et compare, spectre par spectre, les phases et les pressions avec l'analyse
manuelle. Le fichier n'est pas modifié. C'est l'outil de validation sur les archives. La recherche et
l'ajustement se font à la température de l'analyse de référence (médiane de ses phases). `--temperature`
(K) la fixe, et règle aussi `trxrd auto` (298 K par défaut). Sur les runs chauffés à 490 K, une recherche
laissée à 298 K décalait la glace VII de Water10_scan0015 de +1,1 GPa.

```bash
trxrd auto analyse/water02_0001_scan0027.trxrd.h5 --phases H2O_Ice_VI H2O_Ice_VII --pmax 10 --frames 0:200
```

### 6.6 Comparaison avec l'ancienne chaîne (spectres synthétiques, Poisson, pics Pearson IV asymétriques)

| | Ancien | Nouveau |
|---|---|---|
| Détection : 20 spectres × 10 pics (dont un pic faible, un pic voisin d'un pic 300 fois plus fort et un épaulement non résolu) | 40/200 trouvés, 0 faux | **161/200 trouvés, 0 faux** |
| Composition : 8 spectres, bibliothèque H2O + Sn (14 phases), 0–20 GPa, un pic parasite, texture aléatoire ; mêmes pics détectés pour les deux | 0/8 correctes, 113 s | **8/8 correctes, 2,7 s** |
| Série avec transition glace VI → VII (6 spectres), après ajustement | – | composition juste partout, P à ±0,02 GPa de la vérité |

Les pics manqués par la nouvelle détection sont au niveau du bruit : 3σ par point, ou noyés dans la queue
d'un pic 300 fois plus intense. L'algorithme génétique a été lancé avec les réglages d'`Auto_dif`
(300 générations, population de 150). Sur ces essais, il confond les glaces de basse pression (II, III,
IV) avec la glace VII.

### 6.7 Validation sur des données réelles : Water05, scan 9 (ID09, janvier 2026)

Décompression rapide de ~6,4 à ~0,2 GPa en 0,25 ms (16 spectres), avec les transitions glace VII → VI → Ih
et Sn β tout du long. Le fichier `.CEDX` analysé à la main avec la v1 a été converti, puis réanalysé **à
l'aveugle** par `trxrd auto`, avec toute la bibliothèque H2O + Sn (14 phases), 0–10 GPa, et la zone du pic
fixe (18,48–18,68°) exclue, telle que proposée par `find_static_peaks`.

| | Résultat |
|---|---|
| Détection | 59 des 61 pics ajustés à la main retrouvés (les 2 manqués sont des raies faibles de la glace VI) |
| Composition | **identique à l'analyse manuelle dans 15 spectres sur 16** ; dans le spectre 8 (coexistence VII + VI), la VI n'est pas retenue : un seul pic faible (SNR 10) la soutient |
| Pression des glaces VII, VI, Ih | écart ≤ 0,03 GPa avec l'analyse manuelle |
| Pression de Sn β | écart ≤ 0,17 GPa (0,165), la v2 donnant en moyenne ~0,1 GPa de moins : écart systématique du modèle de profil, expliqué en §6.8 |
| Temps | ~5 s pour les 16 spectres en relançant la recherche partout, ~3 s avec le suivi des pics (8 spectres sur 16 suivis, résultats identiques) |

Ce run a révélé les points corrigés ci-dessus : un pic fixe non modélisé qui faussait l'ajustement, la glace
XXI qui expliquait tout par hasard, la continuité trop chère pendant une transition, les pics partagés
entre deux phases, et un ajustement qui ne convergeait pas quand la forme Pearson IV était mal déterminée.

### 6.8 Validation sur toute la campagne HC-6664 (janvier 2026, 143 runs)

Tous les `.CEDX` de `F:\ESRF\2026 Janvier HC-6664 id09\Fichier_CED_DRX` ont été réanalysés à l'aveugle
(novembre 2026), soit 143 runs et 6 969 spectres analysés à la main. Deux runs ont été écartés : un sans
analyse, et un doublon (`D2O_01_scan0008`). Réglages :
- bibliothèque selon l'échantillon : H2O + Sn pour Water/D2O, Sn pour Sn01, Bi + KCl, Ga + KCl ;
- 0 ≤ P ≤ max(10, 1,3 × P max de la référence arrondi au multiple de 5 GPa supérieur) ;
- une passe sans exclusion, puis une passe avec les pics fixes retenus par `screen_static_peaks` ;
- température de la référence.

Critère : mêmes phases que l'analyse manuelle. Les scripts ne sont pas dans le dépôt : ils lisent les
données de l'utilisateur.

| Famille | Runs | Spectres | Avant | Après |
|---|---|---|---|---|
| D2O_01 | 9 | 632 | 66 % | 91 % |
| D2O_02 | 21 | 336 | 37 % | 85 % |
| D2O_03 | 8 | 121 | 100 % | 100 % |
| D2O_04 | 7 | 112 | 15 % | 83 % |
| Sn01 | 7 | 112 | 92 % | 97 % |
| Water01 | 14 | 3 332 | 81 % | 87 % |
| Water04 | 15 | 240 | 34 % | 68 % |
| Water05 | 11 | 176 | 72 % | 80 % |
| Water06 | 5 | 67 | 97 % | 97 % |
| Water07 | 8 | 89 | 93 % | 92 % |
| Water08 | 12 | 150 | 75 % | 91 % |
| Water09 | 3 | 41 | 22 % | 51 % |
| Water10 | 12 | 180 | 47 % | 77 % |
| Water11 | 4 | 54 | 15 % | 22 % |
| Water12 | 4 | 1 280 | 15 % | 63 % |
| Bi03, Ga02, scan_jf1m_0000 | 3 | 47 | 21 % | 34 % |
| **Total** | **143** | **6 969** | **61 %** | **81 %** |

« Avant » correspond à l'état de la PR #16, mais avec les 8 fichiers en mode Jungfrau « continu »
désormais convertis, et avec l'exclusion de toutes les zones proposées. Les causes
corrigées, dans l'ordre de leur effet :

| Cause | Correction |
|---|---|
| Une famille qui apparaît coûtait autant qu'une famille qui disparaît : Sn ou la glace VII ne revenaient plus | `cost_family_appear = 0` |
| Le suivi reportait une solution qui laissait un pic fort inexpliqué | `track_unexplained_snr = 10` |
| Les pics fixes proposés étaient des raies de Sn dans les runs à pression presque constante | `screen_static_peaks` |
| `Vmin = 0,8` de la v1 repris par le convertisseur : glace VII plafonnée à 7,03 GPa, VI à 4,9, KCl B2 à 7,4 | `io/legacy.py` ne le reprend plus (§12) |
| Bosse large (liquide) expliquée par la glace Ic | `max_fwhm = 0,5°` |
| Glace VII 0,4 à 0,5 GPa trop basse sur Water10, Water11 et Water12_scan0015 : ces runs ont été analysés à 370-490 K | température de la référence dans `compare` |
| `KCL_B1` et `KCl_B2` vus comme deux échantillons | familles sans casse |

Bilan, spectre par spectre, par rapport à « Avant » : 1 462 spectres gagnés, 42 perdus. Les spectres perdus
sont surtout des cas ambigus : une phase faible en plus ou en moins. Sur Water05 scan 9, rien ne change
(15/16, glaces ≤ 0,03 GPa, Sn ≤ 0,17 GPa). Le temps de calcul passe de ~0,07 à ~0,11 s par spectre en
moyenne, parce qu'il y a moins de spectres suivis.

Une grande partie des désaccords restants vient de l'**analyse de référence** elle-même :
- **Sn ajusté sur une seule raie.** Une maille quadratique est alors indéterminée. Cela vaut pour ~330
  spectres de Water10, Water11 et Water12. Sur Water12, la référence met Sn γ 1,2 GPa sous la glace VII, la
  v2 0,26 GPa sous.
- **Valeurs copiées d'un spectre au suivant.** Dans Water01_scan0005, 347 des 461 spectres répètent la
  pression précédente à 10⁻⁶ près.
- **Pics isolés et pressions de Sn négatives.** On en trouve dans Water01_scan0074 et D2O_01.
- **Bi03.** La phase « Bi_BetaSn » de la référence est le KCl B1 du milieu transmetteur.
- **Ga02.** Ga liquide n'a pas de pic de Bragg : une recherche par pics ne peut pas le trouver.

L'écart systématique de Sn est de −0,07 GPa pour Sn β et −0,10 GPa pour Sn γ (médianes sur les ajustements
manuels crédibles, ~1 050 pour Sn β). Il vient des centres de pics : environ −0,004° avec le profil
Pearson IV. Il ne vient ni de la pondération (+0,000 GPa), ni de l'EoS, ni du choix des raies
(−0,003 GPa). C'est donc une incertitude systématique du modèle de profil, d'environ 0,07 GPa.

**Domaines de Sn**, décision de l'utilisateur : la transition β ↔ γ a une hystérésis entre 8 et 12 GPa, et
pas au-delà. Les fiches de la bibliothèque passent donc de P[−1, 9] à P[−1, 12] pour Sn β, et de P[9, 100]
à P[8, 100] pour Sn γ. Les fiches des fichiers v1 gardent leur EoS mais prennent ces plages : le `--library`
de `compare` et `auto` remplace la plage de pression, qui ne sert qu'à borner la recherche. Ce changement
est compris dans la colonne « Après ».

Restent à trancher par l'utilisateur (§13) :
- Sn bcc, dont la fiche commence à 9 GPa : il se fait passer pour un Sn γ texturé ;
- la fiche `Bi-BetaSn` : sa maille et son EoS sont celles de Sn β.

## 7. Lire un run dans un notebook : `XRDRun`

```python
from trxrd import XRDRun

run = XRDRun("water02_0001_scan0027.trxrd.h5")
run.summary              # une ligne par spectre : spectrum, time, P_<phase>, sigma_P_<phase>, V_<phase>, a_<phase>...
run.phase_fits           # une ligne par (spectre, phase)
run.peaks                # une ligne par pic : center (2θ), d, q, largeur, hkl, incertitudes
run.pressure(with_errors=True)    # P(t) par phase
run.spectrum(12)         # DataFrame two_theta, q, d, intensity, baseline, corrected, model
run.axis("q_A^-1")       # axe x en Q (Å⁻¹) ; "2th_deg", "q_A^-1" ou "d_A"
run.plot_spectrum(12, unit="q_A^-1"); run.plot_pressure(); run.plot_map(unit="q_A^-1")

runs = XRDRun.find(r"F:\ESRF\2025 Juin HC -6205 id09", "*water*")   # plusieurs runs
editable = run.to_run()  # objet du domaine complet, pour modifier puis sauvegarder
```

## 8. Conversion des anciens fichiers dill

```bash
trxrd convert "F:\ESRF\2024 Juin HC-5719 id09\Fichier_CED_DRX"        # tout un dossier
trxrd convert run24.CEDXUp -o converted/
```

**Le convertisseur n'a besoin ni de l'ancien programme, ni de dill, ni de lmfit.** Chaque classe ou
fonction de l'ancien code (et de lmfit, pyFAI, PyQt...) trouvée dans le fichier est remplacée par un
objet passif qui garde seulement les attributs. Seuls numpy, pandas et quelques types de base sont
reconstruits. Aucune fonction contenue dans le fichier n'est exécutée. Le chemin de module utilisé à la
sauvegarde (`CL_FD_Update`, `Bibli_python...`, `src.cedapp...`) n'a donc plus d'importance. Un test vérifie
la conversion en bloquant tous ces modules.

Ce qui est repris : spectres, fonds, paramètres de traitement, temps, oscilloscope, masque, fiches de phase,
a, b, c, V et P avec leurs σ, chaque pic (modèle, centre, hauteur, largeur, forme, hkl), courbe ajustée
recalculée et note. Les jauges de spectroscopie sont ignorées, avec un avertissement.

Limites :
- L'ancien fichier ne contient pas le `.poni`, seulement son chemin. Si ce fichier n'est plus accessible,
  le texte de calibration reste vide (le masque est bien repris).
- `Pics.ctr[1]` contient soit les bornes du fit, soit ±σ après le fit, sans qu'on puisse distinguer.
  `center_err` prend `sigma_ctr_total` s'il existe, sinon la demi-largeur de cet intervalle.
- Ne convertir que des fichiers de confiance : le risque est réduit mais un pickle reste un pickle.

## 9. Interface graphique PySide6

```bash
trxrd-gui                          # ou : python -m trxrd.gui
trxrd-gui "F:/ESRF/.../Water05_scan0009.CEDX"
```

Architecture **MVP** au-dessus de `services.Session` :

- **Session** (sans Qt) : run courant, spectre sélectionné, indicateur « modifié ». Elle émet les
  événements `run_changed`, `spectrum_changed`, `data_changed` et `saved`. C'est la seule source de vérité.
  `Session.read` charge un fichier sans toucher à la session, ce qui permet de le faire dans un thread ;
  `Session.attach` installe ensuite le run, depuis le thread de l'interface.
- **Données affichées** (`gui/presenters/display.py`, sans Qt) : courbes du spectre, raies attendues de
  chaque phase à sa pression, P, V, a, c/a… de chaque phase en fonction du temps, traces de l'oscilloscope,
  carte des diffractogrammes en fonction du temps, mise à l'échelle des images, lignes des tableaux. Tout est
  en 2θ, Q ou d.
- **Presenters** (`gui/presenters/viewer.py`, sans Qt) : un par vue. Ils écoutent la Session et les
  réglages d'affichage (`DisplaySettings` : l'unité de l'axe), remplissent la vue, et reçoivent les gestes
  de l'utilisateur par des fonctions de rappel (`view.on_pick`, `view.on_select`). Ils sont testés avec de
  fausses vues (`tests/test_gui_presenters.py`).
- **Vues** (`gui/views/`) : widgets PySide6/pyqtgraph passifs. Ils ne font aucun calcul.
- **Actions et raccourcis** (`gui/actions.py`) : un registre unique de `QAction`, d'où viennent les menus,
  la barre d'outils et la fenêtre des raccourcis (F1). Les lettres seules restent libres pour l'ajustement
  interactif, qui reprendra les raccourcis de la v1 ; F3 enregistre, comme dans la v1.
- **Calculs longs** (`gui/tasks.py`) : exécutés dans un thread (`QThreadPool`). Le résultat ou l'erreur
  revient dans le thread de l'interface. La tâche reçoit `progress(done, total) -> continuer ?`, le
  paramètre de `build_run`, `fit_spectra` et `auto_analyse`, ce qui servira à la barre de progression et au
  bouton d'arrêt.

**Le lecteur (étape 1)** :
- ouvrir un `.trxrd.h5` ou un ancien `.CEDX`, en arrière-plan (~2 s pour les 1 600 spectres de
  Water01_scan0074) ; un `.CEDX` est converti en mémoire et s'enregistre en `.trxrd.h5` à côté ;
- liste des spectres (temps, phases et pressions) et curseur ;
- spectre courant : données, fond, ajustement, et sous le diagramme une rangée de traits par phase, aux
  positions attendues à sa pression, de hauteur proportionnelle à l'intensité de la fiche (hkl au survol).
  Le zoom est gardé d'un spectre à l'autre ;
- P(t) par phase avec barres d'erreur, et une voie de l'oscilloscope sur l'axe de droite (choix de la voie) ;
- carte intensité (fond retiré), en échelle log par défaut : un pic de joint 100 fois plus fort cache sinon
  l'échantillon (P(t) et la carte forment depuis l'étape 2 le panneau dDAC, voir plus bas) ;
- tableaux des phases (P, σP, V, maille, T, état) et des pics (hkl, 2θ, d, hauteur, largeur, profil) ;
- axe 2θ / Q / d (Ctrl+1/2/3), réservé à 2θ si le run n'a pas d'énergie ;
- un clic sur P(t) ou sur la carte sélectionne le spectre ;
- avant de fermer un run modifié ou de quitter, l'interface demande confirmation.

**Étape 2, première partie (novembre 2026)** :
- **Nouveau run depuis RAW_DATA** (Ctrl+N). Le dialogue lit la structure ID09
  `RAW_DATA/<échantillon>/<échantillon>_0001/` : le fichier maître `<échantillon>_0001.h5` (une entrée par
  scan, avec la commande BLISS, le nombre d'images et l'oscilloscope) et les dossiers `scanNNNN/` (images).
  On y choisit :
  - un jeu de données et un ou plusieurs scans ;
  - la calibration (`.poni` + masque, ou « Calibrer… ») et la plage 2θ ;
  - les fiches de phase et le dossier de sortie.

  L'intégration tourne dans un thread, avec une barre de progression et un bouton « Arrêter ». Le dernier
  run est ouvert ensuite. Les champs sont gardés d'une session à l'autre.
- **Aperçu du scan avant l'intégration**, demandé par l'utilisateur pour choisir la plage 2θ
  (`services/images.ScanPreview`, `gui/presenters/preview.py`, `gui/views/preview.py`). À droite du dialogue,
  le scan courant est montré sur tout le détecteur :
  - l'image brute ou son cake. Par défaut, c'est le maximum pixel par pixel sur le scan (32 images au plus,
    également espacées) : on y voit toutes les raies, à toutes les pressions de la compression. On peut
    aussi choisir une seule image ;
  - le diagramme intégré sur tous les pixels utilisables ;
  - en pointillés, le nombre de degrés d'azimut vus pour chaque 2θ. Aux bords du détecteur, cette
    couverture tombe, et le diagramme devient bruité ;
  - la plage 2θ, en vert. On la tire sur le diagramme ou sur le cake, on la saisit, ou on prend « Tout le
    détecteur ». Sur l'image brute, les pixels hors de la plage sont voilés.

  Sur Water05 scan 9 (`calib_ddac2_correct.poni`), les pixels utilisables vont de 5,21 à 29,30°. Les
  anneaux entre 8 et 25° sont vus sur 59 à 139° d'azimut. Le maximum de 32 images sur 1 600 (Water01
  scan 74) prend 2,4 s, en arrière-plan ; seule la dernière demande est affichée.
- **Oscilloscope** : `ch5` est le trigger (`Channel2` dans la v1), `ch6` est probablement la consigne du
  piézo (0-7,4 V, `Channel3`). `ch2` est lue quand elle existe. Toutes les voies ont la même base de temps :
  `ch6_time` est décalé de 19 ns, pour un pas de 160 ns.
- **Intégration sous le spectre** (Ctrl+K), comme dans la v1 : le cake (2θ × azimut, pyFAI `integrate2d`)
  ou l'image brute du spectre courant, sur le même axe x que le spectre. Les anneaux tombent ainsi sous les
  traits des phases. Le Jungfrau 1M ne voit qu'un secteur d'environ 140° : seul ce secteur est montré, d'un
  seul tenant. Le calcul se fait en arrière-plan (0,9 s la première fois, puis 0,03 s par spectre), et seule
  la dernière demande est affichée. L'image brute d'un ancien `.CEDX` est retrouvée par le nom du run dans
  le `RAW_DATA` voisin, car la v1 n'a souvent pas gardé son chemin.
- **Calibration** (Outils, Ctrl+L, ou « Calibrer… » dans le dialogue de création), à la Dioptas :
  1. charger l'image du calibrant (moyenne de la pile HDF5) et le masque ;
  2. choisir LaB6 ou CeO2 et l'énergie ;
  3. partir d'un `.poni` ou d'une estimation (distance, centre du faisceau, qui peut être hors du
     détecteur) ;
  4. choisir un numéro d'anneau et cliquer dessus ; recommencer sur un deuxième anneau ;
  5. « Calibrer ».

  Le moteur (`processing/geometry.py`) :
  - **suit l'anneau** secteur par secteur depuis le clic, en recentrant la fenêtre sur le pic précédent :
    l'anneau peut s'écarter de la géométrie de départ ;
  - prend dans chaque secteur de 1° le **centroïde du profil moyen** I(2θ), ce qui résiste aux anneaux
    granuleux ;
  - **affine par étapes** avec `GeometryRefinement` de pyFAI : d'abord rotations fixées, puis libérées sur
    une fenêtre étroite. Avec un centre hors du détecteur, la distance, le centre et l'inclinaison sont très
    corrélés ;
  - enregistre le `.poni`, qui sert directement à créer les runs.

  Validation sur l'image réelle de LaB6 (`LaB6_ddac2`, scan 3), comparée au `.poni` de Dioptas
  (`calib_ddac2_correct.poni`), avec quatre départs :
  - le `.poni` de Dioptas ;
  - Dioptas avec la distance +5 % et le centre +10 px ;
  - Dioptas avec la distance −8 % et le centre −15 px ;
  - une estimation grossière (160 mm, faisceau en (1 200 ; 500)), avec deux clics.

  Résultats :
  - distance de 164,9 à 165,4 mm, contre 165,2 mm pour Dioptas ;
  - centre à ±1 px de celui de Dioptas ;
  - pics de LaB6 intégrés à ≤ 0,035° des raies sur les cinq premières, comme avec la géométrie de Dioptas ;
  - environ 2,5 s.

  Les inclinaisons diffèrent de celles de Dioptas dans une direction que ces données ne contraignent pas. La
  dispersion de 0,03° vient des grains du LaB6. `pyFAI.goniometer` n'est pas utilisé : Windows (contrôle
  d'application) bloque une bibliothèque de silx dont il dépend.

**Étape 2, deuxième partie : panneau dDAC et images (novembre 2026)**. La fenêtre reprend la disposition de
la v1 :
- à gauche, la liste des spectres et, en onglets, les tableaux des phases et des pics ;
- au centre, le spectre et son cake ;
- à droite, le panneau dDAC de la v1, de haut en bas : une grandeur de chaque phase en fonction du temps,
  dP/dt, et la carte des diffractogrammes intégrés (temps en x, 2θ / Q / d en y). Les trois graphes
  partagent l'axe du temps, et un clic sur l'un d'eux sélectionne le spectre.

La grandeur se choisit dans une liste : P, V, a, b, c, c/a, b/a, T. La liste ne garde que les grandeurs
présentes dans le run. b, c et les rapports ne sont donnés que pour les systèmes où ils sont libres
(`physics.lattice.free_lengths`, partagé avec `processing/cell.py`) : le c/a d'une phase cubique vaut 1 et
n'est pas tracé. « ÷ fiche (X/X₀) » divise par la valeur de la fiche à pression ambiante (V₀ à la
température de l'ajustement), comme la v1, ce qui permet de comparer des phases différentes sur un même
axe. L'incertitude d'un rapport combine celles de ses deux termes, supposées indépendantes.

La voie de l'oscilloscope (le piézo par défaut) est sur l'axe de droite. Avec « Plage : oscilloscope
entier », on voit toute la rampe du piézo autour des spectres, dont la plage est marquée par une bande grise.
La carte a une colonne par spectre quand les temps sont réguliers. Sinon (trous, spectres statiques), les
colonnes sont régulières et chacune montre le spectre le plus proche (`display.time_columns`, 4 000 colonnes
au plus). Sur Water01_scan0074 (1 600 spectres), la carte se calcule en 0,03 s.

Les images (cake, carte, image du calibrant) ont toutes le même composant (`gui/views/images.py`), à la
Dioptas :
- un histogramme à côté de l'image, dont on tire les niveaux ;
- l'échelle : log, linéaire ou racine ;
- la carte des couleurs : viridis, inferno, magma, plasma, cividis, turbo, gris, gris inversé ;
- « niveaux auto » (percentiles 1 et 99,8 %, recalculés à chaque image), décoché dès qu'on tire les niveaux,
  qui sont alors gardés d'un spectre à l'autre ; « Auto » les recalcule.

Ces choix sont gardés d'une session à l'autre, image par image. Les axes de gauche ont une largeur fixe et un
graphe placé au-dessus d'une image dont l'histogramme est affiché reçoit la même marge à droite : le spectre
reste aligné sur le cake, et la courbe sur la carte. Les tests de l'interface écrivent leurs réglages dans un
`.ini` temporaire (`conftest.py`), jamais dans ceux de l'utilisateur.

**dP/dt** (`processing/rates.py`, `services/rates.py`, graphe sous la grandeur, comme dans la v1) :
- **Courbes par échantillon** (par défaut) : les phases d'une même substance (glaces VII, VI, Ih) forment
  une seule courbe de pression, continue à travers les transitions. Quand deux de ces phases coexistent dans
  un spectre, on prend la moyenne de leurs pressions. Le marqueur (Sn β puis Sn γ) forme une autre courbe.
  « Par phase » donne une courbe par phase. La v1 faisait la moyenne de toutes les phases : l'échantillon et
  le marqueur ne sont pas à la même pression, et cette moyenne sautait à chaque apparition ou disparition
  d'une phase.
- **dP/dt local** : pente d'une droite ajustée par moindres carrés sur le spectre et ses ±k voisins
  (« lissage », k = 2 par défaut, comme les « points glissants » de la v1). Son erreur vient de la
  dispersion des points autour de la droite. La différence de deux spectres voisins, elle, reporte tout le
  bruit des deux spectres sur dP/dt. Les erreurs annoncées de P ne servent pas de poids, car elles sont
  souvent trop petites (`docs/PASSATION.md` §5, point 1). Le calcul passe par des sommes cumulées : 6 ms pour
  1 600 spectres.
- **dP/dt moyen sur un intervalle** (« Moyenne sur un intervalle ») : on déplace un intervalle, sur le graphe
  de la grandeur ou sur celui de dP/dt, et le calcul suit en direct. Deux valeurs par courbe :
  - la **pente** de la droite ajustée sur tous les spectres de l'intervalle, avec son erreur, tracée en
    pointillés sur P(t) ;
  - **ΔP/Δt** entre le premier et le dernier spectre de l'intervalle, sur la pression lissée par la droite
    glissante.

  Les deux concordent quand P(t) est droite sur l'intervalle. Sur un P(t) courbe, la pente n'est plus le
  taux moyen : sur tout Water05 scan 9, la pente vaut −29 ± 2 GPa/ms et ΔP/Δt −24 GPa/ms (6,2 GPa en
  0,257 ms). Sur la partie droite (4,60–4,72 ms), on a −30,8 ± 3,8 et −30,8 GPa/ms. « Copier » met le
  tableau dans le presse-papiers, pour un tableur.

En notebook : `from trxrd.services import rates`, puis `rates.dpdt(run)` ou
`rates.mean_dpdt(run, t1_ms, t2_ms)`, avec `run = XRDRun(fichier).to_run()`. Le lecteur `XRDRun` reste
limité à `io`, `domain` et `physics` (`tests/test_architecture.py`).

**Analyse automatique depuis l'interface** (Analyse, Ctrl+R ; `services/analysis.py`,
`gui/presenters/analysis.py`, `gui/views/analysis.py`) :
- **Le dialogue** propose :
  - les phases candidates du run, cochées par défaut, avec leur plage de pression. On peut ajouter un
    dossier de fiches : les fiches du run restent prioritaires (§5, point 9), mais la recherche prend la
    plage de P de la bibliothèque, comme `trxrd auto --library` ;
  - les spectres : tous, le spectre courant, ou une plage ;
  - la plage de P, la température, les zones exclues (`18.48:18.68; 21.1:21.3`, virgule ou point) ;
  - des réglages avancés, repliés par défaut : plage 2θ, tolérance, nombre de phases, seuil, écart et saut
    de P, suivi des pics.
- **Les réglages sont gardés dans le run** (`Run.ui_state["auto_analysis"]`, enregistré dans le fichier),
  zones exclues comprises : elles dépendent du montage, donc du run. Pour un ancien `.CEDX`, la température
  par défaut est la médiane de celles de ses ajustements, comme dans `trxrd compare`.
- **Le calcul** tourne dans un thread, sur une copie des spectres analysés et des fiches
  (`analysis.analyse`). Le résultat est ensuite mis dans le run, dans le thread de l'interface
  (`Analysis.apply_to`) : les vues ne lisent jamais un spectre en cours d'ajustement. Une barre de
  progression permet d'arrêter : les spectres déjà analysés sont gardés, les autres ne changent pas. Si des
  spectres ont déjà des phases, l'interface demande avant de les remplacer. Le fichier ne change qu'à
  l'enregistrement.
- **Pics fixes** : après l'analyse, les pics qui ne bougent pas et ne sont des raies d'aucune phase trouvée
  (`find_static_peaks` puis `screen_static_peaks`) sont proposés comme zones exclues. On peut relancer
  l'analyse avec ces zones en un clic. Sur une longue série, ces pics sont cherchés sur 200 spectres au
  plus, également espacés. Sur le run synthétique des tests, le pic du joint, inexpliqué, interrompt le
  suivi à chaque spectre ; une fois exclu, le suivi reprend.
- `trxrd auto` et `trxrd compare` passent par les mêmes réglages (`AnalysisSettings`), le même ajout de
  fiches et la même recherche de pics fixes.

Sur Water05 scan 9, l'analyse lancée depuis l'interface (dossiers H2O et Sn, P ≤ 10 GPa, zone
18,479–18,679°) donne le résultat de référence de `trxrd compare` (§6.7) : mêmes phases dans 15 spectres sur
16, |ΔP| ≤ 0,03 GPa pour les glaces et 0,165 GPa pour Sn, en 1,6 s. Sur Water01_scan0074 (1 600 spectres),
l'analyse avance à environ 18 spectres par seconde, et l'arrêt garde les spectres déjà faits.

Remarque : la courbe « ajustement » enregistrée (`Spectrum.model`) ne contient que les pics des phases. Les
pics libres (zones exclues, pics inexpliqués) sont ajustés, mais ni tracés ni gardés. À revoir avec
l'ajustement interactif (étape 3).

**Traitement des spectres et contrôle de la détection** (demandés par l'utilisateur après un essai raté
sur D2O_03_0001_scan0006 ; `services/treatment.py`, `gui/presenters/treatment.py`, `gui/views/treatment.py`) :
- **Affichage** : le spectre est montré **filtré, fond soustrait**, le signal où l'on cherche les pics. Le
  signal brut avec son fond s'affiche à la demande (Affichage > Signal brut, Ctrl+B). La courbe ajustée
  suit : le modèle seul, ou le modèle plus le fond.
- **Onglet Traitement** : lissage du fond SNIP et filtre (Savitzky-Golay, FFT ou aucun) du spectre courant.
  Un changement s'applique tout de suite à ce spectre (au cas par cas). On le propage ensuite à tous les
  spectres, ou à la sélection de la liste (Maj + clic, Ctrl + clic). Les paramètres impossibles (fenêtre
  paire, ordre trop grand...) sont refusés avec un message, et le spectre ne change pas. Les ajustements
  déjà faits sont gardés, mais ils l'ont été avec l'ancien fond : il faut les refaire. Appliquer un
  traitement aux 1 600 spectres de Water01_scan0074 prend 3,2 s ; ce calcul bloque encore l'interface.
- **Onglet Détection** : les pics que l'analyse automatique détecte dans le spectre courant, avec ses
  réglages (seuil, largeur max, fenêtre du bruit, épaulements, signal filtré). Ces réglages sont gardés dans
  le run avec ceux de l'analyse (Ctrl+R), et le dialogue de l'analyse les conserve. Sur le spectre :
  - triangles verts : les pics que la recherche utilise ;
  - orange : trop larges (> 0,5°), ignorés ;
  - gris : dans une zone exclue, ajustés comme pics libres ;
  - triangle vide : épaulement ;
  - pointillés : le seuil (niveau local + seuil × bruit) ;
  - bandes grises : les zones exclues.

  Le tableau donne 2θ, SNR, hauteur, largeur et la raison d'un rejet.
- **Sélection** : la liste des spectres accepte une sélection multiple (`Session.selection`), qui servira
  aussi à l'ajustement interactif.

Ce qu'on a appris sur D2O_03_0001_scan0006 :
- **La calibration était fausse.** Le `.poni` fait ce jour-là avec l'outil de la v2 (`test.poni`) donne
  166,6 mm, contre 162,2 mm pour celui de Dioptas utilisé par la v1 (`calib_ddac2_correct_vendredi`). La
  glace VII sortait alors 2,3 GPa trop bas, et la glace Ih décompressée à −0,55 GPa (a = 4,63 Å) au lieu de
  −0,07 GPa. Avec le `.poni` de Dioptas, la v2 retrouve 13 spectres sur 16, à 0,02 GPa près.
- **Spectre 0** : la raie (200) de la glace VII, vers 23,3°, mesure 0,55° de largeur. Elle est donc ignorée
  par la recherche, qui ne garde que le pic (110) (SNR 217) et deux petits pics de bruit. La solution qui
  laisse la raie (110) inexpliquée (glace VII à 0,01 GPa) coûte 2,18, contre 2,23 pour la bonne (5,91 GPa).
  En effet, le poids d'un pic sature à SNR = 10 : un pic de SNR 217 ne compte pas plus qu'un pic de SNR 10.
  À reprendre (PASSATION, P5).

**Pics larges (liquide, amorphe)**, demandés par l'utilisateur pour étudier des liquides :
- **Fond** (Traitement > Fond : fenêtre, en degrés, `Processing.baseline_window`, format 1.1). Avec des pics
  larges à queues longues, la fenêtre automatique de SNIP (pybaselines) devient trop grande et le fond
  plonge sous zéro par effet de bord. Sur des raies synthétiques de 0,94°, il descend à −3 800 pour un vrai
  fond de 300. Une fenêtre un peu plus large que les pics (1°) le rétablit. Les bosses jusqu'à 2° de large
  restent sinon au-dessus du fond automatique.
- **Détection** (onglet Détection) : relever la largeur max (1,5° par défaut).
- **Recherche des phases** (Détection > Pics larges ignorés au-delà de, `AnalysisSettings.broad_fwhm`) :
  0,5° par défaut ; « jamais » fait compter tous les pics, pour identifier un liquide par ses pics larges.
- **Ajustement** : la fenêtre suit la largeur de chaque pic (§6.2). Sur des raies de glace VII de 0,94°,
  avec une fenêtre de fond de 1°, les largeurs ajustées tombent à 0,05° près et P à 0,01 GPa près. Avant,
  la seconde raie sortait à 1,66°.

**Ensuite** :
- étape 3 : ajustement interactif (pics, phases, zones), gestion des phases, puis le reste
  (`docs/PASSATION.md` §6, P1).

## 10. Séparation en deux dépôts

Le dossier `trxrd/` est autonome : son `pyproject.toml`, ses fiches de phase et ses données de test. Les
seuls tests qui utilisent l'ancien code (comparaisons numériques) seront ignorés automatiquement après la
séparation. Pour créer le nouveau dépôt **en gardant l'historique** :

```bash
git subtree split --prefix=trxrd -b trxrd-only
git push <url-du-nouveau-depot> trxrd-only:main
```

## 11. Feuille de route

| Étape | Contenu | État |
|---|---|---|
| 1 | Domaine, physique, format HDF5, lecture ESRF, intégration, fond, timing, notebook, conversion, CLI | ✅ |
| 2 | Profils de pics, ajustement `curve_fit`, affinement de maille, P et incertitudes, séries de spectres, axes Q/d | ✅ |
| 3 | Détection de pics, recherche de composition déterministe, suivi des pics, analyse automatique (`trxrd auto`), validation sur un run réel (`trxrd compare`) | ✅ |
| 4 | GUI « lecteur » PySide6 sur `Session` | ✅ |
| 5 | GUI complet : création de run, ajustement interactif, zones, calibration, vue oscilloscope | 🔜 |
| 6 | Nouveau dépôt (§10) | 🔜 |

## 12. Écarts avec l'ancien code

- **Positions 2θ attendues** : écart ≤ 0,003°. Il vient de hc : l'ancien code utilise 1239,8 eV·nm au lieu
  de 1239,84.
- **Bug corrigé, maille monoclinique** (Bi-II, Ice V) : la formule de d(hkl) était fausse, avec jusqu'à ~4°
  d'écart en 2θ.
- **Bug corrigé, V0 rhomboédrique sans `VOLUME:`** (Ice IV) : 1 213 Å³ au lieu de 375,4 Å³.
- **Fiche Ga-II** (K0' = 0) : P(V) plafonne à 0,93 GPa, alors que la fiche annonce 1–10 GPa. **La fiche est à
  vérifier.**
- **Intégration** : pas automatique égal à la taille d'un pixel (48 points/degré sur 8–25°), au lieu des
  601 points de l'ancien code.
- **Ajustement** : tous les pics sont ajustés ensemble, au lieu d'une phase après l'autre. La pondération de
  la maille est 1/σ_d², alors que l'ancien code y ajoutait un poids proportionnel à l'amplitude relative.
- **Ajustement sur le signal brut** moins le fond, au lieu du signal lissé (§6.2).
- **Détection des pics** : seuils en σ du bruit local au lieu de fractions du maximum (§6.5).
- **Recherche de composition** déterministe au lieu de l'algorithme génétique (§6.5, §6.6).
- **Fond SNIP et détection des triggers** : résultats identiques à l'ancien code.
- **`Vmin` non repris** : quand son inversion numérique (pynverse) échouait, l'ancien code fixait Vmin = 0,8,
  ce qui arrête l'EoS à V = 0,8 V0 (glace VII ≈ 7 GPa). Le convertisseur ne garde que le `VMIN` écrit dans
  la fiche, sinon 0,6.

## 13. Questions ouvertes

1. **Sn bcc.** Sa fiche commence à 9 GPa, et il se fait passer pour un Sn γ texturé. Faut-il le remonter
   vers 35 GPa ? Les domaines de Sn β et γ sont décidés (§6.8).
2. **Phases sur une seule raie** : une phase non cubique (Sn) vue sur une seule raie a une pression
   indéterminée. Faut-il la signaler « présente, P indéterminée » plutôt que de l'ajuster ou de l'écarter ?
3. **Fiches à revoir** : `Bi-BetaSn` (maille et EoS de Sn β), Ga-II (K0' = 0, P plafonne à 0,93 GPa) ;
   et `Ga_liq`, qu'une recherche par pics ne peut pas trouver.
4. **Écart de Sn** : environ −0,07 GPa, qui vient du modèle de profil (§6.8). Avec les images brutes de
   Water05 (scan + `.poni` + masque), on pourrait comparer les résidus des deux profils.
