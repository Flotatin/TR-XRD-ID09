# Architecture du programme `trxrd`

`trxrd` est le nouveau programme de traitement DRX résolu en temps (ID09, DAC dynamique). Il est écrit de
zéro et vit pour l'instant dans le dossier `trxrd/` du dépôt historique. L'ancien programme (`src/cedapp`,
`main.py`) n'est pas modifié : il reste disponible pour relire les anciennes données.

Ce document décrit les couches du programme, le format de fichier, l'accès depuis un notebook, la conversion
des anciens fichiers, l'interface graphique prévue et la feuille de route.

---

## 0. Décisions prises

| Sujet | Décision |
|---|---|
| Sauvegarde | Un fichier d'analyse HDF5 **séparé** des données brutes, avec un lien externe vers elles (§5). Les fichiers bruts ne sont jamais modifiés. |
| Ancien programme | Conservé tel quel pour les anciennes données. Le nouveau programme ira dans un **dépôt séparé** une fois bien avancé (§10). |
| Jauges de spectroscopie (rubis, Sm, SrFCl, rhodamine) | **Abandonnées.** Seules les phases DRX existent. Le convertisseur les ignore en le signalant. |
| Interface graphique | **PySide6** + pyqtgraph : nouveau programme, nouvelle interface (§8). |
| Ajustement des pics | `scipy.optimize.curve_fit`. **lmfit n'est plus utilisé.** |
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
│   │   └── phase_search.py     🔜 détection de pics + recherche automatique de composition
│   ├── services/
│   │   ├── run_builder.py      ✅ scan brut → Run (intégration, fond, oscilloscope, base de temps)
│   │   ├── fitting.py          ✅ ajuster un spectre ou une série (en suivant l'évolution)
│   │   └── session.py          ✅ état de l'application pour le GUI (run courant, sélection, événements)
│   ├── notebook.py             ✅ XRDRun : lecture seule pour notebooks
│   ├── cli.py                  ✅ trxrd build | convert | info
│   └── gui/                    🔜 vues PySide6 + presenters (§8)
└── tests/                      78 tests ; data/ = scan TUTO, calibration, ancien fichier .CEDX d'exemple
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
/                                   NXroot   file_format="trxrd-run", format_version="1.0", creator
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
    │   └── processing/             table : baseline_degree, filter_type, filter_p1, filter_p2
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

## 9. Interface graphique PySide6 (étape suivante)

Architecture **MVP** au-dessus de `services.Session` :

- **Session** (✅ écrite, sans Qt) : run courant, spectre sélectionné, indicateur « modifié ». Elle émet les
  événements `run_changed`, `spectrum_changed`, `data_changed` et `saved`. C'est la seule source de vérité.
- **Views** (`gui/views/`) : widgets PySide6/pyqtgraph passifs, un par panneau (spectre courant avec choix
  d'axe 2θ/Q/d, carte, P(t), table des phases et des pics, oscilloscope, calibration). Ils ne font aucun
  calcul.
- **Presenters** (`gui/presenters/`) : un par vue. Ils écoutent la Session, mettent la vue à jour et
  appellent les services. Ils sont testables sans afficher de fenêtre.
- **Actions et raccourcis** : un registre unique de `QAction`.
- **Calculs longs** : exécutés dans un `QThread`, avec le paramètre `progress(done, total) -> continuer?`
  qui existe déjà dans `build_run` et `fit_spectra`.

L'ordre proposé : d'abord un **lecteur** (ouvrir un run ou un ancien fichier, naviguer, spectre + fit + P(t),
choix de l'axe). Ensuite la création d'un run depuis un scan, puis l'ajustement interactif (ajout et
retrait de pics, fenêtre, phases), puis la recherche automatique.

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
| 3 | GUI « lecteur » PySide6 sur `Session` | 🔜 |
| 4 | `processing/phase_search.py` : détection de pics + recherche de composition | 🔜 |
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
- **Fond SNIP et détection des triggers** : résultats identiques à l'ancien code.

## 13. Questions ouvertes

1. **Recherche automatique de phases** (ancien `F_Find_compo`, algorithme génétique avec deap) : la porter
   telle quelle, ou la remplacer par `scipy.optimize.differential_evolution` (pas de dépendance en plus) ?
2. **Profil par défaut** des nouveaux pics : Pearson IV (ancien choix pour la DRX) ou pseudo-Voigt (plus
   robuste quand le signal est faible) ?
