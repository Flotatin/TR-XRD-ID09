# Nouvelle architecture : paquet `trxrd`

Ce document propose une architecture neuve pour le traitement DRX résolu en temps (ID09, DAC dynamique) :
les couches, le format de sauvegarde HDF5, l'accès depuis un notebook, la migration des anciens fichiers,
et ce qu'il reste à porter. L'ergonomie de l'interface viendra plus tard. Ici, on pose les fondations.

Le nouveau code vit dans `src/trxrd/`. L'ancien (`src/cedapp`, `src/Bibli_python`) reste en place,
inchangé, tant que le nouveau n'a pas atteint la parité fonctionnelle. Une seule exception :
`trxrd/io/legacy.py` importe l'ancien code, uniquement pour relire les fichiers dill.

---

## 1. Constat sur la version actuelle

| Problème | Conséquence |
|---|---|
| Sauvegarde par `dill.dump(objet_python)` | Le fichier ne se relit qu'avec **le même code**, au **même chemin de module** (`CL_FD_Update`, `Bibli_python.CL_FD_Update`, `src.cedapp.drx...`), avec PyQt5, lecroyscope, tkinter, etc. Il est illisible hors Python et non inspectable. Ouvrir un pickle exécute du code, donc on ne peut pas ouvrir sans risque un fichier reçu. |
| `main.py` : 7 800 lignes, ~320 méthodes | L'UI, les calculs, les E/S et l'état sont mélangés. Toute modification touche tout. |
| Objets métier qui dessinent et lisent des fichiers (`Spectre.Print`, `Data_treatement(print_data_QT=...)`, `CED_DRX.__init__` qui intègre les images) | Impossible de tester ou de réutiliser un calcul sans Qt ni matplotlib. |
| Code dupliqué (`Bibli_python/`, `cedapp/drx/`, `drx.py` jamais utilisé) | On ne sait pas quelle version fait foi. |
| État implicite (`bit_fit`, `bit_model`, `study`, `Summary` recalculés à divers endroits) | Les résultats affichés dépendent de l'ordre des clics. |

## 2. Principes

1. **Couches à dépendances descendantes uniquement.** Le domaine ne connaît ni Qt, ni HDF5, ni lmfit.
   La règle est vérifiée automatiquement par `tests/test_architecture.py`.
2. **Le domaine est fait de données, pas de pickles.** Ce sont des dataclasses simples (tableaux numpy, nombres,
   chaînes), qu'on sait écrire dans n'importe quel format.
3. **Un format de fichier ouvert et versionné.** HDF5, avec une structure NeXus lisible par silx view, h5web,
   MATLAB ou un simple `h5py`.
4. **Les données brutes ne sont jamais modifiées.** On les lit, on les référence par lien externe, et on
   copie dans le fichier d'analyse ce qui est nécessaire pour le partager.
5. **Tout calcul se fait sans interface graphique.** Le GUI, le notebook et la ligne de commande appellent
   les mêmes services.

## 3. Les couches

```
          ┌────────────┐   ┌──────────────┐   ┌──────────┐
          │  gui (Qt)  │   │  notebook.py │   │  cli.py  │    ← points d'entrée
          └─────┬──────┘   └──────┬───────┘   └────┬─────┘
                │                 │                │
          ┌─────▼─────────────────┼────────────────▼─────┐
          │ services : cas d'usage (build_run, Session,   │
          │            fit, recherche de phases...)       │
          └─────┬───────────────────────────┬─────────────┘
                │                           │
   ┌────────────▼───────────┐   ┌───────────▼──────────────┐
   │ io : fichiers          │   │ processing : algorithmes │
   │ h5store, esrf, jcpds,  │   │ intégration pyFAI, fond, │
   │ calibration, legacy    │   │ timing, (fit, maille, GA)│
   └────────────┬───────────┘   └───────────┬──────────────┘
                └────────────┬──────────────┘
                     ┌───────▼────────┐
                     │ domain : model │   dataclasses Run, Spectrum, PhaseFit, Peak, PhaseCard...
                     └───────┬────────┘
                     ┌───────▼────────┐
                     │ physics        │   EoS Birch-Murnaghan, géométrie de maille (numpy/scipy)
                     └────────────────┘
```

| Couche | Peut importer | Interdit |
|---|---|---|
| `physics` | numpy, scipy | le reste |
| `domain` | `physics`, numpy | h5py, Qt, lmfit, pyFAI |
| `io` | `domain`, `physics`, h5py... | `processing`, `services`, Qt |
| `processing` | `domain`, `physics`, pyFAI, lmfit... | `io`, `services`, Qt |
| `services` | tout sauf `gui` | Qt |
| `notebook` | `io`, `domain` | pyFAI, lmfit, Qt |
| `gui` (à venir) | `services`, `domain` | h5py, pyFAI, lmfit en direct |

Seul `io/legacy.py` a le droit d'importer `cedapp`, `Bibli_python` ou `dill`.

## 4. Arborescence

```
src/trxrd/
├── __init__.py            expose XRDRun (import paresseux)
├── domain/model.py        ✅ Run, Spectrum, PhaseFit, Peak, PhaseCard, Calibration, Processing, Source, Oscilloscope
├── physics/
│   ├── eos.py             ✅ Birch-Murnaghan, V0(T), inversion P→V sur la branche physique
│   └── lattice.py         ✅ volume, d(hkl) (tenseur métrique général), 2θ ↔ d
├── io/
│   ├── h5store.py         ✅ save_run / load_run : le format de fichier (§5)
│   ├── h5utils.py         ✅ tables HDF5 « colonnes », chaînes UTF-8
│   ├── esrf.py            ✅ piles d'images LIMA (lecture paresseuse), fichier maître BLISS (ch5/ch6)
│   ├── calibration.py     ✅ .poni + masque Dioptas → Calibration
│   ├── jcpds.py           ✅ fiches de phase
│   └── legacy.py          ✅ relecture des .CEDX/.CEDXUp (dill) et conversion
├── processing/
│   ├── integration.py     ✅ intégration azimutale pyFAI
│   ├── background.py      ✅ lissage (Savitzky-Golay / FFT) + fond SNIP, identiques à l'ancien code
│   ├── timing.py          ✅ détection des triggers sur l'oscilloscope (identique à Temps_Pression)
│   ├── peakfit.py         🔜 ajustement des pics (lmfit), sans état caché
│   ├── cell.py            🔜 affinement de maille → V → P (ancien Element.minimisation)
│   └── phase_search.py    🔜 détection de pics + recherche de composition (algorithme génétique, deap)
├── services/
│   ├── run_builder.py     ✅ scan brut → Run (intégration, fond, oscilloscope, base de temps)
│   ├── session.py         ✅ état de l'application pour le GUI (run courant, sélection, événements)
│   └── fitting.py         🔜 cas d'usage « ajuster ce spectre / cette plage de spectres »
├── notebook.py            ✅ XRDRun : lecture seule pour notebooks
├── cli.py                 ✅ trxrd build | convert | info
└── gui/                   🔜 vues Qt + presenters (§8)
```

## 5. Le format de fichier `trxrd-run` (HDF5)

### 5.1 Fichier compagnon plutôt qu'écriture dans le fichier brut

La demande initiale était d'écrire l'analyse **dans** le fichier HDF5 qui porte la DRX. C'est techniquement
possible, mais je recommande un **fichier d'analyse à côté**, qui pointe vers le brut. Quatre raisons :

- **RAW_DATA est en lecture seule à l'ESRF.** La politique de données ferme le dossier après l'expérience, et
  le fichier `scan_jf1m_0000.h5` est le flux écrit par LIMA. Le modifier risque de corrompre la seule copie des
  images.
- **La taille de partage.** Le scan du TUTO pèse 27 Mo pour 16 images. Le fichier d'analyse correspondant
  (spectres intégrés, fonds, masque, calibration, fiches de phase) pèse **240 Ko**. On partage l'analyse sans
  traîner les images.
- **Un run peut venir de plusieurs fichiers.** Les images viennent du fichier LIMA, l'oscilloscope du fichier
  maître BLISS, et on peut ajouter des spectres statiques.
- **On garde quand même un seul fichier à ouvrir.** Un *lien externe HDF5* `entry/source/raw_data` pointe vers
  la pile d'images brutes. Dans silx view, on voit l'analyse et les images côte à côte. Si on partage le fichier
  sans le brut, le lien est simplement cassé et tout le reste se lit normalement.

Le format est un sous-arbre (`/entry`), donc il pourra être écrit dans un autre fichier HDF5 si un jour c'est
utile, sans rien changer au lecteur.

### 5.2 Arborescence

Extension conseillée : `<dataset>_<scan>.trxrd.h5`, par exemple `water02_0001_scan0027.trxrd.h5`.

```
/                                   NXroot   file_format="trxrd-run", format_version="1.0", creator
└── entry                           NXentry  default → results (ou spectra)
    ├── title, note, created        texte
    ├── ui_state                    JSON libre (zones, préférences d'affichage du GUI)
    ├── source/                     raw_file, raw_dataset, master_file, scan ("27.1")
    │   └── raw_data  ──────────►   lien externe vers la pile d'images brute (optionnel)
    ├── calibration/                energy_ev, wavelength_m, poni (TEXTE du .poni), poni_file,
    │                               mask (uint8, compressé, non nul = pixel masqué), mask_file,
    │                               two_theta_range, points_per_degree, flip_axis, sigma_distance_m
    ├── phases/<nom>/               attributs : symmetry, a, b, c, alpha, beta, gamma, v0, k0, k0p, alpha_t...
    │   ├── jcpds                   texte original de la fiche (traçabilité)
    │   └── reflections/            table d, intensity, h, k, l
    ├── oscilloscope/               NXdata : time, Channel2, Channel3..., trigger_channel
    ├── spectra/                    NXdata signal=intensity, axes=[., two_theta]
    │   ├── two_theta [npt]         (ou [n, npt] si les axes diffèrent, complété par NaN)
    │   ├── intensity [n, npt]
    │   ├── baseline  [n, npt]      fond SNIP (ligne NaN = non calculé)
    │   ├── model     [n, npt]      courbe ajustée totale, sans le fond (NaN = pas de fit)
    │   ├── n_points, time [s], frame, label, fit_range [n, 2]
    │   └── processing/             table : baseline_degree, filter_type, filter_p1, filter_p2
    ├── fits/
    │   ├── phases/                 table, une ligne par (spectre, phase) :
    │   │                           spectrum, phase, state, a, a_err, b, b_err, c, c_err,
    │   │                           volume, volume_err, pressure, pressure_err, temperature, temperature_err
    │   └── peaks/                  table, une ligne par pic : fit (ligne de fits/phases), spectrum, phase,
    │                               name, model, center, center_err, height, height_err, width, width_err,
    │                               shape_0, shape_1..., has_hkl, h, k, l, inter, enabled
    └── results/                    NXdata dérivé : pressure_<phase>(time) et ses _errors
                                    → silx view trace directement P(t) à l'ouverture
```

**Conventions.** Les unités sont en attribut `units` : degrés pour 2θ, s, eV, GPa, Å, Å³. Une valeur absente
vaut `NaN`, jamais un objet Python. Les tables sont stockées comme des **groupes de colonnes 1D**, avec
l'ordre des colonnes dans l'attribut `columns` : n'importe quel outil HDF5 sait les lire, et
`pandas.DataFrame(...)` les reconstruit directement.

**Versions.** `format_version` suit le schéma « majeur.mineur ». Un ajout compatible incrémente le mineur
(les anciens lecteurs ignorent ce qu'ils ne connaissent pas). Un changement incompatible incrémente le majeur,
et `load_run` refuse le fichier avec un message clair au lieu de mal le lire.

**Sauvegarde atomique.** On écrit dans `*.tmp`, puis on renomme. Un plantage pendant la sauvegarde ne
détruit donc jamais le fichier précédent.

### 5.3 Ce qui n'est plus sauvegardé

On ne sauvegarde plus les objets lmfit, les fonctions, les états `bit_*`, ni le `Summary` sous forme de
DataFrame : ce sont des **états de calcul**, pas des données. Les paramètres des pics suffisent pour
reconstruire un modèle lmfit. Le résumé est recalculé à la lecture à partir de `fits/phases`.

## 6. Lire un run dans un notebook : `XRDRun`

`XRDRun` n'a besoin que de h5py, numpy et pandas (plus matplotlib pour les tracés). Il n'utilise ni pyFAI,
ni lmfit, ni Qt.

```python
from trxrd import XRDRun

run = XRDRun("water02_0001_scan0027.trxrd.h5")
run                      # <XRDRun 'water02_0001_scan0027': 120 spectra, phases=['H2O_Ice_VI', 'H2O_Ice_VII'], ...>
run.summary              # une ligne par spectre : spectrum, time, P_<phase>, sigma_P_<phase>, V_<phase>, a_<phase>...
run.phase_fits           # format « long » : une ligne par (spectre, phase)
run.peaks                # une ligne par pic (centre, largeur, hkl, incertitudes)
run.pressure(with_errors=True)    # P(t) par phase, indexé par le temps
run.spectrum(12)         # DataFrame two_theta, intensity, baseline, corrected, model
run.intensity            # tableau [n, npt], lu au premier accès seulement
run.oscilloscope         # DataFrame time + canaux
run.phase_card("H2O_Ice_VII").two_theta_lines(pressure=5, energy_ev=run.energy_ev)

run.plot_spectrum(12); run.plot_pressure(); run.plot_map()

# Analyse multi-runs (remplace la boucle LOAD_CEDd du notebook actuel)
runs = XRDRun.find(r"F:\ESRF\2025 Juin HC -6205 id09", "*water*")
for r in runs:
    r.pressure().plot(ax=ax)

editable = run.to_run()  # objet du domaine complet, pour modifier puis sauvegarder
```

Les colonnes de `summary` reprennent les noms de l'ancien `Summary` (`P_H2O_Ice_VII`...). Le code existant
du notebook `Use_drx.ipynb` (par exemple `detect_h2o_phases`) s'adapte donc presque tel quel.

## 7. Migration des anciens fichiers dill

```bash
trxrd convert "F:\ESRF\2024 Juin HC-5719 id09\Fichier_CED_DRX"        # tout un dossier, résultats à côté
trxrd convert run24.CEDXUp -o converted/                               # un fichier, vers un autre dossier
trxrd info converted/run24.trxrd.h5
```

- La relecture fonctionne **quel que soit le chemin de module au moment de la sauvegarde** : `CL_FD_Update`,
  `Bibli_python.CL_FD_Update`, `src.cedapp.drx.CL_FD_Update` ou `cedapp.drx.CL_FD_Update`.
- `lecroyscope` et `tkinter` ne sont pas nécessaires. Le `matplotlib.use('Qt5Agg')` de l'ancien code est
  neutralisé, pour ne pas casser le backend d'un notebook.
- Le convertisseur reprend : les spectres, fonds et paramètres de traitement ; les temps ; l'oscilloscope ;
  la calibration (texte .poni régénéré depuis l'intégrateur pyFAI, plus le masque) ; les fiches de phase
  utilisées ; pour chaque phase de chaque spectre, a, b, c, V et P avec leurs σ ; chaque pic (modèle, centre,
  hauteur, largeur, paramètres de forme, hkl) ; la courbe ajustée ; la note.
- **Limite connue.** L'ancien `Pics.ctr[1]` contient soit les bornes du fit, soit ±σ après le fit, sans qu'on
  puisse distinguer les deux cas. `center_err` prend `sigma_ctr_total` quand il existe, sinon la
  demi-largeur de cet intervalle.
- Le fichier dill d'origine n'est jamais modifié.

Ouvrir un ancien fichier dans le futur GUI (`Session.open("x.CEDXUp")`) le convertit en mémoire. La
sauvegarde écrit `x.trxrd.h5` à côté.

## 8. Interface graphique (étape suivante)

L'architecture retenue est **MVP** (Model-View-Presenter) au-dessus de `services.Session` :

- **Session** (✅ déjà écrite, sans Qt) : run courant, chemin, spectre sélectionné, indicateur « modifié ».
  Elle émet les événements `run_changed`, `spectrum_changed`, `data_changed` et `saved`. C'est la seule source
  de vérité : aucun widget ne garde de copie des données.
- **Views** (`gui/views/*.py`) : widgets Qt passifs, un par panneau (spectre courant, carte 2θ × temps, P(t),
  table des phases et des pics, oscilloscope, calibration). Ils savent afficher et émettent des signaux
  d'intention (« l'utilisateur veut ajuster »), sans aucun calcul.
- **Presenters** (`gui/presenters/*.py`) : un par vue. Ils écoutent la Session, mettent à jour leur vue et
  traduisent les intentions en appels de services. Ils sont testables sans afficher de fenêtre.
- **Actions et raccourcis** : un registre unique de `QAction` (nom, raccourci, handler). La documentation
  des raccourcis se génère à partir de ce registre.
- **Calculs longs** (intégration d'un scan, fit multi-spectres, recherche génétique) : exécutés dans un
  `QThread` qui appelle le service avec une fonction `progress(done, total) -> continuer?`. Le paramètre
  existe déjà dans `build_run`.

L'ordre proposé : d'abord une fenêtre « lecteur » (ouvrir un run, naviguer, voir spectre, fit et P(t)), puis
la création depuis un scan, puis l'ajustement manuel des pics, puis la recherche automatique.

## 9. Feuille de route

| Étape | Contenu | État |
|---|---|---|
| 1 | Domaine, physique, format HDF5, lecture ESRF, intégration, fond, timing, notebook, conversion dill, CLI, tests | ✅ ce commit |
| 2 | `processing/peakfit.py` + `processing/cell.py` + `services/fitting.py` : fit des pics, affinement de maille, P et incertitudes (portage de `FIT_One_Jauge`, `Element.minimisation`, `Corr_Summary`) | 🔜 |
| 3 | GUI « lecteur » MVP sur `Session` | 🔜 |
| 4 | `processing/phase_search.py` : `F_Find_peaks` + `F_Find_compo` (algorithme génétique), `fit_selected_spectra` | 🔜 |
| 5 | GUI complet : création de run, fit interactif, zones, calibration, vue DDAC/oscilloscope | 🔜 |
| 6 | Retrait de `cedapp`/`Bibli_python` (on garde seulement ce qu'il faut pour `io/legacy.py`) | 🔜 |

## 10. Écarts constatés avec l'ancien code

Chaque point a été vérifié par comparaison directe avec l'ancien code (tests `test_physics.py`,
`test_timing.py`, `test_legacy.py`).

- **Positions 2θ attendues** : écart ≤ 0,003° pour les mailles cubique, tétragonale, hexagonale,
  orthorhombique et rhomboédrique (Ice II). Il vient de hc : l'ancien code utilise 1239,8 eV·nm au lieu de
  1239,84.
- **Bug corrigé, maille monoclinique** (Bi-II, Ice V) : l'ancienne formule de d(hkl) était fausse, avec des
  écarts jusqu'à ~4° en 2θ. Le nouveau calcul passe par le tenseur métrique général, vérifié contre la formule
  de référence.
- **Bug corrigé, V0 rhomboédrique sans `VOLUME:`** (Ice IV) : l'ancien code donnait 1 213 Å³. Le nouveau
  donne 375,4 Å³, cohérent avec la densité de la glace IV.
- **EoS non monotone** (Ga-II, K0' = 0) : l'ancien code échouait. Le nouveau prend la racine physique, la plus
  proche de V0. Avec K0' = 0, P(V) plafonne vers 0,9 GPa, alors que la fiche annonce P ∈ [1, 10] : **la fiche
  Ga-II est à vérifier.**
- **Intégration** : `points_per_degree` veut maintenant vraiment dire « points par degré » (850 points sur
  8–25°). L'ancien code répartissait les points sur toute la plage du détecteur avant de découper, ce qui
  donnait 601 points. L'écart sur l'intensité interpolée est de 0,3 % en médiane.
- **Fond SNIP** et **détection des triggers** : résultats strictement identiques à l'ancien code.

## 11. Questions ouvertes

1. Fichier compagnon (recommandé) ou écriture dans le HDF5 brut ?
2. Les jauges de spectroscopie (rubis, Sm, SrFCl, Rhodamine 6G de `gauge.py`) sont-elles encore utilisées
   dans ce logiciel ? Le modèle `PhaseFit` les accueille déjà, mais leur portage en dépend.
3. Qt : rester sur PyQt5, ou passer à PySide6/PyQt6, PyQt5 n'étant plus maintenu ? La couche `gui` est la
   seule concernée.
4. Faut-il garder une option « intégration à l'identique de l'ancien code » (601 points), pour comparer avec
   des analyses déjà publiées ?
