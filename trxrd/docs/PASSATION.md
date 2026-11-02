# Passation — TR-XRD v2 : ce qui est fait, ce qui reste à faire

Document de reprise pour Claude en local (VS Code), rédigé à la fin du travail en session cloud
(novembre 2026). Il complète `trxrd/CLAUDE.md` (règles et conventions) et `docs/ARCHITECTURE.md`
(conception détaillée, format de fichier, algorithmes, validation). **À mettre à jour à chaque étape
importante.** La section 6 sert de liste de tâches.

- Branche de travail : `claude/h5-data-architecture-twebkb`. `main` contient la v1, inchangée.
- Utilisateur : Florian Dembele, physicien des hautes pressions (DAC dynamique, ID09 à l'ESRF). Il écrit en
  français : lui répondre en français. Le code et les docstrings sont en anglais, la documentation en
  français.

---

## 1. Le projet en bref

La **v1** (`cedapp`, à la racine du dépôt) traite la DRX résolue en temps : intégration des images
Jungfrau, détection et ajustement des pics, identification des phases et pression (EoS), le tout dans une
interface PyQt5. Elle a été écrite par l'utilisateur, puis modifiée par Codex. Deux problèmes ont motivé
la réécriture :

1. La sauvegarde par `dill` (fichiers `.CEDX`) : impossible à partager, et illisible sans le même code au
   même chemin de module.
2. Une architecture non maintenable : `main.py` fait 7 800 lignes et mélange interface, calcul et
   entrées/sorties.

La **v2** (`trxrd`, dossier `trxrd/`, version `2.0.0a1`) repart de zéro : couches séparées, format HDF5
ouvert et versionné, lecteur pour notebook, ligne de commande, et bientôt une interface PySide6. La v1 est
**figée** : elle sert à relire les anciennes données. Le dossier `trxrd/` deviendra un dépôt séparé.

## 2. Décisions déjà prises par l'utilisateur

Ne pas les remettre en cause sans lui en parler.

| Sujet | Décision |
|---|---|
| Sauvegarde | Fichier d'analyse `.trxrd.h5` **séparé** du brut (le brut n'est jamais modifié), avec un lien externe HDF5 vers les images |
| v1 | Figée, conservée pour les anciennes données ; ne modifier ni `main.py`, ni `src/cedapp`, ni `src/Bibli_python`, ni le `pyproject.toml` ou `requirements.txt` de la racine |
| Dépôt | La v2 deviendra un dépôt séparé « une fois bien avancée » (`git subtree split --prefix=trxrd`) |
| Jauges de spectroscopie (rubis, Sm, SrFCl, rhodamine) | **Inutiles**, abandonnées |
| Interface | **PySide6** + pyqtgraph, entièrement nouvelle ; l'ergonomie vient après les fonctionnalités |
| Ajustement | `scipy.optimize.curve_fit`, **pas de lmfit** ; supprimer toutes les dépendances inutiles (tkinter, etc.) |
| Profil des pics | **Pearson IV par défaut** : pics asymétriques du synchrotron et de la compression dynamique |
| Résolution en 2θ | Inutile de reproduire les 601 points de la v1 ; garder une bonne résolution (le pas = la taille d'un pixel, automatique) |
| Axes | Affichage en 2θ, **Q (Å⁻¹)** ou d |
| Recherche de phases | L'algorithme génétique de la v1 est remplacé par la recherche déterministe (l'utilisateur a accepté « si scipy fait mieux ») |
| Suivi des pics | Repris de la v1 à sa demande : tant qu'aucun pic n'apparaît ni ne disparaît, on garde les phases et on n'ajuste que la pression |
| Versions | v1 = `cedapp`, v2 = `trxrd` ; à afficher clairement (README racine, `trxrd --version`) |

## 3. Ce qui est fait

### 3.1 Historique (commits de la branche)

| Commit | Contenu |
|---|---|
| `c04d7ea` | Architecture en couches, format HDF5, lecture ESRF (LIMA/BLISS), intégration pyFAI, fond SNIP, triggers de l'oscilloscope, `XRDRun`, conversion des `.CEDX`, CLI |
| `16a3926` | `trxrd/` rendu autonome (dépendances minimales) ; profils de pics et ajustement `curve_fit` ; maille → V → P avec incertitudes ; axes Q/d ; pas en 2θ automatique ; convertisseur sans l'ancien code |
| `f759f1a` | Nouvelle détection des pics (bruit local, épaulements) ; recherche de composition déterministe ; analyse automatique `trxrd auto` ; ajustement sur le signal brut |
| `8cc1c8c` | Validation sur un run réel (Water05) et cinq corrections qui en découlent ; nommage v1 / v2 |
| `e135918` | Documentation : commit de référence de la v1 |
| `68a1b7c` | Suivi des pics ; `trxrd compare` ; `CLAUDE.md` ; environnement virtuel séparé |

### 3.2 Par fonctionnalité

Tout est testé : `python -m pytest trxrd` passe 105 tests, dont des comparaisons numériques avec la v1,
ignorées automatiquement si la v1 est absente.

| Fonctionnalité | Où | État / preuve |
|---|---|---|
| Modèle de données (Run, Spectrum, PhaseFit, Peak, PhaseCard, Calibration) | `domain/model.py` | ✅ |
| EoS Birch-Murnaghan, maille, d(hkl), conversions 2θ/Q/d, tables 2θ(P) vectorisées | `physics/` | ✅ identique à la v1 à 0,003° près, sauf deux bugs v1 corrigés (formule monoclinique, V0 rhomboédrique) |
| Format `.trxrd.h5` (NeXus, tables en colonnes, versionné, sauvegarde atomique) | `io/h5store.py` | ✅ aller-retour sans perte testé ; s'ouvre dans silx view / h5web |
| Lecture des images LIMA et de l'oscilloscope BLISS | `io/esrf.py` | ✅ sur le scan TUTO |
| Calibration (.poni texte embarqué + masque), fiches `.jcpds` | `io/calibration.py`, `io/jcpds.py` | ✅ |
| Conversion des `.CEDX` sans l'ancien code, sans dill ni lmfit | `io/legacy.py` | ✅ testée avec ces modules bloqués ; fonctionne sur un vrai fichier |
| Intégration pyFAI, pas automatique (48 points/° pour le TUTO) | `processing/integration.py` | ✅ |
| Fond SNIP, lissage | `processing/background.py` | ✅ identique à la v1 |
| Détection des triggers sur l'oscilloscope | `processing/timing.py` | ✅ identique à la v1 |
| Détection des pics | `processing/peaks.py` | ✅ 161/200 pics contre 40/200 en v1 (synthétique) ; 59/61 sur le run réel |
| Ajustement des pics (deux temps : forme fixe, puis forme libre) | `processing/peakfit.py` | ✅ |
| Maille → V → P avec incertitudes | `processing/cell.py` | ✅ |
| Recherche de composition déterministe | `processing/phase_search.py` | ✅ 8/8 contre 0/8 pour l'algorithme génétique v1, 40 fois plus rapide (synthétique) |
| Ajustement d'un spectre ou d'une série | `services/fitting.py` | ✅ |
| Analyse automatique + suivi des pics + pics fixes | `services/identification.py` | ✅ |
| Construction d'un run depuis un scan brut | `services/run_builder.py` | ✅ (fichiers HDF5 uniquement) |
| État de l'application pour la future interface | `services/session.py` | ✅ (sans Qt) |
| Comparaison avec une analyse de référence | `services/validation.py` | ✅ |
| Lecture en notebook (`XRDRun` : summary, peaks, pressure, tracés en 2θ/Q/d, `find`) | `notebook.py` | ✅ |
| CLI `trxrd build / auto / compare / convert / info / --version` | `cli.py` | ✅ |
| Règles d'architecture et de dépendances vérifiées par test | `tests/test_architecture.py` | ✅ |

## 4. Validation sur données réelles

Fichier fourni par l'utilisateur : `Water05_scan0009.CEDX` (janvier 2026, HC-6664). C'est une
décompression de ~6,4 à 0,2 GPa en 0,25 ms (16 spectres), avec les transitions glace VII → VI → Ih et Sn β
tout du long. **Ce fichier n'est pas dans le dépôt**, et aucune donnée de l'utilisateur ne doit y entrer.

```bash
trxrd compare ".../Water05_scan0009.CEDX" --library trxrd/resources/phases/H2O trxrd/resources/phases/Sn \
      --pmax 10 --exclude 18.479:18.679
```

Résultat de référence, à ne pas dégrader :

- mêmes phases que l'analyse manuelle dans **15 spectres sur 16** ;
- |ΔP| ≤ 0,03 GPa pour les glaces VII, VI et Ih ;
- |ΔP| ≤ 0,16 GPa pour Sn ;
- ~3 s, dont 8 spectres sur 16 suivis sans recherche.

Sans `--exclude`, Sn dérive jusqu'à 0,9 GPa : le pic fixe à 18,58° (joint ou diamant) est alors pris pour
la raie Sn (220). `trxrd compare` et `trxrd auto` signalent ce pic eux-mêmes.

## 5. Leçons apprises : pièges déjà corrigés, à ne pas réintroduire

1. **Ne jamais ajuster le signal lissé.** Le lissage déplace le maximum d'un pic asymétrique : P était biaisée
   de −0,01 à −0,02 GPa, avec une incertitude annoncée de 0,002. On ajuste l'intensité brute moins le fond.
2. **Fenêtres locales.** Ajuster tout le diagramme laisse un pic intense non modélisé tirer les autres pics :
   la glace VII partait à 9 GPa. On n'ajuste qu'à moins de 0,3° des pics, et les pics détectés sans phase
   sont ajustés comme pics libres.
3. **Une zone exclue n'est exclue que de l'identification.** Son pic reste modélisé comme pic libre,
   sinon sa queue biaise les raies voisines.
4. **Les phases à « forêt de raies »** (glace XXI : 72 raies) expliquent tout par hasard. La pénalité de
   coïncidence fortuite (`cost_chance`) les écarte.
5. **La continuité se compte par famille** (`H2O_…`, `Sn_…`). Sinon une transition VII → VI coûte trop cher
   et une phase fantôme subsiste.
6. **L'écart de pression entre phases** est limité de façon souple (1,5 GPa par défaut). Sinon la glace VII
   restait vers 3 GPa alors que le reste était sous 1 GPa.
7. **Raies qui se chevauchent** : le pic va à la raie attendue la plus forte. Partager un pic entre deux
   phases rend l'ajustement dégénéré.
8. **Forme Pearson IV mal déterminée** quand un pic fait 3 à 6 points : sans ajustement en deux temps et
   sans limite d'itérations, `curve_fit` tournait plus de 100 s sur un spectre.
9. **Les fiches de phase du run sont prioritaires** sur `--library` : ce sont celles réellement utilisées.
   Les remplacer changeait l'EoS et décalait P jusqu'à 0,1 GPa.
10. **Les vieux fichiers** contiennent des noms en `numpy.str_`, que les attributs HDF5 refusent : toujours
    convertir avec `str()`.
11. **Détection** : seuils en σ du bruit local, jamais en fraction du pic le plus fort ; ignorer 0,25° aux
    bords, où le fond SNIP et les filtres ne sont pas fiables.
12. **Reproduire `peakutils.baseline`** demande `scipy.linalg.pinv`, et non `numpy.linalg.pinv` : à degré 50,
    le calcul est mal conditionné.

## 6. Reste à faire (par priorité)

### P1 — Interface PySide6 (prochaine étape convenue)

Architecture prévue (`docs/ARCHITECTURE.md` §9) : **MVP** au-dessus de `services.Session`, déjà écrite.
Vues passives dans `trxrd/gui/views/`, presenters testables sans fenêtre dans `trxrd/gui/presenters/`,
un registre unique de `QAction` pour les raccourcis, et les calculs longs dans un `QThread` avec le
paramètre `progress(done, total)`, qui existe déjà. Seul `trxrd/gui/` a le droit d'importer
PySide6/pyqtgraph (règle vérifiée par test).

Étape 1, le lecteur :
- [ ] ouvrir un `.trxrd.h5` ou un `.CEDX` (`Session.open` le convertit en mémoire ; la sauvegarde écrit un `.trxrd.h5` à côté) ;
- [ ] liste et curseur des spectres ;
- [ ] spectre courant : données, fond, ajustement, raies attendues de chaque phase ; axe 2θ / Q / d ;
- [ ] P(t) par phase avec barres d'erreur, et oscilloscope en superposition (axe piézo, comme la vue DDAC de la v1) ;
- [ ] carte intensité (2θ × temps) ;
- [ ] tables des phases et des pics du spectre courant.

Étape 2 : lancer depuis l'interface `trxrd build` (scan + calibration), `trxrd auto` (phases candidates,
plage de P, zones exclues, pics fixes signalés) et le suivi, avec une barre de progression et la
possibilité d'arrêter.

Étape 3, l'ajustement interactif (fonctions de la v1 à reprendre, d'après ses raccourcis) :
- [ ] ajouter / retirer une phase sur un spectre, à une P de départ ; déplacer les raies attendues avec P ;
- [ ] ajouter, remplacer, confirmer ou supprimer un pic à la souris ; annuler (undo) ;
- [ ] « fit total » d'un spectre, puis « fit all » d'une plage de spectres (`fit_spectra`, avec suivi) ;
- [ ] zones d'ajustement / zones exclues dessinées à la souris (à stocker dans `Run.ui_state` ou dans le modèle) ;
- [ ] copier / coller le modèle de phases d'un spectre sur une sélection de spectres ; supprimer les phases d'une zone de spectres ;
- [ ] détection automatique des pics affichée (marqueurs), avec SNR ;
- [ ] note du run.

Plus tard : dialogue de calibration (charger le .poni et le masque, vérifier sur l'image intégrée, plage
2θ), éditeur de fiches `.jcpds` (export de la maille ajustée), thème clair/sombre, fenêtre des raccourcis.
La v1 avait aussi une console Python intégrée, qui n'est pas prioritaire.

### P2 — Valider sur les archives de l'utilisateur

- [ ] Lancer `trxrd compare` sur une dizaine de `.CEDX` variés : compressions, décompressions, autres
  échantillons (Ga, Bi, Sn seul), signal faible. Les dossiers probables sont listés en section 7.
- [ ] Noter les résultats (même tableau qu'en section 4) dans `docs/ARCHITECTURE.md` §6.7.
- [ ] Ajuster les réglages par défaut si besoin (`SearchOptions`, `DetectionOptions`, `AutoOptions`) **sans
  dégrader Water05**. Chaque correction doit venir avec un test synthétique reproduisant le cas
  (`tests/synthetic.py`).

### P3 — Écart sur Sn à basse pression

La v2 donne ~0,1 GPa de moins que l'analyse manuelle pour Sn β (jusqu'à 0,16 GPa). Pistes : la raie (301)
utilisée en plus, la forme Pearson IV ajustée différemment, la pondération de la maille. Il faut les images
brutes de Water05 (scan `scan_jf1m_0000.h5` du scan 9, `.poni` `calib_ddac2_correct.poni`, masque) : on
reconstruit le run avec `trxrd build`, puis on compare les résidus des ajustements.

### P4 — Fonctions de la v1 pas encore portées

- [ ] **Corrections manuelles de la note** : la v1 appliquait les lignes `[colonne] -> valeur` de la note
      comme décalages dans le Summary (`Corr_Summary`). La v2 conserve la note mais ne l'applique pas.
- [ ] **Annotations** de la v1 (`run.analyse` : points marqués sur P(t), avec id, kind, label, spec_idx,
      time_s, P_GPa…) et plage de temps `multi_time_range_ms` : ni convertis ni stockés. Prévoir une table
      `/entry/annotations` dans le format, avec une version mineure 1.1.
- [ ] **Spectres statiques ajoutés** avant ou après la série (`Add_Static`).
- [ ] **Entrées DRX autres que HDF5** dans `run_builder` : dossier ou liste de fichiers EDF/TIFF via fabio.
      La v1 le gérait.
- [ ] **Mode « température »** : la v1 avait `fixe="T"` (T calculée à P fixée, via `ALPHAT`). La v2 gère
      V0(T) mais pas ce calcul inverse.
- [ ] **Acquisition directe** de l'oscilloscope LeCroy (`Oscillo_autosave`) : probablement hors du périmètre
      de la v2, à confirmer avec l'utilisateur.

### P5 — Améliorations d'algorithmes repérées

- [ ] **Suivi** : une nouvelle phase qui n'apparaît qu'avec des pics faibles (SNR < 10) ne déclenche pas de
      recherche. Option possible : une recherche forcée tous les N spectres.
- [ ] **Détection** : une seconde passe sur le résidu (données − ajustement) pour trouver les raies faibles
      cachées. Les 2 pics manqués sur Water05 sont des raies faibles de la glace VI.
- [ ] **Performances** sur de longs runs (des centaines à des milliers de spectres) : à mesurer, paralléliser
      les spectres suivis si nécessaire.
- [ ] **Fiche Ga-II** (K0' = 0) : P(V) plafonne à 0,93 GPa alors que la fiche annonce 1–10 GPa. À signaler à
      l'utilisateur, qui doit vérifier la fiche.

### P6 — Dépôt et versions

- [ ] **Tag `v1-cedapp`** sur `6f579f5` : la session cloud n'a pas pu le pousser. Commande :
      `git tag -a v1-cedapp 6f579f5 -m "TR-XRD v1 (cedapp)" && git push origin v1-cedapp`.
- [ ] **Pull request** de la branche vers `main`, ou séparation directe en nouveau dépôt
      (`docs/ARCHITECTURE.md` §10). C'est à décider avec l'utilisateur.

## 7. Environnement local de l'utilisateur

- Windows. Son Python historique est `f:\CODE\python_3_10_15` (Python 3.10). Pour la v2, conseiller un
  environnement séparé (`.venv-v2`), en Python **3.11 ou plus** de préférence. En 3.10, le test qui compare
  les imports au `pyproject.toml` est ignoré, faute de `tomllib`.
- La v1 et la v2 dans deux environnements distincts : PyQt5 et PySide6 ne cohabitent pas.
- Archives probables, d'après le notebook `Use_drx.ipynb` :
  `F:\ESRF\2023 Mai HC-5293 id09`, `F:\ESRF\2024 Juin HC-5719 id09`, `F:\ESRF\2025 Juin HC -6205 id09`,
  `F:\ESRF\2026 Janvier HC-6664 id09`. Chacune a un sous-dossier `Fichier_CED_DRX\` contenant les `.CEDX`
  et `.CEDXUp`. Le laboratoire a aussi `F:\Aquisition_Banc_CEDd\Fichier_CEDd\`, des fichiers `.CEDUp` du
  banc de spectroscopie, probablement inutiles pour la v2. Les données brutes sont dans `RAW_DATA\<échantillon>\<dataset>\scanNNNN\`.
- Données de test dans le dépôt : `trxrd/tests/data/` (scan TUTO, sa calibration, un `.CEDX` d'exemple
  généré par `make_legacy_sample.py`).

## 8. Façon de travailler attendue

- Répondre en français, et expliquer les choix (l'utilisateur est physicien, pas développeur de métier).
- Ne jamais toucher à la v1. Tout se passe dans `trxrd/`.
- Garder `python -m pytest trxrd` au vert. Chaque nouveau comportement vient avec un test, et chaque bug
  trouvé sur des données réelles avec un test synthétique qui le reproduit.
- Respecter les couches et les dépendances : les tests échouent sinon. Pas de lmfit, dill, deap, tkinter
  ni PyQt5.
- Tout changement du format `.trxrd.h5` met à jour `FORMAT_VERSION`, `docs/ARCHITECTURE.md` §5 et les tests
  d'aller-retour.
- Ne jamais committer de données de l'utilisateur. Messages de commit explicites, en anglais. Pousser sur
  la branche de travail.
- Mettre à jour ce fichier (sections 3, 4 et 6) à chaque étape importante.
