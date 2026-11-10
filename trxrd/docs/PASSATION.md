# Passation — TR-XRD v2 : ce qui est fait, ce qui reste à faire

Document de reprise pour Claude en local (VS Code), rédigé à la fin du travail en session cloud
(novembre 2026). Il complète `trxrd/CLAUDE.md` (règles et conventions) et `docs/ARCHITECTURE.md`
(conception détaillée, format de fichier, algorithmes, validation). **À mettre à jour à chaque étape
importante.** La section 6 sert de liste de tâches.

- La v2 est dans `main` depuis la PR #16 (dossier `trxrd/`) ; la v1 n'y a pas changé, et son dernier état
  seul est le commit `6f579f5`. Travailler sur une branche et fusionner dans `main` par pull request.
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
ouvert et versionné, lecteur pour notebook, ligne de commande et interface PySide6. La v1 est
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
| Suivi des pics | Repris de la v1 à sa demande : tant qu'aucun pic n'apparaît ni ne disparaît, on garde les phases et on n'ajuste que la pression. Ajout validé par lui en novembre 2026 : le suivi s'arrête aussi si un pic fort (SNR ≥ 10) reste inexpliqué (§5, point 16) |
| Transition Sn β ↔ γ | Hystérésis entre 8 et 12 GPa, pas au-delà : Sn β jusqu'à 12 GPa, Sn γ à partir de 8 GPa (fiches de la bibliothèque) ; pour les fichiers v1, la recherche prend la plage de la bibliothèque |
| Interface | Commencer maintenant (novembre 2026), avec les briques existantes ; ajuster à l'usage |
| Versions | v1 = `cedapp`, v2 = `trxrd` ; à afficher clairement (README racine, `trxrd --version`) |
| dP/dt moyen sur un intervalle | Pas besoin d'enregistrer l'intervalle ni ses résultats dans le run (novembre 2026) : la copie dans le presse-papiers suffit |
| Finitions de l'interface | Avancer d'abord sur les fonctions ; on reviendra peaufiner à la fin (novembre 2026) |
| Liquides | Ils seront étudiés par moments : les pics larges doivent pouvoir être détectés, compter dans la recherche et être ajustés (seuils réglables, fenêtre du fond, fenêtre d'ajustement selon la largeur ; novembre 2026) |
| Liste des spectres | Dans un onglet (novembre 2026) : en haut de la colonne elle prenait trop de place, mais elle reste utile pour choisir des spectres. En haut, un curseur entre deux flèches, sans boutons précédent / suivant dans la barre d'outils |
| Mode Live | Utile, repris de la v1 (novembre 2026). Rien ne bascule à l'écran tant que le run affiché a des modifications non enregistrées |
| Affichage du spectre et détection | Montrer par défaut le signal filtré, fond soustrait, le brut seulement sur demande ; chercher les pics sur ce signal filtré (novembre 2026). L'ajustement reste sur le brut moins le fond (§5, point 1) |
| Temps de l'acquisition | Afficher l'intégration, l'inter-frame (le temps entre deux intégrations) et la fréquence d'acquisition, qui en est la somme : 1 / (intégration + inter-frame) (novembre 2026) |
| Peritheos (bibliothèque d'EoS de Clemens Prescher) | Pas de dépendance, `physics/eos.py` reste (novembre 2026). Son catalogue sert à vérifier nos fiches et à en créer une de temps en temps, quand un matériau utile y apparaît (§7) |

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
| branche `gui-viewer` | Interface :<br>- étape 1, le lecteur : `trxrd-gui`, presenters sans Qt, registre d'actions, ouverture en arrière-plan (`Session.read` / `attach`) ;<br>- étape 2 : nouveaux runs depuis RAW_DATA, cake sous le spectre, calibration à la Dioptas validée sur le LaB6 réel ; aperçu du scan brut pour choisir la plage 2θ ; analyse automatique depuis l'interface (Ctrl+R) ; spectre affiché filtré, fond soustrait, onglets Traitement et Détection, sélection de spectres ;<br>- étape 3 : travail manuel sur les phases (ajout, P, ajustement, série avec suivi, modèle copié), pics libres gardés, annuler / rétablir ;<br>- étape 2 (suite) : panneau dDAC de la v1 (grandeur au choix P, V, a, b, c, c/a, b/a, T, ou X/X₀, avec le piézo ; dP/dt par échantillon, moyen sur un intervalle ; carte des diffractogrammes en fonction du temps), images à échelle, niveaux et couleurs réglables ;<br>- étape 3 (suite) : pics à la souris, ajustement personnalisé, zones à la souris (ajustement, exclues, zone de spectres), curseur entre deux flèches et liste en onglet, raies et repère 2θ sur le cake et l'image, mode Live ;<br>- intégration, inter-frame et fréquence d'acquisition dans la barre d'état (format 1.3), temps des images d'un scan continu au-delà de l'oscilloscope |
| branche `p2-validation-2026` | Validation sur les 143 runs de janvier 2026 (61 % → 81 % de spectres aux mêmes phases) : conversion des runs Jungfrau « continu », `Vmin` de la v1 non repris, coût d'apparition d'une famille nul, suivi interrompu par un pic fort inexpliqué, pics larges ignorés, tri des pics fixes, `--temperature`, familles sans casse, CLI sous Windows |

### 3.2 Par fonctionnalité

Tout est testé : `python -m pytest trxrd` compte 430 tests. Les 7 comparaisons numériques avec la v1 sont
ignorées quand la v1 n'est pas importable, par exemple dans `.venv-v2`, qui n'a pas PyQt5 : 423 tests
passent alors.

| Fonctionnalité | Où | État / preuve |
|---|---|---|
| Modèle de données (Run, Spectrum, PhaseFit, Peak, PhaseCard, Calibration) | `domain/model.py` | ✅ |
| EoS Birch-Murnaghan, maille, d(hkl), conversions 2θ/Q/d, tables 2θ(P) vectorisées | `physics/` | ✅ identique à la v1 à 0,003° près, sauf deux bugs v1 corrigés (formule monoclinique, V0 rhomboédrique) |
| Format `.trxrd.h5` (NeXus, tables en colonnes, versionné, sauvegarde atomique) | `io/h5store.py` | ✅ aller-retour sans perte testé ; s'ouvre dans silx view / h5web |
| Lecture des images LIMA et de l'oscilloscope BLISS | `io/esrf.py` | ✅ sur le scan TUTO |
| Calibration (.poni texte embarqué + masque), fiches `.jcpds` | `io/calibration.py`, `io/jcpds.py` | ✅ |
| Conversion des `.CEDX` sans l'ancien code, sans dill ni lmfit | `io/legacy.py` | ✅ testée avec ces modules bloqués ; les 145 `.CEDX` de janvier 2026 se convertissent |
| Intégration pyFAI, pas automatique (48 points/° pour le TUTO) | `processing/integration.py` | ✅ |
| Fond SNIP, lissage | `processing/background.py` | ✅ identique à la v1 |
| Détection des triggers sur l'oscilloscope | `processing/timing.py` | ✅ identique à la v1 |
| Détection des pics | `processing/peaks.py` | ✅ 161/200 pics contre 40/200 en v1 (synthétique) ; 59/61 sur le run réel |
| Ajustement des pics (deux temps : forme fixe, puis forme libre) | `processing/peakfit.py` | ✅ |
| Maille → V → P avec incertitudes | `processing/cell.py` | ✅ |
| Recherche de composition déterministe | `processing/phase_search.py` | ✅ 8/8 contre 0/8 pour l'algorithme génétique v1, 40 fois plus rapide (synthétique) |
| Ajustement d'un spectre ou d'une série | `services/fitting.py` | ✅ |
| Analyse automatique + suivi des pics + pics fixes | `services/identification.py` | ✅ |
| Analyse automatique depuis l'interface : phases candidates, spectres, plage de P, zones exclues gardées dans le run, pics fixes proposés puis exclus, progression et arrêt | `services/analysis.py`, `gui/presenters/analysis.py`, `gui/views/analysis.py` | ✅ Water05 scan 9 : même résultat que `trxrd compare` (15/16, 0,03 / 0,165 GPa) en 1,6 s ; ~18 spectres/s sur Water01_scan0074 |
| Construction d'un run depuis un scan brut | `services/run_builder.py` | ✅ (fichiers HDF5 uniquement) |
| État de l'application pour l'interface | `services/session.py` | ✅ (sans Qt) |
| Interface, étape 1 : le lecteur (`trxrd-gui`) | `gui/` | ✅ presenters testés avec de fausses vues, fenêtre testée hors écran ; essayée sur Water05 scan 9 et Water01_scan0074 (1 600 spectres) |
| Parcours de RAW_DATA, nouveaux runs depuis l'interface | `services/raw_data.py`, `run_builder.build_and_save`, `gui/views/new_run.py` | ✅ 43 jeux de données de janvier 2026 lus en 0,7 s ; scan 9 de Water05 intégré en 2 s, temps identiques au `.CEDX` |
| Aperçu d'un scan brut avant l'intégration : image ou cake (maximum du scan ou une image), diagramme et couverture en azimut sur tout le détecteur, plage 2θ à tirer | `services/images.ScanPreview`, `gui/presenters/preview.py`, `gui/views/preview.py` | ✅ Water05 scan 9 : pixels utilisables de 5,21 à 29,30° ; maximum de 32 images sur 1 600 en 2,4 s |
| Traitement des spectres (fond SNIP, filtre) au cas par cas ou propagé à tous / à la sélection ; spectre affiché filtré, fond soustrait (brut sur demande) ; contrôle de la détection (pics, seuil, raisons des rejets) avec les réglages de l'analyse | `services/treatment.py`, `gui/presenters/treatment.py`, `gui/views/treatment.py` | ✅ détection sur le signal filtré du spectre : 805/1 053 spectres contre 797 sur 29 runs ; Water05 inchangé ; 35 ms par spectre sur Water01_scan0074 |
| Travail manuel sur les phases : ajouter, retirer, changer P, ajuster le spectre, ajuster une série en suivant à partir du spectre courant, copier le modèle ; pics libres gardés et tracés ; annuler / rétablir | `services/editing.py`, `Session.remember`, `gui/presenters/editing.py`, `gui/views/editing.py` | ✅ D2O_03 scan 6 : 14 spectres à moins de 0,05 GPa de la v1, sauf le spectre 4 (0,09 GPa), en 2 s |
| Ajustement personnalisé : paramètres et modèle de chaque pic, ajustement d'un pic seul, vue zoomée, pics remplis, résidu, cases par raie, comparaison au fit précédent, touches de la v1 | `editing.set_peak`, `fit_peak`, `set_line`, `misfit`, `PeaksPresenter`, `PeaksPanel` | ✅ D2O_03 scan 6 #0 : résidu 4,45e6 → 2,17e6, P = 5,75 GPa ; le résidu montre la bosse non expliquée vers 19° |
| Pics à la souris : poignées, ajout (raie ou pic libre), suppression, désactivation | `editing.add_peak`, `PeaksPresenter`, `SpectrumView.show_handles` | ✅ Water05 scan 9 #8 : pic du joint ajouté, ajusté à 18,64° ; Sn passe de 2,07 à 2,01 GPa |
| Zones à la souris : zone d'ajustement d'un spectre (propagée à la sélection ou à tous), zones exclues du run, zone de spectres sur les trois graphes dDAC, qui choisit la sélection et donne le dP/dt moyen | `editing.set_fit_range`, `analysis.merged_zones`, `gui/presenters/zones.py`, `gui/views/zones.py`, `plots.drag_draws` | ✅ Water01_scan0074 (1 600 spectres) : 28 ms par déplacement de la zone, 0,2 s pour une zone donnée à tous ; Water05 inchangé |
| Curseur des spectres entre deux flèches, sélection écrite dessous, liste en onglet ; l'analyse automatique propose la sélection | `gui/views/panels.py` (`Navigator`, `SpectraList`), `SpectraListPresenter`, `display.position_text`, `gui/views/analysis.py` | ✅ |
| Raies des phases en pointillés sur le cake ; repère 2θ au clic : trait sur le spectre et le cake, arc exact sur l'image | `CakePresenter`, `CakeView`, `processing/geometry.iso_curve`, `FrameSource.ring` | ✅ D2O_03 : arc en 11 ms, à ±0,012° du 2θ demandé |
| Intégration (réglage du détecteur, gardé dans le run, format 1.3), inter-frame (temps entre deux intégrations : période − intégration) et fréquence d'acquisition (1 / période), à droite de la barre d'état, détail en infobulle, et dans `trxrd info` ; lus dans le brut pour un run plus ancien ou un `.CEDX` | `services/acquisition.py`, `processing/timing.py`, `AcquisitionPresenter` | ✅ Water05 scan 9 : 10 µs, 7,12 µs, 58,41 kHz ; Water01 scan 74 : 10 µs, 1,003 ms, 986,7 Hz, lus dans le fichier maître pour son `.CEDX` |
| Temps des images d'un scan continu après la fin de l'oscilloscope (100 ms enregistrées pour 1,6 s) : prolongés à la période mesurée | `processing/timing.frame_times`, `run_builder.build_run` | ✅ Water01 scan 74 : les 1 502 spectres sans temps en ont un, le dernier à 1 621,37 ms (1 621,39 d'après le `count_time` BLISS) ; Water05 scan 9 : temps identiques au `.CEDX` |
| Mode Live : chaque nouveau scan intégré, analysé, enregistré et ouvert | `services/live.py`, `gui/presenters/live.py`, `gui/views/live.py` | ✅ essai de bout en bout sur un scan réel recopié par morceaux ; **pas encore sur une vraie acquisition** |
| Limites de zoom de tous les graphes | `limit_view`, `keep_in_limits`, `ImagePanel.update_limits` | ✅ |
| Cake / image brute sous le spectre | `services/images.py`, `CakePresenter` | ✅ image brute d'un `.CEDX` retrouvée par le nom du run ; 0,03 s par spectre |
| Panneau dDAC : grandeur de chaque phase en fonction du temps (P, V, a, b, c, c/a, b/a, T, ou ÷ fiche), piézo, carte des diffractogrammes sur le même axe du temps | `gui/presenters/display.py`, `EvolutionPresenter`, `MapPresenter`, `gui/views/plots.py` | ✅ Water05 scan 9 ; carte de Water01_scan0074 (1 600 spectres) en 0,03 s |
| dP/dt par échantillon ou par phase (droite glissante), dP/dt moyen sur un intervalle (pente et ΔP/Δt) | `processing/rates.py`, `services/rates.py`, `RatePresenter` | ✅ Water05 scan 9 : eau et Sn culminent vers −45 à −50 GPa/ms ; 6 ms pour 1 600 spectres |
| Images : histogramme, échelle log / linéaire / racine, couleurs, niveaux gardés d'un spectre à l'autre | `gui/views/images.py` | ✅ cake, carte et image du calibrant |
| Calibration du détecteur (LaB6 / CeO2) | `processing/geometry.py`, `services/calibration.py`, `gui/views/calibration.py` | ✅ synthétique : géométrie retrouvée à 0,01° ; LaB6 réel : distance à 0,3 mm de Dioptas, pics à ≤ 0,035° |
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
- |ΔP| ≤ 0,17 GPa pour Sn (0,169) ;
- ~3 s, dont 8 spectres sur 16 suivis sans recherche.

Sans `--exclude`, Sn dérive jusqu'à 0,9 GPa : le pic fixe à 18,58° (joint ou diamant) est alors pris pour
la raie Sn (220). `trxrd compare` et `trxrd auto` signalent ce pic eux-mêmes.

**Toute la campagne de janvier 2026** (143 runs, 6 969 spectres analysés à la main ; détail par famille dans
`docs/ARCHITECTURE.md` §6.8) : **81 %** des spectres ont les mêmes phases que l'analyse manuelle, contre
61 % avant. On a 100 % pour D2O_03, plus de 90 % pour D2O_01, Sn01, Water06, Water07 et Water08, et 22 %
pour Water11. Une bonne part des désaccords restants est due à la référence :
- Sn ajusté sur une seule raie ;
- valeurs copiées d'un spectre au suivant ;
- « Bi_BetaSn » de Bi03, qui est en fait le KCl B1 ;
- Ga liquide.

Les scripts de cette mesure ne sont pas encore dans le dépôt (voir P2). Ils reprennent `trxrd compare`
run par run, avec une passe sans exclusion puis une passe avec les zones fixes retenues.

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
13. **Ne pas reprendre les paramètres numériques de la v1.** Son `Vmin = 0,8` était un repli après un échec de
    pynverse ; repris tel quel, il plafonnait la glace VII à 7 GPa. Dans les runs en mode Jungfrau « continu »,
    son `time_index` est un tableau, et non un nom de canal.
14. **Un pic fixe n'est pas forcément un joint.** À pression presque constante, les raies de Sn sont fixes
    aussi. Avant de proposer une zone, vérifier qu'aucune raie de l'échantillon, fixe elle aussi, ne s'y
    trouve (`screen_static_peaks`).
15. **Une famille qui apparaît ne doit rien coûter.** Le coût de continuité est fait pour qu'une phase ne
    disparaisse pas sans raison. Appliqué aussi à l'apparition, il empêchait Sn ou la glace VII de revenir.
16. **Le suivi doit vérifier ce qui reste inexpliqué.** Une raie forte jamais attribuée est « appariée »
    d'un spectre au suivant, et une solution incomplète pouvait alors durer des centaines de spectres.
17. **Une bosse large n'est pas une raie.** Au-delà de 0,5° de largeur (liquide, amorphe), un pic n'explique
    rien et ne coûte rien dans la recherche. Ce seuil est réglable (« jamais » pour un liquide qu'on veut
    identifier par ses pics larges).
18. **Température.** Certains runs ont été analysés à 370-490 K. Comparer à 298 K faussait la glace VII
    de −0,5 GPa.
19. **L'analyse manuelle n'est pas la vérité.** Avant de corriger un réglage, vérifier la référence :
    - Sn sur une seule raie ;
    - pressions copiées d'un spectre au suivant ;
    - phases mal nommées ;
    - cohérence des pressions de Sn et de la glace dans une même cellule.
20. **Calibration avec le centre du faisceau hors du détecteur** (Jungfrau 1M à ID09) : on ne voit
    qu'environ 80° d'azimut de chaque anneau. La distance, le centre et l'inclinaison se compensent :
    - affiner par étapes, rotations fixées d'abord ;
    - juger sur la position des pics intégrés, pas seulement sur l'écart quadratique (plusieurs solutions
      ont le même) ;
    - le LaB6 est granuleux : prendre le centroïde du profil moyen par secteur, et non le pixel le plus
      brillant.
21. **Windows bloque `silx.image.marchingsquares`** (contrôle d'application), donc `pyFAI.goniometer` :
    ne pas en dépendre. `pyFAI.geometryRefinement`, `calibrant` et `massif` restent utilisables.
    `twoThetaArray` et `chiArray` sont dépréciés : utiliser `center_array(unit=...)`.
22. **dP/dt moyen : la pente d'une droite n'est pas ΔP/Δt** quand P(t) est courbe. Sur tout Water05 scan 9,
    elles valent −29 et −24 GPa/ms. On affiche les deux. La moyenne de toutes les phases, comme dans la v1,
    mélange l'échantillon et le marqueur : on calcule une courbe par échantillon (famille de phases).
23. **Vérifier la calibration sur une phase connue.** Un `.poni` faux de 4,4 mm (D2O_03, `test.poni`)
    décale toutes les pressions de plusieurs GPa sans rien casser d'autre. La glace Ih décompressée doit
    sortir vers 0 GPa.
24. **Le fond SNIP automatique échoue avec des pics larges.** Sa fenêtre devient trop grande, et le fond
    plonge sous zéro par effet de bord (−3 800 pour 300 sur des raies de 0,94°). Il faut alors régler la
    fenêtre du fond, un peu plus large que les pics. La fenêtre d'ajustement doit aussi suivre leur
    largeur, sans toucher les raies cristallines : une largeur à mi-hauteur, pas 1,5.
25. **Ne pas changer le contenu de `trxrd-gui.exe`.** Le contrôle intelligent des applications de Windows
    refuse un `.exe` non signé dont le contenu est nouveau (« Device Guard »). Le nom du point d'entrée est écrit
    dedans : garder `trxrd.gui.app:main` (`docs/ARCHITECTURE.md` §9). En cas de refus : `python -m trxrd.gui`. Pour
    voir ce qui se passe au lancement : `%LOCALAPPDATA%\trxrd\trxrd-gui.log`, et le journal CodeIntegrity.
26. **Un long texte élargit toute la fenêtre.** Le message de dP/dt et le chemin de la barre d'état l'amenaient
    à 2 196 pixels, hors de l'écran : retour à la ligne, ou texte coupé avec une infobulle.
27. **Un pic hors de la zone d'ajustement ne doit plus compter.** Il n'était pas ajusté, mais sa vieille position
    entrait encore dans la maille : P dépendait d'un pic que personne ne regardait. De même, placer une phase élargissait
    la zone à chaque cran de molette, et pouvait attribuer une raie à un pic de zone exclue.
28. **Tester la souris avec de vrais événements, et vérifier que le test attrape le défaut.** `QTest.mouseMove` déplace
    le curseur sans bouton tenu : pyqtgraph ne voit aucun glissement. Il faut des `QMouseEvent`, et `mouseRateLimit` à −1,
    car pyqtgraph jette les mouvements à moins de 10 ms d'intervalle (`test_gui_window.py`, `_drag`). Remettre un défaut
    corrigé pour voir un test tomber a aussi servi : le premier test du point 27 passait avec le défaut, sa tolérance de
    0,1 GPa étant trop lâche. Il compare maintenant P pour deux positions du pic hors de la zone.
29. **Une sélection doit toujours se voir.** Sinon « Coller » ou « Ajuster la sélection » agissent sur des spectres qu'on
    n'a plus en tête. Elle est écrite sous le curseur, quel que soit l'onglet, et la zone retirée l'emporte avec elle.
30. **Une tâche d'arrière-plan peut survivre à la fenêtre.** Émettre son résultat lève alors une exception dans le thread
    (`Signal source has been deleted`) : `_Job._emit` s'arrête en silence. Le mode Live, qui lance une tâche toutes les 2 s,
    rendait le cas fréquent. Vu seulement en essai réel : les tests avec `ImmediateRunner` n'ont pas de thread.
31. **Mode Live : ne regarder que le dernier scan en saute.** Ce que fait la v1 ne suffit plus : l'analyse d'un scan de 1 600
    spectres prend 90 s, et deux tirs rapprochés en perdaient un sans le dire. Les nouveaux scans passent à tour de rôle.
32. **Une mutation qui survit n'est pas toujours un test trop faible.** Pour le mode Live, c'en était trois : deux tests
    simplement pas lancés, un vrai trou (scan terminé juste après l'arrêt). Pour le curseur, c'était du code mort, retiré.
    Pour l'arc, il a fallu construire un croisement exactement sur la limite ±180° d'azimut.
33. **Mode Jungfrau continu : l'oscilloscope s'arrête avant le scan.** Il n'enregistre que 100 ms, soit 98 impulsions pour
    1 600 images. Les spectres suivants d'un run construit depuis le brut restaient sans temps (le panneau dDAC n'en montrait
    que 98) : ils suivent maintenant la période mesurée. La v1 comptait 1 ms par image ; la période vraie est de 1,0135 ms.
    Les `.CEDX` convertis gardent les temps de la v1, et n'ont même plus d'oscilloscope.
34. **Une voie de trigger peut ne porter que du bruit.** Sur 744 traces de janvier 2026, 41 n'ont que du bruit sur `ch5`
    (D2O_01 scans 1 à 20, Water01 scans 7 à 41) : des milliers d'« impulsions » à quelques µs. L'inter-frame n'est donc
    affiché que pour des impulsions d'une même horloge, pas plus rapprochées que l'intégration. « Même horloge » doit
    tolérer le pas de l'oscilloscope : à 1 µs, 17,1 µs se lit 17 ou 18 µs, et 276 scans passaient pour irréguliers. Pour la
    même raison, la période se prend en moyenne de la première impulsion à la dernière, pas en médiane : 1,013 ms au lieu de
    1,01347 ms pour Water01 scan 74, soit 0,7 ms d'écart après 1 500 images prolongées.

## 6. Reste à faire (par priorité)

**Où on en est (9 novembre 2026).** L'utilisateur a essayé les zones (« tout est bien »). Faits ensuite à sa demande, à
essayer : la liste des spectres dans un onglet, les raies des phases sur le cake, le repère 2θ (trait ou arc), le mode Live, et
l'intégration, l'inter-frame et la fréquence d'acquisition dans la barre d'état. Les documents ont été comparés au code et à toute l'interface de la
v1 ; ce qui manquait est dans P4. Pour reprendre :
0. **Temps de l'acquisition** : intégration, inter-frame (temps entre deux intégrations) et fréquence, comme il l'a défini
   (10 novembre). À lui demander encore : les `.CEDX` du mode continu
   gardent les temps de la v1 (1 ms par image au lieu de 1,0135 ms) : faut-il les corriger, ou refaire ces runs depuis le
   brut ? Les scans dont `ch5` n'est que du bruit (D2O_01 1 à 20, Water01 7 à 41, §5 point 34) prendraient, construits
   depuis le brut, leurs temps dans ce bruit, comme avant : piste à lui proposer, la période de 1,0135 ms (le `count_time`
   BLISS vaut nombre d'images × 1,0135 ms pour ces scans).
1. **Essayer le mode Live sur une vraie acquisition** : un arrêt de plus de 2 s de l'écriture au milieu d'un scan en ferait
   lire une partie. Si cela arrive, comparer le nombre d'images au fichier maître. Puis lui faire choisir la suite dans P4
   (l'export du résumé en CSV d'abord, à notre avis). Il n'a pas encore dit si ses campagnes ont des fichiers d'oscilloscope séparés.
2. La zone d'ajustement est par spectre, alors que la v1 en avait une par jauge : il n'a rien demandé, ne pas changer sans lui.
3. Défaut de la recherche sur D2O_03 scan 6 #0 (P5, poids des pics forts).
4. Outil de calibration : `test.poni` faux de 4,4 mm. L'utilisateur n'a pas encore dit sur quelle image et quels
   anneaux il l'a fait.
5. « Ne se lance plus avec l'exe » (P1) : pas revu. Le journal `trxrd-gui.log` montre des lancements normaux depuis
   (4 novembre, 02:22, 02:44 et 03:02, fermés avec le code 0). À regarder si cela revient.
6. La pull request `gui-viewer` vers `main` reste à ouvrir par l'utilisateur (`gh` n'est pas installé). Une seule
   suffit : `gui-viewer` contient `p2-validation-2026`, et `main` n'a rien de plus. Les derniers commits ne sont pas
   poussés : il n'a pas encore dit de le faire.

### P1 — Interface PySide6 (en cours)

Architecture (`docs/ARCHITECTURE.md` §9) : **MVP** au-dessus de `services.Session`. Données affichées et
presenters sans Qt dans `trxrd/gui/presenters/`, vues passives dans `trxrd/gui/views/`, registre unique de
`QAction` (`gui/actions.py`), calculs longs dans un thread (`gui/tasks.py`, avec `progress(done, total)`).
Seul `trxrd/gui/` a le droit d'importer PySide6/pyqtgraph (règle vérifiée par test). Lancer : `trxrd-gui`.

Étape 1, le lecteur (fait, branche `gui-viewer`) :
- [x] ouvrir un `.trxrd.h5` ou un `.CEDX`, en arrière-plan ; un `.CEDX` s'enregistre en `.trxrd.h5` à côté ;
- [x] curseur des spectres, entre deux flèches, et leur liste dans l'onglet Liste ;
- [x] spectre courant : données, fond, ajustement ; raies attendues de chaque phase dans un bandeau sous le
  spectre, qui reste visible au zoom, et en traits fins à travers le spectre (Ctrl+G) ; axe 2θ / Q / d ;
- [x] P(t) par phase avec barres d'erreur, et une voie de l'oscilloscope sur l'axe de droite ;
- [x] carte intensité (échelle log par défaut) ;
- [x] tables des phases et des pics du spectre courant ;
- [x] fenêtre des raccourcis (F1), confirmation avant de perdre des modifications.

Étape 2, première partie (fait, branche `gui-viewer`) :
- [x] nouveau run depuis RAW_DATA (Ctrl+N) : jeu de données, scans du fichier maître, calibration,
  phases, dossier ; intégration en arrière-plan avec progression et arrêt ;
- [x] cake (2θ × azimut) ou image brute du spectre courant sous le spectre, aligné sur son axe (Ctrl+K) ;
- [x] calibration à la Dioptas (Ctrl+L, ou « Calibrer… » à la création d'un run) : image du calibrant,
  LaB6 / CeO2, départ `.poni` ou estimation, clic sur deux anneaux, calibration, `.poni` enregistré ;
- [x] panneau dDAC, comme dans la v1 : à droite, une grandeur de chaque phase en fonction du temps (P, V, a,
  b, c, c/a, b/a, T ; « ÷ fiche » pour X/X₀) avec une voie de l'oscilloscope (le piézo) et la plage de
  l'oscilloscope entier ; en dessous, dP/dt, puis la carte des diffractogrammes intégrés (temps en x,
  2θ / Q / d en y), sur le même axe du temps ;
- [x] dP/dt (demandé par l'utilisateur) : par échantillon (glaces VII, VI, Ih ensemble) ou par phase, pente
  d'une droite glissante sur ±k spectres ; dP/dt moyen sur un intervalle déplaçable : pente de la droite et
  ΔP/Δt de la pression lissée, à copier dans un tableur ;
- [x] images (cake, carte, calibrant) : histogramme pour tirer les niveaux, échelle log / linéaire / racine,
  carte des couleurs, niveaux auto ou gardés d'un spectre à l'autre ; choix mémorisés ;
- [x] aperçu du scan brut à la création d'un run (demandé par l'utilisateur) : image ou cake sur tout le
  détecteur (maximum du scan ou une image), diagramme et couverture en azimut, plage 2θ tirée à la souris ;
- [x] spectre affiché filtré, fond soustrait (brut sur demande, Ctrl+B) ; onglet Traitement (fond, filtre,
  au cas par cas puis propagé à tous ou à la sélection) ; onglet Détection (pics détectés, seuil, raisons
  des rejets, réglages de l'analyse) ; sélection de plusieurs spectres (onglet Liste, ou zone de spectres) ;
- [ ] **outil de calibration à vérifier** : le `.poni` fait par l'utilisateur pour D2O_03 (`F:\ESRF\TEST\test.poni`)
  est faux de 4,4 mm sur la distance (glace Ih décompressée à −0,55 GPa). Savoir sur quelle image et avec
  quels anneaux il a été fait ; ajouter une vérification (glace Ih ou calibrant intégré, écart des raies) ;
- [ ] traitement de tous les spectres dans un thread (3,2 s pour 1 600 spectres, l'interface est bloquée) ;
- [ ] « le script ne se lance plus avec l'exe » (4 novembre 2026) : cause non trouvée. Lancé depuis les outils de
  Claude, l'exécutable démarre, avec ou sans fichier, et aucun blocage ni plantage n'était inscrit dans les
  journaux de Windows. Regarder `trxrd-gui.log` au prochain échec. Vu une seule fois, non reproduit : avec
  D2O_03, le processus ne s'est pas terminé après la fermeture ;
- [x] pics larges (liquides) : fenêtre du fond SNIP réglable (format 1.1), largeur max de la détection,
  seuil des pics larges de la recherche (« jamais »), fenêtre d'ajustement qui suit la largeur des pics.
  Reste à essayer sur un vrai liquide (Ga liquide de janvier 2026, fiche `Ga_liq`) ;
- [x] lancer l'analyse automatique depuis l'interface (Ctrl+R) : phases candidates, spectres, plage de P,
  zones exclues (gardées dans le run), pics fixes proposés puis exclus en relançant, avec progression et arrêt ;
- [ ] onglet « masque » de la calibration : dessiner ou modifier un masque, comme dans Dioptas. Pour
  l'instant, on charge celui de Dioptas.

À reprendre selon l'usage de l'utilisateur :
- la voie « piézo » : `ch6` (`Channel3`) varie de 0 à 7,4 V, c'est probablement la consigne du piézo. Il est
  montré par défaut, mais il faut le confirmer avec l'utilisateur. Le rôle de `ch2` est inconnu ;
- des icônes pour la barre d'outils ;
- l'axe de P(t) passe à « P (GPa) (x0.001) » quand la plage est petite (préfixe automatique de pyqtgraph) ;
- la largeur minimale de la fenêtre : 1 841 pixels, imposés par les barres de réglages du cake (748) et de la carte
  (695). Elle tient sur 1 920 pixels, pas sur un écran plus étroit ;
- le zoom du panneau dDAC, qui revient à la plage des spectres à chaque modification du run ;
- aperçu du scan brut, à peaufiner à la fin si besoin : raies attendues des phases sur le diagramme,
  contour de la plage sur l'image ;
- superposer plusieurs diffractogrammes (sélection de spectres, décalage vertical) : pas demandé, à proposer
  si la carte ne suffit pas.

Étape 3, l'ajustement interactif et la gestion des phases (demandés ensuite par l'utilisateur) (fonctions de la v1 à reprendre, d'après ses raccourcis) :
- [x] ajouter / retirer une phase sur un spectre, à une P de départ ; déplacer les raies attendues avec P ;
- [x] annuler / rétablir (Ctrl+Z, Ctrl+Y), y compris une analyse automatique et le traitement ;
- [x] garder et tracer les pics libres (zones exclues, pics inexpliqués) : `Spectrum.free_peaks`, format 1.2 ;
- [x] « fit total » d'un spectre (Ctrl+F), puis « fit all » de la sélection ou de tous, en suivant à partir du
  spectre courant ;
- [x] pics à la souris (Ctrl+E) : poignées à tirer, Ctrl + clic pour ajouter (raie de la phase choisie, ou pic
  libre), Suppr, activer / désactiver ; la zone d'ajustement s'étend pour couvrir ce qu'on ajoute ;
- [x] ajustement personnalisé, comme dans la v1 : modèle et paramètres de chaque pic, ajustement d'un seul pic
  (Maj+R, fenêtre en σ, Var), vue zoomée, pics remplis, résidu, cases par raie, comparaison avec le fit
  précédent, curseur et touches de la v1 ;
- [x] zones d'ajustement et zones exclues dessinées à la souris (onglet Zones, Z), zone sur les graphes dDAC pour
  choisir les spectres. La zone d'ajustement est par spectre et non par jauge comme dans la v1 ;
- [x] copier / coller le modèle de phases d'un spectre sur une sélection de spectres ; supprimer les phases d'une zone de spectres ;
- [x] détection automatique des pics affichée (marqueurs), avec SNR (onglet Détection) ;
- [ ] note du run.

Plus tard : éditeur de fiches `.jcpds` (export de la maille ajustée, P4), thème clair/sombre. La v1 avait aussi une
console Python intégrée, qui n'est pas prioritaire.

### P2 — Valider sur les archives de l'utilisateur

- [x] `trxrd compare` sur **tous** les `.CEDX` de janvier 2026 (143 runs) : 61 % → 81 % (§4,
  `docs/ARCHITECTURE.md` §6.8). Réglages corrigés avec un test synthétique par cas, sans dégrader Water05.
- [x] Domaines de pression de Sn β et γ : décidés par l'utilisateur (§2) et appliqués.
- [ ] **Encore à décider avec l'utilisateur** (`docs/ARCHITECTURE.md` §13) :
  - Sn bcc (fiche P[9,100]) : il se fait passer pour un Sn γ texturé ; le remonter vers 35 GPa ?
  - phase non cubique vue sur une seule raie : la signaler « présente, P indéterminée » ?
  - fiches `Bi-BetaSn` et `Ga_liq` ;
  - fiche `Cr` : elle annonce « EoS Anzellini » avec V0 = 23,985 Å³, alors que le catalogue Peritheos donne 24,08 pour
    Anzellini 2022 (table 3), avec les mêmes K0 et K0'. Cela fait +0,8 GPa à 5 GPa et +1,6 GPa à 50 GPa. À vérifier dans
    l'article ;
  - glace VII au-dessus de 10 GPa : la fiche suit Bezacier 2014 (mesures jusqu'à 10,1 GPa), mais donne environ 1 GPa de
    moins que Frank 2004 à 14 GPa (Water11). Comparer la glace et Sn d'une même cellule avant de choisir.
- [ ] Intégrer le script de validation d'une campagne (dossier de `.CEDX` → tableau par famille) dans le
  dépôt, par exemple `trxrd compare` sur un dossier.
- [ ] Mesurer aussi les campagnes 2023, 2024 et 2025 (fichiers `.CEDXUp`, section 7).
- [ ] Désaccords restants à examiner :
  - Water11 (22 %) : la référence place Sn γ entre 5 et 7,8 GPa, sous la limite de 8 GPa, sur une seule
    raie ; la v2 prend ces raies pour de la glace VI. Sn β seul est pris pour de la glace VII à 14-15 GPa ;
  - Water12_scan0015 (7 %) et Water12_scan0022 (32 %) : sauts de la glace VII entre deux pics (110)
    candidats ; glace VI en plus là où la référence n'a que Sn (oubli de la référence ?) ;
  - coexistence glace VI + VII sur des raies faibles ;
  - Water01_scan0057 #8-9 : Sn β seul à 5,2 GPa au lieu de la glace VI.
- [ ] Pistes mesurées mais non retenues :
  - poids logarithmique des pics forts : il fait gagner D2O_02 mais effondre Water01 + D2O_01 (87 % → 59 %) ;
  - marge commune sur les domaines de toutes les fiches : elle fait apparaître Sn bcc.

### P3 — Écart sur Sn à basse pression

Expliqué (`docs/ARCHITECTURE.md` §6.8). Par rapport aux ajustements manuels crédibles (~1 050 pour Sn β),
la v2 donne −0,07 GPa pour Sn β et −0,10 GPa pour Sn γ (médianes). L'écart vient des centres de pics (−0,004°, profil
Pearson IV), et non de la pondération, de l'EoS ou des raies utilisées. Les grands écarts de Sn01 viennent
d'ajustements manuels jamais refaits (modèle copié, forme par défaut). Pour aller plus loin, il faut les
images brutes de Water05 :
- scan `scan_jf1m_0000.h5` du scan 9 ;
- `.poni` `calib_ddac2_correct.poni` ;
- masque.

On reconstruit le run avec `trxrd build`, puis on compare les résidus des deux profils.

### P4 — Fonctions de la v1 pas encore portées

- [ ] **Corrections manuelles de la note** : la v1 appliquait les lignes `[colonne] -> valeur` de la note
      comme décalages dans le Summary (`Corr_Summary`). La v2 conserve la note mais ne l'applique pas.
- [ ] **Annotations** de la v1 (`run.analyse` : points marqués sur P(t), avec id, kind, label, spec_idx,
      time_s, P_GPa…) et plage de temps `multi_time_range_ms` : ni convertis ni stockés. Prévoir une table
      `/entry/annotations` dans le format, avec une version mineure 1.1.
- [ ] **Spectres statiques ajoutés** avant ou après la série (`Add_Static`).
- [ ] **Entrées DRX autres que HDF5** dans `run_builder` : dossier ou liste de fichiers EDF/TIFF via fabio.
      La v1 le gérait.
- [ ] **Édition d'une fiche** (`f_edit_gauge`) et export `.jcpds` depuis l'interface ; **T d'une phase** (`T_All`,
      spinbox T) : la v2 prend une seule température par analyse.
- [ ] **Mode « température »** : la v1 avait `fixe="T"` (T calculée à P fixée, via `ALPHAT`). La v2 gère
      V0(T) mais pas ce calcul inverse.
- [ ] **Acquisition directe** de l'oscilloscope LeCroy (`Oscillo_autosave`) : probablement hors du périmètre
      de la v2, à confirmer avec l'utilisateur.

Trouvé en comparant toute l'interface et le guide de la v1 (novembre 2026), par utilité décroissante :
- [ ] **Export du résumé en CSV** (`summary.to_csv`) : aucune commande dans l'interface ni dans `trxrd`. Pour l'instant, il
      faut passer par `XRDRun(...).summary` en notebook, ou par « Copier » pour le dP/dt moyen.
- [ ] **Oscilloscope dans des fichiers séparés** (`.trc`, CSV) : la v2 ne lit que le fichier maître BLISS. À vérifier
      pour les campagnes 2023 à 2025.
- [ ] **Spectres ASCII** (deux colonnes, découpés en spectres).
- [ ] **Retirer une fiche du run** (Maj+D) : on peut en ajouter, pas en retirer.
- [ ] **Recharger le fichier** (F4), **détacher un panneau** (Undock) et **masquer la colonne de gauche** (Maj+P).

Rien à reprendre : Y (les cases de raies agissent tout de suite), la ligne de seuil « exclusion line » (seuil en pointillés de
l'onglet Détection), `vslmfit` (lmfit abandonné), Maj+U (Ctrl+Z), Maj+B (le fond suit l'onglet Traitement), F5 (Ctrl+W).

### P5 — Améliorations d'algorithmes repérées

- [x] **Suivi** : il s'interrompt maintenant dès qu'un pic de SNR ≥ 10 reste inexpliqué. Une recherche
      forcée tous les N spectres a été mesurée : elle n'apporte rien.
- [ ] **Poids des pics forts dans la recherche** : le poids d'un pic sature à SNR = 10. Sur
      D2O_03_0001_scan0006 #0, une solution qui laisse inexpliqué le pic (110) de la glace VII (SNR 217) bat la
      bonne de peu (2,18 contre 2,23), car la raie (200), large de 0,55°, est ignorée (> 0,5°). À examiner
      avec l'onglet Détection ; le poids logarithmique, déjà mesuré (P2), n'a pas été retenu.
- [ ] **Point de départ du suivi** : un premier spectre à un seul pic mal indexé se propage (Ga02 : KCl B2 à
      5,8 GPa au lieu de B1 à 1 GPa). Piste : partir du spectre le mieux déterminé, ou analyser dans les deux
      sens.
- [ ] **Détection** : une seconde passe sur le résidu (données − ajustement) pour trouver les raies faibles
      cachées. Les 2 pics manqués sur Water05 sont des raies faibles de la glace VI.
- [x] **Performances** mesurées : ~0,1 s par spectre en moyenne ; 102 s pour les 1 600 spectres de
      Water01_scan0074. Paralléliser n'est pas nécessaire.
- [ ] **Fiche Ga-II** (K0' = 0) : P(V) plafonne à 0,93 GPa alors que la fiche annonce 1–10 GPa. À signaler à
      l'utilisateur, qui doit vérifier la fiche.
- [ ] **Erreur de l'EoS dans σP** : σP ne tient compte que de l'erreur sur V. Pour la glace VII, à 6 GPa, les erreurs de
      Bezacier 2014 sur V0 et K0 pèsent 0,03 et 0,05 GPa (sans la covariance). À faire dans notre code, si l'utilisateur le veut.

### P6 — Dépôt et versions

- [x] **Tag `v1-cedapp`** sur `6f579f5`, poussé sur GitHub en novembre 2026.
- [x] Fusion dans `main` (PR #16).
- [ ] **Séparation en nouveau dépôt** (`docs/ARCHITECTURE.md` §10), quand l'utilisateur le décidera.

## 7. Environnement local de l'utilisateur

- Windows. Son Python historique est `f:\CODE\python_3_10_15` (Python 3.10). Pour la v2, conseiller un
  environnement séparé (`.venv-v2`), en Python **3.11 ou plus** de préférence. En 3.10, le test qui compare
  les imports au `pyproject.toml` est ignoré, faute de `tomllib`.
- Environnement en place depuis novembre 2026 : `.venv-v2` en Python 3.14, créé avec `uv`
  (`uv venv .venv-v2 --python 3.14` puis `uv pip install --python .venv-v2/Scripts/python.exe -e "./trxrd[notebook,gui,dev]"`).
- Rediriger ou chaîner la sortie de `trxrd` sous Windows (`> fichier`, `| more`) faisait planter le programme
  sur les caractères θ et Δ. C'est corrigé ; si un script Python lit cette sortie, mettre `PYTHONUTF8=1`.
- La v1 et la v2 dans deux environnements distincts : PyQt5 et PySide6 ne cohabitent pas.
- Archives probables, d'après le notebook `Use_drx.ipynb` :
  `F:\ESRF\2023 Mai HC-5293 id09`, `F:\ESRF\2024 Juin HC-5719 id09`, `F:\ESRF\2025 Juin HC -6205 id09`,
  `F:\ESRF\2026 Janvier HC-6664 id09`. Chacune a un sous-dossier `Fichier_CED_DRX\` contenant les `.CEDX`
  et `.CEDXUp`. Le laboratoire a aussi `F:\Aquisition_Banc_CEDd\Fichier_CEDd\`, des fichiers `.CEDUp` du
  banc de spectroscopie, probablement inutiles pour la v2. Les données brutes sont dans `RAW_DATA\<échantillon>\<dataset>\scanNNNN\`.
- Données de test dans le dépôt : `trxrd/tests/data/` (scan TUTO, sa calibration, un `.CEDX` d'exemple
  généré par `make_legacy_sample.py`).
- Catalogue Peritheos : `pip download peritheos --no-deps --only-binary :all: -d <dossier>` donne un `.whl`, qui est un zip.
  Ses fichiers `peritheos/data/materials/*.eosmat` sont du JSON : on les lit sans exécuter Peritheos, donc sans son module
  Rust non signé (jamais essayé sur ce PC). Vérifié sur la version 0.17.4, pour faire une fiche :
  - V0 est en Å³ par maille conventionnelle, comme `VOLUME` ;
  - prendre un enregistrement `BM3`, ou `BM2` avec K0' = 4 : nos fiches ne connaissent que Birch-Murnaghan, et 191 des
    685 enregistrements sont en Vinet ;
  - calculer `A` à partir de V0 : le bloc `lattice` peut décrire un autre état (V0/a³ = 1,155 pour l'argon) ;
  - noter le domaine, l'échelle de pression (`pressure_calibration`) et l'article d'origine, à relire pour une fiche de
    référence ;
  - les raies : seuls 13 matériaux sur 234 en ont (`peaks`). Pour les autres, il faut les calculer à partir de la structure
    (`atom_sites`, 187 matériaux), ce que `trxrd` ne sait pas faire ;
  - droits : le code est sous MIT, mais les données ont des conditions propres à chaque source, parfois restreintes ou non
    précisées (`peritheos/data/LICENSING.md`). Reprendre fiche par fiche en citant la publication, jamais tout le catalogue.

## 8. Façon de travailler attendue

- Répondre en français, et expliquer les choix (l'utilisateur est physicien, pas développeur de métier).
- Ne jamais toucher à la v1. Tout se passe dans `trxrd/`.
- Garder `python -m pytest trxrd` au vert. Chaque nouveau comportement vient avec un test, et chaque bug
  trouvé sur des données réelles avec un test synthétique qui le reproduit.
- Respecter les couches et les dépendances : les tests échouent sinon. Pas de lmfit, dill, deap, tkinter
  ni PyQt5.
- Tout changement du format `.trxrd.h5` met à jour `FORMAT_VERSION`, `docs/ARCHITECTURE.md` §5 et les tests
  d'aller-retour.
- Ne jamais committer de données de l'utilisateur. Messages de commit explicites, en anglais. Travailler
  sur une branche, puis fusionner dans `main` par pull request.
- Mettre à jour ce fichier (sections 3, 4 et 6) à chaque étape importante.
