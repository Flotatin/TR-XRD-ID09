# DRX

DRX est une application Python à interface graphique dédiée au traitement et à
la visualisation de données de diffraction des rayons X. L’outil centralise le
pilotage du détecteur, l’intégration 1D des images et l’analyse des spectres pour
les expériences en laboratoire ou sur ligne de lumière.

## Prérequis

- Python 3.8 ou version ultérieure
- `dill`
- `numpy`
- `pandas`
- `PyQt5`
- `pyqtgraph`
- `scipy`
- `matplotlib`
- `pynverse`
- `pyFAI`
- `fabio`
- `scikit-image`

Installez les dépendances avec :

```bash
pip install -r requirements.txt
```

## Guide d’utilisation

### 1. Charger la calibration (.poni) et le masque

1. Depuis le menu de configuration (icône « ⚙ » ou `Shift+P`), pointez le fichier
   de configuration texte à utiliser. Celui-ci mémorise les chemins vers les
   dossiers de données ainsi que la calibration courante (`calib_file_mask`,
   `calib_file_poni`).
2. Ouvrez la boîte *Calibration DRX* afin de (re)charger le masque et le fichier
   `.poni`. L’utilitaire `Calib_DRX` lit le masque, charge la géométrie PyFAI et
   permet de vérifier le résultat sur l’image intégrée.【F:Bibli_python/Calibration.py†L10-L145】【F:ID09_CEDX_vtravaux.py†L1274-L1286】
3. Ajustez éventuellement la fenêtre angulaire via la région interactive avant de
   valider. Les paramètres sont appliqués à toutes les intégrations suivantes.【F:Bibli_python/Calibration.py†L96-L120】

### 2. Importer les fichiers DRX et oscilloscope

1. Sélectionnez les dossiers racines pour les DRX et l’oscilloscope (`Shift+S`
   pour créer un nouveau spectre). Les boutons de navigation permettent de
   parcourir les répertoires configurés et de filtrer la liste des fichiers.【F:ID09_CEDX_vtravaux.py†L1162-L1208】
2. La fonction *Select DRX* accepte les formats gérés par `fabio` (HDF5, EDF,
   TIFF, etc.) et détecte automatiquement le nombre d’images contenus dans un
   fichier multi-trames.【F:ID09_CEDX_vtravaux.py†L1260-L1267】
3. Le bouton *Select oscillo* enregistre le fichier brut (par exemple `.trc` ou
   CSV). L’import ASCII des scans est pris en charge : un fichier deux colonnes
   est automatiquement re-découpé en spectres individuels et indexé dans la
   liste déroulante.【F:ID09_CEDX_vtravaux.py†L1162-L1185】【F:ID09_CEDX_vtravaux.py†L1269-L1272】
4. Lorsque l’on sélectionne d’abord un tir oscilloscope, l’interface tente de
   retrouver le DRX correspondant dans l’arborescence en respectant la
   nomenclature `xxx_##_scan####`. Un message apparaît si la correspondance
   échoue.【F:ID09_CEDX_vtravaux.py†L1216-L1244】

### 3. Créer et sauvegarder un objet CEDX

1. Chargez ou intégrez un premier spectre (commande `Shift+S`) puis lancez
   `Shift+E` pour générer un CEDX vide à partir des fichiers DRX/oscillo ouverts.
   L’objet `CL.CED_DRX` encapsule les spectres, la calibration et les métadonnées
   d’acquisition.【F:ID09_CEDX_vtravaux.py†L888-L905】【F:ID09_CEDX_vtravaux.py†L2803-L2824】
2. Pour automatiser la détection de pics et l’initialisation des jauges sur un
   lot, utilisez *New CEDd* (bouton ou raccourci) qui exécute une intégration
   complète, détecte les pics et associe les phases candidates via
   `F_Find_compo`.【F:ID09_CEDX_vtravaux.py†L2724-L2801】
3. Une fois les spectres traités ou corrigés manuellement, sauvegardez l’objet
   avec `F3` (*Save CEDd*) : toutes les jauges, séries temporelles et résumés
   sont écrits sur disque via `CL.SAVE_CEDd`. Le chemin de sauvegarde est issu de
   `folder_CED` et du nom du tir oscilloscope.【F:ID09_CEDX_vtravaux.py†L898-L906】【F:ID09_CEDX_vtravaux.py†L2790-L2799】【F:ID09_CEDX_vtravaux.py†L3065-L3068】
4. `F4` recharge les données depuis le disque pour rafraîchir les graphiques, et
   `F5` efface l’état courant sans quitter la session.【F:ID09_CEDX_vtravaux.py†L892-L905】【F:ID09_CEDX_vtravaux.py†L2487-L2503】

### 4. Lancer les pipelines de fit

1. Les zones d’intérêt se tracent via le bouton *Add zone* ou le raccourci `Z`.
   Les régions contrôlent la plage angulaire utilisée lors des recherches de
   pics.【F:controllers/spectrum_controller.py†L64-L123】【F:ID09_CEDX_vtravaux.py†L1293-L1316】
2. Le bouton *run_fit_selected_spectra* (ou le raccourci associé) délègue au
   `RUN` actif un ajustement multi-spectres. Les paramètres de l’algorithme
   génétique (NGEN, MUTPB, etc.) et des pics (hauteur, largeur) proviennent du
   panneau *FindCompo*。【F:controllers/spectrum_controller.py†L127-L163】【F:widgets/find_compo_widget.py†L6-L63】【F:ID09_CEDX_vtravaux.py†L1293-L1316】
3. Le bouton « Multi fit » appelle `_CEDX_multi_fit` : chaque spectre de l’intervalle
   est rechargé, ajusté via `FIT_lmfitVScurvfit` puis réinjecté dans le RUN. Une
   barre de progression permet d’interrompre la boucle si nécessaire.【F:ID09_CEDX_vtravaux.py†L1970-L2033】
4. Pour rechercher automatiquement les phases sur une série temporelle, utilisez
   `_CEDX_auto_compo`. Le pipeline détecte les pics filtrés par les zones actives
   et propose les meilleures combinaisons de jauges tout en alimentant le tableau
   `Summary`.【F:ID09_CEDX_vtravaux.py†L2826-L3016】

## Gestion des jauges

- **Sélection** : la table de gauche liste les jauges disponibles. Un double
  clic ou `Shift+A` charge l’élément depuis la bibliothèque (`Bibli_elements`) ou
  depuis le spectre courant. `f_gauge_select` assure la synchronisation entre la
  table et le panneau de détail.【F:ID09_CEDX_vtravaux.py†L1361-L1398】
- **Réglages P/T** : les spinbox de pression et température pilotent
  `GaugeController`. Chaque modification met à jour les lignes d-hkl sur le
  graphique principal et la vue différentielle.【F:controllers/gauge_controller.py†L173-L309】
- **d-hkl de référence** : `f_Gauge_Load` génère une case à cocher par pic, en
  initialisant leur visibilité selon la fenêtre angulaire du spectre. Les lignes
  peuvent être figées pour les autres jauges via `refresh_fixed_lines`, et leurs
  états sont mémorisés pour la sauvegarde.【F:controllers/gauge_controller.py†L314-L390】
- **Export / impression** : `F3` puis `Save CEDd` persiste les jauges modifiées,
  tandis que `I` (Output bib gauge) permet d’exporter un élément au format JCPDS
  pour enrichir la bibliothèque.【F:ID09_CEDX_vtravaux.py†L845-L883】【F:ID09_CEDX_vtravaux.py†L3065-L3068】

## Formats supportés et configuration

- **Images DRX** : ouverture générique via `fabio.open` (HDF5 `scan_jf1m_0000.h5`,
  EDF, TIFF…), avec gestion multi-image par index de trame.【F:ID09_CEDX_vtravaux.py†L1260-L1267】【F:ID09_CEDX_vtravaux.py†L3052-L3063】
- **Spectres ASCII** : lecture de fichiers texte à colonnes, découpe automatique
  en spectres et remplissage du sélecteur `Spec n°i`.【F:ID09_CEDX_vtravaux.py†L1162-L1185】
- **Oscilloscope / piezo** : les fichiers sont référencés dans `RUN.data_oscillo`
  et utilisés pour superposer la tension piézo aux courbes de pression.【F:ID09_CEDX_vtravaux.py†L2087-L2095】【F:ID09_CEDX_vtravaux.py†L2110-L2140】
- **Configuration** : les fichiers `config/*.txt` définissent les dossiers
  (`folder_DRX`, `folder_OSC`, `folder_CED`), les fichiers récemment chargés,
  l’énergie en keV et la bibliothèque de phases via la clé `bib_files`. Modifiez
  ces valeurs pour personnaliser votre session.【F:config/config_labo.txt†L1-L11】
- **Thèmes** : la boîte de paramètres propose un thème clair ou sombre.
  L’option met à jour la feuille de style globale et la palette PyQtGraph.【F:ID09_CEDX_vtravaux.py†L778-L793】

## Raccourcis et références utiles

- `Ctrl+Entrée` : exécuter le code Python dans la console embarquée.【F:ID09_CEDX_vtravaux.py†L824-L869】
- `Shift+S` : intégrer un nouveau spectre à partir de l’image courante.【F:ID09_CEDX_vtravaux.py†L850-L858】
- `Shift+F` : lancer un fit global sur le spectre affiché.【F:ID09_CEDX_vtravaux.py†L855-L858】
- `Shift+A` / `D` : ajouter ou supprimer une jauge du spectre.【F:ID09_CEDX_vtravaux.py†L872-L883】
- `Shift+E`, `F3`, `F4`, `F5` : créer, sauvegarder, recharger ou vider un objet
  CEDX.【F:ID09_CEDX_vtravaux.py†L887-L906】
- `M` : détection automatique de pics sur le spectre courant.【F:ID09_CEDX_vtravaux.py†L924-L927】
- `Z` : afficher ou masquer les régions de fit.【F:ID09_CEDX_vtravaux.py†L939-L940】

Les références complémentaires sont listées dans `txt_file/Command.txt` : on y
retrouve les attributs accessibles depuis la console (`self.RUN.Spectra`,
`self.Spectrum.Gauges`, etc.) pour inspecter ou modifier l’état interne.【F:txt_file/Command.txt†L1-L39】

## Dépannage / FAQ

- **« warn: no calibration loaded »** : assurez-vous d’avoir chargé un couple
  masque/poni valide avant de lancer l’intégration. Sans calibration, la
  modification est bloquée.【F:ID09_CEDX_vtravaux.py†L1274-L1286】
- **Impossible de créer un CEDd** : le message « Calibration absente… » apparaît
  si la calibration n’est pas initialisée ou si aucun fichier DRX n’est chargé.
  Chargez un spectre et vérifiez les chemins configurés.【F:ID09_CEDX_vtravaux.py†L2724-L2808】
- **« No RUN loaded » lors d’un multi-fit** : ouvrez ou créez un CEDX avant
  d’exécuter `run_fit_selected_spectra` ou `_CEDX_multi_fit` (les pipelines
  s’appuient sur `self.RUN`).【F:ID09_CEDX_vtravaux.py†L1293-L1317】【F:ID09_CEDX_vtravaux.py†L1970-L2033】
- **Chemin de sauvegarde invalide** : définissez `folder_CED` dans le fichier de
  configuration pour permettre l’écriture des CEDX générés.【F:ID09_CEDX_vtravaux.py†L2790-L2799】【F:config/config_labo.txt†L1-L11】
- **Prérequis logiciels** : vérifiez que toutes les dépendances listées dans
  `requirements.txt` (PyQt5, pyFAI, fabio, h5py, etc.) sont installées. Certaines
  fonctionnalités (lecture HDF5, intégration pyFAI) échoueront sinon.【F:requirements.txt†L1-L23】

## Journalisation

Le module configure désormais automatiquement le logging Python au démarrage
avec un format horodaté, un gestionnaire console et un niveau `INFO` par
défaut.【F:vDRX.py†L94-L108】【F:logging_config.py†L1-L38】 Pour activer un niveau
`DEBUG` plus verbeux, utilisez l’argument de ligne de commande `--debug` :

```bash
python vDRX.py --debug
```

L’option peut également être activée sans modifier la commande en définissant la
variable d’environnement `DRX_DEBUG=1` avant de lancer l’application.【F:vDRX.py†L100-L115】

## Licence

Ce projet est distribué sous licence MIT. Voir [LICENSE](LICENSE) pour plus de
détails.
