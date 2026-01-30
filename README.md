Python graphical application dedicated to X-ray diffraction (XRD) data processing and visualization.
The software centralizes detector control, 1D image integration, and spectral analysis for laboratory experiments or synchrotron beamlines.

Requirements

Python 3.8 or newer

Dependencies listed in requirements.txt

Install the dependencies with:

pip install -r requirements.txt

User Guide
1. Load calibration (.poni) and mask

From the configuration menu (⚙ icon or Shift+P), select the text configuration file to use.
This file stores paths to data directories as well as the current calibration (calib_file_mask, calib_file_poni).

Open the Calibration DRX dialog to (re)load the mask and the .poni file.
The Calib_DRX utility reads the mask, loads the PyFAI geometry, and allows verification on the integrated image.

Optionally adjust the angular window using the interactive region before validating.
The selected parameters are applied to all subsequent integrations.

2. Import XRD and oscilloscope files

Select the root directories for XRD and oscilloscope data (Shift+S to create a new spectrum).
Navigation buttons allow browsing configured directories and filtering file lists.

The Select DRX function supports formats handled by fabio (HDF5, EDF, TIFF, etc.) and automatically detects the number of frames in multi-image files.

The Select oscillo button loads raw oscilloscope files (e.g. .trc or CSV).
ASCII scan imports are supported: two-column files are automatically split into individual spectra and indexed in the selector.

If an oscilloscope shot is selected first, the interface attempts to locate the corresponding XRD file in the directory tree using the naming convention xxx_##_scan####.
A warning message is displayed if the match fails.

3. Create and save a CEDX object

Load or integrate a first spectrum (Shift+S), then press Shift+E to generate an empty CEDX object from the currently loaded XRD and oscilloscope files.
The CL.CED_DRX object encapsulates spectra, calibration, and acquisition metadata.

To automate peak detection and gauge initialization over a batch, use New CEDd (button or shortcut).
This runs a full integration, detects peaks, and associates candidate phases using F_Find_compo.

Once spectra have been processed or manually corrected, save the object with F3 (Save CEDd).
All gauges, time series, and summary tables are written to disk via CL.SAVE_CEDd.
The save location is derived from folder_CED and the oscilloscope shot name.

F4 reloads data from disk to refresh plots, and F5 clears the current state without exiting the session.

4. Run fitting pipelines

Regions of interest are defined using the Add zone button or the Z shortcut.
These regions control the angular range used during peak searches.

The run_fit_selected_spectra button (or its shortcut) delegates a multi-spectrum fit to the active RUN.
Genetic algorithm parameters (NGEN, MUTPB, etc.) and peak constraints (height, width) are defined in the FindCompo panel.

The Multi fit button calls _CEDX_multi_fit: each spectrum in the selected interval is reloaded, fitted using FIT_lmfitVScurvfit, and reinjected into the RUN.
A progress bar allows interruption of the loop if needed.

To automatically search for phases in a time series, use _CEDX_auto_compo.
This pipeline detects peaks filtered by active regions and proposes optimal gauge combinations while populating the Summary table.

Gauge management

Selection: the left-hand table lists available gauges.
A double-click or Shift+A loads an element from the phase library (Bibli_elements) or from the current spectrum.
f_gauge_select synchronizes the table and the detail panel.

Pressure / Temperature control: pressure and temperature spin boxes drive the GaugeController.
Each modification updates the d-hkl lines on the main plot and in the differential view.

Reference d-hkl lines: f_Gauge_Load creates one checkbox per peak and initializes visibility based on the current angular window.
Lines can be frozen across gauges, and their states are stored for saving.

Export / printing: F3 followed by Save CEDd persists modified gauges.
The I shortcut (Output bib gauge) exports an element in JCPDS format to enrich the phase library.

Supported formats and configuration

XRD images: generic opening via fabio.open (HDF5, EDF, TIFF, etc.), with multi-image handling through frame indexing.

ASCII spectra: reading of column-based text files with automatic splitting into individual spectra and population of the spectrum selector.

Oscilloscope / piezo data: files are referenced in RUN.data_oscillo and used to overlay piezo voltage on pressure curves.

Configuration: text files located in config/ define data directories (folder_DRX, folder_OSC, folder_CED), recently loaded files, beam energy (keV), and phase libraries via the bib_files key.
Edit these values to customize your session.

Themes: the settings dialog allows switching between light and dark themes, updating both the global Qt stylesheet and the PyQtGraph palette.

Shortcuts and references

Ctrl+Enter: execute Python code in the embedded console

Shift+S: integrate a new spectrum from the current image

Shift+F: run a global fit on the displayed spectrum

Shift+A / D: add or remove a gauge from the spectrum

Shift+E, F3, F4, F5: create, save, reload, or clear a CEDX object

M: automatic peak detection on the current spectrum

Z: show or hide fit regions

Additional references are listed in txt_file/Command.txt, which documents console-accessible attributes (self.RUN.Spectra, self.Spectrum.Gauges, etc.) for inspecting or modifying the internal state.

Troubleshooting / FAQ

“warn: no calibration loaded”: ensure that a valid mask/poni pair is loaded before running integration.
Without calibration, integration is blocked.

Unable to create a CEDd: the message “Calibration missing…” appears if calibration is not initialized or if no XRD file is loaded.
Load a spectrum and verify configured paths.

“No RUN loaded” during multi-fit: open or create a CEDX object before running run_fit_selected_spectra or _CEDX_multi_fit.

Invalid save path: define folder_CED in the configuration file to enable writing generated CEDX objects.

Software dependencies: ensure all packages listed in requirements.txt (PyQt5, pyFAI, fabio, h5py, etc.) are installed.
Some features (HDF5 reading, pyFAI integration) will fail otherwise.

License

This project is distributed under the MIT License.
See the LICENSE file for details.
