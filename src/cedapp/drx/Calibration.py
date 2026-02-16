from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QDoubleSpinBox,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt
import pyFAI
import fabio
import numpy as np
import pyqtgraph as pg
from PIL import Image
import os

ENERGY_CONSTANT_KEV_M = 1.239841984e-6

class Calib_DRX:
    def __init__(self,file_mask=None,file_poni=None,theta_range=[0,40], energy=None):
        self.file_mask = file_mask
        self.file_poni = file_poni
        self.mask = None
        self.ai = None
        self.theta_range=theta_range
        self.energy = energy
        self.Load_calib()

    def Load_calib(self,file_mask=None,file_poni=None):
        if file_mask:
            self.file_mask = file_mask
        
        if file_poni:
            self.file_poni = file_poni
        
        try:
            img = Image.open(self.file_mask)
            mask = np.array(img)
            # Conversion en masque binaire (0 = masqué, 1 = gardé)
            mask_bin = (mask > 0).astype(np.uint8)
            self.mask = mask_bin
            self.ai = pyFAI.load(self.file_poni)
            if getattr(self.ai, "wavelength", None):
                try:
                    self.energy = ENERGY_CONSTANT_KEV_M / self.ai.wavelength
                except Exception:
                    pass

        except Exception as e:
            print("ERROR Load calib",e)

    def Change_calib(self,file, energy=None):
        if energy is not None:
            self.energy = energy
        dialog = CalibDialog(
            file,
            mask=self.mask,
            ai=self.ai,
            theta_range=self.theta_range,
            folder_start=os.path.dirname(self.file_poni) if self.file_poni is not None else None,
            energy=self.energy,
        )
        if dialog.exec_() == QDialog.Accepted:
            self.file_mask = dialog.file_mask
            self.file_poni = dialog.file_poni
            self.mask = dialog.mask
            self.ai = dialog.ai
            self.theta_range=dialog.theta_range
            self.energy = dialog.energy
            return self.energy
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ai"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "file_poni"):
            try:
                self.ai = pyFAI.load(self.file_poni)
            except Exception:
                self.ai = None

class CalibDialog(QDialog):
    def __init__(
        self,
        file_img,
        mask=None,
        ai=None,
        theta_range=None,
        folder_start=None,
        energy=None,
    ):  # image d'entrée fournie ici
        super().__init__()
        self.setWindowTitle("Choix de calibration DRX")
        self.setMinimumSize(1000, 800)

        self.file_mask = None
        self.file_poni = None
        self.folder_start=folder_start
        self.mask = mask
        self.ai = ai
        self.theta_range=theta_range
        self.energy = float(energy) if energy is not None else 0.0
        if os.path.isfile(file_img):
            self.img_data = fabio.open(file_img).data  # image DRX d’entrée
        else:
            self.img_data = file_img
        self.tth = None
        self.intens = None
        self.label = QLabel("Sélectionnez un .mask et un .poni")
        self.label.setAlignment(Qt.AlignCenter)

        self.btn_mask = QPushButton("Charger fichier .mask")
        self.btn_mask.clicked.connect(self.load_mask)

        self.btn_poni = QPushButton("Charger fichier .poni")
        self.btn_poni.clicked.connect(self.load_poni)

      
        self.btn_update_image = QPushButton("update image")
        self.btn_update_image.clicked.connect(self.update_image)
        self.btn_ok = QPushButton("Valider")
        self.btn_ok.clicked.connect(self.accept)

        self.energy_input = QDoubleSpinBox()
        self.energy_input.setDecimals(2)
        self.energy_input.setRange(0.0, 200000.0)
        self.energy_input.setSingleStep(10.0)
        self.energy_input.setValue(self.energy)

        # Affichage image 2D + masque
        self.view = pg.GraphicsLayoutWidget()
        self.img_view = self.view.addViewBox()
        self.img_item = pg.ImageItem()
        self.img_view.addItem(self.img_item)


        self.theta_range_label = QLabel("Pressure range : [0 - 100] GPa")


        # Affichage du profil 1D
        self.plot_1D = pg.PlotWidget(title="Intégration radiale (2θ)")
        self.plot_1D.setLabel('bottom', "2θ (°)")
        self.plot_1D.setLabel('left', "Intensité (a.u.)")


        self.curve_1D = self.plot_1D.plot([], [], pen='b')
        self.region = pg.LinearRegionItem(values=self.theta_range, orientation=pg.LinearRegionItem.Vertical)
        self.region.setBrush(pg.mkBrush(0, 255, 0, 40))  # Vert clair transparent
        self.region.setZValue(10)
        self.region.sigRegionChanged.connect(self.update_theta_range)

        
        self.plot_1D.addItem(self.region)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_mask)
        layout.addWidget(self.btn_poni)
        layout.addWidget(self.view)
        layout.addWidget(self.plot_1D)
        energy_layout = QHBoxLayout()
        energy_layout.addWidget(QLabel("E (keV):"))
        energy_layout.addWidget(self.energy_input)
        layout.addLayout(energy_layout)
        layout.addWidget(self.btn_ok)
        layout.addWidget(self.theta_range_label)
        layout.addWidget(self.btn_update_image)

        self.setLayout(layout)
        self.update_image()
        self._update_energy_from_ai()

    def accept(self):  # type: ignore[override]
        self.energy = float(self.energy_input.value())
        super().accept()

    def _update_energy_from_ai(self):
        if self.ai is None:
            return
        wavelength = getattr(self.ai, "wavelength", None)
        if not wavelength:
            return
        try:
            energy = ENERGY_CONSTANT_KEV_M / wavelength
        except Exception:
            return
        self.energy = float(energy)
        blocked = self.energy_input.blockSignals(True)
        self.energy_input.setValue(self.energy)
        self.energy_input.blockSignals(blocked)


    def update_theta_range(self):
        self.theta_range = list(self.region.getRegion())
        self.theta_range_label.setText(f"Theta range : [{self.theta_range[0]:.2f} - {self.theta_range[1]:.2f}] °")

    def load_mask(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Fichier .mask",directory=self.folder_start)
        if fname:
            self.file_mask = fname
            try:
                img = Image.open(self.file_mask)
                mask = np.array(img)
                # Conversion en masque binaire (0 = masqué, 1 = gardé)
                mask_bin = (mask > 0).astype(np.uint8)
                self.mask = mask_bin
                self.label.setText("Mask chargé.")
                self.update_image()
            except Exception as e:
                self.label.setText(f"Erreur lecture mask : {e}")

    def load_poni(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Fichier .poni",directory=self.folder_start)
        if fname:
            self.file_poni = fname
            try:
                self.ai = pyFAI.load(fname)
                self.label.setText("PONI chargé.")
                self.update_image()
                self._update_energy_from_ai()
            except Exception as e:
                self.label.setText(f"Erreur lecture poni : {e}")

    def update_image(self):
        if self.img_data is None:
            return

        flipped_img = np.flipud(self.img_data)

        # Affichage de l’image principale (niveaux de gris ou viridis)
        self.img_item.setImage(flipped_img, autoLevels=True)

        # Supprimer ancien masque s'il existe
        if hasattr(self, 'mask_item'):
            self.img_view.removeItem(self.mask_item)

        # Ajout de la couche de masque en rouge transparent
        if self.mask is not None:
            flipped_mask = np.flipud(self.mask)
            mask_rgba = np.zeros((flipped_mask.shape[0], flipped_mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[..., 0] = 255  # Rouge
            mask_rgba[..., 3] = 100  # Alpha (transparence), sur 255

            # Appliquer la transparence seulement où le masque est actif
            mask_rgba[flipped_mask == 0, 3] = 0  # Complètement transparent là où pas de masque

            # Créer un nouvel item pour le masque
            self.mask_item = pg.ImageItem(mask_rgba)
            self.mask_item.setZValue(1)  # S'assurer qu’il est au-dessus de l’image
            self.img_view.addItem(self.mask_item)

        # Courbe 1D si masque et geometry valides
        if self.mask is not None and self.ai is not None:
            try:
                self.tth, self.intens = Integrate_DRX(self.img_data, self.mask, self.ai)
                self.curve_1D.setData(self.tth, self.intens)
                self._update_energy_from_ai()
            except Exception as e:
                self.label.setText(f"Erreur intégration : {e}")


def Integrate_DRX(file_img,mask,ai,theta_range=None,pby2theta=50):
    try:
        if theta_range is not None:
            nb_point=int((theta_range[-1]-theta_range[0])*pby2theta)
        else:
            nb_point=9000 #90*10
        #flipped_img = np.flipud(file_img)
        tth, intens = ai.integrate1d(
            file_img,
            nb_point,
            mask=mask,
            unit="2th_deg"
        )
        if theta_range is not None:
            mask_range = (tth >= theta_range[0]) & (tth <= theta_range[1])
            tth = tth[mask_range]
            intens = intens[mask_range]
        return tth, intens
    except Exception as e:
        print(f"Erreur intégration : {e}")
        return None, None
    
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    import os
    app = QApplication(sys.argv)

    file=r"d:\ESRF\2025 Juin HC -6205 id09\RAW_DATA\water02\water02_0001\scan0027\scan_jf1m_0000.h5"

    file_mask=r"d:\ESRF\2025 Juin HC -6205 id09\dioptas_files\mask_jf1M_flo.mask"
    file_poni=r"d:\ESRF\2025 Juin HC -6205 id09\dioptas_files\calib_LaB6_11062025_correct.poni"

    calib = Calib_DRX(file_mask=file_mask, file_poni=file_poni)
    calib.Change_calib(file=file)
    sys.exit(app.exec_())