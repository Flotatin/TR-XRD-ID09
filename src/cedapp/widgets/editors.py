"""Editor dialogs for DRX widgets."""

from __future__ import annotations


from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class JcpdsEditor(QDialog):
    """Dialog to edit JCPDS element parameters."""

    def __init__(self, element, parent=None):
        super().__init__(parent)
        self.element = element
        self.setWindowTitle("Edit JCPDS")
        self._init_ui()

    def _init_ui(self):
        self.k0_edit = QLineEdit(self._fmt(self.element.K0))
        self.k0p_edit = QLineEdit(self._fmt(self.element.K0P))
        self.v0_edit = QLineEdit(self._fmt(self.element.V0))
        self.a_edit = QLineEdit(self._fmt(self.element.A))
        self.b_edit = QLineEdit(self._fmt(self.element.B))
        self.c_edit = QLineEdit(self._fmt(self.element.C))
        self.alpha_edit = QLineEdit(self._fmt(self.element.ALPHA))
        self.beta_edit = QLineEdit(self._fmt(self.element.BETA))
        self.gamma_edit = QLineEdit(self._fmt(self.element.GAMMA))
        self.alphakt_edit = QLineEdit(self._fmt(getattr(self.element, "ALPHAKT", None)))
        self.vmin_edit = QLineEdit(self._fmt(getattr(self.element, "Vmin", None)))
        self.fu_edit = QLineEdit(self._fmt(getattr(self.element, "fu", None)))
        t_range = self._fmt_range(getattr(self.element, "T_range", None))
        p_range = self._fmt_range(getattr(self.element, "P_range", None))
        self.tmin_edit = QLineEdit(t_range[0])
        self.tmax_edit = QLineEdit(t_range[1])
        self.pmin_edit = QLineEdit(p_range[0])
        self.pmax_edit = QLineEdit(p_range[1])

        form = QFormLayout()
        form.addRow(QLabel("K0"), self.k0_edit)
        form.addRow(QLabel("K0P"), self.k0p_edit)
        form.addRow(QLabel("V0"), self.v0_edit)
        form.addRow(QLabel("a"), self.a_edit)
        form.addRow(QLabel("b"), self.b_edit)
        form.addRow(QLabel("c"), self.c_edit)
        form.addRow(QLabel("alpha"), self.alpha_edit)
        form.addRow(QLabel("beta"), self.beta_edit)
        form.addRow(QLabel("gamma"), self.gamma_edit)
        form.addRow(QLabel("alphaKt"), self.alphakt_edit)
        form.addRow(QLabel("Vmin"), self.vmin_edit)
        form.addRow(QLabel("f.u."), self.fu_edit)
        form.addRow(QLabel("T min"), self.tmin_edit)
        form.addRow(QLabel("T max"), self.tmax_edit)
        form.addRow(QLabel("P min"), self.pmin_edit)
        form.addRow(QLabel("P max"), self.pmax_edit)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_changes)
        self.apply_save_btn = QPushButton("Apply + Save JCPDS")
        self.apply_save_btn.clicked.connect(self.apply_and_save)
        self.save_to_jcpds = False

        layout = QVBoxLayout()
        layout.addLayout(form)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.apply_save_btn)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    @staticmethod
    def _fmt(value):
        return "" if value is None else str(value)
    
    @staticmethod
    def _fmt_range(value) -> Tuple[str, str]:
        if not value or len(value) < 2:
            return "", ""
        return str(value[0]), str(value[1])

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        value = text.strip()
        return float(value) if value else None


    

    def apply_changes(self):
        try:
            self.element.K0 = float(self.k0_edit.text()) if self.k0_edit.text() else None
            self.element.K0P = float(self.k0p_edit.text()) if self.k0p_edit.text() else None
            self.element.V0 = float(self.v0_edit.text()) if self.v0_edit.text() else None
            self.element.A = float(self.a_edit.text()) if self.a_edit.text() else None
            self.element.B = float(self.b_edit.text()) if self.b_edit.text() else None
            self.element.C = float(self.c_edit.text()) if self.c_edit.text() else None
            self.element.ALPHA = float(self.alpha_edit.text()) if self.alpha_edit.text() else None
            self.element.BETA = float(self.beta_edit.text()) if self.beta_edit.text() else None
            self.element.GAMMA = float(self.gamma_edit.text()) if self.gamma_edit.text() else None
            self.element.ALPHAKT = self._parse_float(self.alphakt_edit.text())
            self.element.Vmin = self._parse_float(self.vmin_edit.text())

            fu = self._parse_float(self.fu_edit.text())
            self.element.fu = int(fu) if fu is not None else None

            t_min = self._parse_float(self.tmin_edit.text())
            t_max = self._parse_float(self.tmax_edit.text())
            self.element.T_range = (t_min, t_max) if t_min is not None and t_max is not None else None

            p_min = self._parse_float(self.pmin_edit.text())
            p_max = self._parse_float(self.pmax_edit.text())
            if p_min is not None and p_max is not None:
                self.element.P_range = (p_min, p_max)


            self.element.Eos_Pdhkl(self.element.P_start)
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, "Error", str(exc))

    def apply_and_save(self):
        self.save_to_jcpds = True
        self.apply_changes()
