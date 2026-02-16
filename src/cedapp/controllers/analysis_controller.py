"""Analysis controller for draggable markers on CEDX plots."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox

logger = logging.getLogger(__name__)

ANALYSE_COLUMNS = [
    "id",
    "kind",
    "label",
    "spec_idx",
    "time_s",
    "P_GPa",
    "x",
    "y",
    "source",
    "locked",
    "meta",
]


def ensure_analyse_dataframe(run) -> Optional[pd.DataFrame]:
    """Ensure RUN.analyse exists and return it."""
    if run is None:
        return None
    analyse = getattr(run, "analyse", None)
    if analyse is None or not isinstance(analyse, pd.DataFrame):
        run.analyse = pd.DataFrame(columns=ANALYSE_COLUMNS)
    else:
        missing_cols = [col for col in ANALYSE_COLUMNS if col not in analyse.columns]
        if missing_cols:
            for col in missing_cols:
                analyse[col] = np.nan
            run.analyse = analyse
    return run.analyse


class AnalysisController:
    """
    Gère:
    - checkbox mode analyse
    - overlay items (draggables)
    - synchro avec self.RUN.analyse (DataFrame)
    """

    def __init__(
        self,
        main,
        plot_item: pg.PlotItem,
        *,
        dpdt_plot: Optional[pg.PlotItem] = None,
        x_axis: str = "time_ms",
    ):
        self.main = main  # ex: MainWindow
        self.plot = plot_item  # le plot où tu poses les markers (P vs t ou P vs idx)
        self.dpdt_plot = dpdt_plot
        self.enabled = False
        self.x_axis = x_axis

        self.cb_analysis = QCheckBox("Analyse")
        self.cb_analysis.setChecked(False)
        self.cb_analysis.toggled.connect(self.set_enabled)

        self._items_by_id: dict[str, pg.TargetItem] = {}
        self._text_by_id: dict[str, pg.TextItem] = {}
        self._dpdt_lines_by_id: dict[str, pg.InfiniteLine] = {}
        self._dpdt_text_by_id: dict[str, pg.TextItem] = {}
        self._curve_by_id: dict[str, pg.PlotCurveItem] = {}
        self._dpdt_curve_by_id: dict[str, pg.PlotCurveItem] = {}
        self._syncing = False

    # ---------- Mode ON/OFF ----------
    def set_enabled(self, state: bool) -> None:
        self.enabled = bool(state)
        df = None
        run = getattr(self.main, "RUN", None)
        if run is not None:
            df = ensure_analyse_dataframe(run)
        for it in self._items_by_id.values():
            it.setVisible(self.enabled)
        for txt in self._text_by_id.values():
            txt.setVisible(self.enabled)
        for curve in self._curve_by_id.values():
            curve.setVisible(self.enabled)
        for line in self._dpdt_lines_by_id.values():
            line.setVisible(self.enabled)
        for txt in self._dpdt_text_by_id.values():
            txt.setVisible(self.enabled)
        for curve in self._dpdt_curve_by_id.values():
            curve.setVisible(self.enabled)
        if df is not None and not df.empty:
            for mid, item in self._items_by_id.items():
                row = self._find_row_for_id(df, mid)
                if row is None:
                    continue
                locked = bool(row.get("locked", False))
                movable = self.enabled and (not locked)
                self._set_target_movable(item, movable)
                dpdt_line = self._dpdt_lines_by_id.get(mid)
                if dpdt_line is not None:
                    dpdt_line.setMovable(movable)
        if hasattr(self.main, "_set_analysis_overlays_visible"):
            try:
                self.main._set_analysis_overlays_visible(self.enabled)
            except Exception:
                logger.exception("Unable to toggle analysis overlays")

    # ---------- Init / refresh depuis RUN.analyse ----------
    def refresh_from_run(self) -> None:
        run = getattr(self.main, "RUN", None)
        if run is None:
            return

        df = ensure_analyse_dataframe(run)
        if df is None:
            return

        if df.empty:
            stale_ids = set(self._items_by_id.keys()) | set(self._curve_by_id.keys())
            for mid in list(stale_ids):
                self._delete_marker_item(mid)
            self.set_enabled(self.cb_analysis.isChecked())
            return
        df_ids = set(df["id"].astype(str))

        # créer / mettre à jour items
        for _, row in df.iterrows():
            mid = str(row["id"])
            kind = str(row.get("kind", ""))
            if kind == "analysis_curve":
                if mid not in self._curve_by_id:
                    self._create_curve_item(mid)
                self._sync_curve_from_row(mid, row)
                continue
            if mid not in self._items_by_id:
                self._create_marker_item(mid)
            self._sync_item_from_row(mid, row)

        # supprimer items qui n’existent plus
        stale_ids = [
            mid
            for mid in (set(self._items_by_id.keys()) | set(self._curve_by_id.keys()))
            if mid not in df_ids
        ]
        for mid in stale_ids:
            self._delete_marker_item(mid)

        # appliquer visibilité selon mode
        self.set_enabled(self.cb_analysis.isChecked())

    def _create_marker_item(self, mid: str) -> None:
        if self.plot is None:
            return
        # TargetItem draggable
        ti = pg.TargetItem(pos=(0, 0), movable=True)
        ti.sigPositionChanged.connect(lambda _item: self._on_marker_moved(mid, source="target"))
        self.plot.addItem(ti)
        self._items_by_id[mid] = ti

        # texte (optionnel)
        txt = pg.TextItem("", anchor=(0, 1))
        self.plot.addItem(txt)
        self._text_by_id[mid] = txt

        if self.dpdt_plot is not None:
            dpdt_line = pg.InfiniteLine(angle=0, movable=True)
            dpdt_line.sigPositionChanged.connect(
                lambda _item: self._on_marker_moved(mid, source="dpdt")
            )
            self.dpdt_plot.addItem(dpdt_line)
            self._dpdt_lines_by_id[mid] = dpdt_line

            dpdt_txt = pg.TextItem("", anchor=(0, 1))
            self.dpdt_plot.addItem(dpdt_txt)
            self._dpdt_text_by_id[mid] = dpdt_txt

    def _create_curve_item(self, mid: str) -> None:
        if self.plot is None:
            return
        curve = pg.PlotCurveItem([], [])
        self.plot.addItem(curve)
        self._curve_by_id[mid] = curve

        if self.dpdt_plot is not None:
            dpdt_curve = pg.PlotCurveItem([], [])
            self.dpdt_plot.addItem(dpdt_curve)
            self._dpdt_curve_by_id[mid] = dpdt_curve

    def _delete_marker_item(self, mid: str) -> None:
        it = self._items_by_id.pop(mid, None)
        if it is not None:
            self.plot.removeItem(it)
        txt = self._text_by_id.pop(mid, None)
        if txt is not None:
            self.plot.removeItem(txt)
        dpdt_line = self._dpdt_lines_by_id.pop(mid, None)
        if dpdt_line is not None and self.dpdt_plot is not None:
            self.dpdt_plot.removeItem(dpdt_line)
        dpdt_txt = self._dpdt_text_by_id.pop(mid, None)
        if dpdt_txt is not None and self.dpdt_plot is not None:
            self.dpdt_plot.removeItem(dpdt_txt)
        curve = self._curve_by_id.pop(mid, None)
        if curve is not None:
            self.plot.removeItem(curve)
        dpdt_curve = self._dpdt_curve_by_id.pop(mid, None)
        if dpdt_curve is not None and self.dpdt_plot is not None:
            self.dpdt_plot.removeItem(dpdt_curve)

    def _sync_curve_from_row(self, mid: str, row) -> None:
        curve = self._curve_by_id.get(mid)
        dpdt_curve = self._dpdt_curve_by_id.get(mid)
        if curve is None:
            return
        meta = row.get("meta", None)
        if not isinstance(meta, dict):
            curve.setData([], [])
            if dpdt_curve is not None:
                dpdt_curve.setData([], [])
            return
        t_fine = np.asarray(meta.get("t_fine", []), dtype=float)
        p_fine = np.asarray(meta.get("P_fine", []), dtype=float)
        if t_fine.size == 0 or p_fine.size == 0:
            curve.setData([], [])
            if dpdt_curve is not None:
                dpdt_curve.setData([], [])
            return
        color = meta.get("color")
        pen = pg.mkPen(color, width=3) if color is not None else pg.mkPen("w", width=3)
        curve.setPen(pen)
        curve.setData(t_fine, p_fine)

        if dpdt_curve is not None:
            dpdt_fine = np.asarray(meta.get("dPdt_fine", []), dtype=float)
            if dpdt_fine.size == t_fine.size:
                dpdt_curve.setPen(pen)
                dpdt_curve.setData(t_fine, dpdt_fine)
            else:
                dpdt_curve.setData([], [])

    def _sync_item_from_row(self, mid: str, row) -> None:
        self._syncing = True
        try:
            it = self._items_by_id[mid]
            time_s = row.get("time_s", np.nan)
            if pd.notna(row.get("x", np.nan)):
                x = float(row["x"])
            elif pd.notna(time_s):
                x = float(time_s) * 1000.0 if self.x_axis == "time_ms" else float(time_s)
            else:
                x = 0.0
            if pd.notna(row.get("y", np.nan)):
                y = float(row["y"])
            elif pd.notna(row.get("P_GPa", np.nan)):
                y = float(row["P_GPa"])
            else:
                y = 0.0
            it.setPos(x, y)

            locked = bool(row.get("locked", False))
            movable = self.enabled and (not locked)
            self._set_target_movable(it, movable)

            label = str(row.get("label", mid))
            display_label = self._format_display_label(label, row)
            txt = self._text_by_id[mid]
            txt.setText(f"{display_label}\nP={y:.3f} GPa\nt={self._format_time_value(x)}")
            txt.setPos(x, y)

            color = self._resolve_color(row)
            if color is not None:
                pen = pg.mkPen(color)
                it.setPen(pen)
                txt.setColor(color)

            dpdt_line = self._dpdt_lines_by_id.get(mid)
            dpdt_txt = self._dpdt_text_by_id.get(mid)
            dpdt_value = self._resolve_dpdt(row)
            if dpdt_line is not None:
                dpdt_line.setValue(float(dpdt_value) if dpdt_value is not None else 0.0)
                dpdt_line.setMovable(movable)
                if color is not None:
                    dpdt_line.setPen(pg.mkPen(color))
            if dpdt_txt is not None:
                dpdt_label = (
                    f"{display_label}\ndP/dt={dpdt_value:.3f}"
                    if dpdt_value is not None
                    else display_label
                )
                dpdt_txt.setText(dpdt_label)
                dpdt_txt.setPos(x, float(dpdt_value) if dpdt_value is not None else 0.0)
                if color is not None:
                    dpdt_txt.setColor(color)
        finally:
            self._syncing = False

    # ---------- Drag handler ----------
    def _on_marker_moved(self, mid: str, *, source: str) -> None:
        if not self.enabled:
            return
        if self._syncing:
            return

        run = getattr(self.main, "RUN", None)
        df = ensure_analyse_dataframe(run)
        if df is None or df.empty:
            return

        x = None
        y = None
        if source == "target":
            it = self._items_by_id[mid]
            pos = it.pos()
            x = float(pos.x())
            y = float(pos.y())
        elif source == "dpdt":
            dpdt_line = self._dpdt_lines_by_id.get(mid)
            if dpdt_line is None:
                return
            dpdt_value = float(dpdt_line.value())
            idx = df.index[df["id"].astype(str) == mid]
            if len(idx) == 0:
                return
            i = idx[0]
            meta = df.at[i, "meta"]
            if not isinstance(meta, dict):
                meta = {}
            meta["dpdt"] = float(dpdt_value)
            df.at[i, "meta"] = meta
            df.at[i, "source"] = "manual"
            self._sync_item_from_row(mid, df.loc[i])
            return

        if source == "target":
            current_item = self._items_by_id.get(mid)
            if current_item is not None:
                y = float(current_item.pos().y())
        if x is None or y is None:
            return

        # retrouver la ligne
        idx = df.index[df["id"].astype(str) == mid]
        if len(idx) == 0:
            return
        i = idx[0]

        time_s, spec_idx = self._convert_x_to_spec(run, x)

        df.at[i, "x"] = x
        df.at[i, "y"] = y
        df.at[i, "time_s"] = float(time_s)
        df.at[i, "spec_idx"] = int(spec_idx)
        df.at[i, "P_GPa"] = float(y)
        df.at[i, "source"] = "manual"

        # rafraîchir texte
        self._sync_item_from_row(mid, df.loc[i])

    def _convert_x_to_spec(self, run, x_value: float) -> tuple[float, int]:
        if self.x_axis == "index":
            spec_idx = int(np.clip(round(x_value), 0, max(len(getattr(run, "Time_spectrum", []) or []) - 1, 0)))
            time_s = self._time_from_index(run, spec_idx)
            return time_s, spec_idx

        time_s = float(x_value) / 1000.0 if self.x_axis == "time_ms" else float(x_value)
        spec_idx = self._nearest_spec_index_from_time(run, time_s)
        return time_s, spec_idx

    def _time_from_index(self, run, spec_idx: int) -> float:
        time_arr = np.asarray(getattr(run, "Time_spectrum", []), dtype=float)
        if time_arr.size == 0 or spec_idx < 0 or spec_idx >= time_arr.size:
            return float(spec_idx)
        return float(time_arr[spec_idx])

    def _nearest_spec_index_from_time(self, run, time_s: float) -> int:
        t = np.asarray(getattr(run, "Time_spectrum", []), dtype=float)
        if t.size == 0:
            return 0
        return int(np.argmin(np.abs(t - time_s)))

    def _resolve_color(self, row):
        meta = row.get("meta", None)
        color = None
        if isinstance(meta, dict):
            color = meta.get("color")
        if color is None and hasattr(self.main, "_get_gauge_color"):
            label = row.get("label", None)
            try:
                color = self.main._get_gauge_color(label)
            except Exception:
                color = None
        return color

    def _resolve_dpdt(self, row):
        dpdt = row.get("dpdt", None)
        if dpdt is not None and pd.notna(dpdt):
            return float(dpdt)
        meta = row.get("meta", None)
        if isinstance(meta, dict):
            dpdt = meta.get("dpdt")
            if dpdt is not None:
                try:
                    return float(dpdt)
                except Exception:
                    return None
        return None

    def _find_row_for_id(self, df: pd.DataFrame, mid: str):
        idx = df.index[df["id"].astype(str) == str(mid)]
        if len(idx) == 0:
            return None
        return df.loc[idx[0]]

    def _format_time_value(self, x_ms: float) -> str:
        value_ms = float(x_ms)
        if abs(value_ms) < 1.0:
            return f"{value_ms * 1000.0:.2f} µs"
        return f"{value_ms:.3f} ms"

    def _format_display_label(self, label: str, row) -> str:
        meta = row.get("meta", None)
        if not isinstance(meta, dict):
            return label
        if label.endswith(":t_lim_dt"):
            dt_ms = meta.get("dt_ms")
            if dt_ms is not None and pd.notna(dt_ms):
                return f"{label.split(':')[0]}:Δt={float(dt_ms):.3f} ms"
        return label

    def _set_target_movable(self, item: pg.TargetItem, movable: bool) -> None:
        """Compatibility helper across pyqtgraph versions.

        Some TargetItem implementations do not expose ``setMovable``.
        """
        if hasattr(item, "setMovable"):
            item.setMovable(movable)
            return
        if hasattr(item, "setMouseEnabled"):
            try:
                item.setMouseEnabled(movable, movable)
                return
            except Exception:
                logger.debug("setMouseEnabled failed on TargetItem", exc_info=True)
        if hasattr(item, "setAcceptedMouseButtons"):
            buttons = Qt.LeftButton if movable else Qt.NoButton
            item.setAcceptedMouseButtons(buttons)
