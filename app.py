import streamlit as st
import matplotlib.pyplot as plt
import json
import io
import numpy as np
from Pynite import FEModel3D
from solucionador_matricial import BeamMatrixSolver
from datetime import datetime
import pandas as pd
import sympy as sp
from pathlib import Path
import glob

st.set_page_config(
    page_title="ANALISIS ESTRUCTURAL ACADÉMICO",
    layout="centered",
    page_icon=None,
    initial_sidebar_state="expanded"
)



# --------------------------------------------------------------
#  Estado de sesión
# --------------------------------------------------------------
if "beam_data" not in st.session_state:
    st.session_state.beam_data = {
        "version": "3.0.0",
        "name": "Nuevo Proyecto",
        "created": datetime.now().isoformat(),
        "units": {"length": "m", "force": "kN", "moment": "kN·m"},
        "materials": {"E_default": 200e6, "A_default": 0.01, "I_default": 5e-5},
        "beam": {
            "nodes": [],
            "elements": [],
            "supports": [],
            "loads": {"point": [], "distributed": []}
        }
    }

if "solved_model" not in st.session_state:
    st.session_state.solved_model = None

if "modelo_cargado" not in st.session_state:
    seleccionado = Path(__file__).parent / "ejemplos" / "_seleccionado.json"
    if seleccionado.exists():
        try:
            with open(seleccionado, "r", encoding="utf-8") as _f:
                st.session_state.beam_data = json.load(_f)
            seleccionado.unlink()
            st.session_state.pagina = "app"
        except Exception:
            pass
    st.session_state.modelo_cargado = True

# Migrar formato antiguo de cargas distribuidas (start_node/end_node -> element_id)
def migrar_distribuidas(beam_data):
    beam = beam_data.get("beam", beam_data)
    distribuidas = beam.get("loads", {}).get("distributed", [])
    elements = beam.get("elements", [])
    for dl in distribuidas:
        if "start_node" in dl and "end_node" in dl and "element_id" not in dl:
            sn = dl["start_node"]
            en = dl["end_node"]
            elem = next((e for e in elements if e["start_node"] == sn and e["end_node"] == en), None)
            if elem:
                dl["element_id"] = elem["id"]
                del dl["start_node"]
                del dl["end_node"]

def beam_data_actual():
    return st.session_state.beam_data

try:
    migrar_distribuidas(st.session_state.beam_data)
except (AttributeError, KeyError):
    pass

if "pagina" not in st.session_state:
    st.session_state.pagina = "home"

if "history" not in st.session_state:
    st.session_state.history = []

if "history_index" not in st.session_state:
    st.session_state.history_index = -1

data = st.session_state.beam_data["beam"]
materials = st.session_state.beam_data["materials"]

# ----------------------------------------------------------------
# ----------------------------------------------------------------
#  PÁGINA DE INICIO
# ----------------------------------------------------------------
if st.session_state.pagina == "home":

    st.markdown("<h1 style='text-align: center; font-family: \"Palatino Linotype\", \"Book Antiqua\", Palatino, serif; letter-spacing: 2px;'>ANÁLISIS ESTRUCTURAL ACADÉMICO</h1>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; font-family: \"Segoe Script\", \"Comic Sans MS\", cursive; font-size: 1.2rem; color: gray; margin-top: -10px;'>Sebastian M.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("VIGAS", type="primary", use_container_width=True):
            st.session_state.pagina = "app"
            st.rerun()
    with col2:
        if st.button("CERCHAS", use_container_width=True):
            st.warning("Proximamente.")
    with col3:
        if st.button("PORTICOS", use_container_width=True):
            st.warning("Proximamente.")

    st.divider()
    st.markdown("### Modelos de ejemplo")

    autores = [
        ("hibbeler", "Hibbeler"),
        ("kassimali", "Kassimali"),
        ("millan", "Millán"),
        ("causil", "Causil"),
    ]

    autor_models = {}
    for key, label in autores:
        folder = Path(__file__).parent / "ejemplos" / key
        files = sorted(folder.glob("*.json")) if folder.exists() else []
        models = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    models.append((fp.stem, json.load(f)))
            except Exception:
                continue
        autor_models[key] = models

    for i in range(0, len(autores), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(autores):
                key, label = autores[i + j]
                with cols[j]:
                    models = autor_models.get(key, [])
                    with st.expander(f"**{label}**", expanded=False):
                        if not models:
                            st.caption("Sin modelos disponibles.")
                        else:
                            for stem, model_data in models:
                                nombre = model_data.get("name", stem)
                                if st.button(nombre, key=f"hm_{key}_{stem}", use_container_width=True):
                                    st.session_state.beam_data = model_data
                                    st.session_state.solved_model = None
                                    st.session_state.pagina = "app"
                                    st.rerun()

    st.stop()

# ----------------------------------------------------------------
#  Funciones de historial (Undo/Redo)
# ----------------------------------------------------------------
def save_to_history():
    """Guarda el estado actual en el historial"""
    snapshot = json.dumps(st.session_state.beam_data)
    # Eliminar estados futuros si estamos en medio del historial
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
    st.session_state.history.append(snapshot)
    st.session_state.history_index = len(st.session_state.history) - 1
    # Limitar historial a 50 estados
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)
        st.session_state.history_index -= 1

def undo():
    """Deshace la última acción"""
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.beam_data = json.loads(st.session_state.history[st.session_state.history_index])
        st.session_state.solved_model = None
        st.rerun()

def redo():
    """Rehace una acción"""
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history_index += 1
        st.session_state.beam_data = json.loads(st.session_state.history[st.session_state.history_index])
        st.session_state.solved_model = None
        st.rerun()

# Inicializar historial
if not st.session_state.history:
    save_to_history()

# --------------------------------------------------------------
#  Funciones utilitarias
# --------------------------------------------------------------
def add_node(x):
    """Añade un nodo y guarda en historial"""
    used_ids = [n["id"] for n in data["nodes"]]
    new_id = max(used_ids) + 1 if used_ids else 1
    data["nodes"].append({"id": new_id, "x": float(x)})
    st.session_state.solved_model = None
    save_to_history()

def remove_node(node_id):
    """Elimina un nodo y guarda en historial"""
    data["nodes"] = [nd for nd in data["nodes"] if nd["id"] != node_id]
    data["supports"] = [s for s in data["supports"] if s["node_id"] != node_id]
    data["loads"]["point"] = [l for l in data["loads"]["point"] if l["node_id"] != node_id]
    deleted_elements = [e["id"] for e in data["elements"] if e["start_node"] == node_id or e["end_node"] == node_id]
    data["elements"] = [e for e in data["elements"] if e["start_node"] != node_id and e["end_node"] != node_id]
    data["loads"]["distributed"] = [
        l for l in data["loads"]["distributed"]
        if l.get("element_id") not in deleted_elements
    ]
    st.session_state.solved_model = None
    save_to_history()

def add_element(start_node, end_node):
    """Añade un elemento conectando dos nodos"""
    used_ids = [e["id"] for e in data["elements"]]
    new_id = max(used_ids) + 1 if used_ids else 1
    data["elements"].append({
        "id": new_id,
        "start_node": start_node,
        "end_node": end_node,
        "E": materials["E_default"],
        "A": materials["A_default"],
        "I": materials["I_default"]
    })
    st.session_state.solved_model = None
    save_to_history()

def remove_element(element_id):
    """Elimina un elemento"""
    data["elements"] = [e for e in data["elements"] if e["id"] != element_id]
    data["loads"]["distributed"] = [
        l for l in data["loads"]["distributed"]
        if l.get("element_id") != element_id
    ]
    st.session_state.solved_model = None
    save_to_history()

def add_support(node_id, support_type):
    """Añade un apoyo"""
    if not any(s["node_id"] == node_id for s in data["supports"]):
        data["supports"].append({"node_id": node_id, "type": support_type})
        st.session_state.solved_model = None
        save_to_history()
        return True
    return False

def remove_support(node_id):
    """Elimina un apoyo"""
    data["supports"] = [sup for sup in data["supports"] if sup["node_id"] != node_id]
    st.session_state.solved_model = None
    save_to_history()

def add_point_load(node_id, fy):
    """Añade una carga puntual en un nodo"""
    data["loads"]["point"].append({"node_id": node_id, "fx": 0.0, "fy": float(fy), "mz": 0.0})
    st.session_state.solved_model = None
    save_to_history()

def add_point_load_on_element(element_id, x_pos, fy):
    """Añade una carga puntual en posicion arbitraria sobre un elemento"""
    data["loads"]["point"].append({"element_id": element_id, "x": float(x_pos), "fy": float(fy), "mz": 0.0})
    st.session_state.solved_model = None
    save_to_history()

def add_distributed_load(element_id, w_start, w_end):
    """Añade una carga distribuida sobre un elemento"""
    data["loads"]["distributed"].append({
        "element_id": element_id,
        "w_start": float(w_start),
        "w_end": float(w_end),
        "direction": "y"
    })
    st.session_state.solved_model = None
    save_to_history()

def remove_point_load(index):
    """Elimina una carga puntual"""
    data["loads"]["point"].pop(index)
    st.session_state.solved_model = None
    save_to_history()

def remove_distributed_load(index):
    """Elimina una carga distribuida"""
    data["loads"]["distributed"].pop(index)
    st.session_state.solved_model = None
    save_to_history()

# --------------------------------------------------------------
# Constructor del modelo PyNite
# --------------------------------------------------------------
def _combo_value(values, combo="Combo 1"):
    """Devuelve un valor numerico desde los diccionarios de resultados de PyNite."""
    if isinstance(values, dict):
        if combo in values:
            return float(values[combo])
        if values:
            return float(next(iter(values.values())))
        return 0.0
    return float(values or 0.0)


class PyNiteBeamModel:
    """Adaptador minimo para que la app trabaje con PyNite sin cambiar la UI."""

    def __init__(self, beam_data):
        self.data = beam_data
        self.model = FEModel3D()
        self.node_names = {}
        self.member_names = {}
        self.member_lengths = {}
        self.reaction_force = {}
        self._build()

    def _node_name(self, node_id):
        return f"N{node_id}"

    def _member_name(self, element_id):
        return f"M{element_id}"

    def _node_by_id(self, node_id):
        return next((n for n in self.data["nodes"] if n["id"] == node_id), None)

    def _build(self):
        for node in self.data["nodes"]:
            name = self._node_name(node["id"])
            self.node_names[node["id"]] = name
            self.model.add_node(name, float(node["x"]), 0.0, 0.0)

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            n2 = self._node_by_id(e["end_node"])
            if not n1 or not n2:
                continue

            element_id = e["id"]
            member_name = self._member_name(element_id)
            material_name = f"Mat{element_id}"
            section_name = f"Sec{element_id}"

            E = float(e["E"])
            A = float(e["A"])
            I = float(e["I"])
            G = E / (2 * (1 + 0.3))
            J = max(2 * I, 1e-12)

            self.model.add_material(material_name, E, G, 0.3, 0.0)
            self.model.add_section(section_name, A, I, I, J)
            self.model.add_member(
                member_name,
                self._node_name(e["start_node"]),
                self._node_name(e["end_node"]),
                material_name,
                section_name,
            )
            self.member_names[element_id] = member_name
            self.member_lengths[member_name] = abs(float(n2["x"]) - float(n1["x"]))

        for support in self.data["supports"]:
            node_name = self._node_name(support["node_id"])
            if support["type"] == "Empotrado":
                self.model.def_support(node_name, True, True, True, True, True, True)
            elif support["type"] == "Articulado":
                self.model.def_support(node_name, True, True, True, True, True, False)
            elif support["type"] == "Rodillo":
                self.model.def_support(node_name, False, True, True, True, True, False)

        for pl in self.data["loads"]["point"]:
            if "element_id" in pl:
                self._add_point_on_member(pl)
            elif "node_id" in pl:
                node_name = self._node_name(pl["node_id"])
                if pl.get("fx", 0):
                    self.model.add_node_load(node_name, "FX", float(pl["fx"]))
                if pl.get("fy", 0):
                    self.model.add_node_load(node_name, "FY", float(pl["fy"]))
                if pl.get("mz", 0):
                    self.model.add_node_load(node_name, "MZ", float(pl["mz"]))

        for dl in self.data["loads"]["distributed"]:
            self._add_distributed_load(dl)

    def _add_point_on_member(self, pl):
        eid = pl.get("element_id")
        if eid is None:
            return
        elem = next((e for e in self.data["elements"] if e["id"] == eid), None)
        if not elem:
            return
        n_start = self._node_by_id(elem["start_node"])
        n_end = self._node_by_id(elem["end_node"])
        if not n_start or not n_end:
            return
        x_start = float(n_start["x"])
        x_end = float(n_end["x"])
        L = abs(x_end - x_start)
        if L < 1e-12:
            return
        x_pos = pl.get("x")
        if x_pos is not None:
            x_min = min(x_start, x_end)
            d = abs(float(x_pos) - x_min)
        else:
            d = float(pl.get("d", L / 2))
        fy = float(pl.get("fy", 0))
        if abs(fy) < 1e-15:
            return
        member_name = self.member_names.get(eid)
        if member_name:
            eps = L / 10000
            self.model.add_member_dist_load(
                member_name, "FY", fy / eps, fy / eps, d - eps / 2, d + eps / 2
            )

    def _add_distributed_load(self, dl):
        eid = dl.get("element_id")
        if eid is None:
            return
        elem = next((e for e in self.data["elements"] if e["id"] == eid), None)
        if not elem:
            return
        n_start = self._node_by_id(elem["start_node"])
        n_end = self._node_by_id(elem["end_node"])
        if not n_start or not n_end:
            return

        x_start = float(n_start["x"])
        x_end = float(n_end["x"])
        if x_start == x_end:
            return

        def q_at(x):
            ratio = (x - x_start) / (x_end - x_start)
            return float(dl["w_start"]) + ratio * (float(dl["w_end"]) - float(dl["w_start"]))

        member_name = self.member_names.get(eid)
        if member_name:
            self.model.add_member_dist_load(member_name, "FY", q_at(x_start), q_at(x_end), 0, x_end - x_start)

    def solve(self):
        self.model.analyze_linear(log=False)
        self.reaction_force = {}
        for support in self.data["supports"]:
            node_id = support["node_id"]
            node = self.model.nodes[self._node_name(node_id)]
            self.reaction_force[node_id] = {
                "Fx": _combo_value(node.RxnFX),
                "Fy": _combo_value(node.RxnFY),
                "M": _combo_value(node.RxnMZ),
            }

    def vertex_id_flection(self, node_id):
        node = self.model.nodes[self._node_name(node_id)]
        return _combo_value(node.DY)

    def _member_points(self, member_name, count=60):
        length = self.member_lengths.get(member_name, 0.0)
        return np.linspace(0.0, length, count)

    def _plot_result(self, title, ylabel, evaluator, baseline=True, color="#1f77b4"):
        fig, ax = plt.subplots(figsize=(12, 5))
        all_x = []

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            member_name = self.member_names.get(e["id"])
            if not n1 or not member_name:
                continue
            x0 = float(n1["x"])
            xs_local = self._member_points(member_name)
            xs_global = x0 + xs_local
            ys = [evaluator(self.model.members[member_name], x) for x in xs_local]
            all_x.extend(xs_global)
            ax.plot(xs_global, ys, color=color, linewidth=2)
            ax.fill_between(xs_global, ys, 0, color=color, alpha=0.18)

        if baseline:
            ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("X (m)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle="--")
        if all_x:
            ax.set_xlim(min(all_x), max(all_x))
        return fig

    def show_shear_force(self):
        return self._plot_result(
            "Diagrama de Fuerza Cortante",
            "V (kN)",
            lambda member, x: float(member.shear("Fy", x)),
            color="#2980b9",
        )

    def show_bending_moment(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        all_x = []

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            member_name = self.member_names.get(e["id"])
            if not n1 or not member_name:
                continue
            x0 = float(n1["x"])
            xs_local = self._member_points(member_name)
            xs_global = x0 + xs_local
            ys = [float(self.model.members[member_name].moment("Mz", x)) for x in xs_local]
            all_x.extend(xs_global)
            ax.plot(xs_global, ys, color="#c0392b", linewidth=2)
            ax.fill_between(xs_global, ys, 0, color="#c0392b", alpha=0.18)

        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title("Diagrama de Momento Flector", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("M (kN*m)")
        ax.grid(True, alpha=0.3, linestyle="--")
        if all_x:
            ax.set_xlim(min(all_x), max(all_x))
        return fig

    def get_bending_moment_stats(self):
        abs_max_moment = 0.0
        abs_max_x = None
        abs_max_member = None

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            member_name = self.member_names.get(e["id"])
            if not n1 or not member_name:
                continue
            x0 = float(n1["x"])
            length = self.member_lengths.get(member_name, 0.0)
            member_obj = self.model.members[member_name]
            for x_local in np.linspace(0, length, 60):
                try:
                    m = abs(float(member_obj.moment("Mz", x_local)))
                    x_global = x0 + x_local
                    if m > abs_max_moment:
                        abs_max_moment = m
                        abs_max_x = x_global
                        abs_max_member = e["id"]
                except:
                    pass

        return {
            "abs_max_moment": abs_max_moment,
            "abs_max_x": abs_max_x,
            "abs_max_member": abs_max_member,
        }

    def show_displacement(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        all_x = []
        all_y_mm = []

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            member_name = self.member_names.get(e["id"])
            if not n1 or not member_name:
                continue
            x0 = float(n1["x"])
            xs_local = self._member_points(member_name)
            xs_global = x0 + xs_local
            ys_mm = [float(self.model.members[member_name].deflection("dy", x)) * 1000 for x in xs_local]
            all_x.extend(xs_global)
            all_y_mm.extend(ys_mm)
            ax.plot(xs_global, ys_mm, color="#27ae60", linewidth=2)
            ax.fill_between(xs_global, ys_mm, 0, color="#27ae60", alpha=0.18)

        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title("Diagrama de Deflexion", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("dy (mm)")
        ax.grid(True, alpha=0.3, linestyle="--")
        if all_x:
            ax.set_xlim(min(all_x), max(all_x))
        return fig

    def get_displacement_stats(self):
        abs_max_disp_mm = 0.0
        abs_max_node = None

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            member_name = self.member_names.get(e["id"])
            if not n1 or not member_name:
                continue
            x0 = float(n1["x"])
            xs_local = self._member_points(member_name, count=120)
            for x in xs_local:
                try:
                    d = abs(float(self.model.members[member_name].deflection("dy", x)) * 1000)
                    xg = x0 + x
                    if d > abs_max_disp_mm:
                        abs_max_disp_mm = d
                        abs_max_node = f"E{e['id']} @ x={xg:.2f}m"
                except:
                    pass

        return {
            "abs_max_disp_mm": abs_max_disp_mm,
            "abs_max_node": abs_max_node,
        }

    def show_reaction_force(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        y_scale = self._plot_scale()
        xs = [float(n["x"]) for n in self.data["nodes"]]

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            n2 = self._node_by_id(e["end_node"])
            if n1 and n2:
                ax.plot([n1["x"], n2["x"]], [0, 0], color="#6b7280", linewidth=4,
                        solid_capstyle="round", zorder=2)

        for node_id, values in self.reaction_force.items():
            node = self._node_by_id(node_id)
            if not node:
                continue
            x = float(node["x"])
            ax.scatter(x, 0, s=120, marker="o", color="#e74c3c",
                       edgecolors="white", linewidths=1.4, zorder=7)
            ax.text(x, y_scale * 0.22, f"N{node_id}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#7f1d1d")

            fy = values.get("Fy", 0.0)
            if abs(fy) > 1e-9:
                direction = 1 if fy >= 0 else -1
                ax.arrow(x, -direction * y_scale * 0.72, 0, direction * y_scale * 0.66,
                         head_width=y_scale * 0.14, head_length=y_scale * 0.12,
                         length_includes_head=True, color="#27ae60", linewidth=2)
                ax.text(x, -direction * y_scale * 0.92, f"{fy:.2f} kN",
                        ha="center", va="center", fontsize=9, color="#1e8449", fontweight="bold")

            mz = values.get("M", 0.0)
            if abs(mz) > 1e-9:
                ax.text(x, y_scale * 0.58, f"M={mz:.2f} kN*m",
                        ha="center", va="center", fontsize=9, color="#8e44ad", fontweight="bold")

        if xs:
            ax.set_xlim(min(xs) - y_scale * 1.2, max(xs) + y_scale * 1.2)
        ax.set_ylim(-y_scale * 1.15, y_scale * 1.15)
        ax.set_xlabel("X (m)")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_title("Diagrama de Reacciones", fontsize=14, fontweight="bold", pad=15)
        return fig

    def show_structure(self):
        fig, ax = plt.subplots(figsize=(12, 5))
        self._draw_structure(ax, show_loads=True)
        ax.set_title("Vista de la Estructura", fontsize=14, fontweight="bold", pad=15)
        return fig

    def _plot_scale(self):
        xs = [float(n["x"]) for n in self.data["nodes"]]
        span = max(xs) - min(xs) if xs else 1.0
        return max(span * 0.075, 0.28)

    def _draw_structure(self, ax, show_loads=True, show_supports=True):
        y_scale = self._plot_scale()
        node_label_y = y_scale * 0.30
        element_label_y = y_scale * 0.12
        xs = [float(n["x"]) for n in self.data["nodes"]]
        if not xs:
            return
        centered_support_node_ids = {
            support["node_id"]
            for support in self.data["supports"]
            if support["type"] == "Empotrado"
        }
        support_markers = {
            "Empotrado": "s",
            "Articulado": "^",
            "Rodillo": "o",
        }
        support_size = 190
        lower_support_y = -y_scale * 0.10

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            n2 = self._node_by_id(e["end_node"])
            if n1 and n2:
                ax.plot([n1["x"], n2["x"]], [0, 0], color="#1e3a5f", linewidth=5, solid_capstyle="round", zorder=2)
                ax.text((n1["x"] + n2["x"]) / 2, element_label_y, f"E{e['id']}",
                        ha="center", va="bottom", fontsize=8, color="#34495e")

        for node in self.data["nodes"]:
            if node["id"] not in centered_support_node_ids:
                ax.scatter(node["x"], 0, s=50, color="#1e3a5f", edgecolors="white", linewidths=1.2, zorder=4)
            ax.text(node["x"], node_label_y, f"N{node['id']}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#1e3a5f")

        if show_supports:
            for support in self.data["supports"]:
                node = self._node_by_id(support["node_id"])
                if not node:
                    continue
                x = float(node["x"])
                marker = support_markers.get(support["type"], "o")
                y = 0 if support["type"] == "Empotrado" else lower_support_y
                ax.scatter(x, y, s=support_size, marker=marker, color="#27ae60",
                           edgecolors="#1e8449", linewidths=1.3, zorder=5)

        if show_loads:
            for load in self.data["loads"]["point"]:
                node = self._node_by_id(load["node_id"])
                if not node:
                    continue
                x = float(node["x"])
                fy = float(load.get("fy", 0.0))
                sign = -1 if fy < 0 else 1
                start_y = -sign * y_scale * 0.66
                end_y = -sign * y_scale * 0.04
                ax.arrow(x, start_y, 0, end_y - start_y,
                         head_width=y_scale * 0.14, head_length=y_scale * 0.12,
                         length_includes_head=True, color="#e74c3c", linewidth=2)
                ax.text(x, start_y - sign * y_scale * 0.18, f"{fy:.2f} kN",
                        ha="center", va="center", fontsize=9, color="#c0392b", fontweight="bold")

            for load in self.data["loads"]["distributed"]:
                eid = load.get("element_id")
                elem = next((e for e in self.data["elements"] if e["id"] == eid), None)
                if not elem:
                    continue
                n1 = self._node_by_id(elem["start_node"])
                n2 = self._node_by_id(elem["end_node"])
                if not n1 or not n2:
                    continue
                x1, x2 = sorted([float(n1["x"]), float(n2["x"])])
                w_start = float(load["w_start"])
                w_end = float(load["w_end"])
                max_abs_w = max(abs(w_start), abs(w_end), 1e-12)
                scale = y_scale * 0.58 / max_abs_w
                h1 = -w_start * scale
                h2 = -w_end * scale
                env_color = "#e67e22"
                ax.plot([x1, x2], [h1, h2], color=env_color, linewidth=1.5, zorder=3)
                n_arrows = max(3, int(round((x2 - x1) / 0.5)))
                for x in np.linspace(x1, x2, n_arrows):
                    ratio = (x - x1) / (x2 - x1) if x2 != x1 else 0
                    h = h1 + ratio * (h2 - h1)
                    if abs(h) < 1e-12:
                        continue
                    arr_h = min(y_scale * 0.085, abs(h) * 0.6)
                    arr_l = arr_h
                    ax.arrow(x, h, 0, -h,
                             head_width=arr_h, head_length=arr_l,
                             length_includes_head=True, color=env_color, linewidth=1.4, zorder=3)
                w_label = f"{w_start:.2f} kN/m" if abs(w_start - w_end) < 1e-10 else f"{w_start:.2f} a {w_end:.2f} kN/m"
                label_offset = y_scale * 0.18
                label_y = (h1 + h2) / 2 + (label_offset if (h1 + h2) >= 0 else -label_offset)
                ax.text((x1 + x2) / 2, label_y, w_label,
                        ha="center", va="center", fontsize=9, color="#d35400", fontweight="bold", zorder=4)

        ax.set_xlim(min(xs) - y_scale * 1.2, max(xs) + y_scale * 1.2)
        ax.set_ylim(-y_scale * 1.25, y_scale * 1.25)
        ax.set_aspect("auto")
        ax.set_xlabel("X (m)")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, linestyle="--")


def build_pynite_model(data):
    return PyNiteBeamModel(data)

# --------------------------------------------------------------
#  Funciones del método matricial de rigidez
# --------------------------------------------------------------
def matriz_local(L, max_dof, a, b, c, d, EI=1):
    K11 =  12 * EI / L**3
    K12 =   6 * EI / L**2
    K13 = -12 * EI / L**3
    K14 =   6 * EI / L**2
    K21 =   6 * EI / L**2
    K22 =   4 * EI / L
    K23 =  -6 * EI / L**2
    K24 =   2 * EI / L
    K31 = -12 * EI / L**3
    K32 =  -6 * EI / L**2
    K33 =  12 * EI / L**3
    K34 =  -6 * EI / L**2
    K41 =   6 * EI / L**2
    K42 =   2 * EI / L
    K43 =  -6 * EI / L**2
    K44 =   4 * EI / L

    K = sp.Matrix.zeros(max_dof, max_dof)
    filas = [a - 1, b - 1, c - 1, d - 1]
    local = sp.Matrix([
        [K11, K12, K13, K14],
        [K21, K22, K23, K24],
        [K31, K32, K33, K34],
        [K41, K42, K43, K44],
    ])
    for i in range(4):
        for j in range(4):
            K[filas[i], filas[j]] += local[i, j]
    return K.evalf(4)


def vector_fuerzas_puntual(P, dist, L, max_dof, a, b, c, d):
    bb = L - dist
    f1 = P * bb**2 * (3 * dist + bb) / L**3
    f2 = P * dist * bb**2 / L**2
    f3 = P * dist**2 * (dist + 3 * bb) / L**3
    f4 = -P * dist**2 * bb / L**2
    local = sp.Matrix([f1, f2, f3, f4])
    F = sp.Matrix.zeros(max_dof, 1)
    filas = [a - 1, b - 1, c - 1, d - 1]
    for i in range(4):
        F[filas[i], 0] += local[i]
    return F


def vector_fuerzas_lineal(q, L, max_dof, a, b, c, d):
    f1 = q * L / 2
    f2 = q * L**2 / 12
    f3 = q * L / 2
    f4 = -q * L**2 / 12
    local = sp.Matrix([f1, f2, f3, f4])
    F = sp.Matrix.zeros(max_dof, 1)
    filas = [a - 1, b - 1, c - 1, d - 1]
    for i in range(4):
        F[filas[i], 0] += local[i]
    return F


def vector_fuerzas_trapecio(q1, q2, L, max_dof, a, b, c, d):
    local = sp.Matrix([
        q1 * L / 2 + 3 * (q2 - q1) * L / 20,
        q1 * L**2 / 12 + (q2 - q1) * L**2 / 30,
        q1 * L / 2 + 7 * (q2 - q1) * L / 20,
        -q1 * L**2 / 12 - (q2 - q1) * L**2 / 20,
    ])
    F = sp.Matrix.zeros(max_dof, 1)
    filas = [a - 1, b - 1, c - 1, d - 1]
    for i in range(4):
        F[filas[i], 0] += local[i]
    return F

def compact_matrix(mat):
    n = mat.rows
    indices = [
        i for i in range(n)
        if not (all(mat[i, j] == 0 for j in range(n)) and
                all(mat[j, i] == 0 for j in range(n)))
    ]
    m = len(indices)
    new_mat = sp.Matrix.zeros(m, m)
    for i_new, i_orig in enumerate(indices):
        for j_new, j_orig in enumerate(indices):
            new_mat[i_new, j_new] = mat[i_orig, j_orig]
    return new_mat, indices

def matrix_to_df(mat, row_indices, col_indices=None, precision=3):
    if col_indices is None:
        col_indices = row_indices
    row_labels = [f"GDL {idx + 1}" for idx in row_indices]
    col_labels = [f"GDL {idx + 1}" for idx in col_indices]
    data = [[float(mat[i, j]) for j in range(mat.cols)] for i in range(mat.rows)]
    df = pd.DataFrame(data, index=row_labels, columns=col_labels)
    return df.map(lambda x: f"{x:.{precision}f}")

# --------------------------------------------------------------
# Funciones de visualización mejoradas
# --------------------------------------------------------------
def plot_and_display(ss, plot_func, title, color_scheme='default'):
    """Función genérica para plotear y mostrar gráficos"""
    try:
        plot_func()
        fig = plt.gcf()
        fig.set_size_inches(12, 5)
        fig.patch.set_facecolor('white')

        if fig.axes:
            ax = fig.axes[0]
            ax.set_facecolor('#fafafa')
            ax.grid(True, alpha=0.3, linestyle='--')

        st.pyplot(fig)
        plt.close(fig)
        return True
    except Exception as e:
        st.error(f"Error al generar {title}: {e}")
        return False

# --------------------------------------------------------------
#  Ejemplos predefinidos
# --------------------------------------------------------------
EXAMPLE_PROJECTS = {
    "Viga simplemente apoyada": {
        "nodes": [
            {"id": 1, "x": 0.0},
            {"id": 2, "x": 3.0},
            {"id": 3, "x": 6.0}
        ],
        "elements": [
            {"id": 1, "start_node": 1, "end_node": 2, "E": 200000000.0, "A": 0.01, "I": 5e-05},
            {"id": 2, "start_node": 2, "end_node": 3, "E": 200000000.0, "A": 0.01, "I": 5e-05}
        ],
        "supports": [
            {"node_id": 1, "type": "Articulado"},
            {"node_id": 3, "type": "Articulado"}
        ],
        "loads": {
            "point": [{"node_id": 2, "fx": 0.0, "fy": -15.0, "mz": 0.0}],
            "distributed": []
        }
    },
    "Viga con voladizo": {
        "nodes": [
            {"id": 1, "x": 0.0},
            {"id": 2, "x": 4.0},
            {"id": 3, "x": 6.0}
        ],
        "elements": [
            {"id": 1, "start_node": 1, "end_node": 2, "E": 200000000.0, "A": 0.01, "I": 5e-05},
            {"id": 2, "start_node": 2, "end_node": 3, "E": 200000000.0, "A": 0.01, "I": 5e-05}
        ],
        "supports": [
            {"node_id": 1, "type": "Empotrado"}
        ],
        "loads": {
            "point": [
                {"node_id": 2, "fx": 0.0, "fy": -10.0, "mz": 0.0},
                {"node_id": 3, "fx": 0.0, "fy": -8.0, "mz": 0.0}
            ],
            "distributed": []
        }
    },
    "Viga con carga distribuida": {
        "nodes": [
            {"id": 1, "x": 0.0},
            {"id": 2, "x": 5.0}
        ],
        "elements": [
            {"id": 1, "start_node": 1, "end_node": 2, "E": 200000000.0, "A": 0.01, "I": 5e-05}
        ],
        "supports": [
            {"node_id": 1, "type": "Articulado"},
            {"node_id": 2, "type": "Rodillo"}
        ],
        "loads": {
            "point": [],
            "distributed": [{"element_id": 1, "w_start": -5.0, "w_end": -5.0, "direction": "y"}]
        }
    }
}

def load_example(example_name):
    """Carga un ejemplo predefinido"""
    if example_name in EXAMPLE_PROJECTS:
        example_data = json.loads(json.dumps(EXAMPLE_PROJECTS[example_name]))

        st.session_state.beam_data["beam"] = {
            "nodes": example_data["nodes"],
            "elements": example_data.get("elements", []),
            "supports": example_data["supports"],
            "loads": example_data["loads"]
        }
        st.session_state.solved_model = None
        save_to_history()
        st.success(f"Ejemplo '{example_name}' cargado")

# ----------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------
with st.sidebar:
    st.markdown("##  Panel de Herramientas")

    # --- Navegación entre secciones ---
    sections = {
        " Nodos": "nodes",
        " Apoyos": "supports",
        " Elementos": "elements",
        " Cargas": "loads",
        " Materiales": "materials",
        " Proyecto": "project"
    }

    selected_section = st.radio(
        "Sección",
        list(sections.keys()),
        label_visibility="collapsed"
    )

    st.divider()

    # ------------------------------------------------------------
    #  SECCIÓN: NODOS
    # ------------------------------------------------------------
    if selected_section == " Nodos":
        st.markdown("###  Gestión de Nodos")

        with st.container():
            col1, col2 = st.columns([3, 1])
            x = col1.number_input(
                "Coordenada X (m)",
                value=0.0,
                step=0.5,
                format="%.2f",
                key="new_node_x",
                label_visibility="collapsed"
            )
            if col2.button("➕", use_container_width=True, help="Añadir nodo"):
                add_node(x)
                st.rerun()

        st.divider()

        # Lista de nodos
        st.markdown("**Nodos definidos:**")

        if not data["nodes"]:
            st.info("No hay nodos. Añada nodos para comenzar.")
        else:
            # Crear tabla de nodos
            node_data = []
            for n in sorted(data["nodes"], key=lambda x: x["id"]):
                has_support = any(s["node_id"] == n["id"] for s in data["supports"])
                has_load = any(l["node_id"] == n["id"] for l in data["loads"]["point"])
                status = []
                if has_support:
                    status.append(" Apoyo")
                if has_load:
                    status.append("Carga")
                node_data.append({
                    "ID": f"N{n['id']}",
                    "X (m)": n["x"],
                    "Estado": ", ".join(status) if status else "—"
                })

            df_nodes = pd.DataFrame(node_data)
            st.dataframe(df_nodes, use_container_width=True, hide_index=True)

            # Eliminar nodos
            st.markdown("**Eliminar nodo:**")
            nodes_to_delete = st.multiselect(
                "Seleccione nodos a eliminar",
                options=[n["id"] for n in data["nodes"]],
                format_func=lambda x: f"N{x}",
                label_visibility="collapsed"
            )
            if nodes_to_delete and st.button("Eliminar seleccionados", type="primary"):
                for nid in nodes_to_delete:
                    remove_node(nid)
                st.rerun()

    # ------------------------------------------------------------
    #  SECCIÓN: APOYOS
    # ------------------------------------------------------------
    elif selected_section == " Apoyos":
        st.markdown("###  Asignar Apoyos")

        node_ids = [n["id"] for n in sorted(data["nodes"], key=lambda x: x["id"])]

        if not node_ids:
            st.warning(" Añada nodos primero en la sección Nodos.")
        else:
            col1, col2 = st.columns(2)
            s_node = col1.selectbox("Nodo", node_ids, format_func=lambda x: f"N{x}")
            s_type = col2.selectbox("Tipo", ["Empotrado", "Articulado", "Rodillo"])

            # Iconos para cada tipo
            type_icons = {
                "Empotrado": "",
                "Articulado": "",
                "Rodillo": ""
            }

            if st.button(f"{type_icons[s_type]} Asignar Apoyo", use_container_width=True):
                if add_support(s_node, s_type):
                    st.success(f"Apoyo {s_type} asignado a N{s_node}")
                    st.rerun()
                else:
                    st.error("Este nodo ya tiene un apoyo asignado")

        st.divider()

        # Lista de apoyos
        st.markdown("**Condiciones de Frontera:**")

        if not data["supports"]:
            st.info("No hay apoyos definidos.")
        else:
            support_data = []
            for s in sorted(data["supports"], key=lambda x: x["node_id"]):
                support_data.append({
                    "Nodo": f"N{s['node_id']}",
                    "Tipo": s["type"],
                    "Restricciones": {
                        "Empotrado": "Dx, Dy, Rz",
                        "Articulado": "Dx, Dy",
                        "Rodillo": "Dy"
                    }.get(s["type"], "—")
                })

            df_supports = pd.DataFrame(support_data)
            st.dataframe(df_supports, use_container_width=True, hide_index=True)

            # Leyenda de símbolos
            st.markdown("""
            **Leyenda:**
            -  Empotrado: Fija traslación y rotación
            -  Articulado: Fija ambas translaciones
            -  Rodillo: Fija una traslación (vertical)
            """)

            # Eliminar apoyo
            support_to_delete = st.selectbox(
                "Eliminar apoyo",
                options=[s["node_id"] for s in data["supports"]],
                format_func=lambda x: f"N{x}",
                key="del_support_select"
            )
            if st.button("Eliminar apoyo", type="primary"):
                remove_support(support_to_delete)
                st.rerun()

    # ------------------------------------------------------------
    #  SECCIÓN: ELEMENTOS
    # ------------------------------------------------------------
    elif selected_section == " Elementos":
        st.markdown("### Gestion de Elementos")

        node_ids = [n["id"] for n in sorted(data["nodes"], key=lambda x: x["id"])]

        if len(node_ids) < 2:
            st.warning("Añada al menos 2 nodos primero en la seccion Nodos.")
        else:
            col1, col2 = st.columns(2)
            sn = col1.selectbox("Nodo inicial", node_ids, format_func=lambda x: f"N{x}", key="elem_start")
            en = col2.selectbox("Nodo final", node_ids, format_func=lambda x: f"N{x}", key="elem_end")
            if sn == en:
                st.warning("Los nodos deben ser diferentes.")
            else:
                if st.button("Anadir Elemento", use_container_width=True):
                    add_element(sn, en)
                    st.rerun()

        st.divider()
        st.markdown("**Elementos definidos:**")

        if not data["elements"]:
            st.info("No hay elementos. Defina al menos un elemento.")
        else:
            elem_data = []
            for e in sorted(data["elements"], key=lambda x: x["id"]):
                n1 = next((n for n in data["nodes"] if n["id"] == e["start_node"]), None)
                n2 = next((n for n in data["nodes"] if n["id"] == e["end_node"]), None)
                x1 = n1["x"] if n1 else "?"
                x2 = n2["x"] if n2 else "?"
                elem_data.append({
                    "ID": f"E{e['id']}",
                    "Nodos": f"N{e['start_node']} - N{e['end_node']}",
                    "X (m)": f"{x1} - {x2}",
                })

            df_elems = pd.DataFrame(elem_data)
            st.dataframe(df_elems, use_container_width=True, hide_index=True)

            elem_to_del = st.selectbox(
                "Eliminar elemento",
                options=[e["id"] for e in data["elements"]],
                format_func=lambda x: f"E{x}",
                key="del_elem_select"
            )
            if st.button("Eliminar elemento", type="primary"):
                remove_element(elem_to_del)
                st.rerun()

    # ------------------------------------------------------------
    #  SECCIÓN: CARGAS
    # ------------------------------------------------------------
    elif selected_section == " Cargas":
        st.markdown("###  Aplicar Cargas")

        node_ids = [n["id"] for n in sorted(data["nodes"], key=lambda x: x["id"])]

        if not node_ids:
            st.warning(" Añada nodos primero.")
        else:
            load_type = st.radio(
                "Tipo de carga",
                ["Puntual", "Distribuida"],
                horizontal=True
            )

            if load_type == "Puntual":
                pt_mode = st.radio("Ubicacion", ["En nodo", "Sobre elemento"], horizontal=True)
                if pt_mode == "En nodo":
                    col1, col2 = st.columns(2)
                    p_node = col1.selectbox("Nodo", node_ids, format_func=lambda x: f"N{x}")
                    fy = col2.number_input("Fy (kN)", value=-10.0, step=1.0)
                    st.caption("Positivo = hacia arriba | Negativo = hacia abajo")
                    if st.button("Aplicar Carga Puntual (nodo)", use_container_width=True):
                        add_point_load(p_node, fy)
                        st.success(f"Carga puntual de {fy} kN aplicada en N{p_node}")
                        st.rerun()
                else:
                    if not data["elements"]:
                        st.warning("Defina elementos primero en la seccion Elementos.")
                    else:
                        elem_opts = {f"E{e['id']} (N{e['start_node']}-N{e['end_node']})": e
                                     for e in data["elements"]}
                        sel_label = st.selectbox("Elemento", list(elem_opts.keys()), index=0)
                        sel_elem = elem_opts[sel_label]
                        n1 = next((n for n in data["nodes"] if n["id"] == sel_elem["start_node"]), None)
                        n2 = next((n for n in data["nodes"] if n["id"] == sel_elem["end_node"]), None)
                        x_min = min(n1["x"], n2["x"]) if n1 and n2 else 0
                        x_max = max(n1["x"], n2["x"]) if n1 and n2 else 1
                        col1, col2 = st.columns(2)
                        x_pos = col1.number_input("X (m)", value=float((x_min+x_max)/2),
                                                  min_value=float(x_min), max_value=float(x_max),
                                                  step=0.5, format="%.2f")
                        fy_el = col2.number_input("Fy (kN)", value=-10.0, step=1.0, key="fy_elem")
                        st.caption(f"Elemento de x={x_min:.2f} a x={x_max:.2f} m")
                        if st.button("Aplicar Carga Puntual (elemento)", use_container_width=True):
                            add_point_load_on_element(sel_elem["id"], x_pos, fy_el)
                            st.success(f"Carga puntual de {fy_el} kN en E{sel_elem['id']} @ x={x_pos:.2f} m")
                            st.rerun()

            else:  # Distribuida
                if not data["elements"]:
                    st.warning("Defina elementos primero en la seccion Elementos.")
                else:
                    elem_options = {f"E{e['id']} (N{e['start_node']}-N{e['end_node']})": e["id"] for e in data["elements"]}
                    sel_elem_label = st.selectbox("Elemento", options=list(elem_options.keys()), index=0)
                    sel_elem_id = elem_options[sel_elem_label]
                    col_q1, col_q2 = st.columns(2)
                    w1 = col_q1.number_input("q inicio (kN/m)", value=-5.0, step=0.5)
                    w2 = col_q2.number_input("q fin (kN/m)", value=-5.0, step=0.5)
                    if st.button(" Aplicar Carga Distribuida", use_container_width=True):
                        add_distributed_load(sel_elem_id, w1, w2)
                        st.success(f"Carga distribuida de {w1} a {w2} kN/m aplicada en E{sel_elem_id}")
                        st.rerun()

        st.divider()

        # Lista de cargas
        st.markdown("**Cargas Definidas:**")

        total_point = len(data["loads"]["point"])
        total_dist = len(data["loads"]["distributed"])

        col_p, col_d = st.columns(2)
        col_p.metric("Puntuales", total_point)
        col_d.metric("Distribuidas", total_dist)

        if total_point > 0:
            st.markdown("**Cargas Puntuales:**")
            for i, l in enumerate(data["loads"]["point"]):
                col1, col2 = st.columns([4, 1])
                if "element_id" in l:
                    eid = l["element_id"]
                    xp = l.get("x", "?")
                    col1.markdown(f"**E{eid}** @ x={xp} m: {l['fy']} kN" + (" (abajo)" if l['fy'] < 0 else " (arriba)"))
                else:
                    col1.markdown(f"**N{l['node_id']}**: {l['fy']} kN" + (" (abajo)" if l['fy'] < 0 else " (arriba)"))
                if col2.button("X", key=f"del_pt_{i}"):
                    remove_point_load(i)
                    st.rerun()

        if total_dist > 0:
            st.markdown("**Cargas Distribuidas:**")
            for i, l in enumerate(data["loads"]["distributed"]):
                eid = l.get("element_id", "?")
                elem = next((e for e in data["elements"] if e["id"] == eid), None)
                w_label = f"{l['w_start']} kN/m" if abs(float(l['w_start']) - float(l['w_end'])) < 1e-10 else f"{l['w_start']} a {l['w_end']} kN/m"
                label = f"**E{eid}**" + (f" (N{elem['start_node']}-N{elem['end_node']})" if elem else "") + f": {w_label}"
                col1, col2 = st.columns([4, 1])
                col1.markdown(label)
                if col2.button("X", key=f"del_dist_{i}"):
                    remove_distributed_load(i)
                    st.rerun()

        if total_point == 0 and total_dist == 0:
            st.info("No hay cargas definidas.")

    # ------------------------------------------------------------
    #  SECCIÓN: MATERIALES
    # ------------------------------------------------------------
    elif selected_section == " Materiales":
        st.markdown("###  Propiedades de Materiales")

        st.info("Configuración de propiedades por defecto para nuevos elementos")

        col1, col2 = st.columns(2)
        with col1:
            E = st.number_input(
                "Módulo E (kN/m²)",
                value=float(materials["E_default"]) / 1000,  # Convertir
                step=1000.0,
                format="%.0f",
                help="Módulo de elasticidad"
            )
            materials["E_default"] = E * 1000

        with col2:
            A = st.number_input(
                "Área A (m²)",
                value=materials["A_default"],
                step=0.001,
                format="%.5f",
                help="Área de la sección transversal"
            )
            materials["A_default"] = A

        I = st.number_input(
            "Inercia I (m4)",
            value=materials["I_default"],
            step=0.00001,
            format="%.6f",
            help="Momento de inercia"
        )
        materials["I_default"] = I

        st.divider()

        st.markdown("**Secciones comunes:**")
        sections_presets = {
            "HEB 200": {"A": 0.0078, "I": 0.0000567},
            "HEA 200": {"A": 0.0069, "I": 0.0000369},
            "IPE 200": {"A": 0.00228, "I": 0.00001944},
            "Viga rectangular 30x50 cm": {"A": 0.15, "I": 0.0003125},
        }

        for name, vals in sections_presets.items():
            if st.button(f"Aplicar {name}"):
                materials["A_default"] = vals["A"]
                materials["I_default"] = vals["I"]
                st.success(f"Sección {name} aplicada")
                st.rerun()

    # ------------------------------------------------------------
    #  SECCIÓN: PROYECTO
    # ------------------------------------------------------------
    elif selected_section == " Proyecto":
        st.markdown("###  Gestión de Proyecto")

        # Nombre del proyecto
        project_name = st.text_input(
            "Nombre del proyecto",
            value=st.session_state.beam_data.get("name", "Nuevo Proyecto")
        )
        st.session_state.beam_data["name"] = project_name

        st.divider()

        # Guardar/Cargar
        st.markdown("**Guardar/Cargar:**")

        json_str = json.dumps(st.session_state.beam_data, indent=2)
        st.download_button(
            " Descargar Modelo (JSON)",
            json_str,
            file_name=f"modelo_{project_name.replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

        uploaded_file = st.file_uploader(
            " Cargar Modelo (JSON)",
            type=["json"],
            help="Seleccione un archivo JSON previamente guardado"
        )

        if uploaded_file:
            try:
                loaded_data = json.load(uploaded_file)
                if "beam" in loaded_data and "nodes" in loaded_data["beam"]:
                    migrar_distribuidas(loaded_data)
                    if st.button("Confirmar carga"):
                        st.session_state.beam_data = loaded_data
                        st.session_state.solved_model = None
                        save_to_history()
                        st.success("Modelo cargado correctamente")
                        st.rerun()
            except Exception as e:
                st.error(f"Error al cargar: {e}")

# ----------------------------------------------------------------
# ÁREA PRINCIPAL
# ----------------------------------------------------------------
    st.markdown("# ANÁLISIS ESTRUCTURAL ACADÉMICO")
st.markdown(f"**Proyecto:** {st.session_state.beam_data.get('name', 'Sin nombre')} | **Versión:** {st.session_state.beam_data['version']}")

# --- Barra de acciones superiores ---
top_col1, top_col2 = st.columns([1, 0.5])

with top_col1:
    solve_btn = st.button(" Resolver Modelo", type="primary", use_container_width=True)
with top_col2:
    if st.button("🏠", use_container_width=True, help="Volver al Inicio"):
        st.session_state.pagina = "home"
        st.rerun()

undo_col1, undo_col2, undo_col3 = st.columns([1, 1, 1])
with undo_col1:
    can_undo = st.session_state.history_index > 0
    st.button(" Deshacer", disabled=not can_undo, use_container_width=True, on_click=undo)
with undo_col2:
    can_redo = st.session_state.history_index < len(st.session_state.history) - 1
    st.button(" Rehacer", disabled=not can_redo, use_container_width=True, on_click=redo)
with undo_col3:
    if st.button(" Borrar Todo", type="secondary", use_container_width=True):
        st.session_state.beam_data["beam"] = {
            "nodes": [],
            "elements": [],
            "supports": [],
            "loads": {"point": [], "distributed": []}
        }
        st.session_state.solved_model = None
        save_to_history()
        st.rerun()

# --- Resolver modelo ---
if solve_btn:
    if not data["nodes"]:
        st.error(" Defina al menos un nodo para resolver.")
    elif not data["supports"]:
        st.error(" Defina al menos un apoyo para resolver.")
    elif len(data["nodes"]) < 2:
        st.error(" Se necesitan al menos 2 nodos para formar un elemento.")
    else:
        with st.spinner(" Calculando... por favor espere"):
            try:
                ss = build_pynite_model(data)
                ss.solve()
                st.session_state.solved_model = ss

                # Mostrar métricas de resultados
                st.success("Modelo resuelto correctamente")

            except Exception as e:
                st.error(f"Error en el cálculo: {str(e)}")
                import traceback
                with st.expander("Detalles del error"):
                    st.code(traceback.format_exc())
                st.session_state.solved_model = None

st.divider()

# --- Pestañas de visualización ---
tab_struct, tab_results, tab_matrix = st.tabs([
    "Estructura",
    " Diagramas",
    " Método Matricial"
])

# --- 1. Vista de Estructura ---
with tab_struct:
    if data["nodes"]:
        try:
            ss_preview = build_pynite_model(data)
            ss_preview.show_structure()
            fig_preview = plt.gcf()
            fig_preview.set_size_inches(12, 5)
            fig_preview.patch.set_facecolor('white')

            if fig_preview.axes:
                ax = fig_preview.axes[0]
                ax.set_facecolor('#f8f9fa')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_title("Vista de la Estructura", fontsize=14, fontweight='bold', pad=15)

            st.pyplot(fig_preview)
            plt.close(fig_preview)

            # Botón para descargar gráfico
            buf = io.BytesIO()
            fig_preview.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                " Descargar gráfico como PNG",
                buf,
                file_name="estructura.png",
                mime="image/png",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error al dibujar la estructura: {e}")
    else:
        st.info(" Utilice el panel lateral para añadir nodos, apoyos y cargas.")

# --- 2. Diagramas de Resultados ---
with tab_results:
    ss = st.session_state.solved_model

    if not ss:
        st.info(" Presione **Resolver Modelo** para generar los diagramas de resultados.")
    else:
        # Selector de diagrama
        diagram_type = st.radio(
            "Seleccione diagrama:",
            ["Reacciones", "Fuerza Cortante (V)", "Momento Flector (M)"],
            horizontal=True
        )

        if diagram_type == "Reacciones":
            plot_and_display(ss, ss.show_reaction_force, "Diagrama de Reacciones")
        elif diagram_type == "Fuerza Cortante (V)":
            plot_and_display(ss, ss.show_shear_force, "Diagrama de Fuerza Cortante")
        elif diagram_type == "Momento Flector (M)":
            try:
                fig_moment = ss.show_bending_moment()
                st.pyplot(fig_moment)
                plt.close(fig_moment)

                stats_m = ss.get_bending_moment_stats()
                st.markdown("---")
                st.markdown("###  Momento Flector Maximo Absoluto")
                st.metric(
                    "Momento maximo",
                    f"{stats_m['abs_max_moment']:.4f} kN*m",
                    delta=f"X = {stats_m['abs_max_x']:.2f} m" if stats_m['abs_max_x'] is not None else None
                )
            except:
                st.warning("No se pudo generar el diagrama de momento flector")

        # Descargar diagrama
        if st.button(" Descargar diagrama", use_container_width=True):
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                "Descargar",
                buf,
                file_name=f"diagrama_{diagram_type.lower().replace(' ', '_')}.png",
                mime="image/png"
            )

# --- 3. Método Matricial ---
with tab_matrix:
    st.markdown("##  Método Matricial de Rigidez")
    st.markdown("Se calculan las matrices de rigidez y el vector de fuerzas a partir de la estructura definida en el panel lateral.")

    if not data["nodes"] or len(data["nodes"]) < 2:
        st.info(" Define al menos 2 nodos en el panel lateral primero.")
    elif not data["elements"]:
        st.info(" Define al menos un elemento (2 nodos) en el panel lateral.")
    elif not data["supports"]:
        st.warning(" Define al menos un apoyo para poder resolver el sistema.")
    else:
        all_nodes_full = sorted(data["nodes"], key=lambda x: x["x"])
        node_x_map = {n["id"]: n["x"] for n in data["nodes"]}

        connected_ids = set()
        for e in data["elements"]:
            connected_ids.add(e["start_node"])
            connected_ids.add(e["end_node"])
        all_nodes = [n for n in all_nodes_full if n["id"] in connected_ids]

        max_dof = 2 * len(all_nodes)
        dof_map = {}
        for i, node in enumerate(all_nodes):
            dof_map[node["id"]] = (2 * i + 1, 2 * i + 2)

        constrained_dofs = []
        dof_data = []
        for node in all_nodes:
            v_dof, r_dof = dof_map[node["id"]]
            support = next((s for s in data["supports"] if s["node_id"] == node["id"]), None)
            support_type = support["type"] if support else "Libre"

            v_constrained = False
            r_constrained = False
            if support:
                if support["type"] == "Empotrado":
                    v_constrained = True
                    r_constrained = True
                elif support["type"] in ("Articulado", "Rodillo"):
                    v_constrained = True

            if v_constrained:
                constrained_dofs.append(v_dof)
            if r_constrained:
                constrained_dofs.append(r_dof)

            dof_data.append({
                "Nodo": f"N{node['id']}",
                "X (m)": node["x"],
                "GDL Deflexión": v_dof,
                "GDL Rotación": r_dof,
                "Apoyo": support_type,
                "Deflexión": "Restringido" if v_constrained else "Libre",
                "Rotación": "Restringido" if r_constrained else "Libre",
            })

        st.markdown("### Asignación de Grados de Libertad")
        df_dof = pd.DataFrame(dof_data)
        st.dataframe(df_dof, use_container_width=True, hide_index=True)

        ei_modo = st.radio(
            "Valores de EI:",
            ["EI = 1 (normalizado)", "EI real (E × I de cada elemento)"],
            horizontal=True,
            key="mat_ei_modo"
        )
        usar_ei_real = ei_modo == "EI real (E × I de cada elemento)"

        valid_elements = []
        for elem in data["elements"]:
            sn = elem["start_node"]
            en = elem["end_node"]
            if sn not in dof_map or en not in dof_map:
                continue
            a, b = dof_map[sn]
            c, d = dof_map[en]
            L = abs(node_x_map[en] - node_x_map[sn])
            EI_val = (elem["E"] * elem["I"]) if usar_ei_real else 1.0
            valid_elements.append({
                "idx": elem["id"],
                "start_node": sn,
                "end_node": en,
                "a": a, "b": b, "c": c, "d": d,
                "L": L, "EI": EI_val,
            })

        # --- Cargas automáticas desde la estructura ---
        elem_loads = {}
        for ve in valid_elements:
            eid = str(ve["idx"])
            sn = ve["start_node"]
            en = ve["end_node"]
            L = ve["L"]
            loads = []

            for pl in data["loads"]["point"]:
                eid_pl = pl.get("element_id")
                if eid_pl == ve["idx"]:
                    loads.append({"type": "PuntualElem", "P": float(pl.get("fy", 0)),
                                  "x": pl.get("x", 0)})
                elif pl.get("node_id") == sn:
                    loads.append({"type": "Puntual", "P": float(pl.get("fy", 0)), "d": 0.0})
                elif pl.get("node_id") == en:
                    loads.append({"type": "Puntual", "P": float(pl.get("fy", 0)), "d": L})
                elif pl.get("node_id") not in connected_ids:
                    nx = node_x_map.get(pl.get("node_id"))
                    if nx is not None:
                        x1 = node_x_map[sn]; x2 = node_x_map[en]
                        if min(x1, x2) - 1e-10 <= nx <= max(x1, x2) + 1e-10:
                            d_pos = abs(nx - min(x1, x2))
                            loads.append({"type": "PuntualElem", "P": float(pl.get("fy", 0)),
                                          "x": nx})

            for dl in data["loads"]["distributed"]:
                if dl.get("element_id") == ve["idx"]:
                    w_start = float(dl.get("w_start", 0))
                    w_end = float(dl.get("w_end", 0))
                    if abs(w_start - w_end) < 1e-10:
                        loads.append({"type": "Lineal", "q": w_start})
                    else:
                        loads.append({"type": "Trapecio", "q1": w_start, "q2": w_end})

            elem_loads[eid] = loads

        elem_display = []
        for ve in valid_elements:
            eid = str(ve["idx"])
            loads_list = elem_loads.get(eid, [])
            cargas_strs = []
            for ld in loads_list:
                if ld["type"] == "PuntualElem":
                    cargas_strs.append(f"P = {ld['P']:.2f} kN @ x={ld['x']:.2f} m")
                elif ld["type"] == "Puntual":
                    pos_str = "nodo inicial" if ld["d"] == 0 else f"d = {ld['d']:.2f} m"
                    cargas_strs.append(f"P = {ld['P']:.2f} kN ({pos_str})")
                elif ld["type"] == "Lineal":
                    cargas_strs.append(f"q = {ld['q']:.2f} kN/m")
                elif ld["type"] == "Trapecio":
                    cargas_strs.append(f"q = {ld['q1']:.2f} a {ld['q2']:.2f} kN/m")
            carga_str = ", ".join(cargas_strs) if cargas_strs else "Sin carga"
            elem_display.append({
                "Elemento": f"E{ve['idx']}",
                "Nodos": f"N{ve['start_node']} - N{ve['end_node']}",
                "L (m)": round(ve["L"], 4),
                "EI": round(ve["EI"], 4),
                "Carga": carga_str,
                "GDL": f"({ve['a']},{ve['b']},{ve['c']},{ve['d']})",
            })
        st.markdown("### Elementos y Cargas")
        df_elem = pd.DataFrame(elem_display)
        st.dataframe(df_elem, use_container_width=True, hide_index=True)

        st.divider()
        col_mat_btn, col_mat_sp = st.columns([1, 3])
        with col_mat_btn:
            resolver_mat = st.button("Resolver Metodo Matricial", type="primary",
                                     use_container_width=True, key="btn_mat_solve")

        if resolver_mat or "matricial_solver" in st.session_state:
            P_global = sp.Matrix.zeros(max_dof, 1)
            Qf_global = sp.Matrix.zeros(max_dof, 1)

            for pl in data["loads"]["point"]:
                eid = pl.get("element_id")
                if eid is not None:
                    elem = next((e for e in data["elements"] if e["id"] == eid), None)
                    if elem and elem["start_node"] in dof_map and elem["end_node"] in dof_map:
                        a, b = dof_map[elem["start_node"]]
                        c, d = dof_map[elem["end_node"]]
                        L = abs(node_x_map[elem["end_node"]] - node_x_map[elem["start_node"]])
                        x_pos = pl.get("x")
                        if x_pos is not None:
                            x_start = min(node_x_map[elem["start_node"]], node_x_map[elem["end_node"]])
                            dist = abs(float(x_pos) - x_start)
                        else:
                            dist = float(pl.get("d", L / 2))
                        fy = float(pl.get("fy", 0))
                        if abs(fy) > 1e-15:
                            F_pt = vector_fuerzas_puntual(fy, dist, L, max_dof, a, b, c, d)
                            Qf_global += F_pt
                else:
                    nid = pl.get("node_id")
                    if nid is not None:
                        fy = float(pl.get("fy", 0))
                        mz = float(pl.get("mz", 0))
                        if nid in dof_map:
                            v_dof, r_dof = dof_map[nid]
                            P_global[v_dof - 1, 0] += fy
                            P_global[r_dof - 1, 0] += mz
                        else:
                            node_x_val = node_x_map.get(nid)
                            if node_x_val is not None:
                                for ve in valid_elements:
                                    x1 = node_x_map[ve["start_node"]]
                                    x2 = node_x_map[ve["end_node"]]
                                    if min(x1, x2) - 1e-10 <= node_x_val <= max(x1, x2) + 1e-10:
                                        a, b, c, d = ve["a"], ve["b"], ve["c"], ve["d"]
                                        L = abs(x2 - x1)
                                        d_pos = abs(node_x_val - min(x1, x2))
                                        if abs(fy) > 1e-15:
                                            F_pt = vector_fuerzas_puntual(fy, d_pos, L, max_dof, a, b, c, d)
                                            Qf_global += F_pt
                                        break

            K_global = sp.Matrix.zeros(max_dof, max_dof)

            st.markdown("### Matrices Locales")
            for ve in valid_elements:
                a, b, c, d = ve["a"], ve["b"], ve["c"], ve["d"]
                L = ve["L"]
                EI_val = ve["EI"]

                K_local = matriz_local(L, max_dof, a, b, c, d, EI=EI_val)
                K_global += K_local
                K_compact, indices = compact_matrix(K_local)
                df_local = matrix_to_df(K_compact, indices)

                st.markdown(f"**K{ve['idx']}** — Elemento E{ve['idx']} (N{ve['start_node']} → N{ve['end_node']}), L = {L:.2f} m, EI = {EI_val:.4f}, GDL: ({a}, {b}, {c}, {d})")
                st.dataframe(df_local, use_container_width=True)

                for dl in data["loads"]["distributed"]:
                    if dl.get("element_id") != ve["idx"]:
                        continue
                    w_start = float(dl.get("w_start", 0))
                    w_end = float(dl.get("w_end", 0))
                    if abs(w_start - w_end) < 1e-10:
                        F_local = vector_fuerzas_lineal(w_start, L, max_dof, a, b, c, d)
                        label = f"Elemento E{ve['idx']} — carga distribuida: q = {w_start:.2f} kN/m"
                    else:
                        F_local = vector_fuerzas_trapecio(w_start, w_end, L, max_dof, a, b, c, d)
                        label = f"Elemento E{ve['idx']} — carga distribuida: q = {w_start:.2f} a {w_end:.2f} kN/m"
                    Qf_global += F_local
                    nonzero_idx = [k for k in range(max_dof) if F_local[k] != 0]
                    nonzero_vals = [float(F_local[k]) for k in nonzero_idx]
                    labels_f = [f"GDL {k+1}" for k in nonzero_idx]
                    df_f = pd.DataFrame(nonzero_vals, index=labels_f, columns=["Fuerza equivalente"])
                    df_f["Fuerza equivalente"] = df_f["Fuerza equivalente"].map(lambda x: f"{x:.4f}")
                    st.markdown(f"**{label}**")
                    st.dataframe(df_f, use_container_width=True)

            F_global = P_global + Qf_global

            st.markdown("### Matriz de Rigidez Global")
            all_indices = list(range(max_dof))
            df_global = matrix_to_df(K_global, all_indices)
            st.markdown(f"**K Global [{max_dof}x{max_dof}]**")
            st.dataframe(df_global, use_container_width=True)

            st.markdown("### Vector de Cargas Global")

            nonzero_P = [k for k in range(max_dof) if P_global[k] != 0]
            if nonzero_P:
                st.markdown("**P — Cargas puntuales en nodos:**")
                vals_P = [float(P_global[k]) for k in nonzero_P]
                labels_P = [f"GDL {k+1}" for k in nonzero_P]
                df_P = pd.DataFrame(vals_P, index=labels_P, columns=["Fuerza (kN)"])
                df_P["Fuerza (kN)"] = df_P["Fuerza (kN)"].map(lambda x: f"{x:.4f}")
                st.dataframe(df_P, use_container_width=True)

            nonzero_Qf = [k for k in range(max_dof) if Qf_global[k] != 0]
            if nonzero_Qf:
                st.markdown("**Qf — Fuerzas equivalentes de cargas distribuidas:**")
                vals_Qf = [float(Qf_global[k]) for k in nonzero_Qf]
                labels_Qf = [f"GDL {k+1}" for k in nonzero_Qf]
                df_Qf = pd.DataFrame(vals_Qf, index=labels_Qf, columns=["Fuerza equivalente (kN)"])
                df_Qf["Fuerza equivalente (kN)"] = df_Qf["Fuerza equivalente (kN)"].map(lambda x: f"{x:.4f}")
                st.dataframe(df_Qf, use_container_width=True)

            nonzero_F = [k for k in range(max_dof) if F_global[k] != 0]
            if nonzero_F:
                st.markdown("**F = P + Qf — Vector de fuerzas combinado:**")
                vals_F = [float(F_global[k]) for k in nonzero_F]
                labels_F = [f"GDL {k+1}" for k in nonzero_F]
                df_F = pd.DataFrame(vals_F, index=labels_F, columns=["Fuerza (kN)"])
                df_F["Fuerza (kN)"] = df_F["Fuerza (kN)"].map(lambda x: f"{x:.4f}")
                st.dataframe(df_F, use_container_width=True)
            else:
                st.info("Vector de fuerzas global nulo (sin cargas).")

            solver = BeamMatrixSolver(
                data["nodes"], data["elements"], data["supports"],
                data["loads"]["point"], data["loads"]["distributed"],
                use_real_ei=usar_ei_real
            )

            st.markdown("---")
            st.markdown("## Verificacion de Estabilidad")
            stability = solver.check_stability()
            if stability["status"] == "error":
                st.error(f"{stability['message']}")
            elif stability["status"] == "warning":
                st.warning(f"{stability['message']}")
            else:
                cond_msg = f", Condicion: {stability.get('condition_number', 0):.2e}" if stability.get('condition_number') else ""
                st.success(f"Matriz K_ff bien condicionada. Rango: {stability['rank']}{cond_msg}")

            if stability["status"] == "error":
                st.stop()

            solved_ok = solver.solve()
            if not solved_ok:
                st.error("No se pudo resolver el sistema.")
                st.stop()

            st.session_state["matricial_solver"] = solver
            summary = solver.get_solution_summary()

            free_indices = solver.free_idx
            constrained_indices = solver.cons_idx

            if not free_indices:
                st.warning("Todos los GDL estan restringidos. No hay incognitas que resolver.")
            else:
                K_ff = K_global.extract(free_indices, free_indices)
                F_f = F_global.extract(free_indices, [0])

                st.markdown("---")
                st.markdown("## Solucion del Sistema")

                free_labels = ", ".join(f"u{i+1}" for i in free_indices)
                cons_labels = ", ".join(f"u{i+1}" for i in constrained_indices) if constrained_indices else "ninguno"
                st.markdown(f"**GDL libres:** {free_labels}  |  **GDL restringidos:** {cons_labels}")

                st.markdown("### 1. Particion $K_{ff} \\cdot u_f = F_f$")
                c1, c2 = st.columns(2)
                with c1:
                    labels_free = [f"GDL {i+1}" for i in free_indices]
                    df_Kff = matrix_to_df(K_ff, free_indices)
                    st.markdown("**$K_{ff}$** — Rigidez GDL libres")
                    st.dataframe(df_Kff, use_container_width=True)
                with c2:
                    df_ff = pd.DataFrame(
                        [float(F_f[i]) for i in range(len(free_indices))],
                        index=labels_free, columns=["Fuerza"]
                    )
                    df_ff["Fuerza"] = df_ff["Fuerza"].map(lambda x: f"{x:.4f}")
                    st.markdown("**$F_f$** — Fuerzas en GDL libres")
                    st.dataframe(df_ff, use_container_width=True)

                st.markdown("### 2. Desplazamientos $u_f = K_{ff}^{-1} \\cdot F_f$")
                disp_data = []
                for node in all_nodes:
                    v_dof, r_dof = dof_map[node["id"]]
                    v_val, r_val = solver.get_node_displacement(node["id"])
                    if v_val is not None:
                        disp_data.append({
                            "Nodo": f"N{node['id']}",
                            "Tipo": "Deflexion",
                            "Valor (m)": f"{v_val:.6f}",
                            "Valor (mm)": f"{v_val*1000:.4f}",
                        })
                    if r_val is not None and (r_dof not in constrained_dofs):
                        disp_data.append({
                            "Nodo": f"N{node['id']}",
                            "Tipo": "Rotacion",
                            "Valor (rad)": f"{r_val:.6f}",
                            "Valor (mrad)": f"{r_val*1000:.4f}",
                        })
                df_disp = pd.DataFrame(disp_data)
                st.dataframe(df_disp, use_container_width=True, hide_index=True)

                st.markdown("### 3. Fuerzas en Extremos $Q_e = k_e \\cdot u_e - Qf_e$")
                ef_data = []
                for ve in valid_elements:
                    eid = ve["idx"]
                    ef = solver.elem_end_forces.get(eid)
                    if ef:
                        ef_data.append({
                            "Elem": f"E{eid}",
                            "Vi": f"{ef['Vi']:.4f}",
                            "Mi": f"{ef['Mi']:.4f}",
                            "Vj": f"{ef['Vj']:.4f}",
                            "Mj": f"{ef['Mj']:.4f}",
                        })
                if ef_data:
                    df_ef = pd.DataFrame(ef_data)
                    st.caption("Vi,Vj > 0 = hacia arriba | Mi,Mj > 0 = sagging")
                    st.dataframe(df_ef, use_container_width=True, hide_index=True)

                if constrained_indices:
                    st.markdown("### 4. Reacciones $R = K_{cf} \\cdot u_f - F_c$")
                    rxn_data = []
                    for node in all_nodes:
                        nid = node["id"]
                        support = next((s for s in data["supports"] if s["node_id"] == nid), None)
                        if not support:
                            continue
                        r = solver.get_reactions().get(nid, {})
                        rxn_data.append({
                            "Nodo": f"N{nid}",
                            "Tipo": support["type"],
                            "Ry": f"{r.get('Fy',0):.4f}",
                            "M": f"{r.get('M',0):.4f}" if abs(r.get('M',0)) > 1e-10 else "-",
                        })
                    if rxn_data:
                        df_rxn = pd.DataFrame(rxn_data)
                        st.dataframe(df_rxn, use_container_width=True, hide_index=True)

                    ss_pynite = st.session_state.get("solved_model")
                    if ss_pynite and hasattr(ss_pynite, 'reaction_force') and ss_pynite.reaction_force:
                        with st.expander("Comparacion con PyNite"):
                            comp_rxn = []
                            for node in all_nodes:
                                nid = node["id"]
                                py_rxn = ss_pynite.reaction_force.get(nid, {})
                                py_ry = float(py_rxn.get("Fy", 0)) if isinstance(py_rxn, dict) else 0
                                mat_rxn = solver.get_reactions().get(nid, {})
                                mat_ry = mat_rxn.get("Fy", 0)
                                if abs(mat_ry) > 1e-10 or abs(py_ry) > 1e-10:
                                    comp_rxn.append({
                                        "Nodo": f"N{nid}",
                                        "M. Matricial Ry": f"{mat_ry:.4f}",
                                        "PyNite Ry": f"{py_ry:.4f}",
                                        "Error": f"{abs(mat_ry-py_ry):.4f}",
                                    })
                            if comp_rxn:
                                st.markdown("**Reacciones:**")
                                st.dataframe(pd.DataFrame(comp_rxn), use_container_width=True, hide_index=True)
                else:
                    st.info("No hay apoyos definidos. No se pueden calcular reacciones.")

                st.markdown("### 5. Equilibrio $\\sum F_y = 0$")
                eq_check = solver.verify_equilibrium()
                col_eq1, col_eq2, col_eq3 = st.columns(3)
                col_eq1.metric("Cargas totales (kN)", f"{eq_check['total_load']:.4f}")
                col_eq2.metric("Reacciones totales (kN)", f"{eq_check['total_reaction']:.4f}")
                col_eq3.metric("Error (kN)", f"{eq_check['error']:.6f}",
                              delta="OK" if eq_check['passed'] else "FALLA")
                if eq_check["passed"]:
                    st.success("Equilibrio vertical verificado: ΣFy ≈ 0")
                else:
                    st.error(f"Equilibrio NO verificado: error = {eq_check['error']:.6f} kN")

# --- Footer ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 10px;'>
    ANALISIS ESTRUCTURAL ACADÉMICO | Desarrollado con Streamlit y PyNite<br>
    <small>Herramienta educativa para análisis de estructuras isostáticas e hiperestáticas</small>
    </div>
    """,
    unsafe_allow_html=True
)
