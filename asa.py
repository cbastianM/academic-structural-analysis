import streamlit as st
import matplotlib.pyplot as plt
import json
import io
import numpy as np
from Pynite import FEModel3D
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="ACADEMIC STRUCTURAL ANALYSIS",
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# 🎨 Estilos CSS personalizados
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tema principal */
    :root {
        --primary-color: #1e3a5f;
        --secondary-color: #3498db;
        --accent-color: #e74c3c;
        --success-color: #27ae60;
        --bg-light: #f8f9fa;
        --text-color: #2c3e50;
    }

    /* Encabezados */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
    }

    /* Contenedores de métricas */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Botones primarios */
    .stButton > button[kind="primary"] {
        background-color: var(--secondary-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        background-color: #e8e8e8;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d0d0d0;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-color) !important;
        color: white !important;
    }

    /* Success/Info boxes */
    .success-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }

    /* Card styling */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 🧠 Estado de sesión
# ──────────────────────────────────────────────────────────────
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

if "history" not in st.session_state:
    st.session_state.history = []

if "history_index" not in st.session_state:
    st.session_state.history_index = -1

data = st.session_state.beam_data["beam"]
materials = st.session_state.beam_data["materials"]

# ──────────────────────────────────────────────────────────────
# 📁 Funciones de historial (Undo/Redo)
# ──────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────
# 🔧 Funciones utilitarias
# ──────────────────────────────────────────────────────────────
def auto_generate_elements():
    data["elements"] = []
    data["nodes"].sort(key=lambda n: n["x"])
    for i, n in enumerate(data["nodes"]):
        n["id"] = i + 1

    for i in range(len(data["nodes"]) - 1):
        n1, n2 = data["nodes"][i]["id"], data["nodes"][i + 1]["id"]
        data["elements"].append({
            "id": i + 1,
            "start_node": n1,
            "end_node": n2,
            "E": materials["E_default"],
            "A": materials["A_default"],
            "I": materials["I_default"]
        })

def add_node(x):
    """Añade un nodo y guarda en historial"""
    new_id = len(data["nodes"]) + 1
    data["nodes"].append({"id": new_id, "x": float(x)})
    auto_generate_elements()
    st.session_state.solved_model = None
    save_to_history()

def remove_node(node_id):
    """Elimina un nodo y guarda en historial"""
    data["nodes"] = [nd for nd in data["nodes"] if nd["id"] != node_id]
    data["supports"] = [s for s in data["supports"] if s["node_id"] != node_id]
    data["loads"]["point"] = [l for l in data["loads"]["point"] if l["node_id"] != node_id]
    data["loads"]["distributed"] = [
        l for l in data["loads"]["distributed"]
        if l["start_node"] != node_id and l["end_node"] != node_id
    ]
    auto_generate_elements()
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
    """Añade una carga puntual"""
    data["loads"]["point"].append({"node_id": node_id, "fx": 0.0, "fy": float(fy), "mz": 0.0})
    st.session_state.solved_model = None
    save_to_history()

def add_distributed_load(start_node, end_node, w_start, w_end):
    """Añade una carga distribuida"""
    data["loads"]["distributed"].append({
        "start_node": start_node,
        "end_node": end_node,
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

# ──────────────────────────────────────────────────────────────
# Constructor del modelo PyNite
# ──────────────────────────────────────────────────────────────
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
            node_name = self._node_name(pl["node_id"])
            if pl.get("fx", 0):
                self.model.add_node_load(node_name, "FX", float(pl["fx"]))
            if pl.get("fy", 0):
                self.model.add_node_load(node_name, "FY", float(pl["fy"]))
            if pl.get("mz", 0):
                self.model.add_node_load(node_name, "MZ", float(pl["mz"]))

        for dl in self.data["loads"]["distributed"]:
            self._add_distributed_load(dl)

    def _add_distributed_load(self, dl):
        n_start = self._node_by_id(dl["start_node"])
        n_end = self._node_by_id(dl["end_node"])
        if not n_start or not n_end:
            return

        x_start = float(n_start["x"])
        x_end = float(n_end["x"])
        if x_start == x_end:
            return
        x_min, x_max = sorted([x_start, x_end])

        def q_at(x):
            ratio = (x - x_start) / (x_end - x_start)
            return float(dl["w_start"]) + ratio * (float(dl["w_end"]) - float(dl["w_start"]))

        for e in self.data["elements"]:
            n1 = self._node_by_id(e["start_node"])
            n2 = self._node_by_id(e["end_node"])
            if not n1 or not n2:
                continue
            ex1, ex2 = sorted([float(n1["x"]), float(n2["x"])])
            if ex1 >= x_min and ex2 <= x_max:
                member_name = self.member_names.get(e["id"])
                if member_name:
                    self.model.add_member_dist_load(member_name, "FY", q_at(ex1), q_at(ex2), 0, ex2 - ex1)

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
        return self._plot_result(
            "Diagrama de Momento Flector",
            "M (kN*m)",
            lambda member, x: float(member.moment("Mz", x)),
            color="#c0392b",
        )

    def show_displacement(self):
        return self._plot_result(
            "Diagrama de Deflexion",
            "dy (m)",
            lambda member, x: float(member.deflection("dy", x)),
            color="#27ae60",
        )

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
                n1 = self._node_by_id(load["start_node"])
                n2 = self._node_by_id(load["end_node"])
                if not n1 or not n2:
                    continue
                x1, x2 = sorted([float(n1["x"]), float(n2["x"])])
                avg_load = (float(load["w_start"]) + float(load["w_end"])) / 2
                sign = -1 if avg_load < 0 else 1
                start_y = -sign * y_scale * 0.58
                end_y = -sign * y_scale * 0.04
                for x in np.linspace(x1, x2, 8):
                    ax.arrow(x, start_y, 0, end_y - start_y,
                             head_width=y_scale * 0.085, head_length=y_scale * 0.085,
                             length_includes_head=True, color="#e67e22", linewidth=1.4)
                ax.plot([x1, x2], [start_y, start_y], color="#e67e22", linewidth=1.5)
                ax.text((x1 + x2) / 2, start_y - sign * y_scale * 0.18,
                        f"{load['w_start']:.2f} a {load['w_end']:.2f} kN/m",
                        ha="center", va="center", fontsize=9, color="#d35400", fontweight="bold")

        ax.set_xlim(min(xs) - y_scale * 1.2, max(xs) + y_scale * 1.2)
        ax.set_ylim(-y_scale * 1.25, y_scale * 1.25)
        ax.set_aspect("auto")
        ax.set_xlabel("X (m)")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, linestyle="--")


def build_pynite_model(data):
    return PyNiteBeamModel(data)

# ──────────────────────────────────────────────────────────────
# 📊 Funciones de visualización mejoradas
# ──────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────
# 📋 Ejemplos predefinidos
# ──────────────────────────────────────────────────────────────
EXAMPLE_PROJECTS = {
    "Viga simplemente apoyada": {
        "nodes": [
            {"id": 1, "x": 0.0},
            {"id": 2, "x": 3.0},
            {"id": 3, "x": 6.0}
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
            {"id": 2, "x": 2.5},
            {"id": 3, "x": 5.0}
        ],
        "supports": [
            {"node_id": 1, "type": "Articulado"},
            {"node_id": 3, "type": "Rodillo"}
        ],
        "loads": {
            "point": [],
            "distributed": [{"start_node": 1, "end_node": 3, "w_start": -5.0, "w_end": -5.0, "direction": "y"}]
        }
    },
    "Pórtico simple": {
        "nodes": [
            {"id": 1, "x": 0.0},
            {"id": 2, "x": 5.0},
            {"id": 3, "x": 5.0}
        ],
        "supports": [
            {"node_id": 1, "type": "Empotrado"},
            {"node_id": 3, "type": "Empotrado"}
        ],
        "loads": {
            "point": [
                {"node_id": 2, "fx": 0.0, "fy": -20.0, "mz": 0.0}
            ],
            "distributed": []
        }
    }
}

def load_example(example_name):
    """Carga un ejemplo predefinido"""
    if example_name in EXAMPLE_PROJECTS:
        example_data = json.loads(json.dumps(EXAMPLE_PROJECTS[example_name]))
        nodes = sorted(example_data["nodes"], key=lambda n: n["x"])
        for i, node in enumerate(nodes):
            node["id"] = i + 1

        elements = []
        for i in range(len(nodes) - 1):
            elements.append({
                "id": i + 1,
                "start_node": nodes[i]["id"],
                "end_node": nodes[i + 1]["id"],
                "E": materials["E_default"],
                "A": materials["A_default"],
                "I": materials["I_default"]
            })

        st.session_state.beam_data["beam"] = {
            "nodes": nodes,
            "elements": elements,
            "supports": example_data["supports"],
            "loads": example_data["loads"]
        }
        st.session_state.solved_model = None
        save_to_history()
        st.success(f"✅ Ejemplo '{example_name}' cargado")

# ════════════════════════════════════════════════════════════════
# 🛠️ BARRA LATERAL
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧰 Panel de Herramientas")

    # ─── Navegación entre secciones ───
    sections = {
        "📍 Nodos": "nodes",
        "🧱 Apoyos": "supports",
        "⬇️ Cargas": "loads",
        "⚙️ Materiales": "materials",
        "📂 Proyecto": "project"
    }

    selected_section = st.radio(
        "Sección",
        list(sections.keys()),
        label_visibility="collapsed"
    )

    st.divider()

    # ────────────────────────────────────────────────────────────
    # 📍 SECCIÓN: NODOS
    # ────────────────────────────────────────────────────────────
    if selected_section == "📍 Nodos":
        st.markdown("### 📍 Gestión de Nodos")

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
                    status.append("🔗 Apoyo")
                if has_load:
                    status.append("⚡ Carga")
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
            if nodes_to_delete and st.button("🗑️ Eliminar seleccionados", type="primary"):
                for nid in nodes_to_delete:
                    remove_node(nid)
                st.rerun()

    # ────────────────────────────────────────────────────────────
    # 🧱 SECCIÓN: APOYOS
    # ────────────────────────────────────────────────────────────
    elif selected_section == "🧱 Apoyos":
        st.markdown("### 🧱 Asignar Apoyos")

        node_ids = [n["id"] for n in sorted(data["nodes"], key=lambda x: x["id"])]

        if not node_ids:
            st.warning("⚠️ Añada nodos primero en la sección Nodos.")
        else:
            col1, col2 = st.columns(2)
            s_node = col1.selectbox("Nodo", node_ids, format_func=lambda x: f"N{x}")
            s_type = col2.selectbox("Tipo", ["Empotrado", "Articulado", "Rodillo"])

            # Iconos para cada tipo
            type_icons = {
                "Empotrado": "🔒",
                "Articulado": "🔺",
                "Rodillo": "⚙️"
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
            - 🔒 Empotrado: Fija traslación y rotación
            - 🔺 Articulado: Fija ambas translaciones
            - ⚙️ Rodillo: Fija una traslación (vertical)
            """)

            # Eliminar apoyo
            support_to_delete = st.selectbox(
                "Eliminar apoyo",
                options=[s["node_id"] for s in data["supports"]],
                format_func=lambda x: f"N{x}",
                key="del_support_select"
            )
            if st.button("🗑️ Eliminar apoyo", type="primary"):
                remove_support(support_to_delete)
                st.rerun()

    # ────────────────────────────────────────────────────────────
    # ⬇️ SECCIÓN: CARGAS
    # ────────────────────────────────────────────────────────────
    elif selected_section == "⬇️ Cargas":
        st.markdown("### ⬇️ Aplicar Cargas")

        node_ids = [n["id"] for n in sorted(data["nodes"], key=lambda x: x["id"])]

        if not node_ids:
            st.warning("⚠️ Añada nodos primero.")
        else:
            load_type = st.radio(
                "Tipo de carga",
                ["Puntual", "Distribuida"],
                horizontal=True
            )

            if load_type == "Puntual":
                with st.container():
                    col1, col2 = st.columns(2)
                    p_node = col1.selectbox("Nodo", node_ids, format_func=lambda x: f"N{x}")
                    fy = col2.number_input("Fy (kN)", value=-10.0, step=1.0)

                col_info = st.columns([1, 3])
                with col_info[0]:
                    st.markdown("**Signo:**")
                with col_info[1]:
                    st.markdown("⬆️ Positivo = Hacia arriba<br>⬇️ Negativo = Hacia abajo", unsafe_allow_html=True)

                if st.button("⚡ Aplicar Carga Puntual", use_container_width=True):
                    add_point_load(p_node, fy)
                    st.success(f"Carga puntual de {fy} kN aplicada en N{p_node}")
                    st.rerun()

            else:  # Distribuida
                if len(node_ids) < 2:
                    st.warning("Se necesitan al menos 2 nodos para carga distribuida.")
                else:
                    col1, col2 = st.columns(2)
                    n1 = col1.selectbox("Nodo Inicio", node_ids, format_func=lambda x: f"N{x}")
                    n2 = col2.selectbox("Nodo Fin", node_ids, format_func=lambda x: f"N{x}")

                    col3, col4 = st.columns(2)
                    w1 = col3.number_input("q inicio (kN/m)", value=-5.0, step=0.5)
                    w2 = col4.number_input("q fin (kN/m)", value=-5.0, step=0.5)

                    if st.button("📊 Aplicar Carga Distribuida", use_container_width=True):
                        if n1 != n2:
                            start, end = (n1, n2) if n1 < n2 else (n2, n1)
                            add_distributed_load(start, end, w1, w2)
                            st.success(f"Carga distribuida de {w1} a {w2} kN/m aplicada")
                            st.rerun()
                        else:
                            st.error("Seleccione nodos diferentes")

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
                col1.markdown(f"• **N{l['node_id']}**: {l['fy']} kN ⬇️" if l['fy'] < 0 else f"• **N{l['node_id']}**: {l['fy']} kN ⬆️")
                if col2.button("🗑️", key=f"del_pt_{i}"):
                    remove_point_load(i)
                    st.rerun()

        if total_dist > 0:
            st.markdown("**Cargas Distribuidas:**")
            for i, l in enumerate(data["loads"]["distributed"]):
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"• **N{l['start_node']} → N{l['end_node']}**: {l['w_start']} a {l['w_end']} kN/m")
                if col2.button("🗑️", key=f"del_dist_{i}"):
                    remove_distributed_load(i)
                    st.rerun()

        if total_point == 0 and total_dist == 0:
            st.info("No hay cargas definidas.")

    # ────────────────────────────────────────────────────────────
    # ⚙️ SECCIÓN: MATERIALES
    # ────────────────────────────────────────────────────────────
    elif selected_section == "⚙️ Materiales":
        st.markdown("### ⚙️ Propiedades de Materiales")

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
            "Inercia I (m⁴)",
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

    # ────────────────────────────────────────────────────────────
    # 📂 SECCIÓN: PROYECTO
    # ────────────────────────────────────────────────────────────
    elif selected_section == "📂 Proyecto":
        st.markdown("### 📂 Gestión de Proyecto")

        # Nombre del proyecto
        project_name = st.text_input(
            "Nombre del proyecto",
            value=st.session_state.beam_data.get("name", "Nuevo Proyecto")
        )
        st.session_state.beam_data["name"] = project_name

        st.divider()

        # Acciones rápidas
        st.markdown("**Acciones rápidas:**")

        col_undo, col_redo = st.columns(2)
        with col_undo:
            can_undo = st.session_state.history_index > 0
            st.button("↩️ Deshacer", disabled=not can_undo, use_container_width=True, on_click=undo)
        with col_redo:
            can_redo = st.session_state.history_index < len(st.session_state.history) - 1
            st.button("↪️ Rehacer", disabled=not can_redo, use_container_width=True, on_click=redo)

        if st.button("🧨 Borrar Todo", type="primary", use_container_width=True):
            st.session_state.beam_data["beam"] = {
                "nodes": [],
                "elements": [],
                "supports": [],
                "loads": {"point": [], "distributed": []}
            }
            st.session_state.solved_model = None
            save_to_history()
            st.rerun()

        st.divider()

        # Guardar/Cargar
        st.markdown("**Guardar/Cargar:**")

        json_str = json.dumps(st.session_state.beam_data, indent=2)
        st.download_button(
            "💾 Descargar Modelo (JSON)",
            json_str,
            file_name=f"modelo_{project_name.replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

        uploaded_file = st.file_uploader(
            "📤 Cargar Modelo (JSON)",
            type=["json"],
            help="Seleccione un archivo JSON previamente guardado"
        )

        if uploaded_file:
            try:
                loaded_data = json.load(uploaded_file)
                if "beam" in loaded_data and "nodes" in loaded_data["beam"]:
                    if st.button("✅ Confirmar carga"):
                        st.session_state.beam_data = loaded_data
                        st.session_state.solved_model = None
                        save_to_history()
                        st.success("Modelo cargado correctamente")
                        st.rerun()
            except Exception as e:
                st.error(f"Error al cargar: {e}")

        st.divider()

        # Ejemplos
        st.markdown("**📚 Ejemplos Predefinidos:**")

        for name in EXAMPLE_PROJECTS.keys():
            if st.button(f"📐 {name}", use_container_width=True):
                load_example(name)
                st.rerun()

# ════════════════════════════════════════════════════════════════
# 🖥️ ÁREA PRINCIPAL
# ════════════════════════════════════════════════════════════════
st.markdown("# ACADEMIC STRUCTURAL ANALYSIS")
st.markdown(f"**Proyecto:** {st.session_state.beam_data.get('name', 'Sin nombre')} | **Versión:** {st.session_state.beam_data['version']}")

# ─── Barra de acciones superiores ───
top_col1, top_col2 = st.columns([1, 4])

with top_col1:
    solve_btn = st.button("▶️ Resolver Modelo", type="primary", use_container_width=True)

# ─── Resolver modelo ───
if solve_btn:
    if not data["nodes"]:
        st.error("⚠️ Defina al menos un nodo para resolver.")
    elif not data["supports"]:
        st.error("⚠️ Defina al menos un apoyo para resolver.")
    elif len(data["nodes"]) < 2:
        st.error("⚠️ Se necesitan al menos 2 nodos para formar un elemento.")
    else:
        with st.spinner("🔄 Calculando... por favor espere"):
            try:
                ss = build_pynite_model(data)
                ss.solve()
                st.session_state.solved_model = ss

                # Mostrar métricas de resultados
                st.success("✅ Modelo resuelto correctamente")

            except Exception as e:
                st.error(f"❌ Error en el cálculo: {str(e)}")
                import traceback
                with st.expander("Detalles del error"):
                    st.code(traceback.format_exc())
                st.session_state.solved_model = None

st.divider()

# ─── Pestañas de visualización ───
tab_struct, tab_results, tab_data = st.tabs([
    "🏗️ Estructura",
    "📈 Diagramas",
    "📋 Datos Numéricos"
])

# ─── 1. Vista de Estructura ───
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
                "📥 Descargar gráfico como PNG",
                buf,
                file_name="estructura.png",
                mime="image/png",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error al dibujar la estructura: {e}")
    else:
        st.info("👈 Utilice el panel lateral para añadir nodos, apoyos y cargas.")

# ─── 2. Diagramas de Resultados ───
with tab_results:
    ss = st.session_state.solved_model

    if not ss:
        st.info("⬆️ Presione **Resolver Modelo** para generar los diagramas de resultados.")
    else:
        # Selector de diagrama
        diagram_type = st.radio(
            "Seleccione diagrama:",
            ["Reacciones", "Fuerza Cortante (V)", "Momento Flector (M)", "Deflexión"],
            horizontal=True
        )

        if diagram_type == "Reacciones":
            plot_and_display(ss, ss.show_reaction_force, "Diagrama de Reacciones")
        elif diagram_type == "Fuerza Cortante (V)":
            plot_and_display(ss, ss.show_shear_force, "Diagrama de Fuerza Cortante")
        elif diagram_type == "Momento Flector (M)":
            plot_and_display(ss, ss.show_bending_moment, "Diagrama de Momento Flector")
        elif diagram_type == "Deflexión":
            try:
                plot_and_display(ss, ss.show_displacement, "Diagrama de Deflexión")
            except:
                st.warning("No se pudo generar el diagrama de deflexión")

        # Descargar diagrama
        if st.button("📥 Descargar diagrama", use_container_width=True):
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

# ─── 3. Datos Numéricos ───
with tab_data:
    ss = st.session_state.solved_model

    if not ss:
        st.info("⬆️ Resuelva el modelo primero para ver los datos numéricos.")
    else:
        # Obtener datos del modelo resuelto
        st.markdown("### 📋 Resultados Detallados")

        # Información del modelo
        with st.expander("📐 Información del Modelo", expanded=True):
            info_col1, info_col2, info_col3 = st.columns(3)
            info_col1.metric("Nodos", len(data["nodes"]))
            info_col2.metric("Elementos", len(data["elements"]))
            info_col3.metric("Apoyos", len(data["supports"]))

            # Propiedades de material
            st.markdown("**Propiedades de Material:**")
            mat_col1, mat_col2, mat_col3 = st.columns(3)
            mat_col1.metric("E", f"{materials['E_default']/1000:.0f} MN/m²")
            mat_col2.metric("A", f"{materials['A_default']*10000:.2f} cm²")
            mat_col3.metric("I", f"{materials['I_default']*1e8:.2f} cm⁴")

        # Tabla de reacciones
        st.markdown("### 🔄 Reacciones en Apoyos")
        try:
            reactions = ss.reaction_force
            if reactions:
                reaction_data = []
                for node_id, values in sorted(reactions.items()):
                    if isinstance(values, dict):
                        reaction_data.append({
                            "Nodo": f"N{node_id}",
                            "Fx (kN)": round(values.get('Fx', 0), 4),
                            "Fy (kN)": round(values.get('Fy', 0), 4),
                            "Mz (kN·m)": round(values.get('M', 0), 4)
                        })
                df_reactions = pd.DataFrame(reaction_data)
                st.dataframe(df_reactions, use_container_width=True, hide_index=True)
            else:
                st.info("No se encontraron reacciones")
        except Exception as e:
            st.error(f"Error al obtener reacciones: {e}")

        # Tabla de nodos con desplazamiento
        st.markdown("### ↕️ Desplazamientos en Nodos")
        try:
            node_deflections = []
            for node in data["nodes"]:
                try:
                    deflection = ss.vertex_id_flection(node["id"])
                    node_deflections.append({
                        "Nodo": f"N{node['id']}",
                        "X (m)": node["x"],
                        "Deflexión (m)": round(deflection, 6)
                    })
                except:
                    node_deflections.append({
                        "Nodo": f"N{node['id']}",
                        "X (m)": node["x"],
                        "Deflexión (m)": 0.0
                    })

            df_deflections = pd.DataFrame(node_deflections)
            st.dataframe(df_deflections, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"No se pudieron obtener todos los desplazamientos: {e}")

        # Exportar a CSV
        st.markdown("### 📤 Exportar Datos")
        if st.button("📊 Exportar resultados a CSV", use_container_width=True):
            try:
                # Crear dataframe combinado
                all_data = []

                # Agregar reacciones
                for node_id, values in sorted(reactions.items()) if 'reactions' in dir() else []:
                    all_data.append({
                        "Tipo": "Reacción",
                        "Nodo": f"N{node_id}",
                        "Fx_kN": values.get('Fx', 0),
                        "Fy_kN": values.get('Fy', 0),
                        "Mz_kNm": values.get('M', 0)
                    })

                # Agregar desplazamientos
                for nd in node_deflections:
                    all_data.append({
                        "Tipo": "Desplazamiento",
                        "Nodo": nd["Nodo"],
                        "Fx_kN": "",
                        "Fy_kN": nd["Deflexión (m)"],
                        "Mz_kNm": ""
                    })

                if all_data:
                    df_export = pd.DataFrame(all_data)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        "Descargar CSV",
                        csv,
                        file_name=f"resultados_{project_name.replace(' ', '_')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No hay datos para exportar")
            except Exception as e:
                st.error(f"Error al exportar: {e}")

# ─── Footer ───
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 10px;'>
    ACADEMIC STRUCTURAL ANALYSIS | Desarrollado con Streamlit y PyNite<br>
    <small>Herramienta educativa para análisis de estructuras isostáticas e hiperestáticas</small>
    </div>
    """,
    unsafe_allow_html=True
)
