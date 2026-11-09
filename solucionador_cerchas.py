from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np


def _number(value, label, default=None):
    if value is None and default is not None:
        return default
    try:
        if isinstance(value, bool):
            raise ValueError
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} debe ser un número.") from None
    if not isfinite(value):
        raise ValueError(f"{label} debe ser un número finito.")
    return value


def _identifier(value, label):
    number = _number(value, label)
    if number < 1 or not number.is_integer():
        raise ValueError(f"{label} debe ser un entero positivo.")
    return int(number)


def _restriction(value, label):
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{label} debe ser una casilla activada o desactivada.")
    return value


def normalize_truss(truss):
    """Valida los datos editables, incluso cuando el modelo está incompleto."""
    if not isinstance(truss, dict):
        raise ValueError("Se esperaba un modelo de cercha.")
    nodes = truss.get("nodes")
    elements = truss.get("elements")
    if not isinstance(nodes, list) or not isinstance(elements, list):
        raise ValueError("El modelo debe contener una lista de nodos y otra de barras.")

    node_map = {}
    for row, node in enumerate(nodes, start=1):
        if not isinstance(node, dict):
            raise ValueError(f"La fila {row} de nodos debe contener los datos de un nodo.")
        node_id = _identifier(node.get("id"), f"ID del nodo en la fila {row}")
        if node_id in node_map:
            raise ValueError(f"El nodo {node_id} está repetido.")
        label = f"Nodo {node_id}"
        node_map[node_id] = {
            "id": node_id,
            "x": _number(node.get("x"), f"{label}, X"),
            "y": _number(node.get("y"), f"{label}, Y"),
            "fix_x": _restriction(node.get("fix_x"), f"{label}, restringir X"),
            "fix_y": _restriction(node.get("fix_y"), f"{label}, restringir Y"),
            "fx": _number(node.get("fx"), f"{label}, Fx", default=0.0),
            "fy": _number(node.get("fy"), f"{label}, Fy", default=0.0),
        }

    bar_map = {}
    for row, element in enumerate(elements, start=1):
        if not isinstance(element, dict):
            raise ValueError(f"La fila {row} de barras debe contener los datos de una barra.")
        bar_id = _identifier(element.get("id"), f"ID de la barra en la fila {row}")
        if bar_id in bar_map:
            raise ValueError(f"La barra {bar_id} está repetida.")
        label = f"Barra {bar_id}"
        start = _identifier(element.get("start_node"), f"{label}, nodo inicial")
        end = _identifier(element.get("end_node"), f"{label}, nodo final")
        if start not in node_map or end not in node_map:
            raise ValueError(f"{label}: ambos nodos deben existir en la tabla de nodos.")
        ni, nj = node_map[start], node_map[end]
        length = hypot(nj["x"] - ni["x"], nj["y"] - ni["y"])
        if not isfinite(length) or length == 0:
            raise ValueError(f"{label}: los extremos deben definir una longitud finita y mayor que cero.")

        modulus = _number(element.get("e_gpa"), f"{label}, E")
        area = _number(element.get("area"), f"{label}, A")
        delta_l = _number(element.get("delta_l_mm"), f"{label}, ΔL₀", default=0.0)
        if modulus <= 0 or area <= 0:
            raise ValueError(f"{label}: E y A deben ser mayores que cero.")
        if length + delta_l / 1000 <= 0:
            raise ValueError(f"{label}: L + ΔL₀ debe ser mayor que cero.")

        bar_map[bar_id] = {
            "id": bar_id,
            "start_node": start,
            "end_node": end,
            "e_gpa": modulus,
            "area": area,
            "delta_l_mm": delta_l,
        }

    return {"nodes": list(node_map.values()), "elements": list(bar_map.values())}


@dataclass
class TrussBar:
    id: int
    dofs: list[int]
    length: float
    b: np.ndarray
    stiffness: float
    delta_l: float

    @property
    def matrix(self):
        return self.stiffness * np.outer(self.b, self.b)

    @property
    def equivalent_load(self):
        return self.stiffness * self.delta_l * self.b


class TrussMatrixSolver:
    """Cercha plana articulada, con deformaciones pequeñas y respuesta axial elástica."""

    def __init__(self, truss):
        truss = normalize_truss(truss)
        if not truss["elements"]:
            raise ValueError("Defina al menos una barra para resolver la cercha.")
        self.nodes = sorted(truss["nodes"], key=lambda node: node["id"])
        self.dofs = {node["id"]: (2 * i, 2 * i + 1) for i, node in enumerate(self.nodes)}
        self.labels = [f"N{node['id']} {axis}" for node in self.nodes for axis in ("X", "Y")]
        self.loads = np.array([[node["fx"], node["fy"]] for node in self.nodes]).ravel()
        fixed = np.array([[node["fix_x"], node["fix_y"]] for node in self.nodes]).ravel()
        self.free = np.flatnonzero(~fixed)
        self.fixed = np.flatnonzero(fixed)

        self.K = np.zeros((len(fixed), len(fixed)))
        self.initial_loads = np.zeros(len(fixed))
        self.bars = []
        node_map = {node["id"]: node for node in self.nodes}
        for element in sorted(truss["elements"], key=lambda bar: bar["id"]):
            i, j = element["start_node"], element["end_node"]
            dx = node_map[j]["x"] - node_map[i]["x"]
            dy = node_map[j]["y"] - node_map[i]["y"]
            length = hypot(dx, dy)
            c, s = dx / length, dy / length
            bar = TrussBar(
                element["id"], [*self.dofs[i], *self.dofs[j]], length,
                np.array([-c, -s, c, s]),
                element["e_gpa"] * 1e6 * element["area"] / length,
                element["delta_l_mm"] / 1000,
            )
            if not isfinite(bar.stiffness) or bar.stiffness <= 0:
                raise ValueError(f"Barra {bar.id}: el valor de EA/L está fuera del rango numérico.")
            self.K[np.ix_(bar.dofs, bar.dofs)] += bar.matrix
            self.initial_loads[bar.dofs] += bar.equivalent_load
            self.bars.append(bar)

        self.rhs = self.loads + self.initial_loads
        if not np.isfinite(self.K).all() or not np.isfinite(self.rhs).all():
            raise ValueError("La rigidez o las cargas equivalentes están fuera del rango numérico.")
        self.Kff = self.K[np.ix_(self.free, self.free)]
        try:
            if self.free.size and np.linalg.matrix_rank(self.Kff, hermitian=True) < self.free.size:
                raise np.linalg.LinAlgError
            self.u = self._solve(self.rhs)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Cercha inestable o con rigidez casi singular. Revise los apoyos, "
                "las barras y los nodos sin conexión."
            ) from None

        self.reactions = self.K @ self.u - self.rhs
        self.axial = np.array([
            bar.stiffness * (bar.b @ self.u[bar.dofs] - bar.delta_l) for bar in self.bars
        ])
        if not all(np.isfinite(values).all() for values in (self.u, self.reactions, self.axial)):
            raise ValueError("Los resultados están fuera del rango numérico. Revise los datos del modelo.")

    def _solve(self, loads):
        u = np.zeros_like(loads)
        if self.free.size:
            u[self.free] = np.linalg.solve(self.Kff, loads[self.free])
        return u

    def virtual_work(self, dof):
        """Coeficientes axiales virtuales y aportes elástico y de fabricación en m."""
        unit_load = np.zeros_like(self.loads)
        unit_load[dof] = 1
        virtual_u = self._solve(unit_load)
        elastic_extensions = self.axial / np.array([bar.stiffness for bar in self.bars])
        virtual_extensions = np.array([bar.b @ virtual_u[bar.dofs] for bar in self.bars])
        virtual_axial = virtual_extensions * np.array([bar.stiffness for bar in self.bars])
        initial_extensions = np.array([bar.delta_l for bar in self.bars])
        return virtual_axial, virtual_axial * elastic_extensions, virtual_axial * initial_extensions
