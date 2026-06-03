import numpy as np


class BeamMatrixSolver:
    """
    Solucionador matricial de rigidez para vigas 2D.
    Implementa el metodo completo: ensamblaje, particion, solucion,
    recuperacion de fuerzas en extremos, diagramas V(x)/M(x),
    reacciones y verificaciones.
    """

    def __init__(self, nodes, elements, supports, point_loads, distributed_loads,
                 use_real_ei=False):
        self.all_nodes = sorted(nodes, key=lambda n: n["x"])
        self.node_x = {n["id"]: n["x"] for n in nodes}
        self.elements = elements
        self.supports = supports
        self.point_loads = point_loads
        self.distributed_loads = distributed_loads
        self.use_real_ei = use_real_ei

        connected = set()
        for elem in elements:
            connected.add(elem["start_node"])
            connected.add(elem["end_node"])
        self.nodes = [n for n in self.all_nodes if n["id"] in connected]
        self.connected_ids = connected

        self.n = len(self.nodes)
        self.N_DOF = 2 * self.n

        self.dof_map = {}
        self.constrained_dofs = set()
        self.free_idx = []
        self.cons_idx = []

        self.K = None
        self.P = None
        self.Qf = None
        self.F = None
        self.elem_Qf_local = {}
        self.elem_point_loads = {}
        self.u_full = None
        self.R = None
        self.elem_end_forces = {}

        self._build_dof_map()
        self._build_global_matrices()
        self.stability = {}
        self.solved = False

    def _build_dof_map(self):
        for i, node in enumerate(self.nodes):
            v_dof = 2 * i + 1
            r_dof = 2 * i + 2
            self.dof_map[node["id"]] = (v_dof, r_dof)

            support = next((s for s in self.supports
                           if s["node_id"] == node["id"]), None)
            if support:
                if support["type"] == "Empotrado":
                    self.constrained_dofs.add(v_dof)
                    self.constrained_dofs.add(r_dof)
                elif support["type"] in ("Articulado", "Rodillo"):
                    self.constrained_dofs.add(v_dof)

        for i in range(self.N_DOF):
            if (i + 1) in self.constrained_dofs:
                self.cons_idx.append(i)
            else:
                self.free_idx.append(i)

    def _build_ke(self, L, EI):
        k = np.zeros((4, 4))
        k[0, 0] = 12 * EI / L ** 3
        k[0, 1] = 6 * EI / L ** 2
        k[0, 2] = -12 * EI / L ** 3
        k[0, 3] = 6 * EI / L ** 2
        k[1, 0] = 6 * EI / L ** 2
        k[1, 1] = 4 * EI / L
        k[1, 2] = -6 * EI / L ** 2
        k[1, 3] = 2 * EI / L
        k[2, 0] = -12 * EI / L ** 3
        k[2, 1] = -6 * EI / L ** 2
        k[2, 2] = 12 * EI / L ** 3
        k[2, 3] = -6 * EI / L ** 2
        k[3, 0] = 6 * EI / L ** 2
        k[3, 1] = 2 * EI / L
        k[3, 2] = -6 * EI / L ** 2
        k[3, 3] = 4 * EI / L
        return k

    def _compute_fe4_uniform(self, q, L):
        fe4 = np.zeros(4)
        fe4[0] = q * L / 2
        fe4[1] = q * L ** 2 / 12
        fe4[2] = q * L / 2
        fe4[3] = -q * L ** 2 / 12
        return fe4

    def _compute_fe4_trapezoidal(self, w1, w2, L):
        fe4 = np.zeros(4)
        dw = w2 - w1
        fe4[0] = w1 * L / 2 + 3 * dw * L / 20
        fe4[1] = w1 * L ** 2 / 12 + dw * L ** 2 / 30
        fe4[2] = w1 * L / 2 + 7 * dw * L / 20
        fe4[3] = -w1 * L ** 2 / 12 - dw * L ** 2 / 20
        return fe4

    def _compute_fe4_point(self, P_val, d, L):
        a_dist = d
        b_dist = L - d
        fe4 = np.zeros(4)
        fe4[0] = P_val * b_dist ** 2 * (2 * a_dist + L) / L ** 3
        fe4[1] = P_val * a_dist * b_dist ** 2 / L ** 2
        fe4[2] = P_val * a_dist ** 2 * (2 * b_dist + L) / L ** 3
        fe4[3] = -P_val * a_dist ** 2 * b_dist / L ** 2
        return fe4

    def _find_element_at(self, x):
        for elem in self.elements:
            sn = elem["start_node"]
            en = elem["end_node"]
            if sn not in self.node_x or en not in self.node_x:
                continue
            x1 = self.node_x[sn]
            x2 = self.node_x[en]
            if (min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10):
                L = abs(x2 - x1)
                d = abs(x - min(x1, x2))
                return elem, L, d
        return None, 0, 0

    def _build_global_matrices(self):
        K = np.zeros((self.N_DOF, self.N_DOF))
        P = np.zeros(self.N_DOF)
        Qf = np.zeros(self.N_DOF)
        self.elem_Qf_local = {}
        self.elem_point_loads = {}

        for pl in self.point_loads:
            nid = pl.get("node_id")
            eid = pl.get("element_id")
            x_pos = pl.get("x")
            fy = float(pl.get("fy", 0))
            mz = float(pl.get("mz", 0))

            if eid is not None:
                elem = next((e for e in self.elements if e["id"] == eid), None)
                if elem is None or elem["start_node"] not in self.dof_map:
                    continue
                sn = elem["start_node"]
                en = elem["end_node"]
                L = abs(self.node_x[en] - self.node_x[sn])
                if L < 1e-12:
                    continue
                if x_pos is not None:
                    x_start = min(self.node_x[sn], self.node_x[en])
                    d = abs(float(x_pos) - x_start)
                else:
                    d = float(pl.get("d", L / 2))
                if abs(fy) > 1e-15:
                    fe4 = self._compute_fe4_point(fy, d, L)
                    qf_elem = self.elem_Qf_local.setdefault(eid, np.zeros(4))
                    qf_elem += fe4
                    pt_list = self.elem_point_loads.setdefault(eid, [])
                    pt_list.append({"d": d, "P": fy})
                if abs(mz) > 1e-15:
                    P[self.dof_map[sn][1] - 1] += mz
            elif nid is not None and nid in self.dof_map:
                v_dof, r_dof = self.dof_map[nid]
                P[v_dof - 1] += fy
                P[r_dof - 1] += mz
            elif nid is not None and nid not in self.dof_map:
                node_x_val = self.node_x.get(nid)
                if node_x_val is not None:
                    found_elem, L, d = self._find_element_at(node_x_val)
                    if found_elem is not None and abs(fy) > 1e-15:
                        eid_found = found_elem["id"]
                        fe4 = self._compute_fe4_point(fy, d, L)
                        qf_elem = self.elem_Qf_local.setdefault(eid_found, np.zeros(4))
                        qf_elem += fe4
                        pt_list = self.elem_point_loads.setdefault(eid_found, [])
                        pt_list.append({"d": d, "P": fy})

        for elem in self.elements:
            sn = elem["start_node"]
            en = elem["end_node"]
            if sn not in self.dof_map or en not in self.dof_map:
                continue

            a_1b, b_1b = self.dof_map[sn]
            c_1b, d_1b = self.dof_map[en]
            dofs_0 = [a_1b - 1, b_1b - 1, c_1b - 1, d_1b - 1]
            L = abs(self.node_x[en] - self.node_x[sn])
            if L < 1e-12:
                continue

            EI_val = (elem.get("E", 200e6) * elem.get("I", 5e-5)
                      if self.use_real_ei else 1.0)

            ke = self._build_ke(L, EI_val)
            for i in range(4):
                for j in range(4):
                    K[dofs_0[i], dofs_0[j]] += ke[i, j]

            qf_e = np.zeros(4)
            for dl in self.distributed_loads:
                if dl.get("element_id") != elem["id"]:
                    continue
                w_start = float(dl.get("w_start", 0))
                w_end = float(dl.get("w_end", 0))
                direction = dl.get("direction", "y")
                if direction != "y":
                    continue
                if abs(w_start - w_end) < 1e-10:
                    fe4 = self._compute_fe4_uniform(w_start, L)
                else:
                    fe4 = self._compute_fe4_trapezoidal(w_start, w_end, L)
                qf_e += fe4

            qf_e += self.elem_Qf_local.get(elem["id"], np.zeros(4))
            self.elem_Qf_local[elem["id"]] = qf_e
            for i in range(4):
                Qf[dofs_0[i]] += qf_e[i]

        self.K = K
        self.P = P
        self.Qf = Qf
        self.F = P + Qf

    def check_stability(self):
        if not self.free_idx:
            self.stability = {
                "status": "error",
                "message": "Todos los GDL estan restringidos. "
                           "No hay incognitas que resolver."
            }
            return self.stability

        K_ff = self.K[np.ix_(self.free_idx, self.free_idx)]
        try:
            rank = np.linalg.matrix_rank(K_ff, tol=1e-8)
        except Exception:
            rank = 0

        deficiency = len(self.free_idx) - rank
        cond_num = None
        if rank == len(self.free_idx):
            try:
                cond_num = np.linalg.cond(K_ff)
            except Exception:
                cond_num = float("inf")

        if rank < len(self.free_idx):
            self.stability = {
                "status": "error",
                "message": (
                    f"Matriz K_ff singular: rango {rank} vs {len(self.free_idx)} GDL libres. "
                    f"Deficiencia: {deficiency}. "
                    f"La estructura es un mecanismo. Verifique los apoyos."
                ),
                "rank": rank,
                "deficiency": deficiency,
            }
        elif cond_num and cond_num > 1e12:
            self.stability = {
                "status": "warning",
                "message": (
                    f"K_ff mal condicionada (cond = {cond_num:.2e}). "
                    f"Resultados pueden ser imprecisos."
                ),
                "rank": rank,
                "condition_number": cond_num,
            }
        else:
            self.stability = {
                "status": "ok",
                "rank": rank,
                "condition_number": cond_num,
            }
        return self.stability

    def solve(self):
        stability = self.check_stability()
        if stability["status"] == "error":
            return False

        K_ff = self.K[np.ix_(self.free_idx, self.free_idx)]
        F_f = self.F[self.free_idx]

        u_f = np.linalg.solve(K_ff, F_f)

        self.u_full = np.zeros(self.N_DOF)
        for i, idx in enumerate(self.free_idx):
            self.u_full[idx] = u_f[i]

        self._compute_reactions()
        self._recover_end_forces()
        self.solved = True
        return True

    def _compute_reactions(self):
        if not self.cons_idx:
            self.R = np.array([])
            return

        K_cf = self.K[np.ix_(self.cons_idx, self.free_idx)]
        F_c = self.F[self.cons_idx]
        u_f = self.u_full[self.free_idx]
        self.R = K_cf @ u_f - F_c

    def _recover_end_forces(self):
        self.elem_end_forces = {}
        for elem in self.elements:
            sn = elem["start_node"]
            en = elem["end_node"]
            if sn not in self.dof_map or en not in self.dof_map:
                continue

            a_1b, b_1b = self.dof_map[sn]
            c_1b, d_1b = self.dof_map[en]
            dofs_0 = [a_1b - 1, b_1b - 1, c_1b - 1, d_1b - 1]
            L = abs(self.node_x[en] - self.node_x[sn])
            if L < 1e-12:
                continue

            EI_val = (elem.get("E", 200e6) * elem.get("I", 5e-5)
                      if self.use_real_ei else 1.0)

            ke = self._build_ke(L, EI_val)
            u_e = self.u_full[dofs_0]
            qf_e = self.elem_Qf_local.get(elem["id"], np.zeros(4))
            Q_e = ke @ u_e - qf_e

            Vi = Q_e[0]
            Mi = -Q_e[1]
            Vj = -Q_e[2]
            Mj = Q_e[3]

            self.elem_end_forces[elem["id"]] = {
                "Vi": Vi, "Mi": Mi, "Vj": Vj, "Mj": Mj,
                "L": L, "x_start": self.node_x[sn],
            }

    def get_node_displacement(self, node_id):
        if self.u_full is None or node_id not in self.dof_map:
            return None, None
        v_dof, r_dof = self.dof_map[node_id]
        v = self.u_full[v_dof - 1]
        r = self.u_full[r_dof - 1]
        return v, r

    def get_reactions(self):
        result = {}
        if self.R is None:
            return result
        for i, idx in enumerate(self.cons_idx):
            gdl = idx + 1
            for node in self.nodes:
                v_dof, r_dof = self.dof_map[node["id"]]
                if gdl == v_dof:
                    if node["id"] not in result:
                        result[node["id"]] = {"Fy": 0.0, "M": 0.0}
                    result[node["id"]]["Fy"] = float(self.R[i])
                elif gdl == r_dof:
                    if node["id"] not in result:
                        result[node["id"]] = {"Fy": 0.0, "M": 0.0}
                    result[node["id"]]["M"] = float(self.R[i])
        return result

    def compute_diagrams(self, n_points=100):
        diagrams = {}
        for elem in self.elements:
            eid = elem["id"]
            ef = self.elem_end_forces.get(eid)
            if ef is None:
                continue

            L = ef["L"]
            x_start = ef["x_start"]
            Vi = ef["Vi"]
            Mi = ef["Mi"]

            w1, w2 = 0.0, 0.0
            for dl in self.distributed_loads:
                if dl.get("element_id") != eid:
                    continue
                if dl.get("direction", "y") != "y":
                    continue
                w1 = float(dl.get("w_start", 0))
                w2 = float(dl.get("w_end", 0))
                break

            pts = sorted([p["d"] for p in self.elem_point_loads.get(eid, [])])
            segments = []
            prev = 0.0
            for d_pt in pts:
                if d_pt > prev + 1e-10:
                    segments.append((prev, d_pt))
                prev = d_pt
            if prev < L - 1e-10:
                segments.append((prev, L))
            if not segments:
                segments = [(0, L)]

            xs_all = []
            Vs_all = []
            Ms_all = []

            for seg_start, seg_end in segments:
                n_seg = max(5, int(n_points * (seg_end - seg_start) / L))
                xs_local = np.linspace(seg_start, seg_end, n_seg)
                xs_global = x_start + xs_local
                Vs = np.zeros(n_seg)
                Ms = np.zeros(n_seg)

                for j, x in enumerate(xs_local):
                    V_base = Vi + w1 * x + (w2 - w1) * x ** 2 / (2 * L)
                    M_base = Mi + Vi * x + w1 * x ** 2 / 2 + (w2 - w1) * x ** 3 / (6 * L)
                    for pt in self.elem_point_loads.get(eid, []):
                        if x >= pt["d"] + 1e-12:
                            V_base += pt["P"]
                            M_base += pt["P"] * (x - pt["d"])
                    Vs[j] = V_base
                    Ms[j] = M_base

                xs_all.extend(xs_global.tolist())
                Vs_all.extend(Vs.tolist())
                Ms_all.extend(Ms.tolist())

            diagrams[eid] = {
                "x_global": np.array(xs_all),
                "x_local": np.array(xs_all) - x_start,
                "V": np.array(Vs_all),
                "M": np.array(Ms_all),
                "Vi": Vi, "Vj": ef["Vj"],
                "Mi": Mi, "Mj": ef["Mj"],
            }
        return diagrams

    def verify_equilibrium(self):
        total_point = sum(float(pl.get("fy", 0)) for pl in self.point_loads)
        total_distributed = 0.0
        for dl in self.distributed_loads:
            if dl.get("direction", "y") != "y":
                continue
            eid = dl.get("element_id")
            elem = next((e for e in self.elements if e["id"] == eid), None)
            if not elem:
                continue
            sn = elem["start_node"]
            en = elem["end_node"]
            if sn not in self.node_x or en not in self.node_x:
                continue
            L = abs(self.node_x[en] - self.node_x[sn])
            w1 = float(dl.get("w_start", 0))
            w2 = float(dl.get("w_end", 0))
            total_distributed += (w1 + w2) / 2 * L

        total_vertical_load = total_point + total_distributed

        total_reaction = 0.0
        reactions = self.get_reactions()
        for nid, rxn in reactions.items():
            total_reaction += rxn.get("Fy", 0.0)

        error = abs(total_vertical_load + total_reaction)
        passed = error < 1e-4 * max(1.0, abs(total_vertical_load))

        return {
            "total_load": total_vertical_load,
            "total_reaction": total_reaction,
            "error": error,
            "passed": passed,
        }

    def get_solution_summary(self):
        displacements = {}
        for node in self.nodes:
            v, r = self.get_node_displacement(node["id"])
            displacements[node["id"]] = {"dy": v, "rz": r}

        reactions = self.get_reactions()
        end_forces = {
            eid: {
                "Vi": ef["Vi"], "Mi": ef["Mi"],
                "Vj": ef["Vj"], "Mj": ef["Mj"],
            }
            for eid, ef in self.elem_end_forces.items()
        }

        eq_check = self.verify_equilibrium()

        return {
            "displacements": displacements,
            "reactions": reactions,
            "end_forces": end_forces,
            "equilibrium": eq_check,
            "stability": self.stability,
            "dof_map": {nid: list(dofs) for nid, dofs in self.dof_map.items()},
            "constrained_dofs": sorted(list(self.constrained_dofs)),
            "free_idx": self.free_idx,
            "cons_idx": self.cons_idx,
            "u_full": self.u_full.tolist() if self.u_full is not None else None,
        }
