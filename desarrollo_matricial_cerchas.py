import numpy as np
import pandas as pd
import streamlit as st


def _tex(value):
    if abs(value) < 1e-12:
        return "0"
    mantissa, _, exponent = f"{value:.8g}".partition("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}" if exponent else mantissa


def _matrix(values):
    rows = [r" & ".join(_tex(value) for value in row) for row in np.atleast_2d(values)]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def _display_order(labels, reference):
    order = reference.get("dof_order")
    if (
        isinstance(order, list)
        and all(isinstance(label, str) for label in order)
        and sorted(order) == sorted(labels)
    ):
        return [labels.index(label) for label in order]
    return list(range(len(labels)))


def _show_triangular_solve(matrix, rhs, headers, numbers, displacement, length_unit):
    with st.expander("Ver la factorización y las sustituciones", expanded=len(rhs) <= 12,
                     key="truss_matrix_solve", on_change="rerun") as details:
        if not details.open:
            return
        try:
            lower = np.linalg.cholesky(matrix)
            intermediate = np.linalg.solve(lower, rhs)
        except np.linalg.LinAlgError:
            st.caption("El sistema se resolvió directamente con la matriz reducida mostrada arriba.")
            return
        st.markdown("La matriz reducida de la cercha estable es simétrica y definida positiva. "
                    "Se aplica la factorización de Cholesky y se resuelven dos sistemas triangulares.")
        st.latex(r"K_{ff}=LL^T,\qquad Lz=b_f,\qquad L^Tu_f=z")
        st.dataframe(pd.DataFrame(lower, index=headers, columns=headers).round(8),
                     alt="Factor triangular inferior L")
        st.markdown("**Sustitución hacia adelante** · se obtiene $z$ de arriba hacia abajo.")
        st.latex(r"z_i=\frac{(b_f)_i-\sum_{j<i} L_{ij}z_j}{L_{ii}}")
        diagonal = lower.diagonal()
        st.dataframe(pd.DataFrame({
            "GDL": numbers,
            "b_f": rhs,
            "Σ de términos anteriores": np.tril(lower, -1) @ intermediate,
            "Lᵢᵢ": diagonal,
            "z": intermediate,
        }).round(9).replace(-0.0, 0.0), hide_index=True, alt="Sustitución hacia adelante para cada grado libre")
        st.markdown("**Sustitución hacia atrás** · se calculan los desplazamientos desde la última fila.")
        st.latex(r"u_i=\frac{z_i-\sum_{j>i} L_{ji}u_j}{L_{ii}}")
        last = numbers[-1]
        st.latex(rf"u_{{{last}}}=\frac{{z_{{{last}}}}}{{L_{{{last},{last}}}}}"
                 rf"=\frac{{{_tex(intermediate[-1])}}}{{{_tex(lower[-1, -1])}}}"
                 rf"={_tex(displacement[-1])}\,\mathrm{{{length_unit}}}")
        st.dataframe(pd.DataFrame({
            "GDL": numbers,
            "z": intermediate,
            "Σ de términos posteriores": np.triu(lower.T, 1) @ displacement,
            "Lᵢᵢ": diagonal,
            f"u ({length_unit})": displacement,
        }).iloc[::-1].round(9).replace(-0.0, 0.0), hide_index=True,
            alt="Sustitución hacia atrás para cada grado libre")


def show_matrix_steps(truss, solution, saved_solution):
    reference = saved_solution.get("matrix", {})
    if not isinstance(reference, dict):
        reference = {}
    if reference.get("units") == "kip_in":
        length_unit, force_unit = "in", "kip"
        metres, kilonewtons = 0.0254, 4.4482216152605
    else:
        length_unit, force_unit = "m", "kN"
        metres, kilonewtons = 1.0, 1.0
    stiffness_factor = metres / kilonewtons
    stiffness_unit = f"{force_unit}/{length_unit}"
    order = _display_order(solution.labels, reference)
    number = {dof: i for i, dof in enumerate(order, start=1)}
    header = {dof: f"GDL {number[dof]}" for dof in order}
    free_set = set(solution.free)
    free = [dof for dof in order if dof in free_set]
    fixed = [dof for dof in order if dof not in free_set]
    bar_ids = [bar.id for bar in solution.bars]
    elements = {element["id"]: element for element in truss["elements"]}
    by_node = {node["id"]: node for node in solution.nodes}
    headers = [header[dof] for dof in order]

    st.caption(f"Longitudes en {length_unit}, fuerzas en {force_unit} y rigidez en {stiffness_unit}. "
               "El sentido positivo es hacia la derecha y hacia arriba.")
    st.subheader("1. Numerar los grados de libertad")
    st.markdown("Cada nodo tiene dos traslaciones. Los apoyos fijan a cero los grados restringidos "
                "y las demás traslaciones se calculan.")
    st.dataframe(pd.DataFrame({
        "GDL": [number[dof] for dof in order],
        "Traslación": [solution.labels[dof] for dof in order],
        "Condición": ["Libre" if dof in free_set else "Restringido" for dof in order],
    }), hide_index=True, alt="Numeración y condición de los grados de libertad")

    st.subheader("2. Calcular la rigidez de cada barra")
    st.latex(r"L_e=\sqrt{(\Delta x)^2+(\Delta y)^2},\qquad c=\Delta x/L_e,\quad s=\Delta y/L_e")
    st.latex(r"\bar k_e=\frac{E_e A_e}{L_e}\begin{bmatrix}1&-1\\-1&1\end{bmatrix},\qquad "
             r"T_e=\begin{bmatrix}c&s&0&0\\0&0&c&s\end{bmatrix},\qquad k_e=T_e^T\bar k_eT_e")
    st.dataframe(pd.DataFrame({
        "Barra": bar_ids,
        "Nodos i → j": [f"{elements[n]['start_node']} → {elements[n]['end_node']}" for n in bar_ids],
        "GDL (iX, iY, jX, jY)": [", ".join(str(number[d]) for d in bar.dofs) for bar in solution.bars],
        f"L ({length_unit})": [bar.length / metres for bar in solution.bars],
        "c": [bar.b[2] for bar in solution.bars],
        "s": [bar.b[3] for bar in solution.bars],
        f"EA/L ({stiffness_unit})": [bar.stiffness * stiffness_factor for bar in solution.bars],
    }).round(8), hide_index=True, alt="Geometría, conectividad y rigidez axial de todas las barras")
    first_misfit = next((i for i, bar in enumerate(solution.bars) if bar.delta_l), 0)
    bar_index = st.selectbox("Barra para desarrollar", list(range(len(bar_ids))), index=first_misfit,
                             format_func=lambda index: f"Barra {bar_ids[index]}")
    bar = solution.bars[bar_index]
    element = elements[bar.id]
    start, end = by_node[element["start_node"]], by_node[element["end_node"]]
    dx, dy = [(end[axis] - start[axis]) / metres for axis in ("x", "y")]
    length = bar.length / metres
    modulus = element["e_gpa"] * 1e6 * metres**2 / kilonewtons
    area = element["area"] / metres**2
    stiffness = bar.stiffness * stiffness_factor
    gdl = ", ".join(str(number[dof]) for dof in bar.dofs)
    st.latex(rf"L_{{{bar.id}}}=\sqrt{{({_tex(dx)})^2+({_tex(dy)})^2}}"
             rf"={_tex(length)}\,\mathrm{{{length_unit}}},\quad"
             rf"\frac{{EA}}{{L}}=\frac{{({_tex(modulus)})({_tex(area)})}}{{{_tex(length)}}}"
             rf"={_tex(stiffness)}\,\mathrm{{{stiffness_unit}}}")
    st.latex(rf"\bar k_{{{bar.id}}}={_matrix(stiffness * np.array([[1, -1], [-1, 1]]))},\qquad "
             rf"k_{{{bar.id}}}={_matrix(bar.matrix * stiffness_factor)}")
    st.caption(f"En la matriz de barra expresada en ejes globales, el orden es {gdl}.")

    st.subheader("3. Convertir el desajuste en cargas nodales")
    st.latex(r"b_e=[-c,-s,c,s],\qquad p_{0,e}=\frac{E_e A_e}{L_e}\Delta L_{0,e} b_e^T")
    delta_l = bar.delta_l / metres
    st.latex(rf"p_{{0,{bar.id}}}=({_tex(stiffness)})({_tex(delta_l)}){_matrix(bar.b[:, None])}"
             rf"={_matrix(bar.equivalent_load[:, None] / kilonewtons)}\,\mathrm{{{force_unit}}}")
    st.markdown("Se suman los aportes por grado de libertad para obtener $P_0$. "
                "En $Ku=P+P_0+R$, $P$ es la carga aplicada y $R$ son las reacciones. "
                "Si se escribe $P+R=Ku+Q_0$, el vector de fuerzas de extremo fijo es $Q_0=-P_0$.")
    st.dataframe(pd.DataFrame({
        "GDL": [number[dof] for dof in order],
        f"P ({force_unit})": solution.loads[order] / kilonewtons,
        f"P₀ ({force_unit})": solution.initial_loads[order] / kilonewtons,
        f"P + P₀ ({force_unit})": solution.rhs[order] / kilonewtons,
    }).round(8), hide_index=True, alt="Cargas aplicadas, cargas equivalentes y suma por grado de libertad")

    st.subheader("4. Ensamblar la matriz global")
    st.markdown("Cada coeficiente de $k_e$ se suma en la fila y la columna globales "
                "indicadas por los cuatro GDL de la barra.")
    st.latex(r"K_{g_e(a),\,g_e(b)}\gets K_{g_e(a),\,g_e(b)}+(k_e)_{ab}")
    # Se desarrolla un término diagonal conectado a la barra seleccionada.
    component = int(abs(bar.b[3]) > abs(bar.b[2]))
    candidates = [bar.dofs[component], bar.dofs[2 + component]]
    dof = next((candidate for candidate in candidates if candidate in free_set), candidates[0])
    contributions = []
    for member in solution.bars:
        if dof in member.dofs:
            local_index = member.dofs.index(dof)
            value = member.stiffness * member.b[local_index] ** 2 * stiffness_factor
            if value:
                contributions.append((member.id, value))
    if contributions:
        st.caption("Aportes de las barras " + ", ".join(str(n) for n, _ in contributions) + ".")
        terms = " + ".join(_tex(value) for _, value in contributions)
        st.latex(rf"K_{{{number[dof]},{number[dof]}}}={terms}={_tex(solution.K[dof, dof] * stiffness_factor)}")
    global_matrix = solution.K[np.ix_(order, order)] * stiffness_factor
    st.dataframe(pd.DataFrame(global_matrix, index=headers, columns=headers).round(6),
                 alt=f"Matriz de rigidez global K en {stiffness_unit}")

    st.subheader("5. Aplicar apoyos y resolver")
    st.latex(r"u_r=0,\qquad K_{ff}u_f=(P+P_0)_f=b_f")
    if fixed:
        st.markdown("Al fijar a cero los GDL " + ", ".join(str(number[dof]) for dof in fixed)
                    + ", se conservan las filas y columnas libres para formar $K_{ff}$.")
    if free:
        free_headers = [header[dof] for dof in free]
        matrix = solution.K[np.ix_(free, free)] * stiffness_factor
        rhs = solution.rhs[free] / kilonewtons
        table = pd.DataFrame(matrix, index=free_headers, columns=free_headers)
        table[f"b_f ({force_unit})"] = rhs
        st.dataframe(table.round(8), alt="Sistema reducido con la matriz Kff y el término independiente")
        _show_triangular_solve(matrix, rhs, free_headers, [number[dof] for dof in free],
                               solution.u[free] / metres, length_unit)
    else:
        st.info("Todos los grados de libertad están restringidos.")
    st.dataframe(pd.DataFrame({
        "GDL": [number[dof] for dof in order],
        "Traslación": [solution.labels[dof] for dof in order],
        f"u ({length_unit})": solution.u[order] / metres,
    }).round(9).replace(-0.0, 0.0), hide_index=True, alt="Vector de desplazamientos resuelto")

    st.subheader("6. Recuperar fuerzas y leer la respuesta")
    st.latex(r"\delta_e=b_eu_e,\qquad N_e=\frac{E_e A_e}{L_e}(\delta_e-\Delta L_{0,e}),"
             r"\qquad R=Ku-(P+P_0)")
    u_bar = solution.u[bar.dofs] / metres
    extension = bar.b @ u_bar
    st.latex(rf"\delta_{{{bar.id}}}={_matrix(bar.b)}{_matrix(u_bar[:, None])}"
             rf"={_tex(extension)}\,\mathrm{{{length_unit}}}")
    st.latex(rf"N_{{{bar.id}}}=({_tex(stiffness)})[({_tex(extension)})-({_tex(delta_l)})]"
             rf"={_tex(solution.axial[bar_index] / kilonewtons)}\,\mathrm{{{force_unit}}}")
    axial = np.round(solution.axial / kilonewtons, 8) + 0.0
    st.dataframe(pd.DataFrame({"Barra": bar_ids, f"N ({force_unit})": axial}),
                 hide_index=True, alt="Fuerza axial calculada en cada barra")
    if fixed:
        st.dataframe(pd.DataFrame({
            "GDL": [number[dof] for dof in fixed],
            "Componente": [solution.labels[dof] for dof in fixed],
            f"R ({force_unit})": solution.reactions[fixed] / kilonewtons,
        }).round(8).replace(-0.0, 0.0), hide_index=True, alt="Reacciones en los grados restringidos")
    st.caption("Una fuerza axial positiva indica tracción y una negativa indica compresión.")
    target = reference.get("target_dof")
    if isinstance(target, str) and target in solution.labels:
        target_index = solution.labels.index(target)
        u = solution.u[target_index]
        st.latex(rf"u_{{{number[target_index]}}}={_tex(u / metres)}\,\mathrm{{{length_unit}}}"
                 rf"={_tex(1000 * u)}\,\mathrm{{mm}}")
    if "answer" in saved_solution:
        st.success(saved_solution["answer"], title="Respuesta del ejercicio")
