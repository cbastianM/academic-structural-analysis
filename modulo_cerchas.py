import json
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import streamlit as st

from desarrollo_matricial_cerchas import show_dof_reference, show_matrix_steps
from solucionador_cerchas import TrussMatrixSolver, normalize_truss


HIBBELER_EXAMPLES = Path(__file__).parent / "ejemplos" / "hibbeler"
NODE_TYPES = {"id": "Int64", "x": "Float64", "y": "Float64", "fix_x": "boolean",
              "fix_y": "boolean", "fx": "Float64", "fy": "Float64"}
BAR_TYPES = {"id": "Int64", "start_node": "Int64", "end_node": "Int64",
             "e_gpa": "Float64", "area": "Float64", "delta_l_mm": "Float64"}


def set_truss_project(project, *, solve=None):
    if not isinstance(project, dict) or "truss" not in project:
        raise ValueError("Seleccione un archivo JSON del módulo de cerchas.")
    if project.get("version") != "1.0.0":
        raise ValueError("La versión del archivo de cercha no es compatible.")
    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("El proyecto necesita un nombre.")
    saved_solution = project.get("solution")
    if saved_solution is not None and (
        not isinstance(saved_solution, dict)
        or not isinstance(saved_solution.get("answer"), str)
        or not isinstance(saved_solution.get("steps"), list)
        or not all(isinstance(step, str) for step in saved_solution["steps"])
    ):
        raise ValueError("La solución guardada debe incluir una respuesta y una lista de pasos.")
    model = {
        "version": "1.0.0", "name": name.strip(), "truss": normalize_truss(project["truss"])
    }
    if saved_solution is not None:
        model["solution"] = saved_solution
    st.session_state.truss_project = model
    st.session_state.truss_solution = None
    st.session_state.truss_solve_error = None
    st.session_state.truss_revision = st.session_state.get("truss_revision", 0) + 1
    st.session_state.truss_next_view = "Modelo"
    if solve is None:
        solve = saved_solution is not None
    if solve:
        try:
            st.session_state.truss_solution = TrussMatrixSolver(model["truss"])
        except ValueError as error:
            st.session_state.truss_solve_error = str(error)
        else:
            st.session_state.truss_next_view = "Resultados"


def _new_project():
    return {"version": "1.0.0", "name": "Nueva cercha", "truss": {"nodes": [], "elements": []}}


def _project_tools(project):
    with st.sidebar:
        if st.button("Volver al inicio", key="truss_home", icon=":material/home:", width="stretch"):
            st.session_state.pagina = "home"
            st.rerun()
        st.subheader("Proyecto de cercha")
        if st.button("Nueva cercha", key="truss_new", width="stretch"):
            set_truss_project(_new_project())
            st.rerun()
        st.caption("Ejemplos resueltos de Hibbeler")
        for problem in ("14-11", "14-14"):
            if st.button(f"Cargar problema {problem}", key=f"truss_example_{problem}", width="stretch"):
                path = HIBBELER_EXAMPLES / f"Problema {problem}.json"
                set_truss_project(json.loads(path.read_text(encoding="utf-8")))
                st.rerun()
        st.divider()
        with st.form("truss_import", enter_to_submit=False):
            uploaded = st.file_uploader("Archivo de cercha", type="json", key="truss_file")
            if st.form_submit_button("Abrir JSON", width="stretch"):
                if uploaded is None:
                    st.warning("Seleccione un archivo JSON.")
                else:
                    try:
                        set_truss_project(json.loads(uploaded.getvalue()))
                    except (ValueError, RecursionError) as error:
                        st.error(f"Error al abrir el modelo. {error}")
                    else:
                        st.rerun()
        st.download_button(
            "Descargar modelo guardado", json.dumps(project, ensure_ascii=False, indent=2),
            file_name="cercha.json", mime="application/json", key="truss_export",
            on_click="ignore", width="stretch",
        )


def _plot_model(truss, solution=None):
    if not truss["nodes"]:
        return
    coords = {node["id"]: np.array([node["x"], node["y"]]) for node in truss["nodes"]}
    points = np.array(list(coords.values()))
    span = np.ptp(points, axis=0).max() or 1.0
    forces = np.array([[node["fx"], node["fy"]] for node in truss["nodes"]])
    max_force = np.hypot(forces[:, 0], forces[:, 1]).max()
    fig = Figure(figsize=(8.5, 3.5), layout="constrained")
    ax = fig.subplots()
    blue, amber, teal = "#204963", "#a65312", "#00827c"

    if solution is not None:
        max_u = np.max(np.abs(solution.u))
        scale = 0.12 * span / max_u if max_u else 1.0
        deformed = {nid: point + scale * solution.u[list(solution.dofs[nid])]
                    for nid, point in coords.items()}

    for bar in truss["elements"]:
        i, j = bar["start_node"], bar["end_node"]
        start, end = coords[i], coords[j]
        color = amber if bar["delta_l_mm"] else blue
        ax.plot(*np.array([start, end]).T, color=color, linewidth=2.3, zorder=2)
        if solution is not None and max_u:
            ax.plot(*np.array([deformed[i], deformed[j]]).T, "--", color=teal, linewidth=1.4)
        ax.annotate(
            f"B{bar['id']}", start + 0.43 * (end - start), xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", color=color, fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
        )

    for node, force in zip(truss["nodes"], forces):
        nid = node["id"]
        x, y = coords[nid]
        if node["fix_x"]:
            ax.plot(x - span * 0.027, y, marker=">", color=blue,
                    markerfacecolor=blue if node["fix_y"] else "white", markersize=9)
        elif node["fix_y"]:
            ax.plot(x, y - span * 0.027, marker="^", color=blue, markerfacecolor="white", markersize=9)
        if np.any(force):
            tail = coords[nid] - 0.13 * span * (force / max_force)
            ax.update_datalim([tail])
            ax.annotate("", xy=(x, y), xytext=tail,
                        arrowprops={"arrowstyle": "-|>", "color": "#ae3153", "lw": 1.7})
        lower = y < (points[:, 1].min() + points[:, 1].max()) / 2
        ax.annotate(f"N{nid}", (x, y), xytext=(6, -7 if lower else 7), textcoords="offset points",
                    va="top" if lower else "bottom", fontsize=9, color=blue, fontweight="bold")
    ax.scatter(*points.T, color=blue, s=20, zorder=3)
    ax.set(xlabel="X (m)", ylabel="Y (m)", aspect="equal")
    ax.grid(alpha=0.2)
    ax.margins(x=0.14, y=0.20)
    st.pyplot(fig, alt="Geometría de la cercha con nodos, barras, apoyos y cargas aplicadas")
    if solution is not None and max_u:
        st.caption(f"Línea discontinua verde · deformada a escala ×{scale:.3g}")
    elif solution is None and any(bar["delta_l_mm"] for bar in truss["elements"]):
        st.caption("La línea naranja identifica una barra con desajuste de longitud.")


def _edit_model(project):
    truss = project["truss"]
    revision = st.session_state.truss_revision
    number = st.column_config.NumberColumn
    with st.form("truss_model", enter_to_submit=False):
        name = st.text_input("Nombre del proyecto", project["name"], key=f"truss_name_{revision}")
        save, solve = st.columns(2)
        saved = save.form_submit_button("Aplicar cambios", width="stretch")
        solved = solve.form_submit_button("Resolver cercha", type="primary", width="stretch")
        st.subheader("Nodos, apoyos y cargas")
        st.caption("Activa ambas casillas para una articulación o una sola para un rodillo. "
                   "X es positiva hacia la derecha e Y hacia arriba.")
        st.caption("Añade datos en la última fila y confirma cada edición con Enter. "
                   "Para eliminar una fila, selecciónala desde el margen izquierdo y pulsa Supr.")
        nodes = st.data_editor(
            pd.DataFrame(truss["nodes"], columns=list(NODE_TYPES)).astype(NODE_TYPES),
            num_rows="dynamic", hide_index=True, key=f"truss_nodes_{revision}", alt="Nodos de la cercha",
            on_change="ignore",
            column_config={
                "id": number("Nodo", min_value=1, step=1),
                "x": number("X (m)"),
                "y": number("Y (m)"),
                "fix_x": st.column_config.CheckboxColumn("Restringir X", default=False),
                "fix_y": st.column_config.CheckboxColumn("Restringir Y", default=False),
                "fx": number("Fx (kN)", default=0.0),
                "fy": number("Fy (kN)", default=0.0),
            },
        )
        st.subheader("Barras")
        st.caption("ΔL₀ es positiva si la barra es más larga que su longitud nominal entre nodos.")
        bars = st.data_editor(
            pd.DataFrame(truss["elements"], columns=list(BAR_TYPES)).astype(BAR_TYPES),
            num_rows="dynamic", hide_index=True, key=f"truss_bars_{revision}", alt="Barras de la cercha",
            on_change="ignore",
            column_config={
                "id": number("Barra", min_value=1, step=1),
                "start_node": number("Nodo i", min_value=1, step=1),
                "end_node": number("Nodo j", min_value=1, step=1),
                "e_gpa": number("E (GPa)", default=200.0, min_value=0.0),
                "area": number("A (m²)", min_value=0.0),
                "delta_l_mm": number("ΔL₀ (mm)", default=0.0),
            },
        )
    if saved or solved:
        edited_truss = {"nodes": nodes.to_dict("records"), "elements": bars.to_dict("records")}
        edited_project = {"name": name, "version": project["version"], "truss": edited_truss}
        if "solution" in project and edited_truss == truss:
            edited_project["solution"] = project["solution"]
        try:
            set_truss_project(edited_project, solve=solved)
        except ValueError as error:
            st.error(str(error))
        else:
            st.rerun()


def _show_results(truss, solution):
    nodes = solution.nodes
    node_ids = [node["id"] for node in nodes]
    first_free = next((i for i, n in enumerate(nodes) if not n["fix_x"] and not n["fix_y"]), 0)
    left, right = st.columns(2)
    node = left.selectbox("Nodo a consultar", node_ids, index=first_free, format_func=lambda nid: f"N{nid}")
    component = right.selectbox("Componente", ["Vertical (Y)", "Horizontal (X)"])
    dof = solution.dofs[node][int(component == "Vertical (Y)")]
    u_mm = round(solution.u[dof] * 1000, 4) + 0.0
    st.metric(f"Desplazamiento de N{node} · {component}", f"{u_mm:+.4f} mm".replace(".", ","))
    _plot_model(truss, solution)

    st.subheader("Desplazamientos y reacciones")
    st.dataframe(pd.DataFrame({
        "Nodo": node_ids,
        "Ux (mm)": solution.u[::2] * 1000,
        "Uy (mm)": solution.u[1::2] * 1000,
        "Rx (kN)": solution.reactions[::2],
        "Ry (kN)": solution.reactions[1::2],
    }).round(6).replace(-0.0, 0.0), hide_index=True, alt="Desplazamientos y reacciones por nodo")
    st.subheader("Fuerzas axiales")
    axial = np.round(solution.axial, 6) + 0.0
    st.dataframe(pd.DataFrame({
        "Barra": [bar.id for bar in solution.bars],
        "Longitud (m)": [bar.length for bar in solution.bars],
        "ΔL₀ (mm)": [bar.delta_l * 1000 for bar in solution.bars],
        "Axial (kN)": axial,
        "Estado": np.select([axial > 0, axial < 0], ["Tracción", "Compresión"], default="Nula"),
    }), hide_index=True, alt="Fuerza axial por barra")

    with st.expander("Comprobación por trabajo virtual", key="truss_virtual", on_change="rerun") as details:
        if details.open:
            st.latex(r"u_q = \sum_e n_e \frac{N_e L_e}{E_e A_e} + \sum_e n_e \Delta L_{0,e}")
            st.caption(f"La carga virtual se aplica en el sentido positivo de {solution.labels[dof]}. "
                       "n es la fuerza axial virtual por unidad de carga.")
            virtual, elastic, initial = solution.virtual_work(dof)
            st.dataframe(pd.DataFrame({
                "Barra": [bar.id for bar in solution.bars],
                "n": virtual,
                "Aporte elástico (mm)": elastic * 1000,
                "Aporte por desajuste (mm)": initial * 1000,
            }).round(6).replace(-0.0, 0.0), hide_index=True, alt="Aportes de cada barra al desplazamiento consultado")
            total_mm = round((elastic + initial).sum() * 1000, 6) + 0.0
            total = f"{total_mm:+.6f}".replace(".", ",")
            st.markdown(f"**Desplazamiento por trabajo virtual · {total} mm**")


def _set_guide_step(key, step):
    st.session_state[key] = step


def _show_matrix_guide(truss, solution, saved_solution):
    steps = saved_solution.get("guide")
    if not isinstance(steps, list) or not steps or any(
        not isinstance(step, dict)
        or not isinstance(step.get("title"), str)
        or not isinstance(step.get("body"), list)
        or not all(isinstance(paragraph, str) for paragraph in step["body"])
        for step in steps
    ):
        return False

    revision = st.session_state.truss_revision
    mode = st.segmented_control(
        "Forma de estudio", ["Guía desde cero", "Matrices completas"], default="Guía desde cero",
        key=f"truss_matrix_mode_{revision}", label_visibility="collapsed", required=True, width="stretch",
    )
    if mode == "Matrices completas":
        return False

    key = f"truss_guide_step_{revision}"
    index = st.selectbox(
        "Paso del ejercicio", list(range(len(steps))), key=key, required=True, persist_state="session",
        format_func=lambda i: f"{i + 1}. {steps[i]['title']}",
    )
    st.progress((index + 1) / len(steps), text=f"Paso {index + 1} de {len(steps)}")
    if st.toggle("Mostrar qué significa cada u", key=f"truss_guide_dofs_{revision}"):
        show_dof_reference(solution, saved_solution)
    step = steps[index]
    with st.container(border=True):
        st.subheader(step["title"])
        st.markdown("\n\n".join(step["body"]))
        if index == 0:
            _plot_model(truss)
    previous, following = st.columns(2)
    previous.button("Anterior", key="truss_guide_previous", icon=":material/arrow_back:",
                    disabled=index == 0, on_click=_set_guide_step, args=(key, index - 1),
                    width="stretch")
    if index + 1 < len(steps):
        following.button("Siguiente", key="truss_guide_next", icon=":material/arrow_forward:",
                         type="primary", on_click=_set_guide_step, args=(key, index + 1),
                         width="stretch")
    else:
        following.button("Volver al primer paso", key="truss_guide_restart",
                         on_click=_set_guide_step, args=(key, 0), width="stretch")
    return True


def render_truss():
    if "truss_project" not in st.session_state:
        set_truss_project(_new_project())
    project = st.session_state.truss_project
    truss = project["truss"]
    solution = st.session_state.truss_solution
    _project_tools(project)

    st.title("Cerchas planas")
    st.caption("Método de rigidez directa · barras articuladas de respuesta axial elástica")
    if "truss_next_view" in st.session_state:
        st.session_state.truss_view = st.session_state.pop("truss_next_view")
    if "truss_view" not in st.session_state:
        st.session_state.truss_view = "Modelo"
    view = st.segmented_control("Vista de cerchas", ["Modelo", "Resultados", "Método matricial"],
                                key="truss_view", label_visibility="collapsed", width="stretch")
    if view in (None, "Modelo"):
        if st.session_state.truss_solve_error:
            st.error(st.session_state.truss_solve_error)
        _plot_model(truss)
        _edit_model(project)
    elif solution is None:
        st.info("Ve a Modelo y pulsa Resolver cercha para calcular los resultados.")
    else:
        st.subheader(project["name"])
        if view == "Resultados":
            if st.button("Ver método matricial paso a paso", key="truss_matrix_steps", type="primary"):
                st.session_state.truss_next_view = "Método matricial"
                st.rerun()
            if "solution" in project:
                saved_solution = project["solution"]
                st.success(saved_solution["answer"], title="Solución guardada")
                with st.expander("Resumen de la solución"):
                    st.markdown("\n\n".join(saved_solution["steps"]))
            _show_results(truss, solution)
        else:
            saved_solution = project.get("solution", {})
            if not _show_matrix_guide(truss, solution, saved_solution):
                show_matrix_steps(truss, solution, saved_solution)
