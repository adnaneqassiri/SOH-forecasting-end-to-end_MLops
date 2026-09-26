"""Streamlit dashboard backed exclusively by the SOH FastAPI service."""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 180

st.set_page_config(
    page_title="Battery SOH Monitor",
    page_icon="",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 16px;
    }
    [data-testid="stDialog"] > div {border-radius: 18px;}
    .vehicle-chip {
        display: inline-block;
        margin: 0 7px 7px 0;
        padding: 5px 11px;
        border-radius: 999px;
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        font-size: 0.88rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str):
    response = requests.get(
        f"{API_URL}{path}", timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def api_predict(vehicle_ids: tuple[int, ...], all_vehicles: bool):
    payload = (
        {"all_vehicles": True}
        if all_vehicles
        else {"vehicle_ids": list(vehicle_ids)}
    )
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def percentage_points(value: float) -> str:
    return f"{value * 100:+.2f} pp"


@st.dialog("Sélectionner les véhicules de test", width="large")
def vehicle_selection_dialog(vehicles: list[dict]) -> None:
    st.write(
        "Choisissez un ou plusieurs véhicules. L’inférence utilisera les "
        "100 événements précédant le snapshot de test de chaque véhicule."
    )
    all_ids = [int(vehicle["vehicle_id"]) for vehicle in vehicles]
    label_to_id = {
        f"{vehicle['display_name']} · {vehicle['available_cycles']} cycles": int(
            vehicle["vehicle_id"]
        )
        for vehicle in vehicles
    }
    select_all = st.checkbox(
        "Tous les véhicules de test",
        value=bool(st.session_state.get("select_all_vehicles", True)),
        key="selection_dialog_all",
    )
    defaults = [
        label
        for label, vehicle_id in label_to_id.items()
        if vehicle_id in st.session_state.get("selected_vehicle_ids", [])
    ]
    selected_labels = st.multiselect(
        "Véhicules",
        options=list(label_to_id),
        default=[] if select_all else defaults,
        disabled=select_all,
        placeholder="Sélectionner les véhicules…",
        key="selection_dialog_vehicles",
    )
    if select_all:
        st.caption(f"Les {len(all_ids)} véhicules du split test seront analysés.")

    left, right = st.columns([1, 1])
    if left.button(
        "Lancer les prédictions",
        type="primary",
        width="stretch",
        key="selection_dialog_confirm",
    ):
        selected_ids = (
            all_ids
            if select_all
            else [label_to_id[label] for label in selected_labels]
        )
        if not selected_ids:
            st.error("Sélectionnez au moins un véhicule.")
        else:
            st.session_state.selected_vehicle_ids = selected_ids
            st.session_state.select_all_vehicles = select_all
            st.session_state.selection_confirmed = True
            st.rerun()
    if right.button(
        "Annuler",
        width="stretch",
        key="selection_dialog_cancel",
    ):
        if st.session_state.get("selected_vehicle_ids"):
            st.session_state.selection_confirmed = True
            st.rerun()
        st.info("Une sélection est nécessaire avant de lancer l’inférence.")


def vehicle_chart(vehicle: dict) -> alt.Chart:
    history = pd.DataFrame(
        {
            "Cycle": vehicle["history_cycles"],
            "SOH": vehicle["historical_soh"],
            "Série": "SOH observé",
        }
    )
    reference = pd.DataFrame(
        {
            "Cycle": vehicle["history_cycles"],
            "SOH": vehicle["reference_soh"],
            "Série": "SOH de référence",
        }
    )
    forecast = pd.DataFrame(
        {
            "Cycle": [
                vehicle["history_cycles"][-1],
                *vehicle["forecast_cycles"],
            ],
            "SOH": [vehicle["current_soh"], *vehicle["forecast"]],
            "Série": "SOH prédit",
        }
    )
    chart_data = pd.concat(
        [history, reference, forecast], ignore_index=True
    )
    lines = (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "Cycle:Q",
                title="Événement de charge",
            ),
            y=alt.Y("SOH:Q", title="SOH", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Série:N",
                scale=alt.Scale(
                    domain=[
                        "SOH observé",
                        "SOH de référence",
                        "SOH prédit",
                    ],
                    range=["#8A8F98", "#4C78A8", "#F58518"],
                ),
            ),
            tooltip=[
                "Série",
                "Cycle",
                alt.Tooltip("SOH:Q", format=".4f"),
            ],
        )
    )
    forecast_start = (
        alt.Chart(
            pd.DataFrame({"Cycle": [vehicle["forecast_cycles"][0]]})
        )
        .mark_rule(color="#808080", strokeDash=[5, 5])
        .encode(x="Cycle:Q")
    )
    return (lines + forecast_start).interactive()


def fleet_chart(vehicles: list[dict]) -> alt.Chart:
    rows = []
    for vehicle in vehicles:
        rows.extend(
            {
                "Cycle": cycle,
                "SOH": soh,
                "Véhicule": vehicle["display_name"],
                "Série": "Observé",
            }
            for cycle, soh in zip(
                vehicle["history_cycles"], vehicle["reference_soh"]
            )
        )
        rows.extend(
            {
                "Cycle": cycle,
                "SOH": soh,
                "Véhicule": vehicle["display_name"],
                "Série": "Prévision",
            }
            for cycle, soh in zip(
                [vehicle["history_cycles"][-1], *vehicle["forecast_cycles"]],
                [vehicle["current_soh"], *vehicle["forecast"]],
            )
        )
    data = pd.DataFrame(rows)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "Cycle:Q",
                title="Événement de charge",
            ),
            y=alt.Y("SOH:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Véhicule:N"),
            strokeDash=alt.StrokeDash("Série:N"),
            tooltip=[
                "Véhicule",
                "Série",
                "Cycle",
                alt.Tooltip("SOH:Q", format=".4f"),
            ],
        )
        .interactive()
    )


def render_vehicle(vehicle: dict) -> None:
    columns = st.columns(4)
    columns[0].metric("SOH actuel", percent(vehicle["current_soh"]))
    columns[1].metric(
        "SOH prédit (+10)", percent(vehicle["predicted_soh_10"])
    )
    columns[2].metric(
        "Variation prévue", percentage_points(vehicle["expected_change"])
    )
    columns[3].metric(
        "Capacité nominale estimée",
        f"{vehicle['nominal_capacity']:.2f}",
    )

    st.altair_chart(vehicle_chart(vehicle), width="stretch")
    forecast_table = pd.DataFrame(
        {
            "Horizon": vehicle["forecast_horizons"],
            "SOH prédit": vehicle["forecast"],
            "Capacité prédite": vehicle["forecast_capacities"],
        }
    )
    st.dataframe(
        forecast_table.style.format(
            {"SOH prédit": "{:.2%}", "Capacité prédite": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Entrée modèle : les 100 événements se terminant à "
        f"charge_segment {vehicle['history_cycles'][-1]}. Prévision : "
        f"{vehicle['forecast_cycles'][0]} → {vehicle['forecast_cycles'][-1]}. "
        "La capacité nominale est le quantile 0,99 des données du véhicule."
    )


def render_fleet_dashboard(data: dict) -> None:
    st.title("Fleet Battery SOH Dashboard")
    st.caption(
        "Véhicules Data 4 du split test · snapshots rétrospectifs fixes · "
        "100 événements en entrée · prévision sur les 10 événements suivants"
    )
    vehicles = data["vehicles"]
    st.markdown(
        " ".join(
            f'<span class="vehicle-chip">{vehicle["display_name"]}</span>'
            for vehicle in vehicles
        ),
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("Véhicules analysés", data["total_vehicles"])
    metric_columns[1].metric(
        "SOH moyen", percent(data["fleet_average_soh"])
    )
    metric_columns[2].metric(
        "SOH inférieur à 90%", data["count_below_090"]
    )
    metric_columns[3].metric(
        "SOH actuel minimal", percent(data["lowest_current_soh"])
    )

    left, right = st.columns([1.45, 1])
    with left:
        st.subheader("Évolution de la flotte sélectionnée")
        st.altair_chart(fleet_chart(vehicles), width="stretch")
    with right:
        st.subheader("Priorité d’attention")
        attention = pd.DataFrame(
            {
                "Véhicule": [item["display_name"] for item in vehicles],
                "SOH actuel": [item["current_soh"] for item in vehicles],
                "SOH prédit (+10)": [
                    item["predicted_soh_10"] for item in vehicles
                ],
                "Variation": [item["expected_change"] for item in vehicles],
            }
        ).sort_values(["Variation", "SOH actuel"], kind="stable")
        st.dataframe(
            attention.style.format(
                {
                    "SOH actuel": "{:.2%}",
                    "SOH prédit (+10)": "{:.2%}",
                    "Variation": lambda value: f"{value * 100:+.2f} pp",
                }
            ),
            hide_index=True,
            width="stretch",
        )


def render_vehicle_analysis(data: dict) -> None:
    st.title("Vehicle Analysis")
    st.caption(
        "La prévision utilise uniquement les 100 événements précédant le "
        "snapshot de test; la courbe affiche tout l’historique disponible "
        "jusqu’à ce point."
    )
    vehicles = data["vehicles"]
    names = {
        vehicle["display_name"]: vehicle for vehicle in vehicles
    }
    selected_name = st.selectbox(
        "Véhicule",
        options=list(names),
        key="vehicle_analysis_selection",
    )
    render_vehicle(names[selected_name])


def main() -> None:
    vehicles = api_get("/vehicles")

    if not st.session_state.get("selection_confirmed", False):
        vehicle_selection_dialog(vehicles)
        st.info("Sélectionnez les véhicules dans la fenêtre pour commencer.")
        st.stop()

    selected_ids = tuple(
        int(value) for value in st.session_state["selected_vehicle_ids"]
    )
    with st.sidebar:
        page = st.radio(
            "Vue",
            options=["Fleet Dashboard", "Vehicle Analysis"],
            key="dashboard_page",
        )
        st.divider()
        st.header("Sélection")
        st.write(f"{len(selected_ids)} véhicule(s)")
        if st.button(
            "Modifier la sélection",
            width="stretch",
            key="change_vehicle_selection",
        ):
            vehicle_selection_dialog(vehicles)

    with st.spinner("Calcul des prédictions…"):
        data = api_predict(
            selected_ids,
            bool(st.session_state.get("select_all_vehicles", False)),
        )
    if page == "Fleet Dashboard":
        render_fleet_dashboard(data)
    else:
        render_vehicle_analysis(data)


try:
    main()
except requests.RequestException as error:
    st.error(
        f"L’API est indisponible à l’adresse {API_URL}. "
        "Démarrez FastAPI avant Streamlit."
    )
    st.exception(error)
