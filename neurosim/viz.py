"""Plotly figures for morphologies and simulation results."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .morphology import Geometry
from .simulation import SimulationResult

NEURITE_COLORS = {
    "axon": "#1f77b4",
    "basal_dendrite": "#d62728",
    "apical_dendrite": "#9467bd",
}
SITE_COLORS = ["#111827", "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
NEURITE_LABELS = {"axon": "Axon", "basal_dendrite": "Basal dendrite", "apical_dendrite": "Apical dendrite"}
VOLTAGE_SCALE = "Turbo"


def _scene(fig: go.Figure, height: int) -> go.Figure:
    axis = dict(showbackground=False, showgrid=False, zeroline=False, title="", showticklabels=False)
    fig.update_layout(
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.0),
        uirevision="morph",  # keep the camera when the figure updates
    )
    return fig


def _soma_trace(geom: Geometry, color: str = "#111827") -> go.Scatter3d:
    c = geom.soma_center
    return go.Scatter3d(
        x=[c[0]], y=[c[1]], z=[c[2]], mode="markers", name="Soma",
        marker=dict(size=max(4, min(geom.soma_radius, 14)), color=color),
        hovertemplate=f"Soma (r = {geom.soma_radius:.1f} µm)<extra></extra>",
    )


def morphology_figure(geom: Geometry, height: int = 600) -> go.Figure:
    """3D view of a neuron coloured by neurite type."""
    fig = go.Figure()
    for name, pts in geom.lines.items():
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
            name=NEURITE_LABELS.get(name, name),
            line=dict(color=NEURITE_COLORS.get(name, "#7f7f7f"), width=3),
            hoverinfo="skip",
        ))
    fig.add_trace(_soma_trace(geom))
    return _scene(fig, height)


def _all_lines(geom: Geometry):
    pts = np.vstack(list(geom.lines.values()))
    idx = np.concatenate(list(geom.section_index.values()))
    return pts, idx


def voltage_figure(
    geom: Geometry, result: SimulationResult, n_frames: int = 60,
    vmin: float = -80.0, vmax: float = 40.0, height: int = 600,
) -> go.Figure:
    """Animated 3D view of membrane potential spreading through the neuron.

    Requires a result produced with ``record_sections_dt``.
    """
    if result.v_sections is None or result.t_sections is None:
        raise ValueError("result has no per-section voltages; run with record_sections_dt")
    pts, idx = _all_lines(geom)
    v_sec, t_sec = result.v_sections, result.t_sections
    frame_ids = np.unique(np.linspace(0, len(t_sec) - 1, n_frames).astype(int))
    soma_t = np.interp(t_sec, result.t, result.v["soma"])

    def colors(k: int) -> np.ndarray:
        c = np.full(len(idx), np.nan)
        ok = idx >= 0
        c[ok] = v_sec[idx[ok], k]
        return c

    def traces(k: int):
        return [
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines", name="Neurites", showlegend=False,
                line=dict(color=colors(k), colorscale=VOLTAGE_SCALE, cmin=vmin, cmax=vmax, width=4,
                          colorbar=dict(title="mV", thickness=12, len=0.6)),
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=[geom.soma_center[0]], y=[geom.soma_center[1]], z=[geom.soma_center[2]],
                mode="markers", name="Soma", showlegend=False,
                marker=dict(size=max(5, min(geom.soma_radius, 14)), color=[soma_t[k]],
                            colorscale=VOLTAGE_SCALE, cmin=vmin, cmax=vmax),
                hoverinfo="skip",
            ),
        ]

    k0 = frame_ids[0]
    fig = go.Figure(
        data=traces(k0),
        frames=[go.Frame(data=traces(k), name=f"{t_sec[k]:.1f}") for k in frame_ids],
    )
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=1.0, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.0, len=1.0, y=0.02, currentvalue=dict(prefix="t = ", suffix=" ms"),
            steps=[dict(method="animate", label=f"{t_sec[k]:.0f}",
                        args=[[f"{t_sec[k]:.1f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                   for k in frame_ids],
        )],
    )
    return _scene(fig, height)


def traces_figure(result: SimulationResult, height: int = 480) -> go.Figure:
    """Membrane potential at every recording site plus the injected current."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04)
    for i, (site, v) in enumerate(result.v.items()):
        fig.add_trace(go.Scattergl(
            x=result.t, y=v, mode="lines", name=site.replace("_", " "),
            line=dict(color=SITE_COLORS[i % len(SITE_COLORS)], width=1.6 if i == 0 else 1.2),
        ), row=1, col=1)
    if len(result.spike_times):
        fig.add_trace(go.Scatter(
            x=result.spike_times, y=np.full(len(result.spike_times), 50.0), mode="markers",
            name="spikes", marker=dict(symbol="line-ns-open", size=10, color="#111827"),
        ), row=1, col=1)
    fig.add_trace(go.Scattergl(
        x=result.t, y=result.i_inj, mode="lines", name="I inj", line=dict(color="#6b7280", width=1.2),
        showlegend=False,
    ), row=2, col=1)
    fig.update_yaxes(title_text="V (mV)", row=1, col=1)
    fig.update_yaxes(title_text="I (nA)", row=2, col=1)
    fig.update_xaxes(title_text="time (ms)", row=2, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.04, x=0), hovermode="x unified")
    return fig


def fi_figure(amps: np.ndarray, rates: np.ndarray, height: int = 380) -> go.Figure:
    fig = go.Figure(go.Scatter(x=amps, y=rates, mode="lines+markers", line=dict(color="#d62728")))
    fig.update_layout(xaxis_title="injected current (nA)", yaxis_title="firing rate (Hz)",
                      height=height, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def sholl_figure(radii: np.ndarray, counts: np.ndarray, height: int = 340) -> go.Figure:
    fig = go.Figure(go.Scatter(x=radii, y=counts, mode="lines", fill="tozeroy", line=dict(color="#9467bd")))
    fig.update_layout(xaxis_title="distance from soma (µm)", yaxis_title="intersections",
                      height=height, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def histogram_figure(values: np.ndarray, xlabel: str, height: int = 300) -> go.Figure:
    values = np.asarray(values, dtype=float)
    fig = go.Figure(go.Histogram(x=values[np.isfinite(values)], marker_color="#1f77b4", nbinsx=30))
    fig.update_layout(xaxis_title=xlabel, yaxis_title="count", bargap=0.05,
                      height=height, margin=dict(l=10, r=10, t=10, b=10))
    return fig
