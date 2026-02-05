"""QNM Analyser — Online dashboard for quasi-normal mode convergence analysis.

Upload eigenvalue files (two-column: Re, Im) and interactively explore
convergence across numerical resolutions.

Author: Dr. Denys Dutykh
        Khalifa University of Science and Technology, Abu Dhabi, UAE
        https://www.denys-dutykh.com/
"""

import re as _re
import base64
import datetime

import numpy as np
import plotly.graph_objects as go
from dash import (
    Dash,
    dcc,
    html,
    dash_table,
    Input,
    Output,
    State,
    callback,
    ctx,
    no_update,
    ALL,
)
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOL = 1e-4
INITIAL_SLOTS = 3
MAX_SLOTS = 6

# Wong 2011 colorblind-safe palette
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "star"]

LEGEND_POSITIONS = {
    "Top-right": dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
    "Top-left": dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
    "Bottom-right": dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
    "Bottom-left": dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom"),
    "Hidden": dict(visible=False),
}

# ---------------------------------------------------------------------------
# Layout templates (light & dark)
# ---------------------------------------------------------------------------
_COMMON_AXIS = dict(
    showgrid=True,
    gridwidth=0.5,
    griddash="dash",
    zeroline=True,
    zerolinewidth=1,
    mirror=True,
    showline=True,
    linewidth=0.5,
    showspikes=True,
    spikemode="across",
    spikethickness=1,
    spikedash="dot",
    exponentformat="power",
)

LAYOUT_LIGHT = dict(
    template="simple_white",
    xaxis=dict(
        title=r"$\mathrm{Re}(\omega)$",
        gridcolor="lightgrey",
        zerolinecolor="grey",
        linecolor="grey",
        spikecolor="grey",
        **_COMMON_AXIS,
    ),
    yaxis=dict(
        title=r"$\mathrm{Im}(\omega)$",
        gridcolor="lightgrey",
        zerolinecolor="grey",
        linecolor="grey",
        spikecolor="grey",
        **_COMMON_AXIS,
    ),
    font=dict(family="Computer Modern, Times New Roman, serif", size=14),
    legend=dict(title="", borderwidth=0, font=dict(size=13)),
    width=900,
    height=700,
    margin=dict(l=80, r=40, t=40, b=80),
    paper_bgcolor="white",
    plot_bgcolor="white",
)

LAYOUT_DARK = dict(
    template="plotly_dark",
    xaxis=dict(
        title=dict(text=r"$\mathrm{Re}(\omega)$", font=dict(color="#ddd")),
        gridcolor="#444",
        zerolinecolor="#666",
        linecolor="#666",
        tickfont=dict(color="#ccc"),
        spikecolor="#888",
        **_COMMON_AXIS,
    ),
    yaxis=dict(
        title=dict(text=r"$\mathrm{Im}(\omega)$", font=dict(color="#ddd")),
        gridcolor="#444",
        zerolinecolor="#666",
        linecolor="#666",
        tickfont=dict(color="#ccc"),
        spikecolor="#888",
        **_COMMON_AXIS,
    ),
    font=dict(
        family="Computer Modern, Times New Roman, serif",
        size=14,
        color="#e0e0e0",
    ),
    legend=dict(
        title="",
        borderwidth=0,
        font=dict(size=13, color="#ccc"),
        bgcolor="rgba(30,30,30,0.8)",
    ),
    width=900,
    height=700,
    margin=dict(l=80, r=40, t=40, b=80),
    paper_bgcolor="#1e1e1e",
    plot_bgcolor="#2a2a2a",
)


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------


def parse_upload(contents, filename):
    """Decode an uploaded file and return (re_list, im_list, inferred_N)."""
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string).decode("utf-8")

    re_vals, im_vals = [], []
    for line in decoded.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                r, i = float(parts[0]), float(parts[1])
                if np.isfinite(r) and np.isfinite(i):
                    re_vals.append(r)
                    im_vals.append(i)
            except ValueError:
                continue

    inferred_n = None
    match = _re.search(r"(\d+)", filename or "")
    if match:
        inferred_n = int(match.group(1))

    return re_vals, im_vals, inferred_n


def compute_converged(ref_points, trees, other_keys, tol_value):
    """Find QNMs in *ref_points* present in ALL other resolutions within *tol_value*."""
    conv_re, conv_im = [], []
    for i in range(len(ref_points)):
        point = ref_points[i]
        found = True
        for n in other_keys:
            dist, _ = trees[n].query(point)
            if dist > tol_value:
                found = False
                break
        if found:
            conv_re.append(point[0])
            conv_im.append(point[1])
    return np.asarray(conv_re), np.asarray(conv_im)


def classify_converged(conv_re, conv_im, tol_value):
    """Classify converged QNMs into general, purely-imaginary, purely-real."""
    if len(conv_re) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
    general_mask = (np.abs(conv_re) >= tol_value) & (
        np.abs(conv_im) >= tol_value
    )
    pure_imag_mask = (np.abs(conv_re) < tol_value) & (
        np.abs(conv_im) >= tol_value
    )
    pure_real_mask = np.abs(conv_im) < tol_value

    def _stack(mask):
        if mask.any():
            return np.column_stack([conv_re[mask], conv_im[mask]])
        return np.empty((0, 2))

    return _stack(general_mask), _stack(pure_imag_mask), _stack(pure_real_mask)


def build_figure(datasets, tol_value, dark=False, conv_re=None, conv_im=None):
    """Build Plotly figure from a list of dataset dicts.  Returns (fig, info_str).

    If *conv_re* and *conv_im* are provided, skip internal convergence
    computation and use the supplied arrays instead.
    """
    layout = LAYOUT_DARK if dark else LAYOUT_LIGHT
    marker_outline = "white" if dark else "black"
    fig = go.Figure()

    if not datasets:
        fig.update_layout(**layout)
        return fig, "Upload data files to begin"

    sorted_ds = sorted(datasets, key=lambda d: d["resolution"])
    num = len(sorted_ds)

    for idx, ds in enumerate(sorted_ds):
        ridx = num - 1 - idx
        n = ds["resolution"]
        fig.add_trace(
            go.Scatter(
                x=ds["re"],
                y=ds["im"],
                mode="markers",
                name=f"N = {n}",
                marker=dict(
                    symbol=SYMBOLS[ridx % len(SYMBOLS)],
                    size=8,
                    color=COLORS[ridx % len(COLORS)],
                    opacity=0.85,
                    line=dict(width=1, color=marker_outline),
                ),
                customdata=[f"N = {n}"] * len(ds["re"]),
                hovertemplate=(
                    "Re(\u03c9) = %{x:.6f}<br>"
                    "Im(\u03c9) = %{y:.6f}<br>"
                    f"N = {n}<extra></extra>"
                ),
            )
        )

    info_str = ""
    # Use pre-computed convergence if available, else compute internally
    if conv_re is None and num >= 2:
        eigs_dict = {}
        for ds in sorted_ds:
            eigs_dict[ds["resolution"]] = np.column_stack([ds["re"], ds["im"]])
        res_list = sorted(eigs_dict.keys())
        highest = res_list[-1]
        others = res_list[:-1]
        trees = {n: cKDTree(eigs_dict[n]) for n in others}
        conv_re, conv_im = compute_converged(
            eigs_dict[highest], trees, others, tol_value
        )

    if conv_re is not None and len(conv_re) > 0:
        fig.add_trace(
            go.Scatter(
                x=conv_re.tolist() if hasattr(conv_re, "tolist") else list(conv_re),
                y=conv_im.tolist() if hasattr(conv_im, "tolist") else list(conv_im),
                mode="markers",
                name="Converged",
                marker=dict(
                    symbol="circle-open",
                    size=16,
                    color="red",
                    line=dict(width=2, color="red"),
                ),
                customdata=["Converged"] * len(conv_re),
                hovertemplate=(
                    "Re(\u03c9) = %{x:.6f}<br>"
                    "Im(\u03c9) = %{y:.6f}<br>"
                    "Converged<extra></extra>"
                ),
            )
        )
        info_str = f"({len(conv_re)} converged)"
    elif conv_re is not None:
        info_str = "(0 converged)"
    elif num < 2:
        info_str = "Upload at least 2 files for convergence analysis"

    fig.update_layout(**layout)
    return fig, info_str


def generate_report_text(conv_data):
    """Return the convergence report as a string from pre-computed data."""
    tol_value = conv_data["tol_value"]
    res_list = conv_data["resolutions"]
    conv_re = conv_data["conv_re"]
    general = np.array(conv_data["general"]) if conv_data["general"] else np.empty((0, 2))
    pure_imag = np.array(conv_data["pure_imag"]) if conv_data["pure_imag"] else np.empty((0, 2))
    pure_real = np.array(conv_data["pure_real"]) if conv_data["pure_real"] else np.empty((0, 2))

    lines = [
        "=" * 60,
        "QNM Convergence Report",
        "=" * 60,
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Tolerance: {tol_value:.1e}",
        f"Resolutions: {', '.join(str(n) for n in res_list)}",
        "",
        "-" * 60,
        "Summary",
        "-" * 60,
        f"Total converged QNMs: {len(conv_re)}",
        f"  General (Re!=0, Im!=0): {len(general)}",
        f"  Purely imaginary (Re~0): {len(pure_imag)}",
        f"  Purely real (Im~0): {len(pure_real)}",
        "",
    ]

    def fmt_table(arr):
        tbl = [
            f"  {'Re(omega)':>26s}  {'Im(omega)':>26s}",
            f"  {'\u2500' * 26}  {'\u2500' * 26}",
        ]
        for row in arr:
            tbl.append(f"  {row[0]:>26.16e}  {row[1]:>26.16e}")
        return tbl

    for label, arr, sort_fn in [
        (
            "General QNMs (Re != 0, Im != 0)",
            general,
            lambda a: np.argsort(-a[:, 1]),
        ),
        (
            "Purely Imaginary QNMs (Re ~ 0)",
            pure_imag,
            lambda a: np.argsort(-a[:, 1]),
        ),
        (
            "Purely Real QNMs (Im ~ 0)",
            pure_real,
            lambda a: np.argsort(a[:, 0]),
        ),
    ]:
        lines.append("-" * 60)
        lines.append(label)
        lines.append("-" * 60)
        if len(arr) > 0:
            lines.extend(fmt_table(arr[sort_fn(arr)]))
        else:
            lines.append("  (none)")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    external_scripts=[
        {
            "src": (
                "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/MathJax.js"
                "?config=TeX-AMS-MML_SVG"
            ),
            "integrity": "sha512-M36RUChWzAh1veeenRZFql7HydLEnkYmoloiCvVrhz402UZgKI93qkV7SsaxtVKdN95Wzajh39ysrXCq34NTsg==",
            "crossorigin": "anonymous",
        }
    ],
    title="QNM Analyser",
)
server = app.server  # entry-point for gunicorn: app:server
server.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def _make_upload_slot(i):
    """Build the layout for upload slot *i*."""
    return html.Div(
        [
            dcc.Upload(
                id={"type": "upload", "index": i},
                children=html.Div(
                    [
                        "Drag & drop or ",
                        html.A("browse", style={"fontWeight": "bold"}),
                    ]
                ),
                className="upload-zone",
            ),
            html.Div(
                [
                    html.Span(
                        "No file",
                        id={"type": "filename", "index": i},
                        className="filename-label",
                    ),
                    html.Label(
                        " N = ",
                        style={"marginLeft": "8px", "fontSize": "13px"},
                    ),
                    dcc.Input(
                        id={"type": "resolution", "index": i},
                        type="number",
                        placeholder="auto",
                        className="resolution-input",
                        debounce=True,
                    ),
                    html.Button(
                        "Clear",
                        id={"type": "clear", "index": i},
                        n_clicks=0,
                        className="clear-btn",
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginTop": "6px",
                },
            ),
        ],
        className="upload-slot",
    )


btn_style = {
    "padding": "8px 20px",
    "fontSize": "14px",
    "marginRight": "10px",
    "cursor": "pointer",
}

app.layout = html.Div(
    [
        # Header with title + theme toggle
        html.Div(
            [
                html.H2(
                    "QNM Analyser",
                    style={
                        "textAlign": "center",
                        "margin": "0",
                        "flex": "1",
                    },
                ),
                html.Button(
                    id="theme-toggle",
                    n_clicks=0,
                    title="Toggle dark / light theme",
                    className="theme-toggle-btn",
                ),
            ],
            className="header-bar",
        ),
        html.P(
            "Upload eigenvalue files (two-column: Re, Im) to analyse "
            "quasi-normal mode convergence across resolutions.",
            className="subtitle",
        ),
        # Dynamic slot controls
        html.Div(
            [
                html.Button(
                    "+ Add dataset",
                    id="btn-add-slot",
                    n_clicks=0,
                    className="slot-btn",
                ),
                html.Button(
                    "\u2212 Remove last",
                    id="btn-remove-slot",
                    n_clicks=0,
                    className="slot-btn",
                ),
            ],
            className="slot-controls",
        ),
        # Upload panel (dynamic children)
        html.Div(
            id="upload-panel",
            children=[_make_upload_slot(i) for i in range(INITIAL_SLOTS)],
            className="upload-panel",
        ),
        # Plot with loading spinner
        html.Div(
            dcc.Loading(
                dcc.Graph(
                    id="qnm-plot",
                    mathjax=True,
                    config={
                        "scrollZoom": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                        "toImageButtonOptions": {
                            "format": "png",
                            "scale": 3,
                            "filename": "qnm_complex_plane",
                        },
                    },
                    style={"height": "70vh"},
                ),
                type="circle",
                color="#0072B2",
            ),
            style={"display": "flex", "justifyContent": "center"},
        ),
        # Click-to-inspect detail panel
        html.Div(
            id="inspect-panel",
            className="inspect-panel",
            style={"display": "none"},
        ),
        # Controls bar
        html.Div(
            [
                html.Label(
                    "Tol (\u00d710\u207b\u2074): ",
                    style={"fontSize": "14px", "marginRight": "4px"},
                ),
                dcc.Input(
                    id="tol-input",
                    type="number",
                    value=1.0,
                    step=0.1,
                    min=0.1,
                    debounce=False,
                    className="tol-input",
                ),
                dcc.Slider(
                    id="tol-slider",
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    value=1.0,
                    marks={i: str(i) for i in range(1, 11)},
                    tooltip={"placement": "bottom", "always_visible": False},
                    className="tol-slider",
                ),
                html.Span(
                    id="convergence-info",
                    className="convergence-info",
                ),
                html.Label(
                    "Legend: ",
                    style={"fontSize": "14px", "marginRight": "5px"},
                ),
                dcc.Dropdown(
                    id="legend-pos",
                    options=[
                        {"label": k, "value": k} for k in LEGEND_POSITIONS
                    ],
                    value="Top-right",
                    clearable=False,
                    className="legend-dropdown",
                ),
                html.Button(
                    "Save PNG",
                    id="btn-png",
                    n_clicks=0,
                    style=btn_style,
                ),
                html.Button(
                    "Save PDF",
                    id="btn-pdf",
                    n_clicks=0,
                    style=btn_style,
                ),
                html.Button(
                    "Download Report",
                    id="btn-report",
                    n_clicks=0,
                    className="report-btn",
                ),
                html.Button(
                    "Export QNMs",
                    id="btn-export-qnms",
                    n_clicks=0,
                    style=btn_style,
                ),
            ],
            className="controls-bar",
        ),
        # Converged QNMs table
        html.Div(
            id="converged-table-container",
            className="converged-table-container",
        ),
        # Footer with author info
        html.Footer(
            [
                html.Hr(style={"margin": "16px 0 8px", "opacity": "0.3"}),
                html.P(
                    [
                        "Dr. Denys Dutykh — Khalifa University of Science "
                        "and Technology, Abu Dhabi, UAE — ",
                        html.A(
                            "www.denys-dutykh.com",
                            href="https://www.denys-dutykh.com/",
                            target="_blank",
                            rel="noopener noreferrer",
                        ),
                    ],
                    className="footer-text",
                ),
                html.P(
                    [
                        "See also: ",
                        html.A(
                            "QNMs Hall of Fame",
                            href="https://www.qnms.denys-dutykh.com/",
                            target="_blank",
                            rel="noopener noreferrer",
                        ),
                    ],
                    className="footer-text",
                ),
            ],
        ),
        # Hidden state & downloads
        dcc.Store(id="data-store", storage_type="session"),
        dcc.Store(id="theme-store", storage_type="local", data="light"),
        dcc.Store(id="convergence-store", storage_type="memory"),
        dcc.Store(id="slot-count-store", storage_type="session", data=INITIAL_SLOTS),
        dcc.Download(id="report-download"),
        dcc.Download(id="image-download"),
        dcc.Download(id="qnm-export-download"),
    ],
    id="app-container",
    className="light-theme",
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("app-container", "className"),
    Output("theme-store", "data"),
    Output("theme-toggle", "children"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
)
def toggle_theme(n_clicks, current_theme):
    """Switch between dark and light theme."""
    if n_clicks and n_clicks > 0:
        new_theme = "dark" if current_theme == "light" else "light"
    else:
        new_theme = current_theme or "light"
    css_class = "dark-theme" if new_theme == "dark" else "light-theme"
    icon = "\u2600\ufe0f" if new_theme == "dark" else "\U0001f319"
    return css_class, new_theme, icon


@callback(
    Output("upload-panel", "children"),
    Output("slot-count-store", "data"),
    Input("btn-add-slot", "n_clicks"),
    Input("btn-remove-slot", "n_clicks"),
    State("slot-count-store", "data"),
    prevent_initial_call=True,
)
def manage_slots(add_clicks, remove_clicks, current_count):
    """Add or remove upload slots dynamically."""
    count = current_count or INITIAL_SLOTS
    triggered = ctx.triggered_id
    if triggered == "btn-add-slot":
        count = min(count + 1, MAX_SLOTS)
    elif triggered == "btn-remove-slot":
        count = max(count - 1, 1)
    return [_make_upload_slot(i) for i in range(count)], count


@callback(
    Output("data-store", "data"),
    Output({"type": "filename", "index": ALL}, "children"),
    Output({"type": "resolution", "index": ALL}, "value"),
    Input({"type": "upload", "index": ALL}, "contents"),
    Input({"type": "clear", "index": ALL}, "n_clicks"),
    Input({"type": "resolution", "index": ALL}, "value"),
    Input("slot-count-store", "data"),
    State({"type": "upload", "index": ALL}, "filename"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def manage_data(
    upload_contents, clear_clicks, res_values, slot_count,
    filenames, store_data,
):
    """Handle uploads, clears, resolution edits, and slot count changes."""
    num_slots = len(upload_contents)
    store = store_data or {"slots": []}
    # Extend or shrink slots to match current slot count
    while len(store["slots"]) < num_slots:
        store["slots"].append(None)
    store["slots"] = store["slots"][:num_slots]

    triggered = ctx.triggered_id

    if triggered and isinstance(triggered, dict):
        idx = triggered["index"]
        action = triggered["type"]

        if action == "clear":
            store["slots"][idx] = None

        elif action == "upload" and upload_contents[idx]:
            try:
                re_vals, im_vals, inferred_n = parse_upload(
                    upload_contents[idx], filenames[idx]
                )
                if not re_vals:
                    raise ValueError("No valid data points")
                n = inferred_n or 0
                store["slots"][idx] = {
                    "filename": filenames[idx] or "unknown",
                    "resolution": n,
                    "re": re_vals,
                    "im": im_vals,
                }
            except Exception:
                pass

        elif action == "resolution":
            if store["slots"][idx] is not None and res_values[idx] is not None:
                store["slots"][idx]["resolution"] = int(res_values[idx])

    # Always rebuild labels/resolutions from the store (handles slot regeneration)
    fname_labels = [
        store["slots"][i]["filename"]
        if i < len(store["slots"]) and store["slots"][i]
        else "No file"
        for i in range(num_slots)
    ]
    res_out = [
        store["slots"][i]["resolution"]
        if i < len(store["slots"]) and store["slots"][i]
        else None
        for i in range(num_slots)
    ]

    return store, fname_labels, res_out


@callback(
    Output("qnm-plot", "figure"),
    Output("convergence-info", "children"),
    Output("convergence-store", "data"),
    Input("data-store", "data"),
    Input("tol-input", "value"),
    Input("legend-pos", "value"),
    Input("theme-store", "data"),
    State("qnm-plot", "relayoutData"),
)
def update_plot(store_data, tol_units, legend_pos, theme, relayout_data):
    """Rebuild the figure when data or controls change."""
    tol_value = (tol_units if tol_units and tol_units > 0 else 1.0) * 1e-4
    dark = theme == "dark"

    datasets = []
    if store_data and store_data.get("slots"):
        for slot in store_data["slots"]:
            if slot and slot.get("re") and slot.get("resolution"):
                datasets.append(slot)

    # Compute convergence once at callback level
    conv_data = None
    conv_re_arr, conv_im_arr = None, None
    if len(datasets) >= 2:
        sorted_ds = sorted(datasets, key=lambda d: d["resolution"])
        eigs_dict = {
            ds["resolution"]: np.column_stack([ds["re"], ds["im"]])
            for ds in sorted_ds
        }
        res_list = sorted(eigs_dict.keys())
        highest = res_list[-1]
        others = res_list[:-1]
        trees = {n: cKDTree(eigs_dict[n]) for n in others}
        conv_re_arr, conv_im_arr = compute_converged(
            eigs_dict[highest], trees, others, tol_value
        )
        general, pure_imag, pure_real = classify_converged(
            conv_re_arr, conv_im_arr, tol_value
        )
        conv_data = {
            "conv_re": conv_re_arr.tolist(),
            "conv_im": conv_im_arr.tolist(),
            "general": general.tolist() if len(general) > 0 else [],
            "pure_imag": pure_imag.tolist() if len(pure_imag) > 0 else [],
            "pure_real": pure_real.tolist() if len(pure_real) > 0 else [],
            "resolutions": res_list,
            "tol_value": tol_value,
        }

    fig, info_str = build_figure(
        datasets, tol_value, dark=dark,
        conv_re=conv_re_arr, conv_im=conv_im_arr,
    )
    fig.update_layout(legend=LEGEND_POSITIONS.get(legend_pos, {}))

    # Preserve zoom/pan
    if relayout_data:
        if "xaxis.range[0]" in relayout_data:
            fig.update_xaxes(
                range=[
                    relayout_data["xaxis.range[0]"],
                    relayout_data["xaxis.range[1]"],
                ]
            )
        if "yaxis.range[0]" in relayout_data:
            fig.update_yaxes(
                range=[
                    relayout_data["yaxis.range[0]"],
                    relayout_data["yaxis.range[1]"],
                ]
            )

    return fig, info_str, conv_data


@callback(
    Output("tol-input", "value", allow_duplicate=True),
    Output("tol-slider", "value"),
    Input("tol-input", "value"),
    Input("tol-slider", "value"),
    prevent_initial_call=True,
)
def sync_tol(input_val, slider_val):
    """Keep tolerance input and slider in sync."""
    triggered = ctx.triggered_id
    if triggered == "tol-input":
        val = input_val if input_val and input_val > 0 else 1.0
        return no_update, val
    if triggered == "tol-slider":
        return slider_val, no_update
    return no_update, no_update


@callback(
    Output("inspect-panel", "children"),
    Output("inspect-panel", "style"),
    Input("qnm-plot", "clickData"),
    State("convergence-store", "data"),
    prevent_initial_call=True,
)
def inspect_point(click_data, conv_data):
    """Display details for a clicked data point."""
    if not click_data or not click_data.get("points"):
        return no_update, no_update

    point = click_data["points"][0]
    re_val = point["x"]
    im_val = point["y"]
    label = point.get("customdata", "")

    # Check if this point is converged
    is_converged = False
    if conv_data and conv_data.get("conv_re"):
        tol = conv_data.get("tol_value", 1e-4)
        for cr, ci in zip(conv_data["conv_re"], conv_data["conv_im"]):
            if abs(cr - re_val) < tol and abs(ci - im_val) < tol:
                is_converged = True
                break

    children = [
        html.Strong("Selected QNM"),
        html.Span(f"  \u2014  {label}", style={"fontSize": "13px"}),
        html.Br(),
        html.Span(f"Re(\u03c9) = {re_val:.16e}"),
        html.Br(),
        html.Span(f"Im(\u03c9) = {im_val:.16e}"),
    ]
    if is_converged:
        children.append(html.Br())
        children.append(
            html.Span(
                "\u2714 Converged",
                style={"color": "red", "fontWeight": "bold"},
            )
        )

    return children, {"display": "block"}


@callback(
    Output("converged-table-container", "children"),
    Input("convergence-store", "data"),
    State("theme-store", "data"),
)
def update_converged_table(conv_data, theme):
    """Populate the converged QNMs DataTable."""
    if not conv_data or not conv_data.get("conv_re") or len(conv_data["conv_re"]) == 0:
        return []

    dark = theme == "dark"

    rows = []
    for arr, type_label in [
        (conv_data["general"], "General"),
        (conv_data["pure_imag"], "Purely imaginary"),
        (conv_data["pure_real"], "Purely real"),
    ]:
        if arr:
            for row in arr:
                rows.append({"re": row[0], "im": row[1], "type": type_label})

    if not rows:
        return []

    header_bg = "#333" if dark else "#f0f0f0"
    header_color = "#e0e0e0" if dark else "#222"
    cell_bg = "#2a2a2a" if dark else "white"
    cell_color = "#e0e0e0" if dark else "#222"
    border_color = "#444" if dark else "#ddd"

    table = dash_table.DataTable(
        id="converged-table",
        columns=[
            {
                "name": "Re(\u03c9)",
                "id": "re",
                "type": "numeric",
                "format": dash_table.Format.Format(
                    precision=12,
                    scheme=dash_table.Format.Scheme.exponent,
                ),
            },
            {
                "name": "Im(\u03c9)",
                "id": "im",
                "type": "numeric",
                "format": dash_table.Format.Format(
                    precision=12,
                    scheme=dash_table.Format.Scheme.exponent,
                ),
            },
            {"name": "Type", "id": "type"},
        ],
        data=rows,
        sort_action="native",
        style_header={
            "backgroundColor": header_bg,
            "color": header_color,
            "fontWeight": "bold",
            "fontFamily": "Computer Modern, Times New Roman, serif",
            "fontSize": "14px",
            "border": f"1px solid {border_color}",
        },
        style_cell={
            "backgroundColor": cell_bg,
            "color": cell_color,
            "fontFamily": "Computer Modern, Times New Roman, serif",
            "fontSize": "13px",
            "textAlign": "right",
            "padding": "6px 12px",
            "border": f"1px solid {border_color}",
        },
        style_cell_conditional=[
            {"if": {"column_id": "type"}, "textAlign": "center"},
        ],
        page_size=20,
        style_table={
            "overflowX": "auto",
            "maxWidth": "900px",
            "margin": "0 auto",
        },
    )

    title = html.H4(
        f"Converged QNMs ({len(rows)} total)",
        style={"textAlign": "center", "margin": "8px 0"},
    )
    return [title, table]


@callback(
    Output("report-download", "data"),
    Input("btn-report", "n_clicks"),
    State("convergence-store", "data"),
    prevent_initial_call=True,
)
def download_report(n_clicks, conv_data):
    """Generate and send the convergence report as a text download."""
    if not conv_data or not conv_data.get("conv_re"):
        return no_update
    report = generate_report_text(conv_data)
    return dcc.send_string(report, filename="qnm_report.txt")


@callback(
    Output("qnm-export-download", "data"),
    Input("btn-export-qnms", "n_clicks"),
    State("convergence-store", "data"),
    prevent_initial_call=True,
)
def export_converged_qnms(n_clicks, conv_data):
    """Export converged QNMs as a two-column .dat file."""
    if not conv_data or not conv_data.get("conv_re") or len(conv_data["conv_re"]) == 0:
        return no_update

    header = [
        f"# Converged QNMs (tolerance = {conv_data.get('tol_value', 1e-4):.1e})",
        f"# Resolutions: {', '.join(str(n) for n in conv_data.get('resolutions', []))}",
        f"# {'Re(omega)':>26s}  {'Im(omega)':>26s}",
    ]
    data_lines = [
        f"{r:>28.16e}  {i:>28.16e}"
        for r, i in zip(conv_data["conv_re"], conv_data["conv_im"])
    ]
    content = "\n".join(header + data_lines) + "\n"
    return dcc.send_string(content, filename="converged_qnms.dat")


@callback(
    Output("image-download", "data"),
    Input("btn-png", "n_clicks"),
    Input("btn-pdf", "n_clicks"),
    State("qnm-plot", "figure"),
    State("qnm-plot", "relayoutData"),
    prevent_initial_call=True,
)
def export_image(png_clicks, pdf_clicks, current_fig, relayout_data):
    """Export the current plot as PNG or PDF and send as a browser download."""
    if current_fig is None:
        return no_update

    triggered = ctx.triggered_id
    if triggered not in ("btn-png", "btn-pdf"):
        return no_update

    export_fig = go.Figure(current_fig)
    export_fig.update_layout(title_text="")

    # Apply current zoom/pan
    if relayout_data:
        if "xaxis.range[0]" in relayout_data:
            export_fig.update_xaxes(
                range=[
                    relayout_data["xaxis.range[0]"],
                    relayout_data["xaxis.range[1]"],
                ]
            )
        if "yaxis.range[0]" in relayout_data:
            export_fig.update_yaxes(
                range=[
                    relayout_data["yaxis.range[0]"],
                    relayout_data["yaxis.range[1]"],
                ]
            )

    if triggered == "btn-png":
        img_bytes = export_fig.to_image(format="png", scale=3)
        return dcc.send_bytes(img_bytes, filename="qnm_complex_plane.png")
    else:
        img_bytes = export_fig.to_image(format="pdf")
        return dcc.send_bytes(img_bytes, filename="qnm_complex_plane.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
