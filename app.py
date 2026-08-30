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
import logging
import math
import os

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
logger = logging.getLogger(__name__)

DEFAULT_TOL = 1e-4
DEFAULT_TOL_UNITS = 1.0
DEFAULT_LEGEND_POS = "Top-right"
INITIAL_SLOTS = 3
MAX_SLOTS = 6

# Upper bound on eigenvalues accepted from one file.  MAX_CONTENT_LENGTH caps
# the request body but not the cost of the KD-tree work that follows.
MAX_ROWS_PER_FILE = 100_000

# Axis titles used for PNG/PDF export.  The on-screen figure uses LaTeX that
# MathJax typesets in the browser; Kaleido has no MathJax, so it would write
# the markup out literally.
EXPORT_AXIS_TITLES = {"x": "Re(ω)", "y": "Im(ω)"}

# Box-drawing rule used in the text report.  Kept as a module constant so
# that the f-strings below hold no backslash escapes, which would require
# Python 3.12 (PEP 701).
_RULE = "\u2500"

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
    """Decode an uploaded file and return (re_list, im_list, inferred_N).

    Raises ValueError if the file carries more than MAX_ROWS_PER_FILE
    eigenvalues.  *inferred_N* is None when the filename holds no digits, so
    that the caller can ask the user for a resolution rather than guessing.
    """
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
        if len(re_vals) > MAX_ROWS_PER_FILE:
            raise ValueError(
                f"more than {MAX_ROWS_PER_FILE:,} eigenvalues; "
                "please coarsen the file before uploading"
            )

    # Take the LAST run of digits: "eigs_2024_90.dat" means N = 90, not 2024.
    inferred_n = None
    matches = _re.findall(r"(\d+)", filename or "")
    if matches:
        inferred_n = int(matches[-1])

    return re_vals, im_vals, inferred_n


def normalise_store_slots(store_data, num_slots):
    """Return a store dict with exactly *num_slots* entries in `slots`."""
    store = store_data or {"slots": []}
    slots = list(store.get("slots") or [])
    while len(slots) < num_slots:
        slots.append(None)
    return {"slots": slots[:num_slots]}


def apply_slot_action(
    store, triggered, upload_contents, filenames, res_values,
):
    """Mutate store for one UI action and return (store, upload_error_message)."""
    num_slots = len(upload_contents)
    upload_error = None

    if triggered == "btn-reset":
        store["slots"] = [None] * num_slots
        return store, upload_error

    if not (triggered and isinstance(triggered, dict)):
        return store, upload_error

    idx = triggered.get("index")
    action = triggered.get("type")
    # The pattern-matching id round-trips through the browser, so treat it as
    # untrusted: a negative index would otherwise address a slot from the end.
    if not isinstance(idx, int) or isinstance(idx, bool):
        return store, upload_error
    if not 0 <= idx < num_slots:
        return store, upload_error

    if action == "clear":
        store["slots"][idx] = None

    elif action == "upload" and upload_contents[idx]:
        try:
            re_vals, im_vals, inferred_n = parse_upload(
                upload_contents[idx], filenames[idx]
            )
            if not re_vals:
                raise ValueError("no valid numeric (Re, Im) rows found")
            store["slots"][idx] = {
                "filename": filenames[idx] or "unknown",
                "resolution": inferred_n,
                "re": re_vals,
                "im": im_vals,
            }
            if inferred_n is None:
                file_label = filenames[idx] or f"slot {idx + 1}"
                upload_error = (
                    f"Could not read a resolution from {file_label}: "
                    "please type N for this dataset."
                )
        except Exception as exc:
            store["slots"][idx] = None
            file_label = filenames[idx] or f"slot {idx + 1}"
            upload_error = f"Upload failed for {file_label}: {exc}"

    elif action == "resolution":
        if store["slots"][idx] is not None and res_values[idx] is not None:
            try:
                store["slots"][idx]["resolution"] = int(res_values[idx])
            except (TypeError, ValueError):
                pass

    return store, upload_error


def compute_converged(ref_points, trees, other_keys, tol_value):
    """Find QNMs in *ref_points* present in ALL other resolutions within *tol_value*.

    One vectorised KD-tree query per lower resolution, rather than one query per
    point per resolution: the per-point loop is what allowed a large upload to
    hold a worker past the gunicorn timeout.
    """
    ref_points = np.asarray(ref_points)
    if len(ref_points) == 0:
        return np.empty(0), np.empty(0)

    mask = np.ones(len(ref_points), dtype=bool)
    for n in other_keys:
        dist, _ = trees[n].query(ref_points)
        mask &= dist <= tol_value

    return ref_points[mask, 0], ref_points[mask, 1]


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
        "Note: only Re(\u03c9) \u2265 0 shown "
        "(spectrum symmetric about imaginary axis)",
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
            f"  {_RULE * 26}  {_RULE * 26}",
        ]
        for row in arr:
            tbl.append(f"  {row[0]:>26.16e}  {row[1]:>26.16e}")
        return tbl

    def fmt_imag_table(arr):
        """Format purely-imaginary QNMs with a Delta Im(omega) gap column."""
        col_w = 26
        tbl = [
            f"  {'Re(omega)':>{col_w}s}  {'Im(omega)':>{col_w}s}  {'Delta Im(omega)':>{col_w}s}",
            f"  {_RULE * col_w}  {_RULE * col_w}  {_RULE * col_w}",
        ]
        sorted_arr = arr[np.argsort(-arr[:, 1])]
        for i, row in enumerate(sorted_arr):
            if i == 0:
                tbl.append(f"  {row[0]:>{col_w}.16e}  {row[1]:>{col_w}.16e}")
            else:
                gap = abs(sorted_arr[i - 1, 1] - row[1])
                tbl.append(
                    f"  {row[0]:>{col_w}.16e}  {row[1]:>{col_w}.16e}  {gap:>{col_w}.16e}"
                )
        return tbl

    for label, arr, sort_fn in [
        (
            "General QNMs (Re != 0, Im != 0)",
            general,
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

    # Purely imaginary section with gap column
    lines.append("-" * 60)
    lines.append("Purely Imaginary QNMs (Re ~ 0)")
    lines.append("-" * 60)
    if len(pure_imag) > 0:
        lines.extend(fmt_imag_table(pure_imag))
    else:
        lines.append("  (none)")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------
# No external_scripts: dcc.Graph(mathjax=True) already serves MathJax v3 from
# the app's own /_dash-component-suites/ path, so nothing is fetched from a CDN
# and the page satisfies a script-src 'self' policy.
app = Dash(__name__, title="QNM Analyser")
server = app.server  # entry-point for gunicorn: app:server
server.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


@server.route("/health")
def health():
    """Cheap liveness probe for the reverse proxy."""
    return "OK", 200, {"Cache-Control": "no-store"}


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
        html.Div(
            id="upload-feedback",
            className="upload-feedback",
            style={"display": "none"},
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
                html.Button(
                    "Reset",
                    id="btn-reset",
                    n_clicks=0,
                    className="reset-btn",
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
    Input("btn-reset", "n_clicks"),
    State("slot-count-store", "data"),
    prevent_initial_call=True,
)
def manage_slots(add_clicks, remove_clicks, reset_clicks, current_count):
    """Add or remove upload slots dynamically."""
    count = current_count or INITIAL_SLOTS
    triggered = ctx.triggered_id
    if triggered == "btn-reset":
        count = INITIAL_SLOTS
    elif triggered == "btn-add-slot":
        count = min(count + 1, MAX_SLOTS)
    elif triggered == "btn-remove-slot":
        count = max(count - 1, 1)
    return [_make_upload_slot(i) for i in range(count)], count


@callback(
    Output("data-store", "data"),
    Output({"type": "filename", "index": ALL}, "children"),
    Output({"type": "resolution", "index": ALL}, "value"),
    Output("upload-feedback", "children"),
    Output("upload-feedback", "style"),
    Input({"type": "upload", "index": ALL}, "contents"),
    Input({"type": "clear", "index": ALL}, "n_clicks"),
    Input({"type": "resolution", "index": ALL}, "value"),
    Input("slot-count-store", "data"),
    Input("btn-reset", "n_clicks"),
    State({"type": "upload", "index": ALL}, "filename"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def manage_data(
    upload_contents, clear_clicks, res_values, slot_count, reset_clicks,
    filenames, store_data,
):
    """Handle uploads, clears, resolution edits, and slot count changes."""
    num_slots = len(upload_contents)
    store = normalise_store_slots(store_data, num_slots)
    store, upload_error = apply_slot_action(
        store=store,
        triggered=ctx.triggered_id,
        upload_contents=upload_contents,
        filenames=filenames,
        res_values=res_values,
    )

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

    feedback_children = upload_error if upload_error else ""
    feedback_style = {"display": "block"} if upload_error else {"display": "none"}

    return store, fname_labels, res_out, feedback_children, feedback_style


def _apply_relayout_ranges(fig, relayout_data):
    """Re-apply the client's zoom/pan window, ignoring anything non-numeric.

    *relayout_data* arrives from the browser, so every value is validated as a
    finite float before it reaches the figure.
    """
    if not relayout_data:
        return

    for axis, update in (("xaxis", fig.update_xaxes), ("yaxis", fig.update_yaxes)):
        lo_key, hi_key = f"{axis}.range[0]", f"{axis}.range[1]"
        if lo_key not in relayout_data or hi_key not in relayout_data:
            continue
        try:
            lo = float(relayout_data[lo_key])
            hi = float(relayout_data[hi_key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(lo) and math.isfinite(hi):
            update(range=[lo, hi])


def build_plot(store_data, tol_units, legend_pos, theme, relayout_data):
    """Build the figure, info string and convergence payload from stored data.

    Shared by the plot callback and the image-export callback so that both
    render a figure derived from the numeric datasets, never from figure JSON
    supplied by the client.
    """
    tol_value = (tol_units if tol_units and tol_units > 0 else 1.0) * 1e-4
    dark = theme == "dark"

    datasets = []
    if store_data and store_data.get("slots"):
        for slot in store_data["slots"]:
            if slot and slot.get("re") and slot.get("resolution") is not None:
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
        # Keep only Re(ω) ≥ -tol (spectrum symmetric about imaginary axis)
        nonneg = conv_re_arr >= -tol_value
        conv_re_arr = conv_re_arr[nonneg]
        conv_im_arr = conv_im_arr[nonneg]
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
    _apply_relayout_ranges(fig, relayout_data)

    return fig, info_str, conv_data


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
    return build_plot(store_data, tol_units, legend_pos, theme, relayout_data)


@callback(
    Output("tol-input", "value", allow_duplicate=True),
    Output("tol-slider", "value"),
    Input("tol-input", "value"),
    Input("tol-slider", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def sync_tol(input_val, slider_val, reset_clicks):
    """Keep tolerance input and slider in sync."""
    triggered = ctx.triggered_id
    if triggered == "btn-reset":
        return DEFAULT_TOL_UNITS, DEFAULT_TOL_UNITS
    if triggered == "tol-input":
        val = input_val if input_val and input_val > 0 else DEFAULT_TOL_UNITS
        return no_update, val
    if triggered == "tol-slider":
        return slider_val, no_update
    return no_update, no_update


@callback(
    Output("legend-pos", "value"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_legend(reset_clicks):
    """Restore legend position to its default."""
    return DEFAULT_LEGEND_POS


@callback(
    Output("qnm-plot", "relayoutData"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_plot_view(reset_clicks):
    """Clear any persisted zoom/pan state on reset."""
    return {}


@callback(
    Output("inspect-panel", "children"),
    Output("inspect-panel", "style"),
    Input("qnm-plot", "clickData"),
    Input("btn-reset", "n_clicks"),
    State("convergence-store", "data"),
    prevent_initial_call=True,
)
def inspect_point(click_data, reset_clicks, conv_data):
    """Display details for a clicked data point."""
    if ctx.triggered_id == "btn-reset":
        return [], {"display": "none"}
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
        for cr, ci in zip(conv_data["conv_re"], conv_data["conv_im"], strict=False):
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
        "# Note: only Re(omega) >= 0 (spectrum symmetric about imaginary axis)",
        f"# {'Re(omega)':>26s}  {'Im(omega)':>26s}",
    ]
    data_lines = [
        f"{r:>28.16e}  {i:>28.16e}"
        for r, i in zip(conv_data["conv_re"], conv_data["conv_im"], strict=False)
    ]
    content = "\n".join(header + data_lines) + "\n"
    return dcc.send_string(content, filename="converged_qnms.dat")


@callback(
    Output("image-download", "data"),
    Input("btn-png", "n_clicks"),
    Input("btn-pdf", "n_clicks"),
    State("data-store", "data"),
    State("tol-input", "value"),
    State("legend-pos", "value"),
    State("theme-store", "data"),
    State("qnm-plot", "relayoutData"),
    prevent_initial_call=True,
)
def export_image(
    png_clicks, pdf_clicks, store_data, tol_units, legend_pos, theme, relayout_data
):
    """Export the current plot as PNG or PDF and send as a browser download.

    The figure is rebuilt server-side from the stored numeric datasets rather
    than taken from the browser.  Kaleido renders through a headless browser,
    so handing it client-supplied figure JSON would let a crafted request point
    an image source at an internal address and have the server fetch it.
    """
    triggered = ctx.triggered_id
    if triggered not in ("btn-png", "btn-pdf"):
        return no_update

    export_fig, _, _ = build_plot(
        store_data, tol_units, legend_pos, theme, relayout_data
    )
    export_fig.update_layout(title_text="")

    # Kaleido renders through a headless browser but does not run MathJax, so
    # the LaTeX axis titles would appear verbatim in the file.  Substitute the
    # Unicode equivalents for export only; the on-screen figure keeps MathJax.
    export_fig.update_xaxes(title_text=EXPORT_AXIS_TITLES["x"])
    export_fig.update_yaxes(title_text=EXPORT_AXIS_TITLES["y"])

    try:
        if triggered == "btn-png":
            img_bytes = export_fig.to_image(format="png", scale=3)
            return dcc.send_bytes(img_bytes, filename="qnm_complex_plane.png")
        img_bytes = export_fig.to_image(format="pdf")
        return dcc.send_bytes(img_bytes, filename="qnm_complex_plane.pdf")
    except Exception:
        # Kaleido needs a working headless browser; a missing or broken one
        # must not take the worker down with it.
        logger.exception("Image export failed")
        return no_update


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # The Werkzeug debugger exposes an interactive console; keep it opt-in.
    app.run(debug=os.environ.get("DASH_DEBUG") == "1")
