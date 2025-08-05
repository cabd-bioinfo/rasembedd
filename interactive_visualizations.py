# --- CLI CONFIG ---
import argparse
import base64
import io
import os
import pickle

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from dash import Input, Output, dcc, html
from dash.dependencies import State

parser = argparse.ArgumentParser(description="Interactive Protein Embedding Visualization")
parser.add_argument(
    "--embeddings",
    default="RAS/embeddings/prost_t5_embeddings.pkl",
    help="Path to embeddings file (.pkl)",
)
parser.add_argument(
    "--metadata",
    default="test/test_species.tsv",
    help="Path to metadata file (.tsv or .csv)",
)
parser.add_argument("--id_column", default="uniprot_id", help="Column name for sequence IDs")
parser.add_argument("--color_column", default="Family.name", help="Default column for coloring")
parser.add_argument("--species_column", default="species", help="Column name for species")
args, unknown = parser.parse_known_args()

EMBEDDINGS_PATH = args.embeddings
METADATA_PATH = args.metadata
ID_COLUMN = args.id_column
DEFAULT_COLOR_COLUMN = args.color_column
DEFAULT_SPECIES_COLUMN = args.species_column


# --- LOAD DATA ---
def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metadata(path_or_buffer):
    if isinstance(path_or_buffer, str):
        sep = "\t" if path_or_buffer.endswith(".tsv") else ","
        return pd.read_csv(path_or_buffer, sep=sep)
    else:
        # Assume buffer - first try to determine if it's tab or comma separated
        # by peeking at the first line
        current_pos = path_or_buffer.tell()
        first_line = path_or_buffer.readline()
        path_or_buffer.seek(current_pos)

        # Determine separator based on content
        if "\t" in first_line and "," not in first_line:
            sep = "\t"
        elif "," in first_line:
            sep = ","
        else:
            # Default to comma
            sep = ","

        return pd.read_csv(path_or_buffer, sep=sep)


embeddings = load_embeddings(EMBEDDINGS_PATH)
df = load_metadata(METADATA_PATH)

# Store uploaded metadata in memory
uploaded_metadata = {"current": df}

# --- Filter to common IDs ---
common_ids = set(df[ID_COLUMN]).intersection(embeddings.keys())
df = df[df[ID_COLUMN].isin(common_ids)].copy()
df = df.set_index(ID_COLUMN).loc[list(common_ids)].reset_index()
emb_array = np.array([embeddings[uid] for uid in df[ID_COLUMN]])

# --- UMAP Projection ---

# --- Projection Methods ---
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import pacmap

    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False


def compute_projection(method, emb_array):
    if method == "UMAP":
        # Ensure n_neighbors is appropriate for small datasets
        n_neighbors = min(15, len(emb_array) - 1) if len(emb_array) > 1 else 1
        reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
        proj = reducer.fit_transform(emb_array)
    elif method == "PCA":
        reducer = PCA(n_components=2, random_state=42)
        proj = reducer.fit_transform(emb_array)
    elif method == "t-SNE":
        # Ensure perplexity is appropriate for small datasets
        perplexity = min(30, len(emb_array) - 1) if len(emb_array) > 1 else 1
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        proj = reducer.fit_transform(emb_array)
    elif method == "PaCMAP" and PACMAP_AVAILABLE:
        # Ensure n_neighbors is appropriate for small datasets
        n_neighbors = min(10, len(emb_array) - 1) if len(emb_array) > 1 else 1
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        proj = reducer.fit_transform(emb_array)
    else:
        raise ValueError(f"Projection method '{method}' not supported or pacmap not installed.")
    return proj


# Default projection
DEFAULT_PROJECTION = "UMAP"
proj = compute_projection(DEFAULT_PROJECTION, emb_array)
df["X"] = proj[:, 0]
df["Y"] = proj[:, 1]

# --- DASH APP ---
app = dash.Dash(__name__)
server = app.server

color_columns = [col for col in df.columns if df[col].nunique() < 30 and col != ID_COLUMN]
species_column = DEFAULT_SPECIES_COLUMN if DEFAULT_SPECIES_COLUMN in df.columns else None
species_options = (
    [{"label": s, "value": s} for s in sorted(df[species_column].unique())]
    if species_column
    else []
)

# Store uploaded embeddings in memory
uploaded_embeddings = {EMBEDDINGS_PATH: embeddings}

app.layout = html.Div(
    [
        html.H2("Interactive Projection"),
        html.Div(
            [
                dcc.Graph(id="projection-plot", style={"width": "100vw", "height": "90vh"}),
                html.Div(
                    [
                        html.Label("Select embeddings:"),
                        dcc.Dropdown(
                            id="embeddings-dropdown",
                            options=[
                                {
                                    "label": os.path.basename(EMBEDDINGS_PATH),
                                    "value": EMBEDDINGS_PATH,
                                }
                            ],
                            value=EMBEDDINGS_PATH,
                            clearable=False,
                        ),
                        html.Label("Projection method:"),
                        dcc.Dropdown(
                            id="projection-method",
                            options=[
                                {"label": m, "value": m}
                                for m in ["UMAP", "PCA", "t-SNE"]
                                + (["PaCMAP"] if PACMAP_AVAILABLE else [])
                            ],
                            value=DEFAULT_PROJECTION,
                            clearable=False,
                        ),
                        html.Label("Classification column:"),
                        dcc.Dropdown(
                            id="color-column",
                            options=[{"label": col, "value": col} for col in color_columns],
                            value=DEFAULT_COLOR_COLUMN,
                            clearable=False,
                        ),
                        html.Label("Filter species:"),
                        dcc.Dropdown(
                            id="species-filter",
                            options=species_options if species_column else [],
                            value=([s["value"] for s in species_options] if species_column else []),
                            multi=True,
                            disabled=not bool(species_column),
                        ),
                        html.Hr(
                            style={
                                "marginTop": "20px",
                                "marginBottom": "20px",
                                "borderColor": "#bbb",
                            }
                        ),
                        html.Div(
                            [
                                html.Label("Upload annotation file (.tsv or .csv):"),
                                dcc.Upload(
                                    id="upload-annotation",
                                    children=html.Button("Upload Annotation"),
                                    multiple=False,
                                ),
                                html.Label("Upload embeddings file(s) (.pkl):"),
                                dcc.Upload(
                                    id="upload-embeddings",
                                    children=html.Button("Upload Embeddings"),
                                    multiple=True,
                                ),
                                html.Div(
                                    id="upload-report",
                                    style={"marginTop": "10px", "color": "red"},
                                ),
                            ],
                            style={"marginTop": "10px"},
                        ),
                    ],
                    style={
                        "width": "20vw",
                        "float": "right",
                        "marginRight": "2vw",
                        "background": "#f9f9f9",
                        "padding": "10px",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
                        "fontFamily": '"Segoe UI", "Helvetica Neue", Arial, "Liberation Sans", sans-serif',
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "fontFamily": '"Segoe UI", "Helvetica Neue", Arial, "Liberation Sans", sans-serif',
            },
        ),
    ],
    style={"fontFamily": '"Segoe UI", "Helvetica Neue", Arial, "Liberation Sans", sans-serif'},
)


@app.callback(
    [
        Output("upload-report", "children"),
        Output("color-column", "options"),
        Output("color-column", "value"),
        Output("species-filter", "options"),
        Output("species-filter", "value"),
        Output("species-filter", "disabled"),
        Output("embeddings-dropdown", "options"),
        Output("embeddings-dropdown", "value"),
    ],
    [Input("upload-annotation", "contents"), Input("upload-embeddings", "contents")],
    [
        State("upload-annotation", "filename"),
        State("upload-embeddings", "filename"),
        State("embeddings-dropdown", "options"),
        State("embeddings-dropdown", "value"),
    ],
)
def handle_uploads(
    annotation_contents,
    embedding_contents,
    annotation_filename,
    embedding_filenames,
    current_options,
    current_value,
):
    report = []
    color_options = dash.no_update
    color_value = dash.no_update
    species_options = dash.no_update
    species_value = dash.no_update
    species_disabled = dash.no_update
    options = (
        current_options
        if current_options
        else [{"label": os.path.basename(EMBEDDINGS_PATH), "value": EMBEDDINGS_PATH}]
    )
    value = current_value if current_value else EMBEDDINGS_PATH
    # Handle annotation upload
    if annotation_contents and annotation_filename:
        content_string = annotation_contents.split(",")[1]
        decoded = base64.b64decode(content_string)
        buffer = io.BytesIO(decoded)
        try:
            new_df = load_metadata(buffer)
            uploaded_metadata["current"] = new_df
            emb_ids = set(embeddings.keys())
            meta_ids = set(new_df[ID_COLUMN])
            missing_in_meta = emb_ids - meta_ids
            missing_in_emb = meta_ids - emb_ids
            if missing_in_meta:
                report.append(f"Proteins in embeddings not in metadata: {len(missing_in_meta)}")
            if missing_in_emb:
                report.append(f"Proteins in metadata not in embeddings: {len(missing_in_emb)}")
            if not report:
                report.append("All proteins matched between embeddings and metadata.")
            color_columns = [
                col for col in new_df.columns if new_df[col].nunique() < 30 and col != ID_COLUMN
            ]
            color_options = [{"label": col, "value": col} for col in color_columns]
            # Ensure the selected color column exists in the new table
            if current_value in [opt["value"] for opt in color_options]:
                color_value = current_value
            elif color_options:
                color_value = color_options[0]["value"]
            else:
                color_value = None
            species_column = (
                DEFAULT_SPECIES_COLUMN if DEFAULT_SPECIES_COLUMN in new_df.columns else None
            )
            if species_column:
                species_options = [
                    {"label": s, "value": s} for s in sorted(new_df[species_column].unique())
                ]
                species_value = [s["value"] for s in species_options]
                species_disabled = False
            else:
                species_options = []
                species_value = []
                species_disabled = True
        except Exception as e:
            report.append(f"Error loading metadata: {str(e)}")
    # Handle embedding upload
    if embedding_contents and embedding_filenames:
        for content, fname in zip(embedding_contents, embedding_filenames):
            if not any(opt["value"] == fname for opt in options):
                content_string = content.split(",")[1]
                decoded = base64.b64decode(content_string)
                try:
                    emb = pickle.load(io.BytesIO(decoded))
                    uploaded_embeddings[fname] = emb
                    options.append({"label": fname, "value": fname})
                    value = fname
                    meta_df = get_current_metadata()
                    emb_ids = set(emb.keys())
                    meta_ids = set(meta_df[ID_COLUMN])
                    missing_in_meta = emb_ids - meta_ids
                    missing_in_emb = meta_ids - emb_ids
                    if missing_in_meta:
                        report.append(
                            f"Proteins in embeddings not in metadata: {len(missing_in_meta)}"
                        )
                    if missing_in_emb:
                        report.append(
                            f"Proteins in metadata not in embeddings: {len(missing_in_emb)}"
                        )
                    if not report:
                        report.append("All proteins matched between embeddings and metadata.")
                except Exception:
                    report.append(f"Error loading embeddings file: {fname}")
    return (
        html.Div([html.P(line) for line in report]) if report else dash.no_update,
        color_options,
        color_value,
        species_options,
        species_value,
        species_disabled,
        options,
        value,
    )


def get_current_metadata():
    return uploaded_metadata.get("current", df)


# Update plot to use uploaded metadata
@app.callback(
    Output("projection-plot", "figure"),
    [
        Input("projection-method", "value"),
        Input("color-column", "value"),
        Input("species-filter", "value"),
        Input("embeddings-dropdown", "value"),
        Input("upload-annotation", "contents"),
    ],
    [State("upload-annotation", "filename")],
)
def update_plot(
    proj_method,
    color_col,
    species_filter,
    emb_file,
    annotation_contents,
    annotation_filename,
):
    emb = uploaded_embeddings.get(emb_file, embeddings)
    dff = get_current_metadata()
    common_ids = set(dff[ID_COLUMN]).intersection(emb.keys())
    dff = dff[dff[ID_COLUMN].isin(common_ids)].copy()
    dff = dff.set_index(ID_COLUMN).loc[list(common_ids)].reset_index()
    emb_array = np.array([emb[uid] for uid in dff[ID_COLUMN]])

    # Robust validation
    error_msg = None
    if emb_array.size == 0:
        error_msg = "No common proteins found between metadata and embeddings. Cannot project."
    elif len(emb_array.shape) != 2 or emb_array.shape[1] < 2:
        error_msg = f"Embeddings array is malformed: shape {emb_array.shape}."
    elif np.isnan(emb_array).any() or np.isinf(emb_array).any():
        error_msg = "Embeddings contain NaN or inf values."

    if error_msg:
        fig = px.scatter(x=[], y=[])
        fig.update_layout(title=error_msg)
        return fig

    try:
        proj = compute_projection(proj_method, emb_array)
        dff["X"] = proj[:, 0]
        dff["Y"] = proj[:, 1]
    except Exception as e:
        fig = px.scatter(x=[], y=[])
        fig.update_layout(title=f"Projection error: {str(e)}")
        return fig

    species_column = DEFAULT_SPECIES_COLUMN if DEFAULT_SPECIES_COLUMN in dff.columns else None
    if species_column and species_filter:
        dff = dff[dff[species_column].isin(species_filter)]
    fig = px.scatter(
        dff,
        x="X",
        y="Y",
        color=color_col,
        hover_data=[ID_COLUMN, species_column] if species_column else [ID_COLUMN],
        title=f"{proj_method} Projection colored by {color_col}",
        symbol=species_column if species_column else None,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="black")))
    fig.update_layout(
        legend_title=color_col,
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            scaleanchor="y",
            scaleratio=1,
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            zeroline=False,
            title="X",
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor="black",
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            zeroline=False,
            title="Y",
        ),
        width=None,
        height=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="black", width=2),
                fillcolor="rgba(0,0,0,0)",
            )
        ],
    )
    return fig


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
