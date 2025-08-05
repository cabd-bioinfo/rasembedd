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


def validate_data_and_columns(df, embeddings, id_column, color_column, species_column):
    """
    Validate that required columns exist and data is properly formatted.

    Args:
        df: DataFrame with metadata
        embeddings: Dictionary of embeddings
        id_column: Column name for protein IDs
        color_column: Column name for default coloring
        species_column: Column name for species

    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []

    # Check if metadata is empty
    if df.empty:
        errors.append(f"Metadata file '{METADATA_PATH}' is empty or could not be loaded.")
        return False, errors

    # Check if embeddings are empty
    if not embeddings:
        errors.append(f"Embeddings file '{EMBEDDINGS_PATH}' is empty or could not be loaded.")
        return False, errors

    # Check required columns exist in metadata
    if id_column not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        errors.append(
            f"ID column '{id_column}' not found in metadata.\n"
            f"Available columns: {available_cols}\n"
            f"Use --id_column to specify the correct column name."
        )

    if color_column not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        errors.append(
            f"Color column '{color_column}' not found in metadata.\n"
            f"Available columns: {available_cols}\n"
            f"Use --color_column to specify the correct column name."
        )

    # Check species column if specified
    if species_column and species_column not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        errors.append(
            f"Species column '{species_column}' not found in metadata.\n"
            f"Available columns: {available_cols}\n"
            f"Use --species_column to specify the correct column name or remove this option."
        )

    # If ID column exists, check for overlapping IDs
    if id_column in df.columns:
        metadata_ids = set(df[id_column])
        embedding_ids = set(embeddings.keys())
        common_ids = metadata_ids.intersection(embedding_ids)

        if not common_ids:
            errors.append(
                f"No common protein IDs found between metadata and embeddings.\n"
                f"Metadata contains {len(metadata_ids)} IDs, embeddings contain {len(embedding_ids)} IDs.\n"
                f"Check that the ID column '{id_column}' contains the correct protein identifiers."
            )
        elif len(common_ids) < 3:
            errors.append(
                f"Too few common protein IDs found ({len(common_ids)}).\n"
                f"At least 3 common IDs are required for visualization.\n"
                f"Metadata IDs: {len(metadata_ids)}, Embedding IDs: {len(embedding_ids)}, Common: {len(common_ids)}"
            )

    # Check if color column has reasonable number of unique values
    if color_column in df.columns:
        n_unique = df[color_column].nunique()
        if n_unique > 50:
            errors.append(
                f"Color column '{color_column}' has too many unique values ({n_unique}).\n"
                f"For visualization purposes, please choose a column with fewer categories (< 50)."
            )
        elif n_unique == 1:
            errors.append(
                f"Color column '{color_column}' has only one unique value.\n"
                f"This will result in all points having the same color. Consider using a different column."
            )

    return len(errors) == 0, errors


def print_error_and_exit(errors):
    """Print formatted error messages and exit the program."""
    print("\n" + "=" * 70)
    print("❌ ERROR: Interactive Visualization Setup Failed")
    print("=" * 70)

    for i, error in enumerate(errors, 1):
        print(f"\n{i}. {error}")

    print("\n" + "=" * 70)
    print("💡 Troubleshooting Tips:")
    print("  • Check that file paths are correct")
    print("  • Verify column names in your metadata file")
    print("  • Ensure protein IDs match between metadata and embeddings")
    print("  • Use --help to see all available options")
    print("=" * 70)

    import sys

    sys.exit(1)


# Load data with error handling
try:
    embeddings = load_embeddings(EMBEDDINGS_PATH)
except FileNotFoundError:
    print_error_and_exit([f"Embeddings file not found: {EMBEDDINGS_PATH}"])
except Exception as e:
    print_error_and_exit([f"Error loading embeddings file '{EMBEDDINGS_PATH}': {str(e)}"])

try:
    df = load_metadata(METADATA_PATH)
except FileNotFoundError:
    print_error_and_exit([f"Metadata file not found: {METADATA_PATH}"])
except Exception as e:
    print_error_and_exit([f"Error loading metadata file '{METADATA_PATH}': {str(e)}"])

# Validate data and columns
is_valid, errors = validate_data_and_columns(
    df, embeddings, ID_COLUMN, DEFAULT_COLOR_COLUMN, DEFAULT_SPECIES_COLUMN
)

if not is_valid:
    print_error_and_exit(errors)

# Store uploaded metadata in memory
uploaded_metadata = {"current": df}

# --- Filter to common IDs ---
print(f"📊 Data Summary:")
print(f"  • Loaded {len(embeddings)} protein embeddings")
print(f"  • Loaded metadata for {len(df)} proteins")

common_ids = set(df[ID_COLUMN]).intersection(embeddings.keys())
missing_in_metadata = set(embeddings.keys()) - set(df[ID_COLUMN])
missing_in_embeddings = set(df[ID_COLUMN]) - set(embeddings.keys())

print(f"  • Found {len(common_ids)} proteins with both metadata and embeddings")

if missing_in_metadata:
    print(f"  ⚠️  {len(missing_in_metadata)} proteins have embeddings but no metadata")
if missing_in_embeddings:
    print(f"  ⚠️  {len(missing_in_embeddings)} proteins have metadata but no embeddings")

# Additional validation for filtered data
if len(common_ids) < 3:
    print_error_and_exit(
        [
            f"Insufficient data for visualization after filtering.\n"
            f"Found only {len(common_ids)} proteins with both metadata and embeddings.\n"
            f"At least 3 proteins are required for meaningful visualization."
        ]
    )

df = df[df[ID_COLUMN].isin(common_ids)].copy()
df = df.set_index(ID_COLUMN).loc[list(common_ids)].reset_index()
emb_array = np.array([embeddings[uid] for uid in df[ID_COLUMN]])

# Validate embedding array
if emb_array.size == 0:
    print_error_and_exit(["Embedding array is empty after filtering."])
elif len(emb_array.shape) != 2:
    print_error_and_exit(
        [f"Embedding array has invalid shape: {emb_array.shape}. Expected 2D array."]
    )
elif emb_array.shape[1] < 2:
    print_error_and_exit(
        [f"Embeddings have insufficient dimensions: {emb_array.shape[1]}. Need at least 2."]
    )
elif np.isnan(emb_array).any():
    print_error_and_exit(["Embeddings contain NaN values."])
elif np.isinf(emb_array).any():
    print_error_and_exit(["Embeddings contain infinite values."])

print(f"✅ Successfully prepared {len(df)} proteins for visualization")
print(f"   Embedding dimensions: {emb_array.shape[1]}")
print(f"   Color by: {DEFAULT_COLOR_COLUMN} ({df[DEFAULT_COLOR_COLUMN].nunique()} categories)")
if DEFAULT_SPECIES_COLUMN in df.columns:
    print(
        f"   Species filter: {DEFAULT_SPECIES_COLUMN} ({df[DEFAULT_SPECIES_COLUMN].nunique()} species)"
    )
print()

# --- UMAP Projection ---

# --- Projection Methods ---
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import pacmap

    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False


def compute_projection(method, emb_array, n_components=2):
    if method == "UMAP":
        # Ensure n_neighbors is appropriate for small datasets
        n_neighbors = min(15, len(emb_array) - 1) if len(emb_array) > 1 else 1
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
        proj = reducer.fit_transform(emb_array)
    elif method == "PCA":
        reducer = PCA(n_components=n_components, random_state=42)
        proj = reducer.fit_transform(emb_array)
    elif method == "t-SNE":
        # Ensure perplexity is appropriate for small datasets
        perplexity = min(30, len(emb_array) - 1) if len(emb_array) > 1 else 1
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        proj = reducer.fit_transform(emb_array)
    elif method == "PaCMAP" and PACMAP_AVAILABLE:
        # Ensure n_neighbors is appropriate for small datasets
        n_neighbors = min(10, len(emb_array) - 1) if len(emb_array) > 1 else 1
        reducer = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        proj = reducer.fit_transform(emb_array)
    else:
        raise ValueError(f"Projection method '{method}' not supported or pacmap not installed.")
    return proj


# Default projection
DEFAULT_PROJECTION = "UMAP"
proj = compute_projection(DEFAULT_PROJECTION, emb_array, n_components=2)
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
                        html.Label("Dimensions:"),
                        dcc.Dropdown(
                            id="projection-dimensions",
                            options=[
                                {"label": "2D", "value": 2},
                                {"label": "3D", "value": 3},
                            ],
                            value=2,
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

            # Validate uploaded metadata
            if new_df.empty:
                report.append("❌ Uploaded metadata file is empty.")
                return (
                    html.Div([html.P(line) for line in report]),
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            if ID_COLUMN not in new_df.columns:
                available_cols = ", ".join(new_df.columns.tolist())
                report.append(f"❌ ID column '{ID_COLUMN}' not found in uploaded metadata.")
                report.append(f"Available columns: {available_cols}")
                return (
                    html.Div([html.P(line) for line in report]),
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            uploaded_metadata["current"] = new_df
            emb_ids = set(embeddings.keys())
            meta_ids = set(new_df[ID_COLUMN])
            missing_in_meta = emb_ids - meta_ids
            missing_in_emb = meta_ids - emb_ids
            common_ids = meta_ids.intersection(emb_ids)

            # Provide detailed feedback
            if not common_ids:
                report.append(
                    "❌ No common protein IDs found between uploaded metadata and embeddings!"
                )
                report.append(
                    f"Metadata contains {len(meta_ids)} IDs, embeddings contain {len(emb_ids)} IDs."
                )
                return (
                    html.Div([html.P(line, style={"color": "red"}) for line in report]),
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )
            elif len(common_ids) < 3:
                report.append(
                    f"⚠️ Only {len(common_ids)} common proteins found. Need at least 3 for visualization."
                )
                report.append(f"Metadata IDs: {len(meta_ids)}, Embedding IDs: {len(emb_ids)}")
                return (
                    html.Div([html.P(line, style={"color": "orange"}) for line in report]),
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                )

            # Success - provide feedback
            report.append(f"✅ Successfully loaded metadata with {len(new_df)} proteins")
            report.append(f"📊 Found {len(common_ids)} proteins with both metadata and embeddings")
            if missing_in_meta:
                report.append(f"ℹ️ {len(missing_in_meta)} proteins have embeddings but no metadata")
            if missing_in_emb:
                report.append(f"ℹ️ {len(missing_in_emb)} proteins have metadata but no embeddings")

            color_columns = [
                col for col in new_df.columns if new_df[col].nunique() < 30 and col != ID_COLUMN
            ]
            color_options = [{"label": col, "value": col} for col in color_columns]

            # Ensure the selected color column exists in the new table
            if current_value in [opt["value"] for opt in color_options]:
                color_value = current_value
            elif color_options:
                color_value = color_options[0]["value"]
                report.append(
                    f"ℹ️ Switched to color column '{color_value}' (original not available)"
                )
            else:
                color_value = None
                report.append("⚠️ No suitable color columns found (need < 30 categories)")

            species_column = (
                DEFAULT_SPECIES_COLUMN if DEFAULT_SPECIES_COLUMN in new_df.columns else None
            )
            if species_column:
                species_options = [
                    {"label": s, "value": s} for s in sorted(new_df[species_column].unique())
                ]
                species_value = [s["value"] for s in species_options]
                species_disabled = False
                report.append(f"✅ Species filtering available: {len(species_options)} species")
            else:
                species_options = []
                species_value = []
                species_disabled = True
                if DEFAULT_SPECIES_COLUMN:
                    report.append(
                        f"ℹ️ Species column '{DEFAULT_SPECIES_COLUMN}' not found in uploaded data"
                    )

        except Exception as e:
            report.append(f"❌ Error loading metadata: {str(e)}")
            return (
                html.Div([html.P(line, style={"color": "red"}) for line in report]),
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )
    # Handle embedding upload
    if embedding_contents and embedding_filenames:
        for content, fname in zip(embedding_contents, embedding_filenames):
            if not any(opt["value"] == fname for opt in options):
                content_string = content.split(",")[1]
                decoded = base64.b64decode(content_string)
                try:
                    emb = pickle.load(io.BytesIO(decoded))

                    # Validate uploaded embeddings
                    if not emb:
                        report.append(f"❌ Uploaded embeddings file '{fname}' is empty.")
                        continue

                    if not isinstance(emb, dict):
                        report.append(
                            f"❌ Embeddings file '{fname}' is not in the expected format (should be a dictionary)."
                        )
                        continue

                    # Check if embeddings contain valid numpy arrays
                    sample_embedding = next(iter(emb.values()))
                    if not isinstance(sample_embedding, np.ndarray):
                        report.append(f"❌ Embeddings in '{fname}' are not numpy arrays.")
                        continue

                    if len(sample_embedding.shape) != 1:
                        report.append(
                            f"❌ Embeddings in '{fname}' should be 1D arrays, got shape {sample_embedding.shape}."
                        )
                        continue

                    uploaded_embeddings[fname] = emb
                    options.append({"label": fname, "value": fname})
                    value = fname

                    meta_df = get_current_metadata()
                    emb_ids = set(emb.keys())
                    meta_ids = set(meta_df[ID_COLUMN])
                    missing_in_meta = emb_ids - meta_ids
                    missing_in_emb = meta_ids - emb_ids
                    common_ids = emb_ids.intersection(meta_ids)

                    # Provide detailed feedback
                    report.append(
                        f"✅ Successfully loaded embeddings '{fname}' with {len(emb)} proteins"
                    )
                    report.append(f"📊 Embedding dimensions: {sample_embedding.shape[0]}")

                    if not common_ids:
                        report.append(
                            f"❌ No common protein IDs found between '{fname}' and metadata!"
                        )
                        report.append(
                            f"Embedding IDs: {len(emb_ids)}, Metadata IDs: {len(meta_ids)}"
                        )
                    elif len(common_ids) < 3:
                        report.append(
                            f"⚠️ Only {len(common_ids)} common proteins found with '{fname}'. Need at least 3."
                        )
                    else:
                        report.append(
                            f"✅ Found {len(common_ids)} proteins with both embeddings and metadata"
                        )
                        if missing_in_meta:
                            report.append(
                                f"ℹ️ {len(missing_in_meta)} proteins in '{fname}' not in metadata"
                            )
                        if missing_in_emb:
                            report.append(
                                f"ℹ️ {len(missing_in_emb)} proteins in metadata not in '{fname}'"
                            )

                except Exception as e:
                    report.append(f"❌ Error loading embeddings file '{fname}': {str(e)}")
            else:
                report.append(f"ℹ️ Embeddings file '{fname}' already loaded")
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
        Input("projection-dimensions", "value"),
        Input("color-column", "value"),
        Input("species-filter", "value"),
        Input("embeddings-dropdown", "value"),
        Input("upload-annotation", "contents"),
    ],
    [State("upload-annotation", "filename")],
)
def update_plot(
    proj_method,
    n_dimensions,
    color_col,
    species_filter,
    emb_file,
    annotation_contents,
    annotation_filename,
):
    try:
        emb = uploaded_embeddings.get(emb_file, embeddings)
        dff = get_current_metadata()

        # Check if required columns exist
        if ID_COLUMN not in dff.columns:
            fig = px.scatter(x=[], y=[])
            fig.update_layout(title=f"❌ Error: ID column '{ID_COLUMN}' not found in metadata")
            return fig

        if color_col not in dff.columns:
            fig = px.scatter(x=[], y=[])
            fig.update_layout(title=f"❌ Error: Color column '{color_col}' not found in metadata")
            return fig

        common_ids = set(dff[ID_COLUMN]).intersection(emb.keys())

        if not common_ids:
            fig = px.scatter(x=[], y=[])
            fig.update_layout(
                title=f"❌ No common proteins found between metadata ({len(dff)} proteins) "
                f"and embeddings ({len(emb)} proteins).<br>"
                f"Check that protein IDs match between files."
            )
            return fig

        if len(common_ids) < 3:
            fig = px.scatter(x=[], y=[])
            fig.update_layout(
                title=f"❌ Too few common proteins ({len(common_ids)}) for visualization.<br>"
                f"Need at least 3 proteins with both metadata and embeddings."
            )
            return fig

        dff = dff[dff[ID_COLUMN].isin(common_ids)].copy()
        dff = dff.set_index(ID_COLUMN).loc[list(common_ids)].reset_index()
        emb_array = np.array([emb[uid] for uid in dff[ID_COLUMN]])

        # Robust validation
        error_msg = None
        if emb_array.size == 0:
            error_msg = "❌ Embedding array is empty after filtering"
        elif len(emb_array.shape) != 2:
            error_msg = (
                f"❌ Embeddings array has invalid shape: {emb_array.shape} (expected 2D array)"
            )
        elif emb_array.shape[1] < 2:
            error_msg = f"❌ Embeddings have insufficient dimensions: {emb_array.shape[1]} (need at least 2)"
        elif np.isnan(emb_array).any():
            error_msg = "❌ Embeddings contain NaN values"
        elif np.isinf(emb_array).any():
            error_msg = "❌ Embeddings contain infinite values"

        if error_msg:
            fig = px.scatter(x=[], y=[])
            fig.update_layout(title=error_msg)
            return fig

        # Attempt projection
        try:
            proj = compute_projection(proj_method, emb_array, n_components=n_dimensions)
            dff["X"] = proj[:, 0]
            dff["Y"] = proj[:, 1]
            if n_dimensions == 3:
                dff["Z"] = proj[:, 2]
        except Exception as e:
            fig = px.scatter(x=[], y=[])
            fig.update_layout(title=f"❌ {proj_method} projection failed: {str(e)}")
            return fig

        # Apply species filter if specified
        species_column = DEFAULT_SPECIES_COLUMN if DEFAULT_SPECIES_COLUMN in dff.columns else None
        if species_column and species_filter:
            before_filter = len(dff)
            dff = dff[dff[species_column].isin(species_filter)]
            if len(dff) == 0:
                fig = px.scatter(x=[], y=[])
                fig.update_layout(
                    title=f"❌ No proteins remain after species filtering.<br>"
                    f"Selected species not found in {before_filter} proteins."
                )
                return fig

        # Create the plot
        if n_dimensions == 3:
            fig = px.scatter_3d(
                dff,
                x="X",
                y="Y",
                z="Z",
                color=color_col,
                hover_data=[ID_COLUMN, species_column] if species_column else [ID_COLUMN],
                title=f"{proj_method} 3D Projection of {len(dff)} proteins colored by {color_col}",
                symbol=species_column if species_column else None,
            )
        else:
            fig = px.scatter(
                dff,
                x="X",
                y="Y",
                color=color_col,
                hover_data=[ID_COLUMN, species_column] if species_column else [ID_COLUMN],
                title=f"{proj_method} 2D Projection of {len(dff)} proteins colored by {color_col}",
                symbol=species_column if species_column else None,
            )

        # Style the plot
        if n_dimensions == 3:
            # Smaller markers for 3D visualization
            fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color="black")))
        else:
            # Standard size for 2D visualization
            fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="black")))

        if n_dimensions == 3:
            # 3D plot styling
            fig.update_layout(
                legend_title=color_col,
                autosize=True,
                margin=dict(l=20, r=20, t=40, b=20),
                scene=dict(
                    xaxis_title=f"{proj_method} 1",
                    yaxis_title=f"{proj_method} 2",
                    zaxis_title=f"{proj_method} 3",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                ),
                width=None,
                height=None,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
        else:
            # 2D plot styling
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
                    title=f"{proj_method} 1",
                ),
                yaxis=dict(
                    showline=True,
                    linewidth=2,
                    linecolor="black",
                    showgrid=True,
                    gridcolor="lightgray",
                    gridwidth=0.5,
                    zeroline=False,
                    title=f"{proj_method} 2",
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

    except Exception as e:
        # Catch-all error handler
        fig = px.scatter(x=[], y=[])
        fig.update_layout(title=f"❌ Unexpected error: {str(e)}")
        return fig


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
