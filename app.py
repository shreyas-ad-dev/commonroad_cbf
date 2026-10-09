import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration & Page Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulation Log & Frame Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS to ensure the slider container fills available width without padding
st.markdown("""
    <style>
    div[data-testid="stSlider"] {
        padding-left: 0px !important;
        padding-right: 0px !important;
    }
    div[data-baseweb="slider"] {
        margin-left: 0px !important;
        margin-right: 0px !important;
        padding-left: 0px !important;
        padding-right: 0px !important;
    }
    .stButton button {
        height: 38px;
        line-height: 1;
        padding: 0 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚦 Autonomous Driving Data Logger & Frame Visualizer")

# -----------------------------------------------------------------------------
# Helper Functions for Data & Image Processing
# -----------------------------------------------------------------------------
def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Recursively flattens nested data structures for signal selection."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], (int, float)):
                items.append((new_key, v))
            elif len(v) > 0 and isinstance(v[0], dict):
                for idx, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(flatten_dict(item, f"{new_key}[{idx}]", sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

@st.cache_data
def load_jsonl_log(jsonl_path: str):
    """Parses JSONL simulation data into flattened structured records."""
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            step = entry.get("step")
            timestamp = entry.get("timestamp")
            payload = entry.get("data", {})
            
            flat_data = flatten_dict(payload)
            flat_data["step"] = step
            flat_data["timestamp"] = timestamp
            records.append(flat_data)
            
    df = pd.DataFrame(records)
    return df

def get_gif_frame(gif_path: str, frame_index: int) -> Image.Image:
    """Extracts a specific static frame image from an animated GIF."""
    img = Image.open(gif_path)
    num_frames = getattr(img, "n_frames", 1)
    target_frame = frame_index % num_frames if num_frames > 0 else 0
    img.seek(target_frame)
    return img.convert("RGB")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
root_dir = Path(".")
json_files = sorted([p.name for p in root_dir.glob("*.json*")])
gif_files = sorted([p.name for p in root_dir.glob("*.gif")])

st.sidebar.header("⚙️ Mode Setup")
compare_runs = st.sidebar.checkbox("Compare Runs", value=False)

if compare_runs:
    st.sidebar.checkbox("Separate Control", value=False, key="separate_control_active")


st.sidebar.markdown("---")
st.sidebar.header("📁 File & Settings Config")

default_json_idx1 = json_files.index("log_zam32.jsonl") if "log_zam32.jsonl" in json_files else 0
default_gif_idx1 = gif_files.index("zam_zip32_v2_merge.gif") if "zam_zip32_v2_merge.gif" in gif_files else 0

if compare_runs:
    st.sidebar.subheader("Run 1 Config")

if json_files:
    jsonl_file1 = st.sidebar.selectbox("Select JSON/JSONL Log (Run 1):", options=json_files, index=default_json_idx1, key="json1")
else:
    jsonl_file1 = st.sidebar.text_input("JSONL Path (Run 1):", value="log_zam32.jsonl", key="json1_txt")

if gif_files:
    gif_file1 = st.sidebar.selectbox("Select GIF File (Run 1):", options=gif_files, index=default_gif_idx1, key="gif1")
else:
    gif_file1 = st.sidebar.text_input("GIF Path (Run 1):", value="zam_zip32_v2_merge.gif", key="gif1_txt")

df1 = load_jsonl_log(jsonl_file1)
total_steps1 = len(df1)

df2 = None
total_steps2 = 0
gif_file2 = None

if compare_runs:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Run 2 Config")
    
    default_json_idx2 = (default_json_idx1 + 1) if len(json_files) > 1 else default_json_idx1
    default_gif_idx2 = (default_gif_idx1 + 1) if len(gif_files) > 1 else default_gif_idx1

    if json_files:
        jsonl_file2 = st.sidebar.selectbox("Select JSON/JSONL Log (Run 2):", options=json_files, index=default_json_idx2, key="json2")
    else:
        jsonl_file2 = st.sidebar.text_input("JSONL Path (Run 2):", value="log_zam32.jsonl", key="json2_txt")

    if gif_files:
        gif_file2 = st.sidebar.selectbox("Select GIF File (Run 2):", options=gif_files, index=default_gif_idx2, key="gif2")
    else:
        gif_file2 = st.sidebar.text_input("GIF Path (Run 2):", value="zam_zip32_v2_merge.gif", key="gif2_txt")

    df2 = load_jsonl_log(jsonl_file2)
    total_steps2 = len(df2)

# Signal Selection
st.sidebar.markdown("---")
st.sidebar.header("📊 Signal Selection")

numeric_cols1 = [
    col for col in df1.columns 
    if col not in ["step", "timestamp"] and pd.api.types.is_numeric_dtype(df1[col])
]

if df2 is not None:
    numeric_cols2 = [
        col for col in df2.columns 
        if col not in ["step", "timestamp"] and pd.api.types.is_numeric_dtype(df2[col])
    ]
    available_signals = sorted(list(set(numeric_cols1).union(set(numeric_cols2))))
else:
    available_signals = numeric_cols1

default_selected = [
    col for col in available_signals 
    if any(k in col for k in ["ego.velocity", "steering.steering", "ego.orientation_deg"])
]

selected_signals = st.sidebar.multiselect(
    "Select signals to plot:",
    options=available_signals,
    default=default_selected if default_selected else available_signals[:2]
)

# Session state initialization
if "current_step1" not in st.session_state:
    st.session_state.current_step1 = 0
if "current_step2" not in st.session_state:
    st.session_state.current_step2 = 0
if "sync_step" not in st.session_state:
    st.session_state.sync_step = 0

max_total_steps = max(total_steps1, total_steps2) if compare_runs else total_steps1

# Callback Functions that directly modify slider keys
def increment_step1():
    if st.session_state.current_step1 < total_steps1 - 1:
        st.session_state.current_step1 += 1

def decrement_step1():
    if st.session_state.current_step1 > 0:
        st.session_state.current_step1 -= 1

def increment_step2():
    if st.session_state.current_step2 < total_steps2 - 1:
        st.session_state.current_step2 += 1

def decrement_step2():
    if st.session_state.current_step2 > 0:
        st.session_state.current_step2 -= 1

def increment_both():
    if st.session_state.sync_step < max_total_steps - 1:
        st.session_state.sync_step += 1
        st.session_state.current_step1 = min(st.session_state.sync_step, total_steps1 - 1)
        if compare_runs:
            st.session_state.current_step2 = min(st.session_state.sync_step, total_steps2 - 1)

def decrement_both():
    if st.session_state.sync_step > 0:
        st.session_state.sync_step -= 1
        st.session_state.current_step1 = min(st.session_state.sync_step, total_steps1 - 1)
        if compare_runs:
            st.session_state.current_step2 = min(st.session_state.sync_step, total_steps2 - 1)

def on_sync_slider_change():
    val = st.session_state.sync_step
    st.session_state.current_step1 = min(val, total_steps1 - 1)
    if compare_runs:
        st.session_state.current_step2 = min(val, total_steps2 - 1)

# -----------------------------------------------------------------------------
# 1. Real-Time Signal Plot (Placed at Top)
# -----------------------------------------------------------------------------
st.subheader("📈 Real-time Signal Plots")

current_step1 = min(st.session_state.current_step1, total_steps1 - 1)
step_row1 = df1[df1["step"] == current_step1].iloc[0]

if compare_runs:
    current_step2 = min(st.session_state.current_step2, total_steps2 - 1)
    step_row2 = df2[df2["step"] == current_step2].iloc[0]
else:
    current_step2 = current_step1
    step_row2 = step_row1

if selected_signals:
    fig = go.Figure()
    
    for sig in selected_signals:
        if sig in df1.columns:
            fig.add_trace(go.Scatter(
                x=df1["step"],
                y=df1[sig],
                mode="lines",
                name=f"{sig} (Run 1)",
                opacity=0.8
            ))
            fig.add_trace(go.Scatter(
                x=[current_step1],
                y=[step_row1[sig]],
                mode="markers",
                marker=dict(size=10, symbol="circle"),
                name=f"{sig} (Run 1 Current)",
                showlegend=False
            ))

        if compare_runs and df2 is not None and sig in df2.columns:
            fig.add_trace(go.Scatter(
                x=df2["step"],
                y=df2[sig],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{sig} (Run 2)",
                opacity=0.8
            ))
            fig.add_trace(go.Scatter(
                x=[current_step2],
                y=[step_row2[sig]],
                mode="markers",
                marker=dict(size=10, symbol="x"),
                name=f"{sig} (Run 2 Current)",
                showlegend=False
            ))

    fig.add_vline(x=current_step1, line_width=2, line_dash="dash", line_color="red", annotation_text="Run 1")
    if compare_runs:
        fig.add_vline(x=current_step2, line_width=2, line_dash="dot", line_color="blue", annotation_text="Run 2")

    fig.update_layout(
        xaxis=dict(
            title="Step",
            range=[0, max_total_steps - 1],
            autorange=False,
            fixedrange=True,
            showgrid=True,
            zeroline=False
        ),
        yaxis_title="Value",
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.18,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(0, 0, 0, 0.0)"
        ),
        margin=dict(l=48, r=48, t=40, b=30),
        height=380
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
else:
    st.info("Please select one or more signals from the sidebar to display plots.")

# -----------------------------------------------------------------------------
# 2. Base Sliders & Control (Fixed Callbacks)
# -----------------------------------------------------------------------------
separate_control = st.session_state.get("separate_control_active", False)

if not separate_control:
    col_prev, col_step, col_next = st.columns([0.4, 12, 0.4])
    with col_prev:
        st.button("◀", on_click=decrement_both, use_container_width=True)
    with col_next:
        st.button("▶", on_click=increment_both, use_container_width=True)
    with col_step:
        st.slider(
            "Synchronized Step / Frame Index",
            min_value=0,
            max_value=max_total_steps - 1,
            key="sync_step",
            on_change=on_sync_slider_change,
            label_visibility="collapsed"
        )
else:
    # Stacked Sliders
    st.markdown("**Run 1 Playback Control**")
    col_p1, col_s1, col_n1 = st.columns([0.4, 12, 0.4])
    with col_p1:
        st.button("◀", on_click=decrement_step1, use_container_width=True, key="prev_r1")
    with col_n1:
        st.button("▶", on_click=increment_step1, use_container_width=True, key="next_r1")
    with col_s1:
        st.slider(
            "Step / Frame (Run 1)",
            min_value=0,
            max_value=total_steps1 - 1,
            key="current_step1",
            label_visibility="collapsed"
        )

    st.markdown("**Run 2 Playback Control**")
    col_p2, col_s2, col_n2 = st.columns([0.4, 12, 0.4])
    with col_p2:
        st.button("◀", on_click=decrement_step2, use_container_width=True, key="prev_r2")
    with col_n2:
        st.button("▶", on_click=increment_step2, use_container_width=True, key="next_r2")
    with col_s2:
        st.slider(
            "Step / Frame (Run 2)",
            min_value=0,
            max_value=total_steps2 - 1,
            key="current_step2",
            label_visibility="collapsed"
        )

# -----------------------------------------------------------------------------
# 3. Frame Visualization
# -----------------------------------------------------------------------------
st.subheader("🖼️ Frame Visualization")

if compare_runs:
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown(f"**Run 1:** `{Path(jsonl_file1).name}`")
        if Path(gif_file1).exists():
            frame1 = get_gif_frame(gif_file1, current_step1)
            st.image(frame1, caption=f"Run 1 Frame: {current_step1}", use_container_width=True)
        else:
            st.warning(f"File `{gif_file1}` not found.")
    with img_col2:
        st.markdown(f"**Run 2:** `{Path(jsonl_file2).name}`")
        if Path(gif_file2).exists():
            frame2 = get_gif_frame(gif_file2, current_step2)
            st.image(frame2, caption=f"Run 2 Frame: {current_step2}", use_container_width=True)
        else:
            st.warning(f"File `{gif_file2}` not found.")
else:
    if Path(gif_file1).exists():
        extracted_frame = get_gif_frame(gif_file1, current_step1)
        st.image(extracted_frame, caption=f"GIF Frame Index: {current_step1}", use_container_width=True)
    else:
        st.warning(f"File `{gif_file1}` not found.")

# -----------------------------------------------------------------------------
# 4. Step Data Inspector
# -----------------------------------------------------------------------------
def build_nested_dict(flat_dict: dict) -> dict:
    """
    Recursively transforms a dictionary with dot-notation keys
    (e.g., 'ego.velocity.x': 5) into a nested dictionary structure,
    handling edge cases where a key exists as both a scalar and a sub-path.
    """
    nested = {}
    for key, val in flat_dict.items():
        parts = str(key).split(".")
        current = nested
        
        for i, part in enumerate(parts[:-1]):
            # If the key doesn't exist or is a scalar, re-initialize it as a dict
            if part not in current or not isinstance(current[part], dict):
                if part in current:
                    # Move existing scalar value to a reserved key so it isn't lost
                    existing_val = current[part]
                    current[part] = {"_value": existing_val}
                else:
                    current[part] = {}
            current = current[part]
            
        leaf = parts[-1]
        # If leaf already contains a dictionary built by earlier paths, store under '_value'
        if leaf in current and isinstance(current[leaf], dict):
            current[leaf]["_value"] = val
        else:
            current[leaf] = val
            
    return nested

def render_dynamic_json_expanders(data: dict):
    """
    Recursively renders nested dictionaries directly into expanders,
    rendering leaf dictionaries as clean JSON key-value pairs without tables or extra levels.
    """
    for key, value in data.items():
        key_label = str(key).replace("_", " ").title()
        
        if isinstance(value, dict) and value:
            # Check if any child elements are themselves dictionaries
            has_nested_children = any(isinstance(v, dict) for v in value.values())
            
            with st.expander(f"{key_label}", expanded=False):
                if has_nested_children:
                    # Recurse down for nested dicts
                    render_dynamic_json_expanders(value)
                else:
                    # Leaf dictionary: renders key-value pairs directly inline
                    st.json(value)
        else:
            # Standalone scalar field fallback
            st.write(f"**{key_label}:** `{value}`")

with st.expander("🔍 Inspect Raw Log Entry Data for Current Step", expanded=False):
    if compare_runs:
        insp_col1, insp_col2 = st.columns(2)
        
        with insp_col1:
            st.markdown("### Run 1 Data")
            row1_nested = build_nested_dict(step_row1.to_dict())
            render_dynamic_json_expanders(row1_nested)

        with insp_col2:
            st.markdown("### Run 2 Data")
            row2_nested = build_nested_dict(step_row2.to_dict())
            render_dynamic_json_expanders(row2_nested)
    else:
        row1_nested = build_nested_dict(step_row1.to_dict())
        render_dynamic_json_expanders(row1_nested)

