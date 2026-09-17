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
    
    # Handle wrap-around or clamp if GIF total frames differ from JSONL log length
    num_frames = getattr(img, "n_frames", 1)
    target_frame = frame_index % num_frames if num_frames > 0 else 0
    
    img.seek(target_frame)
    return img.convert("RGB")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("📁 File & Settings Config")

default_jsonl = "log_zam32.jsonl"
default_frames_dir = "frames_zam32"
default_gif = "zam_zip32_v2_merge.gif"

jsonl_file = st.sidebar.text_input("JSONL Path:", value=default_jsonl)
frames_dir = st.sidebar.text_input("Frames Directory:", value=default_frames_dir)
gif_file = st.sidebar.text_input("GIF Path (Optional):", value=default_gif)

if not Path(jsonl_file).exists():
    st.error(f"Cannot find log file: {jsonl_file}. Please check the path.")
    st.stop()

df = load_jsonl_log(jsonl_file)
total_steps = len(df)

st.sidebar.markdown("---")
st.sidebar.header("📊 Signal Selection")

numeric_cols = [
    col for col in df.columns 
    if col not in ["step", "timestamp"] and pd.api.types.is_numeric_dtype(df[col])
]

default_selected = [
    col for col in numeric_cols 
    if any(k in col for k in ["ego.velocity", "steering.steering", "ego.orientation_deg"])
]

selected_signals = st.sidebar.multiselect(
    "Select signals to plot:",
    options=numeric_cols,
    default=default_selected if default_selected else numeric_cols[:2]
)
# -----------------------------------------------------------------------------
# Frame Control Interface (Fixed Callback Logic)
# -----------------------------------------------------------------------------
st.subheader("🕹️ Simulation Playback Controls")

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

# Callbacks to update state prior to rendering widgets
def increment_step():
    if st.session_state.current_step < total_steps - 1:
        st.session_state.current_step += 1

def decrement_step():
    if st.session_state.current_step > 0:
        st.session_state.current_step -= 1

col_prev, col_step, col_next = st.columns([1, 4, 1])

with col_prev:
    st.button("⬅️ Previous Step", on_click=decrement_step)

with col_next:
    st.button("Next Step ➡️", on_click=increment_step)

with col_step:
    st.slider(
        "Step / Frame Index",
        min_value=0,
        max_value=total_steps - 1,
        key="current_step"  # Binding directly to session_state key
    )

current_step = st.session_state.current_step
step_row = df[df["step"] == current_step].iloc[0]
timestamp = step_row["timestamp"]
st.caption(f"**Current Step:** {current_step} / {total_steps - 1} | **Timestamp:** {timestamp:.2f} s")

# -----------------------------------------------------------------------------
# Frame Control Interface
# -----------------------------------------------------------------------------
#st.subheader("🕹️ Simulation Playback Controls")
#
#col_prev, col_step, col_next = st.columns([1, 4, 1])
#
#if "current_step" not in st.session_state:
#    st.session_state.current_step = 0
#
#with col_prev:
#    if st.button("⬅️ Previous Step"):
#        if st.session_state.current_step > 0:
#            st.session_state.current_step -= 1
#
#with col_next:
#    if st.button("Next Step ➡️"):
#        if st.session_state.current_step < total_steps - 1:
#            st.session_state.current_step += 1
#
#with col_step:
#    current_step = st.slider(
#        "Step / Frame Index",
#        min_value=0,
#        max_value=total_steps - 1,
#        value=st.session_state.current_step,
#        key="step_slider"
#    )
#    st.session_state.current_step = current_step
#
#step_row = df[df["step"] == current_step].iloc[0]
#timestamp = step_row["timestamp"]
#st.caption(f"**Current Step:** {current_step} / {total_steps - 1} | **Timestamp:** {timestamp:.2f} s")
#
#st.markdown("---")

# -----------------------------------------------------------------------------
# Main Visualization View (Frames & Interactive Plots)
# -----------------------------------------------------------------------------
view_col1, view_col2 = st.columns([1, 1])

with view_col1:
    st.subheader("🖼️ Frame Visualization")
    
    # Option 1: Try individual PNG file in directory first
    frame_image_path = Path(frames_dir) / f"frame_{current_step:02d}.png"
    if not frame_image_path.exists():
        # Try without zero-padding naming standard
        frame_image_path = Path(frames_dir) / f"frame_{current_step}.png"
    
    if frame_image_path.exists():
        st.image(str(frame_image_path), caption=f"Frame File: {frame_image_path.name}", use_container_width=True)
    
    # Option 2: Extract frame directly from animated GIF by frame index
    elif Path(gif_file).exists():
        extracted_frame = get_gif_frame(gif_file, current_step)
        st.image(extracted_frame, caption=f"GIF Frame Index: {current_step}", use_container_width=True)
    else:
        st.warning(f"No image found for step {current_step} in `{frames_dir}` or `{gif_file}`.")

with view_col2:
    st.subheader("📈 Real-time Signal Plots")
    
    if selected_signals:
        fig = go.Figure()
        
        for sig in selected_signals:
            # Complete signal trend curve
            fig.add_trace(go.Scatter(
                x=df["step"],
                y=df[sig],
                mode="lines",
                name=sig,
                opacity=0.6
            ))
            
            # Highlight current active step marker point
            fig.add_trace(go.Scatter(
                x=[current_step],
                y=[step_row[sig]],
                mode="markers",
                marker=dict(size=10, symbol="circle"),
                name=f"{sig} (Current)",
                showlegend=False
            ))

        # Vertical indicator cursor matching the visible GIF frame
        fig.add_vline(
            x=current_step,
            line_width=2,
            line_dash="dash",
            line_color="red"
        )

        fig.update_layout(
            xaxis_title="Step",
            yaxis_title="Value",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=20, r=20, t=30, b=20),
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select one or more signals from the sidebar to display plots.")

# -----------------------------------------------------------------------------
# Step Data Inspector (Raw Values)
# -----------------------------------------------------------------------------
with st.expander("🔍 Inspect Raw Log Entry Data for Current Step"):
    raw_payload = step_row.to_dict()
    st.json(raw_payload)
