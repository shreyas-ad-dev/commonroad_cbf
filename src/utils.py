# src/utils.py
from pathlib import Path
from typing import List
from src.visualizer import create_gif_from_frames


def setup_frames_directory(frames_dir: Path) -> None:
    """Cleans up existing PNG frames in the directory and creates a fresh folder."""
    if frames_dir.exists():
        for f in frames_dir.glob("*.png"):
            f.unlink()
    frames_dir.mkdir(exist_ok=True)


def build_gif_and_cleanup(frame_files: List[Path], gif_path: Path, dt: float) -> None:
    """
    Generates a GIF animation from saved frame files and cleans up 
    the frame directory afterwards.
    """
    if not frame_files:
        print("⚠️ No frame files found to generate GIF.")
        return

    # Generate GIF
    create_gif_from_frames(frame_files, gif_path, dt)

    # Clean up temporary frames and folder
    frames_dir = frame_files[0].parent
    for f in frame_files:
        if f.exists():
            f.unlink()
    
    if frames_dir.exists():
        frames_dir.rmdir()

    print(f" Output saved to: '{gif_path.name}'")
