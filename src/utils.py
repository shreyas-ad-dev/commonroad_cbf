# src/utils.py
from pathlib import Path

from PIL import Image


def setup_frames_directory(frames_dir: Path) -> None:
    """
    Prepares the temporary frame directory by purging existing PNGs and recreating the folder.

    Args:
        frames_dir (Path): Path directory where simulation frame images are saved.
    """

    if frames_dir.exists():
        for f in frames_dir.glob("*.png"):
            f.unlink()
    frames_dir.mkdir(exist_ok=True)

def create_gif_from_frames(
        frame_files,
        output_gif_path,
        dt):
    """
    Stitches saved frame PNG images into an animated GIF.

    Args:
        frame_files (List[Path]): Ordered list of paths to individual frame images.
        output_gif_path (Path): Target file path where the generated GIF will be saved.
        dt (float): Simulation time step delta in seconds (converted internally to milliseconds for frame duration).
    """

    images = [Image.open(f).copy() for f in frame_files]
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(dt * 1000),
        loop=0
    )

def build_gif_and_cleanup(
        frame_files: list[Path],
        gif_path: Path, dt: float) -> None:
    """
    Generates a GIF animation from frame files and purges temporary frame images and folders.

    Args:
        frame_files (List[Path]): Ordered list of frame image paths.
        gif_path (Path): Output destination path for the compiled GIF.
        dt (float): Time step delta between consecutive frames in seconds.
    """

    if not frame_files:
        print("  No frame files found to generate GIF.")
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


