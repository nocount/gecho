"""Video processing pipeline for echocardiogram frame extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import GechoConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EchoFrames:
    """Extracted frames and metadata for a single echo video."""
    filename: str
    ed_frame: np.ndarray        # RGB uint8, resized to siglip_frame_size
    es_frame: np.ndarray        # RGB uint8, resized to siglip_frame_size
    ed_frame_idx: int
    es_frame_idx: int
    ef: float | None = None     # Ejection fraction (ground truth)
    esv: float | None = None    # End-systolic volume
    edv: float | None = None    # End-diastolic volume
    ef_category: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_ef(ef: float) -> str:
    """Classify ejection fraction into clinical categories."""
    if ef >= 55:
        return "Normal"
    elif ef >= 45:
        return "Mild Dysfunction"
    elif ef >= 30:
        return "Moderate Dysfunction"
    else:
        return "Severe Dysfunction"


def load_file_list(config: GechoConfig, split: str = "TRAIN") -> pd.DataFrame:
    """Load EchoNet FileList.csv and filter by split.

    Standard columns: FileName, EF, ESV, EDV, FrameHeight, FrameWidth,
                      FPS, NumberOfFrames, Split
    Some versions also include EDFrame, ESFrame directly.
    """
    df = pd.read_csv(config.file_list_path)
    df = df[df["Split"].str.upper() == split.upper()].reset_index(drop=True)
    return df


def load_frame_indices(config: GechoConfig) -> dict[str, tuple[int, int]]:
    """Derive ED and ES frame indices from VolumeTracings.csv.

    VolumeTracings.csv has one row per tracing point with columns:
      FileName, X1, Y1, X2, Y2, Frame

    Each video has tracings at exactly two frames (ED and ES).
    The frame with the larger traced volume is ED (most dilated);
    the smaller is ES (most contracted).

    Returns dict mapping FileName -> (ed_frame_idx, es_frame_idx).
    """
    if not config.volume_tracings_path.exists():
        return {}

    vt = pd.read_csv(config.volume_tracings_path)

    # Each row is one tracing point.  Group by (FileName, Frame) to get
    # a rough volume proxy: count of tracing points per frame, or use
    # the traced coordinates to estimate area.  The simpler approach:
    # the two unique frame numbers per video, the larger volume (more
    # area enclosed) corresponds to ED.
    frame_indices: dict[str, tuple[int, int]] = {}

    for fname, group in vt.groupby("FileName"):
        frames = sorted(group["Frame"].unique())
        if len(frames) < 2:
            # Fallback: only one traced frame
            frame_indices[str(fname)] = (frames[0], frames[0])
            continue

        # Estimate enclosed area for each frame using the Shoelace formula
        # on the tracing points (X1,Y1 -> X2,Y2 are inner/outer wall).
        # Simpler proxy: sum of X2-X1 per frame ≈ cavity diameter sum.
        areas: dict[int, float] = {}
        for frame_num, fgroup in group.groupby("Frame"):
            # X1 = inner wall, X2 = outer wall (or vice versa).
            # Cavity width at each tracing line ≈ |X1 - X2|.
            areas[int(frame_num)] = float((fgroup["X2"] - fgroup["X1"]).abs().sum())

        # ED = largest cavity (most dilated), ES = smallest (most contracted)
        sorted_frames = sorted(areas.keys(), key=lambda f: areas[f], reverse=True)
        ed_idx = sorted_frames[0]
        es_idx = sorted_frames[-1]
        frame_indices[str(fname)] = (ed_idx, es_idx)

    return frame_indices


def extract_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    """Extract a single frame from a video file, returned as RGB uint8."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(
                f"Could not read frame {frame_idx} from {video_path}"
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize frame to (size, size) using bilinear interpolation."""
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_echo_frames(
    video_path: str | Path,
    row: pd.Series,
    config: GechoConfig,
    ed_idx: int | None = None,
    es_idx: int | None = None,
) -> EchoFrames:
    """Extract ED + ES frames for one video.

    Frame indices can come from:
      1. Explicit ed_idx/es_idx arguments (from VolumeTracings)
      2. row["EDFrame"] / row["ESFrame"] columns (some CSV versions)
      3. Heuristic fallback (frame 0 and frame at ~33%)
    """
    # Resolve frame indices
    if ed_idx is None:
        if "EDFrame" in row.index:
            ed_idx = int(row["EDFrame"])
        else:
            ed_idx = 0

    if es_idx is None:
        if "ESFrame" in row.index:
            es_idx = int(row["ESFrame"])
        else:
            # Fallback: read total frame count and pick ~33%
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            es_idx = max(1, int(total * 0.33))

    ed_raw = extract_frame(video_path, ed_idx)
    es_raw = extract_frame(video_path, es_idx)

    ed = resize_frame(ed_raw, config.siglip_frame_size)
    es = resize_frame(es_raw, config.siglip_frame_size)

    ef = float(row["EF"]) if "EF" in row.index else None

    return EchoFrames(
        filename=row["FileName"],
        ed_frame=ed,
        es_frame=es,
        ed_frame_idx=ed_idx,
        es_frame_idx=es_idx,
        ef=ef,
        esv=float(row["ESV"]) if "ESV" in row.index and pd.notna(row.get("ESV")) else None,
        edv=float(row["EDV"]) if "EDV" in row.index and pd.notna(row.get("EDV")) else None,
        ef_category=classify_ef(ef) if ef is not None else None,
    )


def extract_frames_from_upload(
    video_path: str | Path,
    config: GechoConfig,
) -> EchoFrames:
    """Heuristic frame extraction for user-uploaded videos.

    Without CSV metadata we use:
      - ED = frame 0  (heart typically most dilated at start of clip)
      - ES = frame at ~33% of total frames
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            raise RuntimeError(f"Video too short ({total} frames): {video_path}")

        ed_idx = 0
        es_idx = max(1, int(total * 0.33))

        cap.set(cv2.CAP_PROP_POS_FRAMES, ed_idx)
        ret, ed_raw = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read ED frame from {video_path}")
        ed_raw = cv2.cvtColor(ed_raw, cv2.COLOR_BGR2RGB)

        cap.set(cv2.CAP_PROP_POS_FRAMES, es_idx)
        ret, es_raw = cap.read()
        if not ret:
            raise RuntimeError(f"Cannot read ES frame from {video_path}")
        es_raw = cv2.cvtColor(es_raw, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

    ed = resize_frame(ed_raw, config.siglip_frame_size)
    es = resize_frame(es_raw, config.siglip_frame_size)

    return EchoFrames(
        filename=Path(video_path).name,
        ed_frame=ed,
        es_frame=es,
        ed_frame_idx=ed_idx,
        es_frame_idx=es_idx,
    )


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_dataset(
    config: GechoConfig,
    split: str = "TRAIN",
    max_videos: int | None = None,
) -> list[EchoFrames]:
    """Process the EchoNet dataset split, returning extracted frames."""
    df = load_file_list(config, split)
    if max_videos is not None:
        df = df.head(max_videos)

    # Try to load ED/ES frame indices from VolumeTracings.csv
    has_frame_cols = "EDFrame" in df.columns and "ESFrame" in df.columns
    if has_frame_cols:
        frame_map: dict[str, tuple[int, int]] = {}
        print("Using EDFrame/ESFrame columns from FileList.csv")
    else:
        print("EDFrame/ESFrame not in FileList.csv, loading VolumeTracings.csv ...")
        frame_map = load_frame_indices(config)
        if frame_map:
            print(f"Loaded frame indices for {len(frame_map)} videos from VolumeTracings.csv")
        else:
            print("[WARN] No VolumeTracings.csv found, using heuristic frame selection")

    results: list[EchoFrames] = []
    for _, row in df.iterrows():
        fname = row["FileName"]
        # EchoNet filenames may or may not have .avi extension
        video_name = fname if fname.endswith(".avi") else f"{fname}.avi"
        video_path = config.videos_dir / video_name

        if not video_path.exists():
            print(f"[WARN] Video not found, skipping: {video_path}")
            continue

        # Look up frame indices
        ed_idx, es_idx = None, None
        if not has_frame_cols and frame_map:
            # VolumeTracings keys may or may not have .avi
            indices = frame_map.get(fname) or frame_map.get(video_name)
            if indices:
                ed_idx, es_idx = indices

        try:
            frames = extract_echo_frames(
                video_path, row, config, ed_idx=ed_idx, es_idx=es_idx
            )
            results.append(frames)
        except Exception as e:
            print(f"[WARN] Failed to process {fname}: {e}")
            continue

    print(f"Processed {len(results)}/{len(df)} videos from {split} split.")
    return results
