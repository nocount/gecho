"""Configuration for Gecho pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GechoConfig:
    """Central configuration for the Gecho pipeline."""

    # --- Dataset paths (Kaggle defaults) ---
    dataset_root: Path = Path("/kaggle/input/echonet-dynamic")
    output_dir: Path = Path("/kaggle/working/gecho_output")

    # --- Model identifiers ---
    medsiglip_model_id: str = "google/medsiglip-448"
    medgemma_kerashub_preset: str = "medgemma_1.5_instruct_4b"
    medgemma_hf_model_id: str = "google/medgemma-1.5-4b-it"

    # --- Frame processing ---
    siglip_frame_size: int = 448
    medgemma_frame_size: int = 896

    # --- FAISS / retrieval ---
    faiss_top_k: int = 5
    embedding_dim: int = 768  # verified at runtime from first embedding

    # --- Generation parameters ---
    max_new_tokens: int = 1024
    sequence_length: int = 4096
    dtype: str = "bfloat16"

    # --- Derived paths (set in __post_init__) ---
    videos_dir: Path = field(init=False)
    file_list_path: Path = field(init=False)
    faiss_index_path: Path = field(init=False)
    faiss_metadata_path: Path = field(init=False)

    def __post_init__(self) -> None:
        # EchoNet may be nested one level deeper on Kaggle
        if not self.dataset_root.exists():
            alt = Path("/kaggle/input/echonet-dynamic/EchoNet-Dynamic")
            if alt.exists():
                self.dataset_root = alt

        self.videos_dir = self.dataset_root / "Videos"
        self.file_list_path = self.dataset_root / "FileList.csv"
        self.faiss_index_path = self.output_dir / "gecho_faiss.index"
        self.faiss_metadata_path = self.output_dir / "gecho_faiss_meta.pkl"

        os.makedirs(self.output_dir, exist_ok=True)
