"""Configuration for Gecho pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _hf_login() -> None:
    """Authenticate with HuggingFace using Kaggle secrets or env var.

    On Kaggle: stores HF token as a Kaggle Secret named "HF_TOKEN".
    Locally: set the HF_TOKEN environment variable.
    """
    token = os.environ.get("HF_TOKEN")

    # Try Kaggle Secrets API (available inside Kaggle notebooks)
    if token is None:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass

    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("Authenticated with HuggingFace.")
        except Exception as e:
            print(f"[WARN] HF login failed: {e}")
    else:
        print(
            "[WARN] No HF_TOKEN found. Gated models will fail unless "
            "attached locally via Kaggle 'Add Model'."
        )


def _find_local_model(model_name: str) -> str | None:
    """Search common Kaggle input paths for a locally-attached model.

    Kaggle's "Add Model" feature places models under /kaggle/input/.
    Users typically attach them with slugs like 'medsiglip-448' or
    'medgemma-1-5-4b-it'.  We search for directories whose name
    contains the key part of the model identifier.
    """
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None

    # Build search terms from the model name
    # "google/medsiglip-448" -> "medsiglip"
    # "google/medgemma-1.5-4b-it" -> "medgemma"
    search_term = model_name.split("/")[-1].split("-")[0].lower()

    for entry in sorted(kaggle_input.iterdir()):
        if not entry.is_dir():
            continue
        if search_term in entry.name.lower():
            # Kaggle model dirs can be nested: slug/framework/variant/version
            # Walk down to find a directory with config.json or similar
            for root, dirs, files in os.walk(entry):
                if "config.json" in files or "tokenizer.json" in files:
                    local_path = str(Path(root))
                    print(f"Found local model for '{model_name}': {local_path}")
                    return local_path
            # If no config.json found, return the top-level match
            local_path = str(entry)
            print(f"Found local model dir for '{model_name}': {local_path}")
            return local_path

    return None


@dataclass
class GechoConfig:
    """Central configuration for the Gecho pipeline."""

    # --- Dataset paths (Kaggle defaults) ---
    dataset_root: Path = Path("/kaggle/input/datasets/syxlicheng/heartdatabase/EchoNet-Dynamic")
    output_dir: Path = Path("/kaggle/working/gecho_output")

    # --- Model identifiers (HuggingFace Hub IDs) ---
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
    volume_tracings_path: Path = field(init=False)
    faiss_index_path: Path = field(init=False)
    faiss_metadata_path: Path = field(init=False)

    def __post_init__(self) -> None:
        # Authenticate with HuggingFace for gated models
        _hf_login()

        self.videos_dir = self.dataset_root / "Videos"
        self.file_list_path = self.dataset_root / "FileList.csv"
        self.volume_tracings_path = self.dataset_root / "VolumeTracings.csv"
        self.faiss_index_path = self.output_dir / "gecho_faiss.index"
        self.faiss_metadata_path = self.output_dir / "gecho_faiss_meta.pkl"

        os.makedirs(self.output_dir, exist_ok=True)

        # Resolve model paths: prefer locally-attached Kaggle models,
        # fall back to HuggingFace Hub IDs (requires authentication).
        self.medsiglip_model_id = (
            _find_local_model(self.medsiglip_model_id)
            or self.medsiglip_model_id
        )
        self.medgemma_hf_model_id = (
            _find_local_model(self.medgemma_hf_model_id)
            or self.medgemma_hf_model_id
        )
