"""MedSigLIP embedding engine with FAISS retrieval."""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from src.config import GechoConfig
from src.video_processor import EchoFrames, classify_ef


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedCase:
    """A single retrieved case from the FAISS index."""
    filename: str
    ef: float | None
    ef_category: str | None
    similarity_score: float
    frame_type: str  # "ED" or "ES"


@dataclass
class RetrievalResult:
    """Aggregated result from a FAISS query."""
    cases: list[RetrievedCase]
    mean_ef: float | None = None
    ef_std: float | None = None
    consensus_category: str | None = None


# ---------------------------------------------------------------------------
# Index metadata
# ---------------------------------------------------------------------------

@dataclass
class IndexEntry:
    """Metadata stored alongside each FAISS vector."""
    filename: str
    frame_type: str  # "ED" or "ES"
    ef: float | None = None
    ef_category: str | None = None


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """MedSigLIP-based embedding engine with FAISS retrieval."""

    def __init__(self, config: GechoConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading MedSigLIP from {config.medsiglip_model_id} ...")
        self.processor = AutoProcessor.from_pretrained(config.medsiglip_model_id)
        self.model = AutoModel.from_pretrained(
            config.medsiglip_model_id,
            torch_dtype=torch.float16,
        ).to(self.device).eval()
        print(f"MedSigLIP loaded on {self.device}.")

        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[IndexEntry] = []

    # --- Encoding ---------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """Encode a single image to an L2-normalized embedding vector."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32).squeeze()

    @torch.no_grad()
    def encode_batch(
        self, images: list[np.ndarray | Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        """Encode a batch of images to L2-normalized embeddings."""
        all_embs: list[np.ndarray] = []
        pil_images = [
            Image.fromarray(img) if isinstance(img, np.ndarray) else img
            for img in images
        ]

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to an L2-normalized embedding vector."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
            self.device
        )
        emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32).squeeze()

    # --- Index building ---------------------------------------------------

    def build_index(self, echo_frames_list: list[EchoFrames]) -> None:
        """Build a FAISS inner-product index from ED+ES frames."""
        images: list[np.ndarray] = []
        entries: list[IndexEntry] = []

        for ef in echo_frames_list:
            for frame, ftype in [(ef.ed_frame, "ED"), (ef.es_frame, "ES")]:
                images.append(frame)
                entries.append(IndexEntry(
                    filename=ef.filename,
                    frame_type=ftype,
                    ef=ef.ef,
                    ef_category=ef.ef_category,
                ))

        print(f"Encoding {len(images)} frames ...")
        embeddings = self.encode_batch(images)

        # Update embedding_dim from actual data
        dim = embeddings.shape[1]
        if dim != self.config.embedding_dim:
            print(f"Updating embedding_dim: {self.config.embedding_dim} -> {dim}")
            self.config.embedding_dim = dim

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = entries
        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}.")

    # --- Persistence ------------------------------------------------------

    def save_index(self, config: GechoConfig | None = None) -> None:
        """Save FAISS index and metadata to disk."""
        cfg = config or self.config
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index first.")

        faiss.write_index(self.index, str(cfg.faiss_index_path))
        with open(cfg.faiss_metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved to {cfg.faiss_index_path}")

    def load_index(self, config: GechoConfig | None = None) -> None:
        """Load FAISS index and metadata from disk."""
        cfg = config or self.config
        self.index = faiss.read_index(str(cfg.faiss_index_path))
        with open(cfg.faiss_metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"Index loaded: {self.index.ntotal} vectors.")

    # --- Querying ---------------------------------------------------------

    def query(
        self,
        image: np.ndarray | Image.Image,
        frame_type: str = "ED",
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Find the most similar cases in the FAISS index."""
        if self.index is None:
            raise RuntimeError("No index loaded. Call build_index or load_index.")

        k = top_k or self.config.faiss_top_k
        emb = self.encode_image(image).reshape(1, -1)
        scores, indices = self.index.search(emb, k * 2)  # over-fetch to filter

        cases: list[RetrievedCase] = []
        seen: set[str] = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.metadata[idx]
            # Optionally filter by frame type; deduplicate by filename
            if entry.filename in seen:
                continue
            seen.add(entry.filename)
            cases.append(RetrievedCase(
                filename=entry.filename,
                ef=entry.ef,
                ef_category=entry.ef_category,
                similarity_score=float(score),
                frame_type=entry.frame_type,
            ))
            if len(cases) >= (top_k or self.config.faiss_top_k):
                break

        # Aggregate statistics
        efs = [c.ef for c in cases if c.ef is not None]
        mean_ef = float(np.mean(efs)) if efs else None
        ef_std = float(np.std(efs)) if efs else None

        # Consensus category from most-common category
        cats = [c.ef_category for c in cases if c.ef_category]
        consensus = Counter(cats).most_common(1)[0][0] if cats else None

        return RetrievalResult(
            cases=cases,
            mean_ef=mean_ef,
            ef_std=ef_std,
            consensus_category=consensus,
        )

    # --- Zero-shot classification -----------------------------------------

    def zero_shot_classify(
        self,
        image: np.ndarray | Image.Image,
        labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Zero-shot classification using image-text similarity.

        Returns dict mapping label -> probability (sums to 1).
        """
        if labels is None:
            labels = [
                "echocardiogram showing normal cardiac function",
                "echocardiogram showing mild left ventricular dysfunction",
                "echocardiogram showing moderate left ventricular dysfunction",
                "echocardiogram showing severe left ventricular dysfunction",
            ]

        img_emb = self.encode_image(image)
        text_embs = np.array([self.encode_text(lbl) for lbl in labels])

        # Cosine similarities (already L2-normalized)
        sims = text_embs @ img_emb
        # Softmax
        exp_sims = np.exp(sims - sims.max())
        probs = exp_sims / exp_sims.sum()

        # Use short labels for display
        short_labels = ["Normal", "Mild Dysfunction", "Moderate Dysfunction", "Severe Dysfunction"]
        if len(short_labels) == len(labels):
            return dict(zip(short_labels, probs.tolist()))
        return dict(zip(labels, probs.tolist()))

    # --- Cleanup ----------------------------------------------------------

    def unload(self) -> None:
        """Free GPU memory by deleting the model."""
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("MedSigLIP unloaded from GPU.")
