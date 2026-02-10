"""MedGemma report generation with KerasHub (primary) and transformers (fallback)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.config import GechoConfig
from src.embedding_engine import RetrievalResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClinicalReport:
    """Generated clinical report for an echocardiogram analysis."""
    summary: str
    ef_assessment: str
    retrieval_context: str
    full_report: str
    confidence_note: str


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert echocardiography AI assistant helping clinicians "
    "interpret echocardiogram images. You provide structured, evidence-based "
    "assessments. Always include a disclaimer that findings require "
    "verification by a qualified cardiologist."
)


def _build_retrieval_context(result: RetrievalResult) -> str:
    """Format retrieval results into a context string for the prompt."""
    if not result.cases:
        return "No similar cases were found in the reference database."

    lines = [
        f"Retrieved {len(result.cases)} similar cases from EchoNet-Dynamic:"
    ]
    for i, c in enumerate(result.cases, 1):
        ef_str = f"EF={c.ef:.1f}%" if c.ef is not None else "EF=N/A"
        lines.append(
            f"  {i}. {c.filename} ({c.frame_type}) - {ef_str} "
            f"({c.ef_category}), similarity={c.similarity_score:.3f}"
        )

    if result.mean_ef is not None:
        lines.append(
            f"Mean EF of similar cases: {result.mean_ef:.1f}% "
            f"(SD: {result.ef_std:.1f}%)"
        )
    if result.consensus_category:
        lines.append(f"Consensus category: {result.consensus_category}")

    return "\n".join(lines)


def _build_zeroshot_context(scores: dict[str, float]) -> str:
    """Format zero-shot scores into a context string."""
    if not scores:
        return ""
    lines = ["MedSigLIP zero-shot classification:"]
    for label, prob in sorted(scores.items(), key=lambda x: -x[1]):
        lines.append(f"  - {label}: {prob:.1%}")
    return "\n".join(lines)


def _build_single_frame_prompt(
    frame_type: str,
    retrieval_ctx: str,
    zeroshot_ctx: str,
) -> str:
    """Build a prompt for single-frame analysis."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Analyze this {frame_type} (End-Diastole = ED, End-Systole = ES) "
        f"echocardiogram frame.\n\n"
        f"### Context from Similar Cases\n{retrieval_ctx}\n\n"
        f"### Zero-Shot Classification\n{zeroshot_ctx}\n\n"
        f"### Requested Output\n"
        f"Provide a structured report with the following sections:\n"
        f"1. **Visual Assessment**: Describe left ventricle size, wall motion, "
        f"and any visible abnormalities.\n"
        f"2. **EF Estimate**: Based on the visual features and similar-case "
        f"context, estimate the ejection fraction range.\n"
        f"3. **Clinical Impression**: Summarize the key findings and their "
        f"clinical significance.\n"
        f"4. **Limitations**: Note any caveats about this automated analysis."
    )


def _build_comparison_prompt(
    ed_retrieval_ctx: str,
    es_retrieval_ctx: str,
) -> str:
    """Build a prompt for ED vs ES comparison analysis."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"You are given two echocardiogram frames from the same patient:\n"
        f"- **Image 1**: End-Diastole (ED) frame — the heart is maximally dilated.\n"
        f"- **Image 2**: End-Systole (ES) frame — the heart is maximally contracted.\n\n"
        f"### ED Similar Cases\n{ed_retrieval_ctx}\n\n"
        f"### ES Similar Cases\n{es_retrieval_ctx}\n\n"
        f"### Requested Output\n"
        f"Provide a structured report with the following sections:\n"
        f"1. **Visual Assessment**: Compare LV size between ED and ES. "
        f"Describe wall motion and contractility.\n"
        f"2. **EF Estimate**: Based on the visual change between ED and ES "
        f"plus similar-case context, estimate the ejection fraction range.\n"
        f"3. **Clinical Impression**: Key findings and clinical significance.\n"
        f"4. **Limitations**: Caveats about this automated analysis."
    )


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """MedGemma-based clinical report generator."""

    def __init__(self, config: GechoConfig) -> None:
        self.config = config
        self._backend = self._load_model()

    def _load_model(self) -> str:
        """Load MedGemma via KerasHub (preferred) or transformers fallback."""
        # Try KerasHub first
        try:
            return self._load_kerashub()
        except Exception as e:
            print(f"[INFO] KerasHub loading failed ({e}), trying transformers...")
            return self._load_transformers()

    def _load_kerashub(self) -> str:
        """Load via keras_hub."""
        os.environ.setdefault("KERAS_BACKEND", "jax")
        import keras_hub  # noqa: E402

        print(f"Loading MedGemma via KerasHub: {self.config.medgemma_kerashub_preset}")
        self.keras_model = keras_hub.models.Gemma3CausalLM.from_preset(
            self.config.medgemma_kerashub_preset,
            dtype=self.config.dtype,
        )
        print("MedGemma loaded via KerasHub (JAX).")
        return "kerashub"

    def _load_transformers(self) -> str:
        """Load via HuggingFace transformers."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading MedGemma via transformers: {self.config.medgemma_hf_model_id}")

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        torch_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self.hf_processor = AutoProcessor.from_pretrained(
            self.config.medgemma_hf_model_id
        )
        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            self.config.medgemma_hf_model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.hf_model.eval()
        print("MedGemma loaded via transformers (PyTorch).")
        return "transformers"

    # --- Generation backends ----------------------------------------------

    def _generate_kerashub(
        self,
        prompt: str,
        images: list[np.ndarray],
    ) -> str:
        """Generate text using KerasHub backend."""
        # KerasHub Gemma3 expects images passed alongside the prompt
        # The prompt should contain <start_of_image> tokens for each image
        image_tokens = "\n".join(["<start_of_image>"] * len(images))
        full_prompt = f"{image_tokens}\n{prompt}"

        response = self.keras_model.generate(
            {
                "prompts": full_prompt,
                "images": [
                    img.astype("float32") / 255.0 if img.dtype == np.uint8 else img
                    for img in images
                ],
            },
            max_length=self.config.sequence_length,
        )
        # KerasHub returns the full sequence; strip the prompt portion
        if isinstance(response, str):
            return response
        return str(response)

    def _generate_transformers(
        self,
        prompt: str,
        images: list[np.ndarray],
    ) -> str:
        """Generate text using transformers backend."""
        import torch

        pil_images = [
            Image.fromarray(img) if isinstance(img, np.ndarray) else img
            for img in images
        ]

        # Build chat messages with images
        content: list[dict] = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.hf_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.hf_model.device)

        with torch.no_grad():
            output_ids = self.hf_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        return self.hf_processor.decode(generated, skip_special_tokens=True)

    def _generate(self, prompt: str, images: list[np.ndarray]) -> str:
        """Route to the active backend."""
        if self._backend == "kerashub":
            return self._generate_kerashub(prompt, images)
        return self._generate_transformers(prompt, images)

    # --- Public API -------------------------------------------------------

    def generate_single_frame_report(
        self,
        frame: np.ndarray,
        frame_type: str,
        retrieval_result: RetrievalResult,
        zeroshot_scores: dict[str, float] | None = None,
    ) -> ClinicalReport:
        """Generate a clinical report for a single echo frame."""
        retrieval_ctx = _build_retrieval_context(retrieval_result)
        zeroshot_ctx = _build_zeroshot_context(zeroshot_scores or {})
        prompt = _build_single_frame_prompt(frame_type, retrieval_ctx, zeroshot_ctx)

        raw = self._generate(prompt, [frame])
        return self._parse_report(raw, retrieval_ctx)

    def generate_comparison_report(
        self,
        ed_frame: np.ndarray,
        es_frame: np.ndarray,
        ed_retrieval: RetrievalResult,
        es_retrieval: RetrievalResult,
    ) -> ClinicalReport:
        """Generate a clinical report comparing ED and ES frames."""
        ed_ctx = _build_retrieval_context(ed_retrieval)
        es_ctx = _build_retrieval_context(es_retrieval)
        prompt = _build_comparison_prompt(ed_ctx, es_ctx)

        raw = self._generate(prompt, [ed_frame, es_frame])
        combined_ctx = f"--- ED ---\n{ed_ctx}\n\n--- ES ---\n{es_ctx}"
        return self._parse_report(raw, combined_ctx)

    # --- Parsing ----------------------------------------------------------

    @staticmethod
    def _parse_report(raw_text: str, retrieval_ctx: str) -> ClinicalReport:
        """Parse raw model output into a structured ClinicalReport."""
        # Try to extract sections; fallback to raw text
        sections = {
            "Visual Assessment": "",
            "EF Estimate": "",
            "Clinical Impression": "",
            "Limitations": "",
        }

        current_section = None
        lines: list[str] = []

        for line in raw_text.split("\n"):
            matched = False
            for key in sections:
                if key.lower() in line.lower():
                    if current_section and lines:
                        sections[current_section] = "\n".join(lines).strip()
                    current_section = key
                    lines = []
                    matched = True
                    break
            if not matched:
                lines.append(line)

        if current_section and lines:
            sections[current_section] = "\n".join(lines).strip()

        # Build summary from first non-empty section
        summary = (
            sections["Clinical Impression"]
            or sections["Visual Assessment"]
            or raw_text[:500]
        )

        return ClinicalReport(
            summary=summary,
            ef_assessment=sections["EF Estimate"] or "See full report.",
            retrieval_context=retrieval_ctx,
            full_report=raw_text,
            confidence_note=(
                sections["Limitations"]
                or "This is an AI-generated analysis and must be reviewed "
                "by a qualified cardiologist before clinical use."
            ),
        )
