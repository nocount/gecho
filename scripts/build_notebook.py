"""Build a consolidated Kaggle submission notebook from src/ modules.

Reads each src/*.py file, strips relative imports (since everything lives
in notebook global scope), and assembles them into ordered notebook cells.

Usage:
    python scripts/build_notebook.py
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_PATH = PROJECT_ROOT / "notebooks" / "gecho_submission.ipynb"

# Order matters: later modules depend on earlier ones.
MODULE_ORDER = [
    "config.py",
    "video_processor.py",
    "embedding_engine.py",
    "report_generator.py",
    "ui.py",
]

# Imports to strip (they become intra-notebook references)
INTERNAL_IMPORT_RE = re.compile(r"^\s*from\s+src\.\w+\s+import\s+.*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_module(path: Path) -> str:
    """Read a Python file and strip internal src.* imports."""
    source = path.read_text(encoding="utf-8")
    source = INTERNAL_IMPORT_RE.sub("", source)
    # Remove blank lines left behind by stripped imports
    source = re.sub(r"\n{3,}", "\n\n", source)
    return source.strip()


def _make_markdown_cell(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=text)


def _make_code_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=source)


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

def build_notebook() -> nbformat.NotebookNode:
    """Assemble the full Kaggle submission notebook."""
    nb = nbformat.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    })

    cells: list[nbformat.NotebookNode] = []

    # --- Cell 1: Title & description ---
    cells.append(_make_markdown_cell(
        "# Gecho: Gemma Echo -- Automated Echocardiogram Reporting\n\n"
        "A RAG system that extracts key frames from echocardiogram videos, "
        "retrieves similar known cases via **MedSigLIP** embeddings, and "
        "generates clinical reports with **MedGemma**.\n\n"
        "**Models**: MedSigLIP-448 (retrieval) + MedGemma 1.5-4B-IT (generation)\n\n"
        "**Dataset**: [EchoNet-Dynamic](https://echonet.github.io/dynamic/) "
        "(Stanford)"
    ))

    # --- Cell 2: Environment setup ---
    cells.append(_make_code_cell(
        '# Environment setup -- must run before any Keras imports\n'
        'import os\n'
        'os.environ["KERAS_BACKEND"] = "jax"\n\n'
        '!pip install -q transformers keras keras-hub jax[cuda12] '
        'opencv-python faiss-cpu gradio Pillow huggingface_hub'
    ))

    # --- Cells 3-7: Source modules ---
    for module_file in MODULE_ORDER:
        path = SRC_DIR / module_file
        if not path.exists():
            print(f"[WARN] Module not found, skipping: {path}")
            continue

        module_name = module_file.replace(".py", "")
        cells.append(_make_markdown_cell(f"## `{module_name}`"))
        cells.append(_make_code_cell(_read_module(path)))

    # --- Cell 8: Main pipeline ---
    cells.append(_make_markdown_cell("## Main Pipeline"))
    cells.append(_make_code_cell(
        '# ---- Main pipeline: build index, load models, launch UI ----\n\n'
        'from pathlib import Path\n\n'
        '# Initialize configuration (paths auto-adjust for Kaggle)\n'
        'config = GechoConfig()\n\n'
        '# Step 1: Load or build FAISS index\n'
        'engine = EmbeddingEngine(config)\n\n'
        'if config.faiss_index_path.exists() and config.faiss_metadata_path.exists():\n'
        '    print("Found existing FAISS index, loading...")\n'
        '    engine.load_index()\n'
        'else:\n'
        '    print("No existing index found. Processing EchoNet-Dynamic training set...")\n'
        '    echo_frames = process_dataset(config, split="TRAIN")\n'
        '    print("Building embedding index...")\n'
        '    engine.build_index(echo_frames)\n'
        '    engine.save_index()\n'
        '    del echo_frames\n\n'
        '# Step 2: Load report generator (MedGemma)\n'
        'print("Loading MedGemma report generator...")\n'
        'generator = ReportGenerator(config)\n\n'
        '# Step 3: Launch Gradio dashboard\n'
        'print("Launching Gecho dashboard...")\n'
        'launch_dashboard(config, engine, generator, share=True)\n'
    ))

    # --- Cell 9: Technical notes ---
    cells.append(_make_markdown_cell(
        "## Technical Notes & Citations\n\n"
        "### Architecture\n"
        "- **Retrieval**: MedSigLIP-448 encodes echo frames into 768-dim "
        "embeddings. FAISS IndexFlatIP performs cosine-similarity search "
        "against ~15K training vectors.\n"
        "- **Generation**: MedGemma 1.5-4B-IT receives the query frame + "
        "RAG context (similar cases, zero-shot scores) and produces a "
        "structured clinical report.\n"
        "- **VRAM Strategy**: MedSigLIP (float16, ~1.6GB) builds the index "
        "first, then MedGemma (bfloat16, ~8GB) is loaded for generation. "
        "Total < 16GB P100.\n\n"
        "### Citations\n"
        "- Ouyang et al. *Video-based AI for beat-to-beat assessment of "
        "cardiac function.* Nature, 2020. (EchoNet-Dynamic)\n"
        "- Yang et al. *Advancing Multimodal Medical Capabilities of Gemini.* "
        "arXiv:2405.03162, 2024. (MedGemma)\n"
        "- Radford et al. *Learning Transferable Visual Models From Natural "
        "Language Supervision.* ICML, 2021. (SigLIP heritage)\n"
    ))

    nb.cells = cells
    return nb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    nbformat.write(nb, str(OUTPUT_PATH))
    print(f"Notebook written to {OUTPUT_PATH}")
    print(f"  Cells: {len(nb.cells)}")


if __name__ == "__main__":
    main()
