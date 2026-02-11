"""Gradio dashboard for Gecho echocardiogram analysis."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.config import GechoConfig
from src.embedding_engine import EmbeddingEngine, RetrievalResult
from src.report_generator import ClinicalReport, ReportGenerator
from src.video_processor import EchoFrames, extract_frames_from_upload


class GechoDashboard:
    """Three-column Gradio dashboard for echocardiogram analysis."""

    def __init__(
        self,
        config: GechoConfig,
        embedding_engine: EmbeddingEngine,
        report_generator: ReportGenerator,
    ) -> None:
        self.config = config
        self.engine = embedding_engine
        self.generator = report_generator

    # --- Analysis pipeline ------------------------------------------------

    def analyze_video(
        self,
        video_file,
        mode: str,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple:
        """Main pipeline: extract -> embed -> retrieve -> classify -> generate.

        Returns tuple of outputs matching the Gradio component order.
        """
        if not video_file:
            raise gr.Error("Please upload a video file first.")

        # gr.File returns a filepath string
        video_path = video_file if isinstance(video_file, str) else video_file.name

        # Step 1: Extract frames
        progress(0.1, desc="Extracting frames...")
        frames: EchoFrames = extract_frames_from_upload(video_path, self.config)
        ed_img = Image.fromarray(frames.ed_frame)
        es_img = Image.fromarray(frames.es_frame)

        # Step 2: Zero-shot classification on the primary frame
        progress(0.3, desc="Running zero-shot classification...")
        primary_frame = frames.ed_frame if "ED" in mode else frames.es_frame
        zeroshot_scores = self.engine.zero_shot_classify(primary_frame)

        # Step 3: FAISS retrieval
        progress(0.5, desc="Retrieving similar cases...")
        if mode == "Comparison (ED vs ES)":
            ed_result = self.engine.query(frames.ed_frame, frame_type="ED")
            es_result = self.engine.query(frames.es_frame, frame_type="ES")
        elif "ES" in mode:
            es_result = self.engine.query(frames.es_frame, frame_type="ES")
            ed_result = None
        else:
            ed_result = self.engine.query(frames.ed_frame, frame_type="ED")
            es_result = None

        active_result = ed_result or es_result

        # Step 4: Build retrieval table
        retrieval_table = self._format_retrieval_table(active_result)

        # Step 5: Generate report
        progress(0.7, desc="Generating clinical report...")
        if mode == "Comparison (ED vs ES)" and ed_result and es_result:
            report = self.generator.generate_comparison_report(
                frames.ed_frame, frames.es_frame, ed_result, es_result
            )
        else:
            frame = frames.ed_frame if ed_result else frames.es_frame
            ftype = "ED" if ed_result else "ES"
            report = self.generator.generate_single_frame_report(
                frame, ftype, active_result, zeroshot_scores
            )

        progress(1.0, desc="Done!")

        # Format outputs
        report_md = self._format_report_markdown(report)
        zeroshot_label = {k: float(v) for k, v in zeroshot_scores.items()}

        return (
            ed_img,                 # ED frame display
            es_img,                 # ES frame display
            zeroshot_label,         # gr.Label (zero-shot)
            retrieval_table,        # gr.Dataframe
            report_md,              # gr.Markdown (report)
        )

    # --- Formatting helpers -----------------------------------------------

    @staticmethod
    def _format_retrieval_table(
        result: RetrievalResult | None,
    ) -> list[list[str]]:
        """Format retrieval results as a table for gr.Dataframe."""
        if result is None or not result.cases:
            return [["No results", "", "", ""]]
        rows = []
        for c in result.cases:
            rows.append([
                c.filename,
                f"{c.ef:.1f}%" if c.ef is not None else "N/A",
                c.ef_category or "N/A",
                f"{c.similarity_score:.3f}",
            ])
        return rows

    @staticmethod
    def _format_report_markdown(report: ClinicalReport) -> str:
        """Format a ClinicalReport as Markdown for display."""
        return (
            f"## Clinical Report\n\n"
            f"{report.full_report}\n\n"
            f"---\n\n"
            f"### EF Assessment\n{report.ef_assessment}\n\n"
            f"### Retrieval Context\n"
            f"```\n{report.retrieval_context}\n```\n\n"
            f"### Confidence Note\n"
            f"> {report.confidence_note}"
        )

    # --- Gradio app -------------------------------------------------------

    def build(self) -> gr.Blocks:
        """Build and return the Gradio Blocks app."""
        with gr.Blocks(
            title="Gecho - Automated Echocardiogram Analysis",
            theme=gr.themes.Soft(),
        ) as app:
            gr.Markdown(
                "# Gecho: Gemma Echo\n"
                "Automated echocardiogram interpretation powered by "
                "MedSigLIP retrieval and MedGemma report generation."
            )

            with gr.Row():
                # --- Left column: Input ---
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    video_input = gr.File(
                        label="Upload Echo Video (.avi / .mp4)",
                        file_types=[".avi", ".mp4", ".mov", ".mkv"],
                    )
                    mode_selector = gr.Radio(
                        choices=[
                            "Single Frame (ED)",
                            "Single Frame (ES)",
                            "Comparison (ED vs ES)",
                        ],
                        value="Comparison (ED vs ES)",
                        label="Analysis Mode",
                    )
                    analyze_btn = gr.Button(
                        "Analyze", variant="primary", size="lg"
                    )

                    gr.Markdown("### Extracted Frames")
                    ed_display = gr.Image(label="End-Diastole (ED)", type="pil")
                    es_display = gr.Image(label="End-Systole (ES)", type="pil")

                # --- Middle column: Retrieval ---
                with gr.Column(scale=1):
                    gr.Markdown("### MedSigLIP Classification")
                    zeroshot_label = gr.Label(
                        label="Zero-Shot Cardiac Function",
                        num_top_classes=4,
                    )

                    gr.Markdown("### Similar Cases (FAISS Retrieval)")
                    retrieval_table = gr.Dataframe(
                        headers=["Filename", "EF", "Category", "Similarity"],
                        label="Top Retrieved Cases",
                        interactive=False,
                    )

                # --- Right column: Report ---
                with gr.Column(scale=1):
                    gr.Markdown("### Generated Report")
                    report_output = gr.Markdown(
                        value="*Upload a video and click Analyze to begin.*"
                    )

                    gr.Markdown("### Human-in-the-Loop")
                    with gr.Row():
                        approve_btn = gr.Button("Approve Report", variant="secondary")
                        edit_btn = gr.Button("Edit Report", variant="secondary")
                    status_text = gr.Textbox(
                        label="Status",
                        value="Awaiting analysis...",
                        interactive=False,
                    )

            # --- Event handlers ---
            analyze_btn.click(
                fn=self.analyze_video,
                inputs=[video_input, mode_selector],
                outputs=[
                    ed_display,
                    es_display,
                    zeroshot_label,
                    retrieval_table,
                    report_output,
                ],
            )

            approve_btn.click(
                fn=lambda: "Report approved by clinician.",
                outputs=[status_text],
            )
            edit_btn.click(
                fn=lambda: "Report flagged for clinician editing.",
                outputs=[status_text],
            )

        return app


def launch_dashboard(
    config: GechoConfig,
    embedding_engine: EmbeddingEngine,
    report_generator: ReportGenerator,
    share: bool = False,
) -> None:
    """Convenience function to build and launch the dashboard."""
    dashboard = GechoDashboard(config, embedding_engine, report_generator)
    app = dashboard.build()
    app.launch(share=share)
