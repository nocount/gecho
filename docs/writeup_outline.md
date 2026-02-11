# Gecho: Technical Writeup (3-page outline)

## Page 1: Problem & Approach

### The Clinical Challenge
- 6M echocardiograms/year in the US; each requires 20-30 min expert analysis
- Inter-reader variability in EF measurement: up to 13% (Lang et al., JASE 2015)
- Rural/underserved areas face weeks-long delays for cardiologist interpretation
- Sonographer burnout is a growing concern in cardiac imaging departments

### Why Echocardiography + AI?
- Echo is the most common cardiac imaging modality (non-invasive, no radiation)
- EF is the single most impactful metric for heart failure diagnosis and management
- EchoNet-Dynamic provides 10,030 labeled apical-4-chamber videos with ground-truth EF
- Key frames (End-Diastole, End-Systole) capture the essential cardiac cycle information

### Our Approach: RAG for Medical Imaging
- Retrieval-Augmented Generation combines the interpretability of case-based reasoning with the fluency of large language models
- MedSigLIP provides the retrieval signal: "These are the most similar hearts we've seen"
- MedGemma provides the generation: structured clinical reports grounded in evidence
- Human-in-the-loop design: AI drafts, clinician approves or edits

---

## Page 2: Technical Implementation

### Architecture Overview
```
Echo Video (AVI/MP4)
  -> OpenCV frame extraction (ED + ES via metadata or heuristic)
  -> MedSigLIP-448 embedding (768-dim, L2-normalized)
  -> FAISS IndexFlatIP cosine-similarity retrieval (top-5)
  -> MedGemma 1.5-4B-IT structured report generation
  -> Gradio 3-column dashboard
```

### Frame Extraction
- EchoNet-Dynamic provides exact ED/ES frame indices in FileList.csv
- For user uploads without metadata: heuristic extraction (frame 0 = ED, frame at 33% = ES)
- Frames resized to 448x448 for MedSigLIP input via bilinear interpolation

### Embedding & Retrieval (MedSigLIP-448)
- Model: `google/medsiglip-448` via HuggingFace Transformers (PyTorch, float16)
- Encodes both ED and ES frames -> ~14,930 vectors for the training split
- FAISS IndexFlatIP for exact cosine-similarity search (no approximation needed at this scale)
- Retrieval output: top-5 similar cases with mean EF, standard deviation, consensus category
- Zero-shot classification via image-text similarity: Normal / Mild / Moderate / Severe dysfunction

### Report Generation (MedGemma 1.5-4B-IT)
- Primary: KerasHub `Gemma3CausalLM` on JAX backend (bfloat16)
- Fallback: HuggingFace `AutoModelForImageTextToText` (PyTorch, bfloat16)
- RAG prompt includes: system context, query image, retrieval statistics, zero-shot scores
- Structured output: Visual Assessment, EF Estimate, Clinical Impression, Limitations

### VRAM Management (P100 16GB)
- MedSigLIP (float16): ~1.6 GB -- used for index building and per-query embedding
- MedGemma (bfloat16): ~8 GB -- loaded after index is built
- Total peak: ~9.6 GB, well within P100 limits
- Strategy: build full FAISS index with MedSigLIP, persist to disk, then load MedGemma

### Gradio Dashboard
- Three-column layout: Input / Retrieval / Report
- Progress tracking through each pipeline stage
- Human-in-the-loop: Approve / Edit buttons for clinician workflow

---

## Page 3: Results & Impact

### Quantitative Evaluation
- Retrieval accuracy: mean EF of top-5 neighbors vs. ground truth EF (MAE, correlation)
- Zero-shot classification: accuracy of MedSigLIP Normal/Mild/Moderate/Severe vs. ground truth
- Report quality: manual review of generated reports for clinical accuracy and completeness

### Qualitative Assessment
- Case studies showing retrieval + generation pipeline on representative examples:
  - Normal function (EF > 55%)
  - Moderate dysfunction (EF 30-44%)
  - Severe dysfunction (EF < 30%)
- Comparison of retrieval-only vs. RAG-augmented reports

### Clinical Impact
- **Speed**: Preliminary report in ~30 seconds vs. 20-30 minutes manual
- **Access**: Enables same-day echo interpretation in rural/underserved clinics
- **Consistency**: Reduces inter-reader variability with data-driven evidence
- **Transparency**: Every report includes the similar-case evidence trail and confidence scores
- **Safety**: Human-in-the-loop design with mandatory clinician review

### Effective Use of HAI-DEF Models
- MedSigLIP: purpose-built medical vision encoder for accurate retrieval beyond generic CLIP
- MedGemma: clinical language model that generates structured, medically appropriate text
- Synergy: MedSigLIP retrieval grounds MedGemma generation, reducing hallucination risk

### Limitations & Future Work
- Single-view analysis (apical 4-chamber only); multi-view fusion is a natural extension
- Heuristic frame selection for uploads; could integrate a cardiac phase detection model
- Report generation evaluated qualitatively; formal clinical validation study needed
- EchoNet-Dynamic is a single-center dataset; multi-center validation would strengthen claims

### References
1. Ouyang et al. "Video-based AI for beat-to-beat assessment of cardiac function." Nature, 2020.
2. Yang et al. "Advancing Multimodal Medical Capabilities of Gemini." arXiv:2405.03162, 2024.
3. Lang et al. "Recommendations for Cardiac Chamber Quantification." JASE, 2015.
4. Johnson et al. "FAISS: A Library for Efficient Similarity Search." arXiv:1702.08734, 2017.
