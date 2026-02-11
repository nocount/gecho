# Gecho Demo Video Script (3 minutes)

## [0:00 - 0:20] Hook

**Visual**: Split screen -- overworked sonographer on left, rural clinic waiting room on right.

**Narration**:
"Six million echocardiograms are performed in the United States every year. Each one requires 20 to 30 minutes of expert interpretation. In rural communities, patients wait weeks for a cardiologist to review their scan. What if AI could draft that report in seconds?"

## [0:20 - 0:45] The Problem

**Visual**: Statistics overlay on echo footage.

**Narration**:
"Echocardiogram interpretation is one of cardiology's biggest bottlenecks. It's subjective, time-consuming, and the expert shortage is growing. Ejection fraction -- the single most important cardiac metric -- varies by up to 13% between readers. We need a system that's fast, consistent, and transparent."

## [0:45 - 1:15] Introducing Gecho

**Visual**: Architecture diagram animating step by step.

**Narration**:
"Gecho -- Gemma Echo -- is a retrieval-augmented generation system built entirely on Google's HAI-DEF models. It works in three steps:

First, OpenCV extracts the key cardiac frames -- End-Diastole and End-Systole -- from the uploaded video.

Second, MedSigLIP encodes those frames and searches a database of 10,000 labeled echos from EchoNet-Dynamic. If the five most similar hearts all have an ejection fraction around 35%, that's powerful evidence.

Third, MedGemma takes the image plus that retrieval context and generates a structured clinical report."

## [1:15 - 2:15] Live Demo

**Visual**: Screen recording of Gradio dashboard.

**Narration**:
"Let's see it in action. I'll upload an echocardiogram video.

The left panel shows the extracted End-Diastole and End-Systole frames.

In the center, MedSigLIP's zero-shot classification shows 72% confidence for moderate left ventricular dysfunction. Below that, the five most similar cases from our database -- all with ejection fractions between 30 and 40 percent.

On the right, MedGemma has generated a structured report: visual assessment of the left ventricle, an EF estimate of 33 to 38 percent, clinical impression, and -- critically -- a limitations section reminding the clinician that this is AI-assisted, not AI-decided.

Notice the Approve and Edit buttons. This is designed for human-in-the-loop workflow. The AI drafts, the cardiologist reviews."

## [2:15 - 2:40] Technical Highlights

**Visual**: VRAM diagram, model cards.

**Narration**:
"The entire system runs on a single P100 GPU with 16 gigabytes of VRAM. MedSigLIP builds the search index first at just 1.6 gigabytes, then MedGemma loads at 8 gigabytes for report generation. No quantization hacks, no multi-GPU setup -- just smart memory management.

Both models are from Google's HAI-DEF suite, purpose-built for medical imaging and clinical language."

## [2:40 - 3:00] Impact & Close

**Visual**: Before/after workflow comparison.

**Narration**:
"Gecho doesn't replace cardiologists -- it gives them a head start. A 20-minute interpretation becomes a 2-minute review. Rural clinics get same-day preliminary reads. And every report comes with the evidence trail: here are the similar cases, here's the confidence score, here's what the AI saw.

Transparent AI for the hearts that need it most. That's Gecho."
