# Methodology figures — index

This directory contains the design briefs and any prebuilt schematics
for the seven figures referenced from `paper/methods.tex`. Each brief
specifies panel layout, palette, data needed, a draft caption, and a
"not-to-include" checklist so an illustrator can produce the final
figure without further context.

| #   | LaTeX label         | Title                                            | Brief file                              | Prebuilt artefact (if any)                                                                          |
| --- | ------------------- | ------------------------------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1   | `fig:acquisition`   | Acquisition and pairing overview                 | `Fig1_acquisition_brief.txt`            | none (requires anonymised paired 3T/7T magnitude slices)                                            |
| 2   | `fig:registration`  | Motion correction and inter-scan registration    | `Fig2_registration_brief.txt`           | none (requires per-frame motion-correction output + 7T→3T temporal MIPs)                            |
| 3   | `fig:segmentation`  | Angiographic synthesis and vascular masking      | `Fig3_segmentation_brief.txt`           | none (needs A(x), 7T mask provenance stages, and per-dataset Dice numbers)                          |
| 4   | `fig:architecture`  | Network architecture (CBAM-DNS / CBAM-SupRes)    | `Fig4_architecture_brief.txt`           | `fig_sr4dflownet_arch.png` (matplotlib draft — **does NOT yet include the CBAM block**; redraw)     |
| 5   | `fig:loss`          | Loss partition and evaluation regions            | `Fig5_loss_brief.txt`                   | none (needs one masked axial slice + a small surface zoom)                                          |
| 6   | `fig:tasks_loocv`   | Task configurations and patient-level LOOCV      | `Fig6_tasks_loocv_brief.txt`            | none (pure schematic)                                                                               |
| 7   | `fig:inference`     | Inference and qualitative reconstruction         | `Fig7_inference_qualitative_brief.txt`  | none (built from real LOOCV held-out subject's prediction + re-segmented surface)                   |

## How to use this directory

- Each `FigN_*_brief.txt` is self-contained: it states the panel
  breakdown, the imaging slices or model outputs needed, the palette,
  the typography, the caption draft, and the "do not include" list
  (no patient identifiers, no scanner serials, etc.).
- Figures 1–3 and 7 require **real cohort data** (one anonymised
  representative subject; reuse the same subject across figures for
  visual continuity).
- Figures 4–6 are pure schematics and need no patient data.
- The matplotlib script `make_sr4dflownet_arch.py` produced
  `fig_sr4dflownet_arch.png` as a draft of Figure 4, but the CBAM
  attention block is missing from that draft and **must be added
  before the figure is used in the manuscript** (see Fig4 brief).
- The matplotlib script `make_pipeline_overview.py` was retained
  because it is a useful schematic of the overall pipeline; it does
  **not** correspond to any single figure in the current Methodology
  and is not referenced from `methods.tex`. Treat it as supplementary
  material if needed.

## Mapping to the Methodology subsections

| Methodology subsection                                | Figures                                |
| ----------------------------------------------------- | -------------------------------------- |
| Study population and 4D Flow MRI acquisition          | Fig. 1                                 |
| Motion correction and multi-field-strength regist.    | Fig. 2                                 |
| Angiographic volume synthesis and vascular masking    | Fig. 3                                 |
| Network architecture: CBAM-DNS and CBAM-SupRes        | Fig. 4                                 |
| Loss function and evaluation metrics                  | Fig. 5                                 |
| Data preparation, training, and inference             | Fig. 6, Fig. 7                         |
