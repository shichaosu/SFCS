# SFCS manuscript analysis code

This repository contains the retained analysis and figure-generation scripts for
*Brain Structure Shapes Function through higher-order Functional Interactions*.

## Scope and version

The repository has been reduced to manuscript analyses, data preparation, their
dependencies, and result-generation scripts. Exploratory comparisons, unrelated
atlas utilities, alternative group-statistics branches, and redundant previews
have been removed. The retained scientific scripts were not changed by this
cleanup.

The pre-cleanup snapshot remains available at commit
[fafe7503eb9930ef0ef7d9b8ef54c8d524d2b44a](https://github.com/shichaosu/SFCS/tree/fafe7503eb9930ef0ef7d9b8ef54c8d524d2b44a).
Repository cleanup does not establish that the public script defaults are the
final experiment configuration. Match the input files, parameters and analysis
version to the records for the reported result.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

MATLAB is required for the `.m` scripts. The structural communication functions
in `scripts/SC/fcn` are dependencies of the MATLAB pipelines; add that directory
to the MATLAB path. Retain the attribution in third-party source files.

## Data inputs

No ADNI participant-level imaging or clinical data are distributed here.
Researchers must obtain data through the applicable ADNI access process.

Scripts use relative paths under `data/` and `outputs/`. Before running a
script, configure its input and output paths. The analysis inputs include
DK68 regional features and connectivity matrices, HOI tensors, assessment
tables, demographic records and atlas files. Keep ROI order consistent across
all inputs. File or scan identifiers are not automatically unique participant
identifiers.

HOI tensors are computed using the external
[hoi package](https://brainets.github.io/hoi/). The retained HOI script thresholds
existing tensors; it is not a BOLD-to-O-information computation entry point.
Upstream preprocessing, HOI package settings, cohort selection and external
enrichment settings must accompany the corresponding experiment records.
The repository does not contain those input data or final run records.

## Analysis workflow

The scripts are standalone entry points, not an automated end-to-end runner.
Configure the paths at the top of each script (or its command-line arguments)
and select the implementation used for the corresponding result.

1. **Prepare inputs:** `scripts/check`, SC format conversion/normalization and
   cohort organization in `scripts/Coefficient/Group_sort.py`.
2. **Construct structural components:** `scripts/MSN/compute_all.py`, the
   applicable MSN transformation, and the applicable SC pipeline in
   `scripts/SC/fcn`. The two SC pipeline variants have different input handling;
   they are alternatives, not consecutive processing steps.
3. **Threshold HOI tensors:**
   `scripts/HOI/HOI_percent_sparsification_pos_neg_Save.py`.
4. **Fit SFCS and reconstruct HOI:**
   `scripts/HSFC/GLM_Sparsify_Pos.py` and
   `scripts/HSFC/GLM_Sparsify_Neg.py`.
5. **Prepare SFCS summaries:** the retained coefficient transformation, individual
   degree and group-summary scripts in `scripts/Coefficient`. Use the actual
   coefficient transformation and aggregation order for the reported result.
6. **Predict cognition:** `scripts/table/BrainGNN_addLoss_orign_ROI_rank.py`,
   with the applicable target preparation and table assembly in `scripts/table`.
7. **Compare AD and CN:** `scripts/Coefficient/Group_difference_IPW2.py`.
8. **Decode regional statistics:** `scripts/brainstat/extract_T_value.py`,
   `region_to_vertex.py`, `meta_decoding_Single.py`, and
   `Gene_analysis_PLS.py`, using their configured inputs.
9. **Generate manuscript figures and tables:** retained scripts in
   `scripts/fig_result`, `scripts/visualization` and the PLS supplementary
   figure script.

## Result-generation entry points

| Result type | Retained scripts |
| --- | --- |
| Scan/participant counts and demographics | `fig_result/Sub_Info.py` |
| Regional reconstruction correlations | `fig_result/HOI_both_correlation_scatter_boxplot.py` |
| Example slice heat maps | `fig_result/Top_slice_heatmap_ECDF.py`, `fig_result/Top_sub_heatmap_ECDF_0.3.py` |
| Example real/reconstructed HOI scatter | `fig_result/plot_individual_scatter_hist.py` |
| Within/cross-sample reconstruction comparison | `fig_result/hoi_reconstruction_correlation_scatter.py` |
| Group coefficient summaries | `fig_result/plot_average_coffs.py`, `fig_result/plot_raincloud_both.py` |
| Cortical rendering | `fig_result/visualization_aparc_single.m` and the relevant ROI export |
| Cognitive result panels | Retained performance scripts in `visualization` and `fig_result/plot_scatter_behavior.py` |
| HOI threshold rationale and sensitivity | `fig_result/plot_hoi_tail_rationale.py` |
| Structural-component correlations | `fig_result/cn_feature_corr_heatmap.py` |
| PLS supplementary results | `brainstat/plot_gene_pls_pos_supplement.py` |
| Enrichment display | `visualization/plot_go_enrichment_cluster_bubble_nolabel.py` |

Paths in the table are relative to `scripts/`. Supplementary results without
thresholding or with the absolute ±0.3 threshold require their corresponding
inputs and experiment settings; the default 20% configuration is not a substitute.
File names containing `0.3` do not by themselves establish the input threshold.

## License

This repository is released under the MIT License; see [LICENSE](LICENSE).
Preserve any separate licensing and attribution for third-party functions.
