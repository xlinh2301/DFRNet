## ADDED Requirements

### Requirement: Batch PPOCRv5 inference over the test set
The system SHALL provide a script that runs the fine-tuned PPOCRv5 checkpoint (`water_meter_amr/outputs/ppocr_v5_paddle/best_accuracy.pdparams`) over every annotation crop in the DATA_COCO test set on the SLURM server, producing a JSON file mapping each annotation to its predicted text.

#### Scenario: Run inference job
- **WHEN** the prepared SLURM job is submitted on `cndt_thangcpd@slurm.uit.edu.vn`
- **THEN** it writes a JSON file with one predicted text per test annotation, keyed by `file_name` (or `file_name#annotation_id` when an image has multiple annotations)

#### Scenario: Job is not auto-submitted
- **WHEN** this change is applied
- **THEN** the inference script and SLURM submission file are created but not executed automatically — submission requires explicit user action

### Requirement: Eval view shows only mismatched predictions
The web app SHALL provide an Eval view that loads `predictions_ppocrv5_test.json` alongside the working test set and displays only annotations whose predicted text differs from the ground-truth `attributes.text` (exact, case-sensitive match).

#### Scenario: Mismatch is shown
- **WHEN** an annotation's PPOCRv5 prediction differs from its `attributes.text`
- **THEN** the Eval view lists that image with both the predicted text and the ground-truth text visible

#### Scenario: Match is hidden
- **WHEN** an annotation's PPOCRv5 prediction equals its `attributes.text` exactly
- **THEN** the Eval view does not show that annotation

#### Scenario: Annotation without a prediction
- **WHEN** an annotation has no corresponding entry in `predictions_ppocrv5_test.json`
- **THEN** the Eval view does not show that annotation (treated as not-yet-evaluated, not as a mismatch)

### Requirement: Eval view supports the same edit/delete actions as Review
The Eval view SHALL reuse the review app's delete-annotation and edit-text actions, applied to the working copy (`data/DATA_COCO_v2`).

#### Scenario: Delete from Eval view
- **WHEN** the reviewer deletes a mismatched annotation from the Eval view
- **THEN** the system applies the same deletion/cascade behavior as the Review view, against `data/DATA_COCO_v2/annotations/instances_test.json`

#### Scenario: Edit from Eval view
- **WHEN** the reviewer edits a mismatched annotation's text from the Eval view
- **THEN** the system updates `attributes.text` in `data/DATA_COCO_v2/annotations/instances_test.json`, same as the Review view
