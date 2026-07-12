## ADDED Requirements

### Requirement: Batch YOLO-OBB inference over the 100k pool
The system SHALL provide a script that runs the fine-tuned YOLO-OBB checkpoint (`wmr_char_attention/outputs/detect_obb/train/weights/best.pt`) over all images in `data/100k/wm_100k/wm_100k/` on the SLURM server, producing a COCO-shaped JSON of oriented-box detections.

#### Scenario: Run inference job
- **WHEN** the prepared SLURM job is submitted on `cndt_thangcpd@slurm.uit.edu.vn`
- **THEN** it writes a COCO-shaped JSON (`images[]`, `annotations[]`) whose `annotations[].bbox`/`segmentation` are the OBB model's oriented-box detections, and whose `attributes` carries at least `angle` and detection confidence

#### Scenario: Job is not auto-submitted
- **WHEN** this change is applied
- **THEN** the inference script and SLURM submission file are created but not executed automatically — submission requires explicit user action

### Requirement: Supplement candidates use OBB-detected boxes
The supplement view SHALL source candidate bounding boxes from `predictions_obb_100k.json` and join in recognized `text` from `data/100k/results_v2/e2e/label.json` by matching `file_name`.

#### Scenario: Candidate has both an OBB box and existing text
- **WHEN** an image's `file_name` appears in both `predictions_obb_100k.json` and `results_v2/e2e/label.json`
- **THEN** the supplement view shows that image using the OBB-detected box with the text from `results_v2/e2e/label.json`

#### Scenario: Candidate has an OBB box but no matching text entry
- **WHEN** an image's `file_name` appears in `predictions_obb_100k.json` but not in `results_v2/e2e/label.json`
- **THEN** the supplement view shows that candidate with an empty text field rather than excluding it

#### Scenario: Image has no OBB detection
- **WHEN** an image in the 100k pool has no corresponding entry in `predictions_obb_100k.json`
- **THEN** it does not appear in the supplement view
