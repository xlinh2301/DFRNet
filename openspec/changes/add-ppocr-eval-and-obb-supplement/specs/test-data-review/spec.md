## MODIFIED Requirements

### Requirement: Delete a bad annotation
The system SHALL allow the reviewer to delete an annotation from the working copy `data/DATA_COCO_v2/annotations/instances_test.json`, removing the associated image entry if that annotation was the image's last one. The original `data/DATA_COCO/annotations/instances_test.json` is never modified.

#### Scenario: Working copy is created on first edit
- **WHEN** the first delete or edit request of the process lifetime is received and `data/DATA_COCO_v2/` does not yet exist
- **THEN** the system recursively copies `data/DATA_COCO/` to `data/DATA_COCO_v2/` before applying the edit

#### Scenario: Delete annotation, image has other annotations
- **WHEN** the reviewer deletes an annotation for an image that has other remaining annotations
- **THEN** the system removes only that annotation from `data/DATA_COCO_v2/annotations/instances_test.json`'s `annotations[]` and keeps the image entry

#### Scenario: Delete annotation, image has no other annotations
- **WHEN** the reviewer deletes the only annotation of an image
- **THEN** the system removes the annotation from `annotations[]` and also removes the image's entry from `images[]`, both in `data/DATA_COCO_v2/annotations/instances_test.json`

#### Scenario: Deletion does not touch other assets
- **WHEN** an annotation or image entry is deleted
- **THEN** the system does not modify `yolo_obb/test`, `data/DATA_COCO/` (the original), or delete the image file from `data/DATA_COCO_v2/images/test`

### Requirement: Edit a wrong text label
The system SHALL allow the reviewer to edit the `attributes.text` value of an existing annotation in the working copy `data/DATA_COCO_v2/annotations/instances_test.json`.

#### Scenario: Edit text label
- **WHEN** the reviewer submits a new text value for an annotation
- **THEN** the system updates that annotation's `attributes.text` in `data/DATA_COCO_v2/annotations/instances_test.json` and leaves `bbox`/`segmentation` unchanged, without touching `data/DATA_COCO/`

### Requirement: Safe writes to ground truth
The system SHALL create a one-time timestamped backup of `data/DATA_COCO_v2/annotations/instances_test.json` before its first write in a server session, and persist every accepted edit/delete directly to `data/DATA_COCO_v2/annotations/instances_test.json`.

#### Scenario: First mutating request in a session
- **WHEN** the first delete or edit request is processed after server startup (and after the working copy exists)
- **THEN** the system copies the current `data/DATA_COCO_v2/annotations/instances_test.json` to a timestamped backup file in the same directory before writing the change

#### Scenario: Subsequent mutating requests in the same session
- **WHEN** a delete or edit request is processed after the session's backup has already been created
- **THEN** the system writes directly to `data/DATA_COCO_v2/annotations/instances_test.json` without creating another backup
