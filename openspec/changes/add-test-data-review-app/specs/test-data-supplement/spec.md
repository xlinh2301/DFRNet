## ADDED Requirements

### Requirement: Browse and filter 100k candidates
The system SHALL display a paginated, sortable grid of candidates from `data/100k/results_v2/e2e/label.json`, sortable and filterable by `attributes.yolo_conf`.

#### Scenario: Sort candidates by confidence
- **WHEN** the reviewer requests candidates sorted by `yolo_conf` descending
- **THEN** the system returns a page of candidates ordered from highest to lowest `yolo_conf`

#### Scenario: Filter candidates by confidence range
- **WHEN** the reviewer supplies a minimum and/or maximum `yolo_conf` filter
- **THEN** the system returns only candidates whose `attributes.yolo_conf` falls within that range

### Requirement: Manually select and import candidates into the test set
The system SHALL let the reviewer select one or more 100k candidate images and import them into `instances_test.json` in DATA_COCO's COCO schema, copying the source image file into `data/DATA_COCO/images/test`.

#### Scenario: Import a selected candidate
- **WHEN** the reviewer selects a candidate image and confirms import
- **THEN** the system appends a new `images[]` entry (with a fresh id) and corresponding `annotations[]` entry (`category_id: 1`, original `bbox`/`segmentation`, `attributes.text` only) to `instances_test.json`, and copies the image file from `data/100k/wm_100k/wm_100k/` to `data/DATA_COCO/images/test/`

#### Scenario: New ids do not collide with existing test-set ids
- **WHEN** an import occurs
- **THEN** the system assigns the new image id and annotation id(s) as one greater than the current maximum ids present in `instances_test.json`

#### Scenario: Re-importing an already-imported image
- **WHEN** the reviewer selects a candidate whose `file_name` already exists in `instances_test.json`
- **THEN** the system skips adding a duplicate JSON entry and skips re-copying the file if it already exists at the destination

#### Scenario: Imported annotation excludes 100k-only fields
- **WHEN** a candidate annotation containing `attributes.yolo_conf` and `attributes.angle` is imported
- **THEN** the resulting `instances_test.json` annotation's `attributes` contains only `text`
