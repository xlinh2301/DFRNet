## ADDED Requirements

### Requirement: Review test images with bbox and text overlay
The system SHALL display a paginated grid of all images in `instances_test.json`, rendering each image's bbox(es) and `attributes.text` value overlaid on the image.

#### Scenario: Load a page of the review grid
- **WHEN** the reviewer opens the review view
- **THEN** the system shows a page (default 24) of test images, each with its bbox drawn over the image and its text label displayed

#### Scenario: Navigate pages
- **WHEN** the reviewer requests the next/previous page
- **THEN** the system returns the corresponding slice of images from `instances_test.json` in the same format

### Requirement: Delete a bad annotation
The system SHALL allow the reviewer to delete an annotation from `instances_test.json`, removing the associated image entry if that annotation was the image's last one.

#### Scenario: Delete annotation, image has other annotations
- **WHEN** the reviewer deletes an annotation for an image that has other remaining annotations
- **THEN** the system removes only that annotation from `annotations[]` and keeps the image entry

#### Scenario: Delete annotation, image has no other annotations
- **WHEN** the reviewer deletes the only annotation of an image
- **THEN** the system removes the annotation from `annotations[]` and also removes the image's entry from `images[]`

#### Scenario: Deletion does not touch other assets
- **WHEN** an annotation or image entry is deleted
- **THEN** the system does not modify `yolo_obb/test` files or delete the image file from `images/test`

### Requirement: Edit a wrong text label
The system SHALL allow the reviewer to edit the `attributes.text` value of an existing annotation in `instances_test.json`.

#### Scenario: Edit text label
- **WHEN** the reviewer submits a new text value for an annotation
- **THEN** the system updates that annotation's `attributes.text` in `instances_test.json` and leaves `bbox`/`segmentation` unchanged

### Requirement: Safe writes to ground truth
The system SHALL create a one-time timestamped backup of `instances_test.json` before its first write in a server session, and persist every accepted edit/delete directly to `instances_test.json`.

#### Scenario: First mutating request in a session
- **WHEN** the first delete or edit request is processed after server startup
- **THEN** the system copies the current `instances_test.json` to a timestamped backup file in the same directory before writing the change

#### Scenario: Subsequent mutating requests in the same session
- **WHEN** a delete or edit request is processed after the session's backup has already been created
- **THEN** the system writes directly to `instances_test.json` without creating another backup
