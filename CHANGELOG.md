# Changelog

## 1.6.0

- Change `model_id` to `finetuning_id` for API endpoints. "Model" now only refers to the underlying foundation model
- Add dataset management to API

## 1.5.5

- Fix CachingSubCaseDSV2 only shuffling once per training leading to partial training for many supercases

## 1.5.4

- Make default orientation for API volumes fix_orientation=never
- Reduce default number of slices in 3D API training to 1, 3 from 8, 16

## 1.5.3

- Improve logging
- Enable publishing a branch-specific package to internal package registry
- Fix large groups resulting from drawing the same subject multiple times in a batch in API
- Fix default value of `replay_augmentations_for_groups` to True for volume mask in API
- Fix state check of training in API
- Change model download URL

## 1.5.2

- Fix size mismatch of representation tensor and mask from GeoMask in API

## 1.5.1

- Fix torch GradScaler dependency

## 1.5.0

- Support GeoMask targets in API
- Preserve order in extract_ids_from_batch
- Solve out of memory problem with 3D tasks by disconnecting tensor views of slices from their original Volume
- Improve distributed logging via OpenTelemetry
- Implement local logging for Classification and Segmentation

## 1.4.1

- Fixed a bug that broke fine-tuning in 1.4.0. The `FineTuner` wired up the foundation model's modules before copying them for fine-tuning. As a result, the foundation model `ws.fm` was modified during fine-tuning, and the blocks intended for fine-tuning got no training signal.
- Removed Squeezer from list of modules that are fine-tuned by default because it is used for compression and gets no training signal in fine-tuning for data compressed as neural tokens.
- Wired up functional predict in API directly. Gives access to all parameters.
- Added `reset-db` command to CLI to delete everything M3 put into the DB.
- Fixed bug where `attn_weights` were always computed by a module "grouper" instead of the grouper used by a task.
- Fixed bug where attn_weights were only computable for subjects with more than one token
- Improved logging for functionals `FineTune` and `Predict`

## 1.4.0

- Replaced Sphinx docs with static markdown
- Stopped building Docker image with dependencies included
- Major API changes
  - Removed LGBModel. Now only deep learning finetuning is supported.
  - Replaced M3Logger with OpenTelemetry
  - Restructured API to make Labelstudio integration optional
  - API now persists model settings in key-value store
  - Finetuning now only persists modules that were actually updated.
  - Downloading images from data directly imported into Labelstudio was removed. Instead, the Labelstudio integration now rewrites subjects from Labelstudio into self-contained Base64 URLs.
- Changed pipeline to use a public Docker image as base image
- Replaced poetry with uv
- Replaced wandb.Image logging with file-based logging to enable data privacy with online W&B
- Added optional discrimination loss for approximation task
- Set DataLoader's `pin_memory` to True by default

## 1.3.3

- Fixed pytest dependency

## 1.3.2

- If available, return attention weights in API for classification tasks
- Add support for ALiBi positional encoding in MHAGrouper, and use it for 3D processing
- Fix issue where UnifySizes changed the order of elements in a batch
- Replace unused AttentionDecoder with Segformer
- Rename NativeBlocks to M3Model

## 1.3.1

- Fixed keyerror itemindex

## 1.3.0

- Added Model Service REST API

## 1.2.1

- Fixed PATH in Docker base image.

## 1.2.0

- Augmentations can now be replayed over multiple patches (mmm.transforms).
- Added data loading for tomographic imaging (mmm.volume3d).
- Added celery functions (mmm.api.api_worker).
- Introduced `GroupUsage` to control the behavior of the grouper.

## 1.1.0

- Tissue Concepts model released! [link](https://www.sciencedirect.com/science/article/pii/S0010482524017062?via%3Dihub)
- implement SurvivalPredictionTask time-to-event targets
- introduce attention based multiple instance learning components: `CLAMReducer` and `AttentionPoolingReducer`
- CachingSubCaseDS uses CachingSubCaseDSSampler to customize behaviour instead of function `decide_removal`.

## 1.0.3

- use default for local cache path if ML_DATA_CACHE is not set
- remove unused files

## 1.0.2

- initial public release
- add minimal example and documentation
