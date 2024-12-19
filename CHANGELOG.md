# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- You can pass `None` for `lm_logits` in `ICLMetric.update()` to support pipeline parallelism.

## [v0.4.0](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.4.0) - 2024-12-18

### Added

- Added more downstream tasks from the model ladder.

## [v0.3.0](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.3.0) - 2024-12-18

### Added

- Allowed passing additional kwargs to the task through `build_task()`.
- Added the option to fix the context length of every batch to the model's context length.

## [v0.2.0](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.2.0) - 2024-10-29

### Added

- Added `ICLMultiChoiceTaskDataset.max_sequence_length` property.

## [v0.1.0](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.1.0) - 2024-10-28

### Added

- Added in-loop evals from original OLMo repo.
