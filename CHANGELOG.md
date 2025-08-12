# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Pinned datasets, tokenizers, pyarrow versions.

## [v0.8.5](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.8.5) - 2025-07-20

### Removed

- Remove `sklearn` and `numpy` as depedencies. Manual implementation of F1 score.

## [v0.8.4](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.8.4) - 2025-06-05

- Add BOS token, when the BOS token exists in the tokenizer

## [v0.8.3](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.8.3) - 2025-05-27

- Fix speed problem for BPB/RC tasks

## [v0.8.2](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.8.2) - 2025-05-17

- Add few-shot HumanEval
- Fix prompting setup for MT MBPP. For more details, see https://github.com/allenai/oe-eval-internal/pull/489

## [v0.8.1](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.8.1) - 2025-05-17

- Add mt MBPP and Minerva Math 500

## [v0.8.0](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.8.0) - 2025-05-17

- Add fast MCQA

## [v0.7.2](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.7.2) - 2025-05-16

- Add basic skills evals

## [v0.7.1](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.7.1) - 2025-04-02

- Fix normalization to match the OLMES standard

## [v0.7.0](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.7.0) - 2025-03-10

- Add in-loop GSM, Minerva, MBPP, HumanEval

## [v0.6.1](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.6.1) - 2025-02-10

### Fixed

- Ensure queries are always a multiple of 128.

## [v0.6.0](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.6.0) - 2024-12-19

## [v0.5.0](https://github.com/allenai/OLMo-in-loop-evals/releases/tag/v0.5.0) - 2024-12-18

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
