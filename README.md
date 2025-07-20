# OLMo-in-loop-evals

Code for in-loop evaluation tasks used by the OLMo training team.

## Installation

```
pip install ai2-olmo-eval
```

## Release process

### Steps

1. Update the version in `src/olmo_eval/version.py`.
2. Run the release script:

    ```bash
    ./src/scripts/release.sh
    ```

    This will commit the changes to the CHANGELOG and `version.py` files and then create a new tag in git
    which will trigger a workflow on GitHub Actions that handles the rest.

### Fixing a failed release

If for some reason the GitHub Actions release workflow failed with an error that needs to be fixed, you'll have to delete the tag on GitHub. Once you've pushed a fix you can simply repeat the steps above.
