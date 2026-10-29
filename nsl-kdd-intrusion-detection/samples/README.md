# NSL-KDD CSV samples

These small CSVs are extracted from the official `data/KDDTest+.txt` file, not fabricated. The artifact-training command currently combines KDDTrain+ and KDDTest+ before its own random split, so a row from this file is not guaranteed to be unseen during model training. Each contains the 25 selected raw features in the same order used by the research code plus `actual_label` for evaluation. `actual_label` is ground truth only and is never used as a model input.

- `sample_normal.csv`: 8 records labelled Normal
- `sample_attack.csv`: 8 records labelled Attack
- `sample_mixed.csv`: 4 Normal and 4 Attack records

For the Flask CSV endpoint, choose **Raw selected features**. The backend encodes categorical values and scales them using the saved training objects. To use `preprocessed` mode, provide the model-specific expected feature columns already encoded and scaled; the FE model requires all 32 columns, while other models require the 25 selected columns. No second transformation is applied in that mode.

Regenerate samples after placing the dataset files locally:

```bash
python make_samples.py
```
