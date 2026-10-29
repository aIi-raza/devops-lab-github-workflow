"""Extract small, genuine raw-feature examples from the official KDDTest+ holdout file."""
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import COLUMN_NAMES
from src.preprocessing import SELECTED_FEATURES
data = pd.read_csv(ROOT / 'data' / 'KDDTest+.txt', header=None, names=COLUMN_NAMES)
data['actual_label'] = data['class'].map(lambda value: 'Normal' if value == 'normal' else 'Attack')
out = Path(__file__).resolve().parent
for filename, frame in [
    ('sample_normal.csv', data[data.actual_label == 'Normal'].head(8)),
    ('sample_attack.csv', data[data.actual_label == 'Attack'].head(8)),
    ('sample_mixed.csv', pd.concat([data[data.actual_label == 'Normal'].head(4), data[data.actual_label == 'Attack'].head(4)])),
]:
    frame[SELECTED_FEATURES + ['actual_label']].to_csv(out / filename, index=False)
    print(f'Wrote {filename}: {len(frame)} records')
