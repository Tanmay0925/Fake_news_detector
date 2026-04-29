# Dataset Folder

Place your dataset CSV file here and rename it to `news.csv`.

## Required Format

Your CSV must have at least two columns:

| Column Name (any of these)         | Description                    |
|------------------------------------|--------------------------------|
| `text` / `content` / `article`     | The news article body          |
| `label` / `class` / `target`       | `0` = Real, `1` = Fake         |

## Recommended Datasets (Free)

### Option 1 — WELFake Dataset (Kaggle)
- Link: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
- ~72,000 articles, balanced, high accuracy results

### Option 2 — ISOT Fake News Dataset
- Link: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php
- Two CSVs (Fake.csv and True.csv) — merge them with a label column

### Option 3 — FakeNewsNet
- Link: https://github.com/KaiDMML/FakeNewsNet

## How to Merge ISOT Dataset (if you use it)

```python
import pandas as pd

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake['label'] = 1
real['label'] = 0

df = pd.concat([fake[['text','label']], real[['text','label']]], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("news.csv", index=False)
print("Merged dataset saved as news.csv")
```
