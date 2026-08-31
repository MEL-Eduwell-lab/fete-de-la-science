# dataset_reptilia_amphibia

Source: https://huggingface.co/datasets/imageomics/rare-species/tree/main/data

Smaller version I made from that dataset: only the `Reptilia` and `Amphibia`
classes (2784 rows), split into two parquet files:

| file                          | rows | contents                                              |
| ----------------------------- | ---- | ----------------------------------------------------- |
| `train-00000-of-00001.parquet`| 2684 | Reptilia: 1524, Amphibia: 1160                        |
| `test-00000-of-00001.parquet` | 100  | held-out set, balanced 20 rows per `order` (5 orders) |

Only the three columns `mission-finale/score-final.py` uses are kept:

- `file_name` — the image, still a HuggingFace `Image` feature (decodes to PIL)
- `class` — `Reptilia` / `Amphibia`, used to split the train set
- `order` — the training label / test ground truth; one of `Anura`,
  `Caudata`, `Crocodilia`, `Squamata`, `Testudines`

Images are pre-resized to **224×224 JPEG** (~8 KB each) since the script
always feeds them through `transforms.Resize((224, 224))`. Dropping the
10 unused taxonomy/id columns and downscaling takes the pair from
~993 MB to ~23 MB with no change to `score-final.py` behavior.

## How it was generated

```python
import io

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

SRC = "dataset_imageomics_rare-species"  # the 9 original train-*.parquet shards
KEEP = ["Amphibia", "Reptilia"]
TEST_PER_ORDER = 20

source = ds.dataset(SRC, format="parquet")

# Filter on the string `class` column (no image decoding happens here).
table = source.to_table(filter=pc.field("class").isin(KEEP))

# Keep the HuggingFace feature metadata so `datasets` still sees
# `file_name` as an Image column.
meta = pq.ParquetFile(source.files[0]).schema_arrow.metadata
table = table.replace_schema_metadata(meta)

# Hold out a balanced test set: TEST_PER_ORDER rows per taxonomic order.
df = table.to_pandas()
test_df = (
    df.groupby("order", group_keys=False)
    .apply(lambda g: g.sample(n=TEST_PER_ORDER, random_state=0))
)
train_df = df.drop(test_df.index)


def shrink(rec):
    """Re-encode one image record as a 224x224 JPEG."""
    img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {"bytes": buf.getvalue(), "path": rec["path"]}


for name, part in [("train", train_df), ("test", test_df)]:
    part = part[["file_name", "class", "order"]].copy()  # drop unused columns
    part["file_name"] = part["file_name"].map(shrink)    # downscale images
    out = pa.Table.from_pandas(part, preserve_index=False)
    out = out.replace_schema_metadata(meta)  # keeps `file_name` as an Image feature
    pq.write_table(out, f"dataset_reptilia_amphibia/{name}-00000-of-00001.parquet")
```

## How to load it

```python
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={
        "train": "dataset_reptilia_amphibia/train-00000-of-00001.parquet",
        "test": "dataset_reptilia_amphibia/test-00000-of-00001.parquet",
    },
)
train, test = ds["train"], ds["test"]
```
