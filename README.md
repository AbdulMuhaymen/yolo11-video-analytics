# Video Indexing at Speed

Indexes `person`, `car`, `dog`, `laptop`, and `cell phone` detections from MP4
videos stored in Parquet. Output is one JSON object per video-second.

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare.py
python run.py --input data/ --output results.jsonl --batch-size 8
python tools/validate_output.py results.jsonl
```

The default mode evaluates every decoded frame. After measuring accuracy against
that reference, `--samples-per-second 3` (or 5/7) enables evenly spread sampling.
On an M4 Pro use `--device mps --batch-size 2`; on a Colab T4 start with
`--device cuda:0 --batch-size 16`.

Input defaults to columns `id` and `video`. The video value may be MP4 bytes, a
Hugging Face-style `{bytes, path}` struct, or a filesystem path. Override the
names with `--id-column` and `--video-column` once the dataset schema is known.
Inspect the first real shard without dumping its large video bytes:

```bash
python tools/inspect_parquet.py data/train-00000.parquet
```

Use `--resume` to append while skipping video IDs recorded in the adjacent
`.completed` manifest. A video's records are buffered and committed together,
so a corrupt video cannot leave partial output.
Errors are isolated per video and logged; a run containing errors exits with 2.

Shard fragments can be combined and checked with:

```bash
python tools/merge_fragments.py fragments/ results.jsonl.gz
python tools/validate_output.py results.jsonl.gz
```

For the full run, open `notebooks/colab_runner.ipynb` in Colab, set its repository
and schema configuration, add `HF_TOKEN` to Colab Secrets, and choose a GPU
runtime. It downloads one shard at a time to `/content` and persists completed
compressed fragments to Drive.
