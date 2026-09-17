# GLiNER2 extraction experiment

Runs local entity and relation extraction on the blog draft in `../notes/problem_outline.md`.

## Setup

The virtual environment and downloaded model weights live outside the vault:

```bash
python3.13 -m venv ~/.virtualenvs/relgraph
~/.virtualenvs/relgraph/bin/pip install -r code/requirements.txt
```

## Run

From `Filip/Projects/Relational_Graph_Extraction/`:

```bash
~/.virtualenvs/relgraph/bin/python \
  code/run_gliner2.py notes/problem_outline.md \
  --output code/output/problem_outline_gliner2.json
```

The script strips lightweight Markdown syntax, runs the local
`fastino/gliner2-base-v1` checkpoint, and stores the schema, timing, source hash,
entities, relations, confidence scores, and character spans in JSON.
