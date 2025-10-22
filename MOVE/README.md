# MOVE package internals

This directory houses the core MOVE implementation. The public entry points
are:

- `MOVE.move.MOVE` – the evolutionary algorithm class.
- `MOVE.move.main()` – CLI-compatible runner used by `python -m MOVE` and
	`python -m move`.

While developing, you can invoke the module directly from here with:

```bash
python -m MOVE --help
```

In typical usage you should run commands from the repository root so that
relative paths (e.g., `default.json`, `data/`, `output/`) resolve correctly.
