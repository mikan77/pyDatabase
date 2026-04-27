# uspexdb

English | [Russian](README.ru.md)

`uspexdb` is a unified CLI for two structure-database workflows:

- `config`: metadata filtering and structure export
- `query`: contact search for a `.mol2` fragment using the `c_anchor` backend

The project is organized as a normal Python package with a small launcher at the repository root. For local development, the usual entrypoint is `./uspexdb`. For distribution to non-Python users, the project can also be packaged into a standalone `PEX scie` executable.

## Project Layout

```text
pyDatabase/
  uspexdb                     # root launcher for local use
  src/uspexdb/
    cli.py                    # CLI entrypoint: config, query, all
    paths.py                  # path resolution for config/query/database/mol2
    reference.py              # bridge to legacy helpers
    config_search/
      runner.py               # metadata search + export workflow
    query_search/
      runner.py               # MOL2 query workflow
      mol2.py                 # MOL2 parsing and query graph handling
      search.py               # contact search, export payloads, plots
      compact_cache.py        # compact graph cache build/load
      validate.py             # optional validation against reference output
      codes.py                # result/status helpers
      _c_anchor.c             # C extension source
    legacy/
      uspexdb_v2.py           # vendored legacy implementation
  scripts/
    build_extension.sh        # install package dependencies + build C extension
    build_scie.sh             # build standalone PEX scie executable
  example/
    config/
      config.json             # sample metadata search config
    query/
      query.json              # sample MOL2 contact query
      fragment_example.mol2   # sample query fragment
```

## Installation

Recommended local setup:

```bash
python3 -m venv .venv
. .venv/bin/activate
./scripts/build_extension.sh
```

`build_extension.sh` installs the project in editable mode, installs runtime dependencies, and builds the `_c_anchor` extension in place.

The main runtime dependencies are:

- `numpy`
- `pandas`
- `pyarrow`
- `ase`
- `matplotlib`
- `pymatgen`

## Command Line Usage

The root launcher supports three modes:

```bash
./uspexdb query
./uspexdb config
./uspexdb all
```

Optional explicit JSON paths:

```bash
./uspexdb query --query /path/to/query.json
./uspexdb config --config /path/to/config.json
```

`all` runs `query` first and then `config`.

## Config Workflow

`config` mode reads `config.json`, loads the target database from `database.path`, applies metadata filters, and exports the matching structures.

Typical filters come from the legacy search layer in `uspexdb_v2.py`. The exact metadata fields depend on the target database, but the current examples are built around the existing database layout used in this project.

See:

- [example/config/README.md](./example/config/README.md)
- [example/config/config.json](./example/config/config.json)

## Query Workflow

`query` mode reads `query.json`, resolves the `.mol2` fragment, prepares or reuses a compact graph cache, runs the `c_anchor` contact search, and writes structured outputs plus plots.

Important path rules:

- `query.fragment_mol2` is required
- if `fragment_mol2` is an absolute path, it is used as-is
- if `fragment_mol2` is a relative path, it is resolved relative to the directory containing `query.json`
- `database.path` is configured independently inside `query.json`

If the `.mol2` file is missing after resolution, execution stops with a clear error.

Current analysis rules:

- the only distance metric is `anchor ... contact`
- aggregate plots over all contacts are not generated
- distance, angle, and torsion distributions are generated separately for each `contact_element`

See:

- [example/query/README.md](./example/query/README.md)
- [example/query/query.json](./example/query/query.json)

## Building a Standalone Executable

For users who should not install Python packages manually, the project can be packaged into a standalone executable:

```bash
./scripts/build_scie.sh
```

This creates:

- `dist/uspexdb-scie` — standalone `PEX scie` executable

There may also be intermediate build artifacts such as:

- `dist/uspexdb` — regular PEX file
- `dist/*.whl` — wheel built for packaging
- `wheelhouse/` — local wheel cache used during scie assembly

Important:

- the produced executable is platform-specific
- a binary built on macOS will not run on a Linux cluster
- for HPC/Linux deployment, run `build_scie.sh` on a compatible Linux machine

## Examples

The `example/` directory contains minimal sample inputs for both workflows:

- [example/config](./example/config)
- [example/query](./example/query)

These are intended as starting points for real database paths and user-specific search parameters.
