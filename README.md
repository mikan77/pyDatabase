

## Русский

`final_version` — это единая версия проекта для двух задач:

- `config`: поиск по метаданным и экспорт структур
- `query`: поиск контактов по `.mol2`-фрагменту через `c_anchor_v3`

Запуск идёт через один файл:

```bash
./uspexdb config
./uspexdb query
./uspexdb all
```

Перед `query` нужно собрать C-extension:

```bash
python3 setup.py build_ext --inplace
```

Или:

```bash
./scripts/build_extension.sh
```

### Важные правила

- `config` читает `config.json`
- `query` читает `query.json`
- `database.path` задаётся отдельно в каждом JSON
- `query.fragment_mol2` может быть:
  - абсолютным путём
  - относительным путём от каталога `query.json`

Если `.mol2` не найден, выполнение останавливается с явной ошибкой.

### Графики

В текущей версии:

- расстояние считается только как `anchor ... contact`
- суммарные общие распределения не строятся
- графики строятся отдельно по `contact_element`

### Примеры

Смотри папку [example](./example):

- [example/config/README.md](./example/config/README.md)
- [example/query/README.md](./example/query/README.md)

### Минимальные зависимости

- `numpy`
- `pandas`
- `pyarrow`
- `ase`
- `matplotlib`

---

## English

`final_version` is a unified version of the project for two workflows:

- `config`: metadata search and structure export
- `query`: `.mol2`-based contact search using `c_anchor_v3`

Everything is launched through a single entrypoint:

```bash
./uspexdb config
./uspexdb query
./uspexdb all
```

Before running `query`, build the C-extension:

```bash
python3 setup.py build_ext --inplace
```

Or:

```bash
./scripts/build_extension.sh
```

### Important rules

- `config` reads `config.json`
- `query` reads `query.json`
- `database.path` is configured independently in each JSON file
- `query.fragment_mol2` may be:
  - an absolute path
  - a path relative to the directory containing `query.json`

If the `.mol2` file is missing, execution stops with a clear error.

### Plots

In the current version:

- distance is computed only as `anchor ... contact`
- no aggregate distributions over all contacts are generated
- plots are generated separately by `contact_element`

### Examples

See the [example](./example) directory:

- [example/config/README.md](./example/config/README.md)
- [example/query/README.md](./example/query/README.md)

### Minimal dependencies

- `numpy`
- `pandas`
- `pyarrow`
- `ase`
- `matplotlib`
