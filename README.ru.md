# uspexdb

[English](README.md) | Русский

`uspexdb` — это единый CLI для двух сценариев работы с базой структур:

- `config`: фильтрация по метаданным и экспорт структур
- `query`: поиск контактов по `.mol2`-фрагменту через backend `c_anchor`

Проект собран как обычный Python-пакет с небольшим launcher-файлом в корне репозитория. Для локальной работы основная точка входа — `./uspexdb`. Для пользователей без Python проект можно упаковать в standalone `PEX scie` executable.

## Структура проекта

```text
pyDatabase/
  uspexdb                     # launcher для локального запуска
  src/uspexdb/
    cli.py                    # CLI: config, query, all
    paths.py                  # разрешение путей для config/query/database/mol2
    reference.py              # мост к legacy helper-коду
    config_search/
      runner.py               # workflow поиска по метаданным и экспорта
    query_search/
      runner.py               # workflow MOL2-поиска
      mol2.py                 # парсинг MOL2 и работа с query graph
      search.py               # contact search, export payloads, plots
      compact_cache.py        # сборка и загрузка compact graph cache
      validate.py             # опциональная валидация против reference output
      codes.py                # status/result helpers
      _c_anchor.c             # исходник C-extension
    legacy/
      uspexdb_v2.py           # vendored legacy implementation
  scripts/
    build_extension.sh        # установка зависимостей + сборка C-extension
    prepare_wheelhouse.sh     # подготовка offline wheelhouse для PEX/scie
    build_scie.sh             # сборка standalone PEX scie executable
  example/
    config/
      config.json             # пример metadata search config
    query/
      query.json              # пример MOL2 contact query
      fragment_example.mol2   # пример query fragment
```

## Установка

Рекомендуемый вариант для локальной работы:

```bash
python3 -m venv .venv
. .venv/bin/activate
./scripts/build_extension.sh
```

`build_extension.sh` устанавливает проект в editable-режиме, подтягивает runtime-зависимости и собирает `_c_anchor` in-place.

Основные runtime-зависимости:

- `numpy`
- `pandas`
- `pyarrow`
- `ase`
- `matplotlib`
- `pymatgen`

Исходная установка проекта сейчас рассчитана на Python `>=3.9`, как указано в `pyproject.toml`.

## Запуск из CLI

Корневой launcher поддерживает три режима:

```bash
./uspexdb query
./uspexdb config
./uspexdb all
```

Можно явно передать пути к JSON:

```bash
./uspexdb query --query /path/to/query.json
./uspexdb config --config /path/to/config.json
```

Режим `all` сначала запускает `query`, затем `config`.

## Workflow `config`

Режим `config` читает `config.json`, берёт базу из `database.path`, применяет metadata-фильтры и экспортирует найденные структуры.

Сами фильтры сейчас идут через legacy search layer из `uspexdb_v2.py`. Набор доступных metadata-полей зависит от конкретной базы, но текущие примеры ориентированы на тот layout базы, который используется в этом проекте.

Смотри:

- [example/config/README.md](./example/config/README.md)
- [example/config/config.json](./example/config/config.json)

## Workflow `query`

Режим `query` читает `query.json`, резолвит `.mol2`-фрагмент, готовит или переиспользует compact graph cache, запускает `c_anchor`-поиск контактов и пишет результаты плюс графики.

Важные правила для путей:

- `query.fragment_mol2` обязателен
- если `fragment_mol2` абсолютный, он используется как есть
- если `fragment_mol2` относительный, он считается относительно каталога, где лежит `query.json`
- `database.path` задаётся отдельно внутри `query.json`

Если после resolution `.mol2` не найден, выполнение завершается с явной ошибкой.

Текущие правила анализа:

- считается только одно расстояние: `anchor ... contact`
- суммарные графики по всем контактам вместе не строятся
- распределения расстояний, углов и торсионов строятся отдельно по каждому `contact_element`

Смотри:

- [example/query/README.md](./example/query/README.md)
- [example/query/query.json](./example/query/query.json)

## Сборка Standalone Executable

Если пользователю не нужно вручную ставить Python-пакеты, проект можно собрать в standalone executable:

```bash
./scripts/build_scie.sh
```

Теперь `build_scie.sh` делает два отдельных шага:

1. собирает wheel проекта в `dist/`
2. готовит offline-style wheelhouse в `artifacts/wheelhouse/` и использует его для финальной сборки `PEX scie`

На выходе получается:

- `dist/uspexdb-scie` — standalone `PEX scie` executable

Также могут появляться промежуточные артефакты:

- `dist/uspexdb` — обычный PEX
- `dist/*.whl` — wheel для packaging
- `artifacts/wheelhouse/` — локальный wheel cache для сборки scie

`artifacts/wheelhouse/` — это build artifact directory, его не нужно коммитить в Git.

### Точные замечания по сборке

- Установка из исходников: Python `>=3.9`
- Workflow сборки `scie`: сейчас документирован и прогонялся с Python `3.12`
- Итоговые бинарники platform-specific, например:
  - `macOS arm64`, если сборка выполняется на Apple Silicon macOS
  - `Linux x86_64`, если сборка выполняется на Linux x86_64
- Для bootstrap wheelhouse сейчас жёстко зафиксированы версии:
  - `pip==23.2`
  - `setuptools==68.0.0`
  - `wheel==0.40.0`

Важно:

- итоговый executable platform-specific
- бинарник, собранный на macOS, не запустится на Linux-кластере
- для HPC/Linux нужно запускать `build_scie.sh` уже на совместимой Linux-машине

## Примеры

Папка `example/` содержит минимальные входные файлы для обоих режимов:

- [example/config](./example/config)
- [example/query](./example/query)

Это стартовые шаблоны, которые потом нужно подставить под реальные пути к базе и реальные search-параметры.
