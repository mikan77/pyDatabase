# Config Example

Русская версия ниже, English version follows after it.

## Русский

### Назначение

Этот пример показывает, как запускать режим `config`, то есть:

- искать структуры по метаданным
- экспортировать найденные структуры

### Файлы

- `config.json` — пример конфигурации для `./uspexdb config`

### Как использовать

1. Откройте `config.json`.
2. Укажите свой путь в `database.path`.
3. При необходимости измените фильтры в `search`.
4. При необходимости измените `export.output_dir` и форматы экспорта.
5. Запустите:

```bash
./uspexdb config --config /path/to/example/config/config.json
```

### Что важно в этом примере

- `database.path` может быть абсолютным или относительным
- относительные пути интерпретируются относительно `config.json`
- `search` задаёт фильтры
- `export` задаёт путь и форматы записи

### Что делает текущий пример

Пример конфигурации:

- ищет структуры с `C`, `H`, `N`
- допускает дополнительные элементы `Ni`, `Cu`, `Zn`, `Fe`
- ограничивает год диапазоном `1900..2024`
- ограничивает число атомов сверху

## English

### Purpose

This example shows how to run `config` mode, i.e.:

- search structures by metadata
- export the matching structures

### Files

- `config.json` — example configuration for `./uspexdb config`

### How to use

1. Open `config.json`.
2. Set your database path in `database.path`.
3. Adjust the filters in `search` if needed.
4. Adjust `export.output_dir` and output formats if needed.
5. Run:

```bash
./uspexdb config --config /path/to/example/config/config.json
```

### Important notes

- `database.path` may be absolute or relative
- relative paths are resolved relative to `config.json`
- `search` defines metadata filters
- `export` defines output location and export formats

### What this example does

The sample config:

- searches for structures containing `C`, `H`, `N`
- allows additional elements `Ni`, `Cu`, `Zn`, `Fe`
- limits the year range to `1900..2024`
- applies an upper atom-count limit
