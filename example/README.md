# Examples

Русская версия ниже, English version follows after it.

## Русский

В этой папке лежат минимальные примеры конфигурации для двух режимов:

- [config](./config)
- [query](./query)

Назначение этих примеров:

- показать формат JSON
- показать, какие поля обязательны
- показать, как указывать пути

Важно:

- примеры не привязаны к конкретной локальной базе
- перед запуском нужно отредактировать `database.path`
- для `query`-примера нужно также проверить `fragment_mol2`, `graph_cache.path` и `compact_cache.path`

## English

This directory contains minimal examples for the two supported modes:

- [config](./config)
- [query](./query)

Purpose of these examples:

- demonstrate the JSON structure
- show which fields are required
- show how paths should be specified

Important:

- the examples are not bound to any specific local database
- `database.path` must be edited before use
- for the `query` example, also review `fragment_mol2`, `graph_cache.path`, and `compact_cache.path`
