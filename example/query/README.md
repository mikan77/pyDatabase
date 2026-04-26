# Query Example

Русская версия ниже, English version follows after it.

## Русский

### Назначение

Этот пример показывает, как запускать режим `query`, то есть:

- искать контакты по `.mol2`-фрагменту
- использовать backend `c_anchor_v3`
- записывать JSON/CSV и графики

### Файлы

- `query.json` — пример конфигурации
- `fragment_example.mol2` — пример фрагмента

### Как использовать

1. Откройте `query.json`.
2. Укажите свой путь в `database.path`.
3. Проверьте `graph_cache.path` внутри вашей базы.
4. При необходимости измените `compact_cache.path`.
5. При необходимости измените `contact_elements`, `radius_max` и `output`.
6. Запустите:

```bash
./uspexdb query --query /path/to/example/query/query.json
```

### Главное правило для `fragment_mol2`

В этом проекте путь `query.fragment_mol2` трактуется только так:

- если путь абсолютный, используется как есть
- если путь относительный, он считается относительно каталога `query.json`

В данном примере используется относительный путь:

```json
"fragment_mol2": "./fragment_example.mol2"
```

Это означает, что файл ищется рядом с `query.json`.

### Что делает текущий пример

Пример:

- запускает режим `mol2_contact`
- использует backend `c_anchor_v3`
- ищет контакты с `O`, `S`, `N`, `B`
- строит выходные файлы в `./output/query`

### Примечание по графикам

В текущей логике:

- расстояние считается только как `anchor ... contact`
- суммарные общие графики не строятся
- распределения строятся отдельно по `contact_element`

## English

### Purpose

This example shows how to run `query` mode, i.e.:

- search for contacts using a `.mol2` fragment
- use the `c_anchor_v3` backend
- write JSON/CSV results and plots

### Files

- `query.json` — example configuration
- `fragment_example.mol2` — example fragment

### How to use

1. Open `query.json`.
2. Set your database path in `database.path`.
3. Verify `graph_cache.path` inside your database.
4. Adjust `compact_cache.path` if needed.
5. Adjust `contact_elements`, `radius_max`, and `output` if needed.
6. Run:

```bash
./uspexdb query --query /path/to/example/query/query.json
```

### Main rule for `fragment_mol2`

In this project, `query.fragment_mol2` is interpreted in exactly one way:

- an absolute path is used as-is
- a relative path is resolved relative to the directory containing `query.json`

This example uses a relative path:

```json
"fragment_mol2": "./fragment_example.mol2"
```

This means the fragment file is expected to be next to `query.json`.

### What this example does

The sample:

- runs `mol2_contact` mode
- uses the `c_anchor_v3` backend
- searches contacts with `O`, `S`, `N`, `B`
- writes outputs into `./output/query`

### Plot note

In the current logic:

- distance is computed only as `anchor ... contact`
- no aggregate plots are generated
- distributions are generated separately by `contact_element`
