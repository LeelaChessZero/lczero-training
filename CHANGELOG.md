# Changelog

## 2026-05-24

### Breaking changes

- **`movesleft` loss config now requires `component`, `scale`, and
  `huber_delta` to be set explicitly.** The previous defaults no longer apply,
  so existing configs must be updated to spell these fields out. Example:

  ```
  movesleft {
    head_name: "main"
    value_type: RESULT
    component: MOVES_LEFT
    weight: 1.0
    scale: 0.05
    huber_delta: 10.0
  }
  ```
