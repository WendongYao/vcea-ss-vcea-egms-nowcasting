# Scope Decision Apply Report

本次 apply 完整落实了 `VCEA_DSCF_infrastructure_refactor_plan_and_scope_decision_final` 的核心范围：

- 保留 `64×64 regional nowcasting` 为主文主线；
- 把 `128/512` 固定为 `resolution robustness / sensitivity`；
- 把 external tiles 固定为 `locked external validation`；
- 不把论文主线改成 high-resolution inspection；
- 不做破坏式目录搬迁，而是增加可追溯的 schema / manifest / registry / migrated result layer。

## Result Migration Summary

- total rows migrated: **589**
- by tile_role: `{'external_locked_eval': 80, 'main6': 216, 'main6_aggregate': 5, 'resolution_sensitivity': 288}`

## Delivered

- `configs/infra_v2/*`
- `data/manifests/*`
- `splits/*`
- `cache/*_registry.yaml`
- `outputs/tgrs_2019_2023/infrastructure_v2/*`

## Important Applied Caveats

- `proxy` 与 `exact` 已显式拆开；
- `512` 下的 `transfer inference` 与 `128 validation cache reuse` 已显式标注；
- `aggregation-only runtime` 与 `full-pipeline runtime` 已拆口径；
- `external locked eval` 已独立登记，不再与主线搜索混用。
