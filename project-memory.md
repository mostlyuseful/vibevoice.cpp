# Project memory

## Decisions

### KugelAudio GGUF metadata migration contract
- Context: Slice 1 starts with converter-side KugelAudio support, but the current runtime/loader still keys off legacy `vibevoice.*` metadata and the GGUF arch string used by existing code.
- Chosen default: emit a dual metadata contract for KugelAudio conversion now:
  - new explicit `kugelaudio.*` keys, with `kugelaudio.checkpoint = kugelaudio-0-open` and `kugelaudio.schema_version = 1`
  - legacy `vibevoice.*` compatibility keys in the same GGUF during migration
  - keep GGUF writer arch as `vibevoice` until loader migration work lands
- Rejected alternatives:
  - emit only `kugelaudio.*` immediately: cleaner, but likely breaks the current loader before Slice 1 loader work is done
  - keep only `vibevoice.*`: preserves compatibility, but fails the spec/PRD requirement for an explicit KugelAudio metadata contract
  - switch GGUF arch to `kugelaudio` immediately: desirable later, but too risky before loader support exists
- Affected area: `spec.md` GGUF schema migration guidance and `prd.md` Slice 1 / “Define and implement the KugelAudio GGUF contract”.
