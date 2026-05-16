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

### KugelAudio runtime-loader variant normalization
- Context: the current runtime only has behavior branches for `realtime-0.5b`, `1.5b`, and `asr-7b`, but converter-side KugelAudio metadata now identifies the checkpoint as `kugelaudio-0-open`.
- Chosen default: when loader metadata is KugelAudio-only, normalize `kugelaudio-0-open` onto the existing `1.5b` runtime branch for now, while still validating `kugelaudio.schema_version` and `kugelaudio.checkpoint` explicitly.
- Rejected alternatives:
  - add a brand-new runtime variant in this iteration: broader than the current PRD increment and under-tested without the later prompt/inference work
  - map KugelAudio onto `realtime-0.5b`: a worse architectural fit because the raw-reference conditioning path and encoder usage are closer to the current 1.5b path
  - leave the variant unnormalized: would bypass existing branch logic and fail to load any integrated runtime state
- Affected area: `prd.md` Slice 1 / loader compatibility work and upcoming Slice 2/3 runtime adaptation.

### KugelAudio schema documentation location
- Context: the next PRD item requires the GGUF schema to be documented clearly enough for future validation work, but the repo already has `docs/conversion.md` as the conversion-facing document and no existing dedicated schema doc.
- Chosen default: extend `docs/conversion.md` with a KugelAudio-specific schema contract section instead of creating a separate `docs/gguf-schema.md` right now.
- Rejected alternatives:
  - create a brand-new schema-only doc immediately: cleaner separation, but adds another place to keep in sync during rapid migration
  - document only in converter source comments: too discoverability-poor for future agents working from docs first
- Affected area: `prd.md` Slice 1 / "The schema is documented well enough that a future agent can add validation without re-deriving intent.".
