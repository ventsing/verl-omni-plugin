---
name: verl-omni-plugin-guardrails
description: Guardrails for working in verl-omni-plugin without stale-context damage. Use when creating, editing, or verifying any file in this repo — especially docs, model skeletons, or shell scripts — or when delegating such work to a subagent. Anchors to the current fact set (git HEAD, naming, mechanism) so agents do not resurrect deleted files, reintroduce old naming, or claim mechanisms that no longer exist (e.g. "no entry_points", "no monkey-patch"). Mandatory delivery check: working tree must be git-clean after any change.
---

# verl-omni-plugin Guardrails

This repo has been damaged twice by background subagents working from **stale context**. They
recreated deleted files, reintroduced old file names, and asserted mechanisms that the project no
longer uses. This skill exists so every future agent — including freshly spawned subagents with no
conversation history — starts from the **current fact set** instead of guessing.

## Ground truth: current state (git HEAD)

The repo at `verl-omni-plugin/` tracks `git@github.com:ventsing/verl-omni-plugin.git`. The three
commits below ARE the evolution; any claim that contradicts them is stale:

| Commit | What it established |
|--------|--------------------|
| `63bb478` | Three-layer strategy (L1 plugin / L2 monkey-patch / L3 gate-patch), 9 extension points, 3 entry-point groups, GP-004 vllm-omni gate patch |
| `d894133` | **Renamed** `adapter.py` → `thinker_adapter.py`, `rollout.py` → `rollout_adapter.py`; added `features/` for cross-domain features; internalized `probes/` into the package |
| `2505a18` | Added `docs/inject_new_model.md` — the authoritative new-model guide (3+1 steps) |

## Current facts (do not contradict these)

**Mechanism (three-layer, NOT external_lib-only):**
- Loading: `VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext` → `import_external_libs` → `verl_omni_ext/__init__.py _load_all()` → iterates **entry-point groups** `verl_omni.models`, `verl_omni.trainers`, `verl_omni.reward` → triggers `@register` decorators.
- L1 plugin (≥95%): adapters, datasets, config — the default.
- L2 monkey-patch (~4%): `_patchkit.py` + per-model `patches.py` — **it exists and is core**. The claimed "no monkey-patch needed" is WRONG.
- L3 gate-patch (≤1%): `gates/ledger.md` + `gates/vllm_omni_external_modules.patch` (GP-004 adds `VLLM_OMNI_EXTERNAL_MODULES` to vllm-omni registry).
- Entry points ARE used — they are what eliminated 42 lines of upstream `__init__.py` edits.

**File naming (current):**
- `verl_omni_ext/models/<model>/thinker_adapter.py` and `rollout_adapter.py` — NOT `adapter.py` / `rollout.py`. The old names were renamed in `d894133`; creating them again is damage.
- `verl_omni_ext/models/<model>/patches.py` (L2) and optional `dataset.py` (slot ④).
- `verl_omni_ext/models/<model>/vllm_omni/` (GP-004 pipeline definitions, registered into `vllm_omni`'s `_OMNI_MODELS`).
- `verl_omni_ext/features/<feature>/` for cross-domain features (e.g. `fullduplex/` with `trainer.py` + `async_worker.py`).
- Adapters implement **4 methods**: `get_strip_modules`, `configure_processor`, `configure_tokenizer`, `configure_model` — not 3. `configure_model` is the instance-level patch entry (ran AFTER `from_pretrained`; module-level patches must go in package `__init__.py` instead).

**Deleted files — do NOT recreate:**
- `QUICKSTART.md`, `PROJECT_SUMMARY.md`, `docs/README.md`, `plugins/`, `shared/` — all removed in the restructuring. Stale-context agents recreated them; they were deleted again.
- `docs/zero_intrusion_mechanism.md`, `docs/inject_new_model_to_verl_omni.md` (old external_lib-only version) — never part of the current set; the correct guide is `docs/inject_new_model.md`.

**Canonical docs (current):** `docs/three_layer_strategy.md`, `docs/plugin_architecture_design.md`, `docs/inject_new_model.md`, `docs/rollout_adaptation.md`, `docs/data_pipeline.md`, `docs/feature_fullduplex.md`, `docs/vllm_omni_changes.md`, `docs/migration_guide.md`, `docs/gate_patch_ledger.md`. The project's entry README is `README.md` at the repo root — not `docs/README.md`.

**Config keys that must align:** `architecture` (↔ `@OmniModelBase.register` key), `external_lib: verl_omni_ext`, `pipeline_name` (↔ `@OmniRolloutPipelineBase.register` key).

## Rules for any change

1. **Read before write.** Load the current file from git working tree; never write from memory of an older session.
2. **Verify against git HEAD.** If a task description mentions a file/name/mechanism that HEAD does not have, the task is stale — do the current version, not the described one.
3. **Do not resurrect deleted files.** If a file is in `.gitignore`-excluded history but not in HEAD, it was deliberately removed.
4. **Do not create duplicate names.** If `thinker_adapter.py` exists, never create `adapter.py` beside it. If the old file appears untracked after your work, delete it.
5. **Do not strip existing comments.** Editing a shell script must not delete trailing `# 说明` annotations on `export` lines.
6. **Use current mechanism in docs.** Entry points + monkey-patch + gate-patch are real. Never write "no entry_points needed" or "no monkey-patch needed".
7. **Keep the working tree clean when you finish.** A delivery that leaves `git status` non-empty (untracked duplicates, stray modifications) is incomplete.

## Pre-delegation checklist (subagents)

When delegating doc/skeleton work to a subagent, REQUIRE it to:
1. Run `git -C verl-omni-plugin log --oneline -3` and `git -C verl-omni-plugin status --short` first, and report what it sees.
2. Load this skill (`verl-omni-plugin-guardrails`) before writing anything.
3. Verify the exact current naming (`thinker_adapter.py` / `rollout_adapter.py`) and current mechanism from HEAD, not from the prompt alone.
4. End with `git status --short` — must be empty (or contain only the intended new files, no duplicates).

## Post-delivery verification (always, by the orchestrator)

```bash
cd verl-omni-plugin
git status --short                # must show ONLY intended changes
find verl_omni_ext/models -name "adapter.py" -o -name "rollout.py"  # must be empty
grep -c "GP-004" verl_omni_ext/gates/ledger.md    # must be ≥ 1
```

If the working tree shows unexpected modifications, `git checkout -- <file>` to restore, delete stray untracked duplicates, then re-run verification.