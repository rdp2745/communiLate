# communiLate

Heritage-adapted machine translation: a frozen **NLLB-200** baseline, **LoRA**
adapters for register and dialect adaptation, and a **translation-memory +
confidence gate** on top.

Everything is driven by a config file. Swapping corpora, switching experiments,
or changing model hyperparameters is a config edit — or a single CLI flag — not
a code change.

---

## Status

Architecture scaffold. The pipeline runs end to end on the data layer
(`inspect`, `prepare`, `make-regional` are exercised by tests and verified on
fixture data). Training and evaluation are implemented but have not been run
against a real corpus or a GPU yet.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The config, schema and data layers import nothing from torch, so you can develop
and test the data pipeline before installing the ML stack:

```bash
pip install PyYAML pytest
pytest          # 71 tests, no torch required
```

---

## The config is the interface

Every run is described by `configs/<name>.yaml` plus any command-line overrides.
Experiment configs `extends: base.yaml` and state only what differs.

```bash
# swap the corpus — the flag you will reach for most
communilate train -c exp1_register --data-path data/processed/some_other_corpus

# override any key at all
communilate train -c exp1_register --set lora.r=32 --set training.epochs=1

# see exactly what a config resolves to, and how many records it finds
communilate inspect -c exp1_register --count
```

`data.path` is the single swap point. Any corpus written into the `Pair` schema
can be dropped in by pointing `data.path` at its folder; no other config key and
no code has to move. That is deliberate — OpenSubtitles is a placeholder, and
replacing it later should cost one flag.

---

## Commands

| Command | What it does |
|---|---|
| `inspect` | Print the resolved config; `--count` also counts records per split |
| `prepare` | Filter a raw corpus into clean JSONL, with a per-filter drop report |
| `make-regional` | Derive `es-ES` / `es-419` variants for Experiment 2 |
| `train` | LoRA fine-tune |
| `evaluate` | Frozen baseline vs. adapted, writing `report.json` + `report.md` |
| `translate` | Ad-hoc translation; `--compare` shows frozen and adapted side by side |
| `build-memory` | Build the retrieval translation memory |

---

## The schema

One record shape for all three experiments. This is what lets the training and
eval code stay corpus-agnostic.

```json
{
  "id": "opensubtitles-000000042",
  "src_lang": "eng_Latn",
  "tgt_lang": "spa_Latn",
  "src": "Are you all coming over later?",
  "tgt": "¿Vosotros venís luego?",
  "register": "dialogue",
  "region": "es-ES",
  "source_corpus": "opensubtitles",
  "split": "train",
  "meta": {"imdb_id": "1000001", "genres": ["Comedy"]}
}
```

Language codes are FLORES-200 (`eng_Latn`, `spa_Latn`, `guj_Gujr`). NLLB rejects
bare ISO codes, and a wrong code fails at generation time rather than load time,
so the schema validates them up front.

**Direction** is derived, not stored. The JSONL holds one canonical ordering and
`data.direction: both` emits the reverse on the fly, so both-direction training
never duplicates the corpus on disk.

---

## The three experiments

### 1. `exp1_register` — formal vs. dialogue register

Frozen NLLB-200 against a LoRA trained on conversational-style parallel data.

Evaluated primarily **English → Spanish**: target-side adaptation is directly
observable, whereas in the reverse direction the English output looks much the
same either way. Both directions are trained; only one is easy to measure.

> **Corpus caveat.** OpenSubtitles is film and TV subtitles — dialogue *written
> to be spoken*, not spontaneous speech. Disfluencies, self-repairs and
> backchannels are largely absent, and subtitle reading-speed limits mean many
> targets are condensed rather than faithful translations. Describe this
> experiment as **written/formal register vs. dialogue register**, not as
> spontaneous conversational speech.

### 2. `exp2_regional` — Peninsular vs. Latin American Spanish

Two adapters over the same frozen base, trained on data differing on exactly one
axis: *vosotros* (es-ES) vs. *ustedes* (es-419).

```bash
communilate make-regional -c exp2_regional --data-path data/processed/opensubtitles_en_es
communilate train -c exp2_regional --data-path data/processed/regional/es-ES  --set experiment.name=exp2_es
communilate train -c exp2_regional --data-path data/processed/regional/es-419 --set experiment.name=exp2_la
```

Three scope limits, all enforced in code:

- **Not voseo.** *Vos* is not a Latin-America-wide feature — Mexico, most of
  Colombia, Peru and the Caribbean are *tú*-using. It needs country-level labels
  and a third bucket, not a Spain/LatAm binary.
- **One-way only.** *Ustedes* is also the formal plural in Spain, so es-419 →
  es-ES is not recoverable. There is no `to_peninsular`.
- **Lexicon-first, not morphological analysis.** The *vosotros* form is the one
  present-tense form that does *not* stem-change, so `venís → vienen` cannot be
  derived by suffix rule. Stem-changers and irregulars live in
  `src/communilate/data/lexicons/vosotros_ustedes.json`; extend that file rather
  than the code. In strict mode, anything the rules cannot convert confidently
  is reported and dropped rather than guessed at.

> **Metric asymmetry.** es-ES is detectable reliably (*podéis* can only be
> *vosotros*). es-419 often is not: once the pronoun drops, *pueden* is
> identical to the third person plural. Treat the **es-ES rate** as the real
> number — the Spain adapter should score high on it and the Latin American
> adapter near zero.

### 3. `exp3_gujarati` — the family corpus case study

Same pipeline, repointed at hand-collected recordings. This is the only
genuinely spontaneous, unscripted corpus in the project: no public dataset
covers informal, code-switched Gujarati family speech, which is what makes it
the contribution rather than a repeat of the Spanish phase.

Retrieval matters most here — a small corpus of fixed family expressions the
base model has never seen is exactly where a near-exact translation-memory match
beats generation. Realistic target is 500–2000 hand-transcribed pairs;
transcription is the long pole, so start collecting in week 1.

---

## Architecture notes

**Retrieval is translation memory, not few-shot prompting.** NLLB-200 is an
encoder-decoder MT model, not an instruction-following LLM — there is no prompt
to put examples into. What retrieval does here is embed the source, find the
nearest stored source, and return its human translation if the similarity clears
a threshold.

**Gating has three routes**, in priority order: translation-memory hit →
adapter output → frozen baseline. The third route is the one that earns the
architecture its keep: a LoRA trained on a few thousand in-domain pairs will
confidently produce nonsense on out-of-domain input. Whether that actually
happens on your data is empirical, which is why every decision records its route
so the ablation can report how often each path fires.

**chrF++ is the headline metric, not BLEU.** BLEU is close to blind to what
these experiments manipulate — a correct *vosotros → ustedes* rewrite barely
moves n-gram overlap. BLEU is reported because reviewers expect it.

**The targeted checks matter more than the corpus metrics.** `diff_rate` is the
first number to look at after any training run: near zero means the adapter is
doing nothing and no downstream metric will be meaningful.

---

## Layout

```
configs/                  base.yaml + one file per experiment
src/communilate/
  config.py               YAML loading, extends-chains, ${} interpolation, overrides
  schema.py               the Pair contract + JSONL I/O
  registry.py             name -> loader dispatch
  data/
    base.py               loader dispatch, direction handling, stable splitting
    jsonl_dir.py          the default loader — point it at a folder
    opensubtitles.py      raw OPUS Moses reader
    filters.py            the filter chain, with per-filter drop reporting
    regional_rules.py     vosotros -> ustedes, lexicon-first
    lexicons/             extend the lexicon here, not in code
  model.py                base model + LoRA/adapter loading (lazy torch imports)
  train_lora.py           LoRA fine-tuning
  translate.py            inference; frozen vs. adapted through one code path
  retrieval.py            translation memory
  gating.py               three-way route arbitration
  eval/                   metrics, targeted checks, report assembly
tests/                    71 tests, no ML stack required
```
