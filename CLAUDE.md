# Conventions for this repo

## Writing

Applies to markdown files and notebooks.

Keep all text written (in markdown files / notebooks) concise and well-defined. That means generally as few words as suffices. Never use the positive, negative sentence structure for emphasis (e.g. This is sth, it is not sth.), i.e. do not add an extra clause just to emphasise the previous point that does not add additional information. Do not use sentence structures such as "this is the important one" or "The one that matters" these are pointless emphasis that clutters reading and affects interpretation. Any non-objective claim can only be presented if it is backed by rigorous reasoning. Define all relevant mathematical formulae and notation. Use fewer definite articles e.g. "the" or "an". Grammatical rules can be suspended to make way for conciseness. Do not pose a question, as a title or as a regular sentence. No question structures unless it is a real open question, in which case say that it is open. Do not open a sentence or a heading with a negation: put the negated part in brackets or after the subject, e.g. "the one-dimensional algorithm (not implemented)" rather than "Not implemented: the one-dimensional algorithm".

Also:

- Few hyphens and dashes. Colons or brackets instead. Dash beside mathematics reads as minus.
- Define at point of use, not in preamble.
- Quote standard results, do not demonstrate them: citation plus test, not table of numbers.
- No number a test already asserts. Closed forms preferred: they do not go stale.
- No assumption about data. Not necessarily SDE trajectories, smooth, or uniformly sampled. State assumption where used.
- Cite at point of use. `papers/references.bib` is single list.

## Structure

| | goes in |
|---|---|
| mathematics, experiments, results | relevant **notebook** |
| function used more than once | **`src/pathloss/`**, with test |
| decisions, changes, findings | **`docs/logbook/`**, dated |
| an experiment's design, before it is run | **`docs/protocol_*.md`**, pre-registered |
| how to run something | **`README.md`** |

Anything that must be correct lives in `src/pathloss/` with a test. Notebook cell is never sole copy of a function.

## Protocols

An experiment's design is written and committed **before** the data is
downloaded: question, selection rule, scoring, predictions, and what each
outcome would mean. Departures are recorded in the logbook with a date and a
reason. Without this the difference between a prediction and a post-hoc
rationalisation is not recoverable.

## Logbook

Records decisions and changes, dated. Not word counts, code cell counts, test counts, or editorial edits to notebooks: rewording a notebook is not a project event.

Entries are not rewritten to reflect later opinion. Corrected in place: cross-references into notebooks since renumbered.

A claim later **superseded** gets a dated line saying so. A claim that was simply **wrong** is removed, with no trace and no narration of the error: a note is for what is true, not a record of drafting.
