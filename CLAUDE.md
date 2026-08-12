# Conventions for this repo

## Writing

Applies to notebooks, logbook entries, README, docstrings and commit messages.

1. **As few words as possible.** If a sentence survives deletion of a phrase, delete it.
2. **No add-on clauses for emphasis.** Not "what it does, and what it does not", not "X, and this is good news", not "not just a measurement". State the thing once.
3. **Define at the point of use**, not in a preamble. A symbol or term appears with its definition the first time it is needed.
4. **No vague words standing in for undefined concepts.** "The claim", "the fix", "the finding" are not names. Either name the object or state it.
5. **Few hyphens and dashes.** Use colons or brackets. A dash next to mathematics reads as a minus sign.
6. **Quote standard results, do not demonstrate them.** A citation and a test, not a table of numbers. Anything provable analytically is stated analytically.
7. **No numbers that a test already asserts.** Closed forms are preferred to measured values: they do not go stale.
8. **Say what a piece is for.** Each section states its role in the project, not only its content.
9. **No assumptions about the data.** It is not necessarily SDE trajectories, or smooth, or uniformly sampled. Say what is assumed where it is assumed.
10. **Cite at the point of use.** `papers/references.bib` is the single list. No separate reading index.

## Structure

| | goes in |
|---|---|
| mathematics, experiments, results | the relevant **notebook** |
| a function used more than once | **`src/pathloss/`**, with a test |
| decisions, findings, interpretations | **`docs/logbook/`**, dated |
| how to run something | **`README.md`** |

Anything that must be correct lives in `src/pathloss/` and has a test. A notebook cell is never the only copy of a function.

## Logbook

Dated, and a record rather than a document: entries are not rewritten to reflect later opinion. Two things are corrected in place, because they are pointers and not history: cross-references into notebooks that have since been renumbered, and counts (word counts, test counts) that have since changed. A superseded claim gets a dated line saying so, not a deletion.
