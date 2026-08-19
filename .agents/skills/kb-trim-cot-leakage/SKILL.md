---
name: kb-trim-cot-leakage
description: Use when writing or reviewing commit messages, PR descriptions,
  or documentation. Committed prose must be resolvable at HEAD with no
  session vantage. Trims chain-of-thought leakage.
---

# kb-trim-cot-leakage

Chain-of-thought leakage is prose whose vantage is the authoring session
rather than the repository: it cites artifacts only that session could
see, narrates the change instead of the state, or argues with a reviewer
who has left.

## Rules

- Commit messages and PR descriptions must be resolvable at HEAD: a
  reader with the repository alone can reconstruct what changed and why.
- Prefer state over process: describe what the codebase now is, not the
  steps that produced it.
- Never cite conversation artifacts, session IDs, or prompts.
- Never reference internal PR ids, stage/checkpoint labels, or the
  upstream project name (this repo is public-facing).
- After rewriting, re-read the text with only the repository in hand;
  anything that requires session memory is leakage.
