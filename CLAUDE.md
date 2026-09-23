# Transformer-Lab — working agreements

## Commits
- **Never add `Co-Authored-By: Claude ...` trailers.** Commits are authored by the repo
  owner alone, so GitHub shows a single avatar.
- No "Generated with Claude Code" footers in commit messages or PR descriptions.

## README roadmap
- **Do not add dates** to roadmap entries. Checking the box (`- [x]`) is the only
  completion marker; the git history already carries the timestamps.

## Language
- All repo content — README, docstrings, comments, commit messages — is written in English.

## The rule specific to this repo
- Every day that introduces a derivative ships a **finite-difference test** for it.
  A gradient nobody has checked numerically is not finished.
- Randomness is always seeded. A test that cannot be re-run to the same number is
  not a test.
