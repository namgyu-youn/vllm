---
name: git-convention
description: Use this skill whenever writing a git commit message or a pull request title/description. Applies house style rules for tone, length, and content — no AI/tool attribution, keep text concise.
---

# Commit & PR Style Guide

## No AI attribution

Never mention that this commit, PR, or its description was written or assisted by Claude, AI, or any tool. Do not include phrases like:

- "Generated with Claude" / "Generated with AI"
- "AI-assisted" / "Co-authored-by: Claude"
- Any similar attribution, in the title, body, or trailer lines.

All work is reviewed by a human before merge, so no disclaimer is needed.

## Keep it short

**Commit message:**

- Summary line: imperative mood, ≤50 characters if possible (e.g. "Fix race condition in worker pool").
- Body (optional): only if the summary line can't convey the "why". Max 2–3 short bullet points. No restating the diff line-by-line.

**PR description:**

- 1–3 short sentences or bullet points on what changed and why.
- Skip boilerplate sections (e.g. long "Testing" essays, verbose "Motivation" paragraphs) unless the repo's PR template explicitly requires them.
- No filler, no restating file-by-file changes — the diff already shows that.

## Quick checklist before submitting

- [ ] No AI/tool mention anywhere in title, body, or trailers
- [ ] Summary line is one sentence, imperative, concise
- [ ] Body/description is short — if it's more than a few lines, cut it down
