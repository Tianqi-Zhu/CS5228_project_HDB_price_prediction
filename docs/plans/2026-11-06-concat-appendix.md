# Concat exploration appendix implementation plan

**Goal:** Archive the pre-final spatial, temporal and architecture explorations on a branch based on appendix.

**Architecture:** Preserve historical source bytes in an isolated snapshot and selected small evidence under X05/X06. A SHA256 manifest links every copied file to its local source. Archive verification performs no training. Main’s final 418.38657 model remains a separate integration.

**Tech stack:** Python standard library, Git; historical model dependencies recorded in original source/configuration.

1. Inspect X05/X06 and local experimental evidence; separate protocols and smoke runs.
2. Copy historical experiment source and supporting analysis source into an isolated archive.
3. Copy small configuration, summary and curve records; exclude weights, caches, prepared banks, raw predictions and deployment bundles.
4. Write X05/X06 navigation and a matched architecture comparison with evidence paths.
5. Verify source hashes, Python syntax and existing appendix integrity; commit a dedicated branch without pushing.
