# Documentation languages

English is the default language for repository documentation. Start with the [project README](../README.md), [experiment index](../experiment/main/README.md) or [running guide](guides/running_experiments.md).

- Use ordinary filenames such as `README.md`, `report.md` and `SKILL.md` for English documents.
- Keep optional Chinese versions beside them as `README.zh-CN.md`, `report.zh-CN.md` and `SKILL.zh-CN.md`. Existing historical `_zh.md` files retain their names; their English counterparts omit `_zh`.
- Link to the English document in default navigation. Each translated English document links to its Chinese alternative at the top.
- Write new documentation and updates in English first. Update a Chinese companion when maintaining its translation; identify any translation lag rather than treating it as the authoritative version.

The English migration covers all 95 tracked documentation files that contained Chinese prose, including guides, planning notes, experiment reports, archived workflow documentation and the report-writing skill. The [language-pair manifest](documentation-translations.json) lists every English/Chinese pair and the SHA-256 of each original Chinese snapshot. Those snapshots were preserved byte for byte. Documents already in English keep their existing filenames.

Translation preserves scientific results, commands, paths, identifiers and the historical scope of claims. Older status notes still describe their original stage. Translating them does not certify that pending work is now complete.

Raw data, machine-readable experiment evidence, model files, historical logs and embedded figure text retain their original contents. Legacy generators may still produce Chinese historical reports under ignored `output/` directories; those outputs are separate from the English documentation shipped in this repository. When publishing newly generated documentation, provide English under the default filename and keep any Chinese output as an additional version. Do not overwrite frozen source evidence to change its language.
