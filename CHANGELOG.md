# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/pax-project/pax-ai/releases/tag/v0.1.0) - 2026-11-22

### Features

- *(index)* [**breaking**] keep the ocr flag in the store (format v3) ([#74](https://github.com/pax-project/pax-ai/pull/74))
- *(index)* OCR scanned PDFs before falling back to title + venue ([#72](https://github.com/pax-project/pax-ai/pull/72))
- *(index)* store embedding vectors in a binary format ([#71](https://github.com/pax-project/pax-ai/pull/71))
- *(mcp)* rank online candidates by similarity to a library paper ([#68](https://github.com/pax-project/pax-ai/pull/68))
- *(index)* recommend related papers by embedding similarity ([#67](https://github.com/pax-project/pax-ai/pull/67))
- *(index)* add full-text semantic search ([#66](https://github.com/pax-project/pax-ai/pull/66))
- *(index)* read every paper's vector set at once ([#64](https://github.com/pax-project/pax-ai/pull/64))
- *(index)* build the paper index and add the index MCP tool ([#63](https://github.com/pax-project/pax-ai/pull/63))
- *(index)* add chunk-set similarity over normalized vectors ([#61](https://github.com/pax-project/pax-ai/pull/61))
- *(index)* extract and chunk a paper's full text ([#60](https://github.com/pax-project/pax-ai/pull/60))
- *(index)* add the on-disk embedding store ([#59](https://github.com/pax-project/pax-ai/pull/59))
- *(port)* batch Embedder with model id and document/query kind ([#58](https://github.com/pax-project/pax-ai/pull/58))
- *(mcp)* default the PDF backup to research/.pdfs/ and return its warnings ([#49](https://github.com/pax-project/pax-ai/pull/49))
- *(mcp)* back up and restore local PDFs with export pdfs and sync from ([#46](https://github.com/pax-project/pax-ai/pull/46))
- *(mcp)* declare a local PDF with fetch's file input ([#44](https://github.com/pax-project/pax-ai/pull/44))
- *(mcp)* list excluded papers ([#43](https://github.com/pax-project/pax-ai/pull/43))
- *(mcp)* log labelled adds as other-methods sources ([#42](https://github.com/pax-project/pax-ai/pull/42))
- *(mcp)* [**breaking**] return pax-core's shared output types ([#41](https://github.com/pax-project/pax-ai/pull/41))
- *(mcp)* paper aliases ([#36](https://github.com/pax-project/pax-ai/pull/36))
- *(mcp)* related tool ([#35](https://github.com/pax-project/pax-ai/pull/35))
- *(mcp)* send abstracts in search and citation results on request ([#34](https://github.com/pax-project/pax-ai/pull/34))
- *(mcp)* search-log labels and per-label stats ([#33](https://github.com/pax-project/pax-ai/pull/33))
- *(mcp)* exclusion criterion, round and year, and stats by criterion ([#32](https://github.com/pax-project/pax-ai/pull/32))
- *(mcp)* fetch, edit, export, check and show-by-DOI tools ([#27](https://github.com/pax-project/pax-ai/pull/27))
- *(mcp)* [**breaking**] multi-seed refs and cited_by with seed attribution and incompleteness reasons ([#30](https://github.com/pax-project/pax-ai/pull/30))
- *(mcp)* refuse adding excluded papers and add an unexclude tool ([#29](https://github.com/pax-project/pax-ai/pull/29))
- *(mcp)* report papers awaiting screening in stats ([#28](https://github.com/pax-project/pax-ai/pull/28))
- *(mcp)* [**breaking**] move to pax-core 2.0 with paged merged search, refs, cited_by, exclude and stats ([#17](https://github.com/pax-project/pax-ai/pull/17))

### Fixed

- *(index)* strip reference lists the heading rule misses ([#76](https://github.com/pax-project/pax-ai/pull/76))
- *(hooks)* clear git's environment before running the gate ([#50](https://github.com/pax-project/pax-ai/pull/50))
- *(flake)* expose pax-mcp with mainProgram, an overlay and a home-manager module ([#48](https://github.com/pax-project/pax-ai/pull/48))
