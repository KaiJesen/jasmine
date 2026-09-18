# Agent Instructions

## Language conventions (mandatory)

- **All comments in code must be written in English.** This covers every source file in the
  repository (`*.hpp`, `*.h`, `*.cpp`, `*.cc`, `*.cu`, `*.cuh`): block comments, line comments,
  doxygen-style comments, and inline documentation inside test and example files.
- **Conversation with the maintainer is in Chinese.** Chat replies, explanations, design
  trade-off discussions and summaries are written in Chinese.
- Review artifacts stay in the repository's existing language: commit messages and PR titles are
  English, `README.md` is English, and the documentation files `TESTING.md` / `CUDA.md` are
  Chinese (they are prose documents, not code).
- Existing Chinese comments are being migrated to English. When touching a file, translate the
  comments of the code you change, and never introduce new Chinese comments.
- When translating a comment, preserve the technical content and the rationale (the "why"), not
  just the words: these comments carry design decisions, measured numbers and pitfalls that later
  readers rely on. Keep identifiers, code snippets, numbers and units unchanged.
