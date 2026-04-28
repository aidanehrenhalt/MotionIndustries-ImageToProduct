# Project Instructions

## Philosophy
- Prefer simple, minimal solutions over clever abstractions
- Edit existing files before creating new ones
- Only change what was asked — no drive-by refactors
- Validate at system boundaries, trust internal code

## Code Style
- Use consistent naming: snake_case for Python, camelCase for JS/TS
- No comments unless logic is non-obvious
- No empty catch blocks; handle errors or propagate them
- Prefer early returns over deep nesting

## Git Practices
- Commit messages: imperative mood, <72 chars, explain *why* not *what*
- One logical change per commit
- Never commit secrets, .env files, or credentials

## Testing
- Write tests for new behavior; update tests for changed behavior
- Run existing tests before declaring a task done
- Prefer integration tests over mocking internals

## Token Efficiency
- Use Haiku-model agents for simple lookup/search tasks
- Delegate verbose operations (test runs, log parsing) to subagents
- Use `/compact` between unrelated tasks
- Keep CLAUDE.md under 200 lines; move details to @-imported files or skills
- Write specific prompts — broad requests waste tokens on exploration

## Architecture
- @docs/architecture.md for project-specific architecture notes
- @docs/conventions.md for language/framework conventions

## Security
- Never disable pre-commit hooks or bypass verification
- Sanitize all user input at API boundaries
- No hardcoded secrets — use environment variables
