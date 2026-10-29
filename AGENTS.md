# Repository workflow

Before making code changes, ensure there is a GitHub issue for the work. Create
an issue first when no matching issue exists.

## Git branch naming

When creating a branch, use:

- `feature/{issue-number}-{short-description}` for new features
- `bugfix/{issue-number}-{short-description}` for fixes
- `task/{issue-number}-{short-description}` for maintenance, chores, docs,
  infrastructure, or refactors

Use lowercase kebab-case for the short description. If no issue number is
available, ask before creating the branch. Use the `richpsharp` username when a
username is needed; do not use `codex` in branch names or attribution.

## Pull request comments

When addressing a pull request comment, make a commit that fixes it, then reply
to the comment with a clickable link to the commit. Leave the comment unresolved
so the user can resolve it manually.
