Push the current branch to origin, then rebase it on top of the latest main.

Steps:
1. Push the current branch to origin (`git push origin <branch>` — use `-u` if the remote branch doesn't exist yet).
2. Fetch the latest main from origin (`git fetch origin main`).
3. Rebase the current branch onto `origin/main` (`git rebase origin/main`).
4. If the rebase produces conflicts, stop and tell the user which files conflict — do NOT force-push or reset to resolve them automatically.
5. After a clean rebase, push again with `--force-with-lease` to update the remote branch.
6. Report the final branch state: current HEAD commit and how many commits ahead of main it is.

Never use `--force` (without `--lease`). Never skip hooks (`--no-verify`). Never rebase if there are uncommitted changes — stash or commit them first and tell the user what you did.
