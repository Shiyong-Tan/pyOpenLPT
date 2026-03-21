---
description: Ensure changes to function signatures or logic are compatible with all existing call sites.
globs: "**/*.py"
---

# Code Compatibility and Regression Prevention

1.  **Function Signature Changes**: Whenever you change a function definition (change arguments, add/remove return values), you **MUST** immediately search (using `grep_search`) for all call sites of that function in the entire codebase.
2.  **Update Call Sites**: Update every identified call site to match the new signature. Failure to do so leads to `ValueError`, `TypeError`, or logic bugs.
3.  **Variable Usage**: When adding or renaming variables (e.g., `radii`), ensure they are defined in all scopes where they are used (avoid `UnboundLocalError`).
4.  **Side Effects**: Analyze if logic changes (like using re-triangulated points) affect other parts of the system or assumptions made in subsequent steps.
5.  **Check before Proceeding**: After making a structural change, do not assume it works. Proactively check relevant files that might depend on that structure.
