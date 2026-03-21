---
description: Comprehensive coding protocol including indentation standards, variable scope verification, and code consistency rules.
globs: "**/*.py"
---

# OpenLPT Coding Protocol

## 1. Indentation & Formatting
- **Standard**: Always use **4 spaces** for indentation. Never use tabs.
- **Consistency**: Every time you modify or add a block (if, for, while, def, class), double-check that the indentation level matches the surrounding code.
- **Tool Usage**: Before using `replace_file_content` or `multi_replace_file_content`, manually count the leading spaces of the `TargetContent` and ensure the `ReplacementContent` preserves the correct logical indentation level.
- **Validation**: Ensure that the code does not have arbitrary extra spaces that lead to `IndentationError`.

## 2. Variable Dependency & Scope Safety
- **Rule**: Before inserting or moving code blocks, **ALWAYS trace the dependency chain** of all variables used in local scope.
- **Check**: Ensure that every variable read in the new block is guaranteed to be assigned in the lines *preceding* the block within the same local execution flow.
- **Prevention**: Do not blindly insert "guard clauses" or conditional checks (e.g., `if rmse_len < rmse_ray:`) without confirming that the variables (e.g., `rmse_len`) have been computed. If they are computed later in the original code, you MUST move their calculation logic up to before your insertion point.

## 3. Code Modification Integrity
- **Context Awareness**: When editing a function, read enough context to understand the variable lifecycle.
- **No Assumptions**: Do not assume a variable "must exist" because it appears later in the file. Python executes top-to-bottom.
