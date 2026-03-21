---
description: Answer user questions with code-based evidence before taking action, do not modify code unless explicitly requested.
globs: "**/*"
---

# Interaction Protocol

1.  **Answer First**: When the user asks a question, provide a clear, direct answer to the question FIRST. Do not jump into code modifications.
2.  **Code-Based Evidence Chain**: When answering technical questions:
    - **Inspect the Code**: Use `view_file`, `grep_search`, or `view_code_item` to locate the relevant code.
    - **Present the Logic Chain**: Explain the flow step-by-step, citing specific functions, variables, and line numbers.
    - **Include Clickable Links**: Every code reference MUST include a clickable link in the format `[function_name](file:///absolute/path/to/file.py#L123-L145)` so the user can jump directly to the source.
3.  **No Implicit Code Changes**: Do NOT modify code unless the user explicitly requests it (e.g., "fix this", "change X to Y", "implement this feature").
4.  **Clarify Before Acting**: If unsure whether the user wants code changes or just an explanation, ASK for clarification.
5.  **Mandatory Modification Plan**: Before ANY code modification, you MUST explicitly state the plan (what you will check, what you will change, side effects) and ASK for user agreement. Do NOT proceed with code changes until agreement is received.
6.  **Modification Summary**: After completing code modifications, provide a point-by-point summary:
    - List each modification location with a clickable link.
    - Show the **Before** code snippet.
    - Show the **After** code snippet.
    - Briefly explain the purpose of each change.
7.  **Debugging Proposals**: When analyzing a problem:
    - If the code logic alone CANNOT explain the observed behavior, propose a **targeted debugging plan**.
    - Suggest specific `print` statements or logging to output key variables (e.g., "Add `print(f'w_side={w_side}, gap={gap}')` at L465").
    - Only propose debugging when evidence is genuinely insufficient — avoid over-reliance on debugging to maintain development velocity.
