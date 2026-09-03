# Workspace Rules

## Main Project

- The main project is `D:\hochanh\hk252\plagiarism_runtime`.
- Do not modify `D:\hochanh\hk252\doan\project` unless the user explicitly
  requests it.

## Write Access

- Read-only inspection and test execution are allowed by default.
- Before creating, editing, deleting, moving, or renaming any file or folder,
  ask the user for explicit permission.
- Permission applies only to the files and changes named in that request.
- Do not add benchmark scripts, reports, logs, or temporary outputs to the
  project root without explicit permission.

## Performance Outputs

- Prefer a separate location outside the project, such as
  `D:\hochanh\hk252\plagiarism_performance`.
- Ask for permission before creating that folder or writing performance data.
- Keep fixed input manifests and SHA-256 values so comparisons use the same
  files.

## Corpus Text

- Keep normalized extracted text because the current API uses it for TF-IDF,
  full-source comparison, Phase 4 evidence, and faculty index loading.
- Do not remove or change normalized text storage without explicit permission
  and a compatible API/index migration plan.
