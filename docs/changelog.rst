Changelog
=========

All notable changes to this project will be documented in this file.

[0.9.3] - 2026-03-29
--------------------

Changed
~~~~~~~
* Added explicit ``run``, ``generate``, and ``gui`` CLI subcommands, plus
  compatibility warnings for legacy flat CLI usage.
* Added a dedicated synthetic-data generation flow and hardened generator setup
  for small graphs and impossible parent-count requests.
* Aligned package/runtime metadata for installable CLI usage and clarified that
  ``pc`` and ``cam`` remain unsupported public APIs in the user docs.

[0.9.2] - 2025
--------------

Changed
~~~~~~~
* Added SHAP budget batching, bootstrap SHAP caching, and HPO optimization
  controls for ReX workflows.
* Renamed and surfaced the optimization settings in the CLI and GUI Train tab.
* Fixed the bootstrap-cache and gradient SHAP regressions uncovered during
  merge validation.

[0.9.0] - 2025
--------------

Changed
~~~~~~~
* Refactored the NiceGUI app into smaller GUI modules with shared helpers.
* Extracted GUI CSS into static assets and improved Train tab resize behavior.
* Added GUI-focused tests for helper utilities and Cytoscape layouts.
* Packaged GUI CSS assets for distribution.

[0.5.0] - 2024
--------------

Initial public release of CausalExplain, after three years of research.

Added
~~~~~
* Support for various causal inference methods
* Command-line interface
* Basic documentation
