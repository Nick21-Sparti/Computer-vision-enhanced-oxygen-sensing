# Uploading this repository

The ZIP is an upload-ready source tree, not an already published GitHub repository. Extract it and upload the **contents** of `Computer-vision-enhanced-oxygen-sensing/` so that `README.md` sits at the repository root. Keep `scripts/repo_config.py` beside the research scripts.

The supplied repository title can remain `Computer-vision-enhanced-oxygen-sensing`. Its description can be:

> Research code for computer-vision-enhanced oxygen sensing using heterogeneous luminescent films, temporal vision transformers and classical dynamic compensation.

GitHub's web upload or GitHub Desktop can be used. With Git, initialise and commit the source tree locally, add your actual GitHub repository URL as the remote, then push. No remote, account, visibility setting or existing repository was assumed or changed during preparation.

Before making a formal research release, supply the intended software licence, verified paper/citation details, a data availability statement with the real access location, and the environment/configuration used for the reported results. The bundle does not invent these. `CITATION.cff` identifies the associated manuscript authors from the supplied document; confirm the desired software authorship before publication.

The `.gitignore` excludes recordings, weights, caches and result directories from ordinary Git staging. For manual browser upload, select source/documentation files yourself; `.gitignore` does not filter files selected in a browser. Keep machine-specific edits to `repo_config.py` local or restore its portable defaults before uploading. No raw experimental data or manuscripts are bundled here.
