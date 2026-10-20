# Publishing version 1.1.0

This distribution is prepared for a new software release. It does not assign
a DOI or change an existing public archive.

1. Add the complete distribution to the GitHub repository, preserving the
   license, package structure, revision drivers, references and provenance.
   Keep the native `.mat` inputs in the separate dataset archive.
2. Commit the changes and create release/tag `v1.1.0`, preserving the existing
   `v1.0.0` history.
3. Create a new version of the existing Zenodo software record, or use the
   repository's existing Zenodo integration. Upload/archive this software
   version. The input-matrix dataset is a separate record.
4. Obtain the DOI of the published new software version and confirm that its
   downloadable files include the primary-analysis entry points and documentation.
5. Update the manuscript's Code availability statement and software bibliography
   from version 1.0.0 to 1.1.0 and the new version DOI. Keep the dataset citation.

Obtain the new version DOI from the published record; do not infer it from the
old DOI. The old version DOI continues to identify the old software version.
If publication metadata are later added to `CITATION.cff`, regenerate the
distribution checksums. Keep publication metadata distinct from the original
reference-run manifests and numerical source hashes.
