# Actual explicit-path plugin predecessor fixtures

These synthetic fixtures use exact plugin 2a719abfbba4f0ee978516ab14b89e8c937d4213's
`pin_legacy_sqlite_db_path`, then SDK70daf32c's real HTTP publication and durable
worker. Explicit path selection precedes storage construction, so all five
cohorts naturally have the exact 304-object schema without `_dataset_identity`.
No identity table was deleted to manufacture this shape.

Completed and pending-billing cohorts each contain one profile. Queued intake,
pending-window execution and pending billing come from their real pipeline
owners. Real provider dispatch is forbidden; responses are SDK mocks and
embeddings are disabled. No account or customer data is included.

`provenance.json` records exact source commits, source recipe SHA256 and raw/gzip
hashes. Snapshots retain original business columns/rows. The recipe is an owned
capture record, not a CI download/runner; CI unpacks verified fixture copies.
Temporary HOME, SDK/plugin archives and source databases were removed after
capture. These prove the actual path owner/pipeline, not full coding-host
installation or real inference. The upgrade must still refuse without an
explicit namespace, and pending billing/window cohorts remain held even with it.
