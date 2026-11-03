# Original v1 publication predecessor fixtures

These are synthetic SQLite backups produced by the exact SDK commit recorded in
`provenance.json`, through its real HTTP publisher, window executor, journal,
publication receipt, and financial acknowledgement owners. No customer data,
provider response, credentials, or remote database is included. Completion and
embedding boundaries are deterministic local stubs; MOCK_LLM_RESPONSE is false.

`producer.py.txt` records the original capture tests. To regenerate, archive the
recorded predecessor SDK, add that file under tests/cli as a Python test, and run
with PYTHON_DOTENV_DISABLED=1, an empty REFLEXIO_ENV_FILE and
REFLEXIO_SYNTHETIC_CAPTURE_DIR pointing to a temporary directory. The pipeline
fixture is that predecessor's tests.server.services.test_durable_window_pipeline.
Do not fetch or run predecessor code in CI. Fixture gzip bytes use mtime=0; source
and compressed hashes are recorded. Random IDs/clocks make newly regenerated
captures logically equivalent rather than byte-identical.

The candidate tests unpack private temporary copies, verify the original source
hash, and exercise current config admission, schema migration, normal worker and
financial owners. Six phases cover interrupted running work, pending completed
finances, legitimate source projection without a saved body, and crash boundaries
before/after financial acknowledgement including source erasure. Parent history,
original paid journal/quota, and original receipt identities must remain intact.
