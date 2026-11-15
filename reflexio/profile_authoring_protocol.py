"""Source-owned authoring semantics and the supported admission protocol.

These identify this maintained implementation, never a runtime policy switch.
A lifecycle-neutral predecessor backport retains its own actual authoring policy
while adopting the same admission protocol. A descriptor cannot supply missing
original-work instructions or qualify an unpatched historical image.
"""

SOURCE_RETENTION_POLICY = "source-retention-v1"
PROFILE_AUTHORING_POLICY = SOURCE_RETENTION_POLICY
ESTIMATED_LIFETIME_POLICY = "estimated-lifetime-v1"
LEARNING_ADMISSION_PROTOCOL = "learning-transition-v1"
AUTHORING_POLICIES = (ESTIMATED_LIFETIME_POLICY, SOURCE_RETENTION_POLICY)
