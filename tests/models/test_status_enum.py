from reflexio.models.api_schema.domain.enums import Status


def test_expired_status_value():
    assert Status.EXPIRED.value == "expired"
    assert Status("expired") is Status.EXPIRED
