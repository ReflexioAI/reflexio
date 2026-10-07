"""Local dependency doubles for application smoke tests; never contact a service."""

from reflexio import ReflexioClient


class Model:
    def generate(self, message, context):  # noqa: ARG002
        return f"response:{message}"

    def diagnostic(self, message):
        pass


class HTTP:
    def post(self, path, *, json, timeout):  # noqa: ARG002
        if path == "/api/search":
            return {
                "success": True,
                "profiles": [],
                "user_playbooks": [],
                "agent_playbooks": [],
            }
        return {"success": True, "request_id": "local-request", "warnings": []}


def sdk_client():
    client = ReflexioClient(api_key="fixture-only", url_endpoint="http://localhost:1")
    client._make_request = lambda method, endpoint, headers=None, **kw: HTTP().post(
        endpoint, json=kw.get("json", {}), timeout=5
    )
    return client
