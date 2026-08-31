#   Copyright 2018 The Batfish Open Source Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
"""Tests for the HTTPX2 REST transport."""

import io
from unittest.mock import Mock, patch

import httpx2
import pytest

from pybatfish.client import restv2helper
from pybatfish.client.consts import CoordConsts
from pybatfish.client.session import Session

BASE_URL = "http://example.test"


@pytest.fixture
def session():
    session = Session(host="example.test", load_questions=False)
    session.get_base_url2 = Mock(return_value=BASE_URL)
    yield session
    session.close()


def install_transport(session, handler):
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    session._httpx_client = client
    session._httpx_client_config = (True, tuple())
    return client


def test_check_response_status_error():
    response = httpx2.Response(400, text="error detail", request=httpx2.Request("GET", BASE_URL))
    with pytest.raises(httpx2.HTTPStatusError, match="error detail"):
        restv2helper._check_response_status(response)
    assert response.is_closed


def test_proxy_mounts():
    mounts = restv2helper._proxy_mounts({"http": "http://proxy:3128", "https": "socks5://proxy:1080"})
    assert set(mounts) == {"http://", "https://"}
    assert all(isinstance(transport, httpx2.HTTPTransport) for transport in mounts.values())


def test_verbs_and_forwarding(session):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers["x-batfish-apikey"]))
        return httpx2.Response(200, json={}, request=request)

    install_transport(session, handler)
    restv2helper._get(session, "/get", {"a": "b"})
    restv2helper._post(session, "/post", {"value": 1})
    restv2helper._put(session, "/put", json={"value": 2})
    restv2helper._delete(session, "/delete")
    assert [(method, path) for method, path, _ in seen] == [
        ("GET", "/get"),
        ("POST", "/post"),
        ("PUT", "/put"),
        ("DELETE", "/delete"),
    ]
    assert all(api_key == session.api_key for _, _, api_key in seen)


def test_retry_status_and_rewind(session):
    calls = []
    body = io.BytesIO(b"payload")

    def handler(request):
        calls.append(request.content)
        return httpx2.Response(503 if len(calls) == 1 else 200, text="retry", request=request)

    install_transport(session, handler)
    with patch.object(restv2helper.time, "sleep"):
        restv2helper._post(session, "/upload", None, stream=body)
    assert calls == [b"payload", b"payload"]
    assert body.tell() == len(body.getvalue())


def test_get_stream_is_incremental(session):
    reads = []

    def handler(request):
        def chunks():
            reads.append("started")
            yield b"first"
            yield b"second"

        return httpx2.Response(200, content=chunks(), request=request)

    install_transport(session, handler)
    stream = restv2helper._get_stream(session, "/stream")
    assert reads == []
    assert stream.read(5) == b"first"
    assert reads == ["started"]
    assert stream.read() == b"second"
    stream.close()
    assert stream.closed


def test_get_api_version(session):
    def handler(request):
        return httpx2.Response(200, json={CoordConsts.KEY_API_VERSION: "2.1.0"}, request=request)

    install_transport(session, handler)
    assert restv2helper.get_api_version(session) == "2.1.0"


def test_session_context_manager():
    with Session(load_questions=False) as session:
        assert session._httpx_client is None
    assert session._httpx_client is None


if __name__ == "__main__":
    pytest.main()
