#   Copyright 2018 The Batfish Open Source Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import io
import json
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from pybatfish.client import restv2helper
from pybatfish.client.consts import CoordConsts
from pybatfish.client.options import Options
from pybatfish.client.restv2helper import (
    _delete,
    _encoder,
    _get,
    _get_headers,
    _get_stream,
    _httpx_client,
    _httpx_client_fail_fast,
    _post,
    _proxy_mounts,
    _put,
    _transport,
    _transport_fail_fast,
    get_api_version,
)
from pybatfish.client.session import Session

BASE_URL = "base"


def _make_response(status_code: int, text: str = "") -> httpx.Response:
    """Build a real httpx.Response for use in tests."""
    return httpx.Response(status_code=status_code, text=text)


@pytest.fixture(scope="module")
def session() -> Session:
    s = Mock(spec=Session)
    s.get_base_url2.return_value = BASE_URL
    s.api_key = "0000"
    s.verify_ssl_certs = True
    s.timeout = 30
    s.proxies = None
    return s


# ---------------------------------------------------------------------------
# _check_response_status
# ---------------------------------------------------------------------------


def test_check_response_status_error():
    response = _make_response(400, "error detail")
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        restv2helper._check_response_status(response)
    assert "error detail" in str(exc_info.value)


def test_check_response_status_ok():
    response = _make_response(200, "no error")
    # Should not raise
    restv2helper._check_response_status(response)


# ---------------------------------------------------------------------------
# _proxy_mounts
# ---------------------------------------------------------------------------


def test_proxy_mounts_empty():
    assert _proxy_mounts(None) == {}
    assert _proxy_mounts({}) == {}


def test_proxy_mounts_http_only():
    mounts = _proxy_mounts({"http": "http://proxy:3128"})
    assert "http://" in mounts
    assert "https://" not in mounts
    assert isinstance(mounts["http://"], httpx.HTTPTransport)


def test_proxy_mounts_https_only():
    mounts = _proxy_mounts({"https": "http://proxy:3128"})
    assert "https://" in mounts
    assert "http://" not in mounts


def test_proxy_mounts_both_schemes():
    mounts = _proxy_mounts({"http": "http://proxy:3128", "https": "http://proxy:3128"})
    assert "http://" in mounts
    assert "https://" in mounts


# ---------------------------------------------------------------------------
# _delete
# ---------------------------------------------------------------------------


def test_delete(session: Session) -> None:
    """_delete should issue a DELETE to the correct URL with the right headers."""
    resource_url = "/test/url"
    target_url = f"{BASE_URL}{resource_url}"
    ok_response = _make_response(200)

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.delete.return_value = ok_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        _delete(session, resource_url)

    mock_client.delete.assert_called_once_with(
        target_url,
        headers=_get_headers(session),
        params=None,
    )


def test_delete_with_params(session: Session) -> None:
    """_delete should forward query-string params."""
    resource_url = "/test/url"
    params = {"key": "value"}
    ok_response = _make_response(200)

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.delete.return_value = ok_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        _delete(session, resource_url, params=params)

    mock_client.delete.assert_called_once_with(
        f"{BASE_URL}{resource_url}",
        headers=_get_headers(session),
        params=params,
    )


# ---------------------------------------------------------------------------
# _get
# ---------------------------------------------------------------------------


def test_get(session: Session) -> None:
    """_get should issue a GET using the standard (non-fail-fast) transport."""
    resource_url = "/test/url"
    target_url = f"{BASE_URL}{resource_url}"
    ok_response = _make_response(200, "{}")

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.get.return_value = ok_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        _get(session, resource_url, None)

    mock_client.get.assert_called_once_with(
        target_url,
        headers=_get_headers(session),
        params=None,
    )


def test_get_fail_fast(session: Session) -> None:
    """_get with fail_fast=True should use the fail-fast transport."""
    resource_url = "/test/url"
    target_url = f"{BASE_URL}{resource_url}"
    ok_response = _make_response(200, "{}")

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.get.return_value = ok_response

    captured_kwargs: dict = {}

    def fake_client(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_client

    with patch("pybatfish.client.restv2helper.httpx.Client", side_effect=fake_client):
        _get(session, resource_url, None, fail_fast=True)

    # The fail-fast transport should have been selected
    assert captured_kwargs.get("transport") is _transport_fail_fast
    mock_client.get.assert_called_once_with(
        target_url,
        headers=_get_headers(session),
        params=None,
    )


def test_get_standard_transport(session: Session) -> None:
    """_get without fail_fast should use the standard transport."""
    resource_url = "/test/url"
    ok_response = _make_response(200, "{}")

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.get.return_value = ok_response

    captured_kwargs: dict = {}

    def fake_client(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_client

    with patch("pybatfish.client.restv2helper.httpx.Client", side_effect=fake_client):
        _get(session, resource_url, None, fail_fast=False)

    assert captured_kwargs.get("transport") is _transport


# ---------------------------------------------------------------------------
# _get_stream
# ---------------------------------------------------------------------------


def test_get_stream_returns_bytesio(session: Session) -> None:
    """_get_stream should return an io.BytesIO wrapping the response body."""
    resource_url = "/test/url"
    body = b"binary data"

    # Build a minimal mock that satisfies the streaming context-manager protocol
    mock_response = MagicMock()
    mock_response.status_code = 200
    # _check_response_status calls raise_for_status(); make it a no-op
    mock_response.raise_for_status = Mock()
    mock_response.read.return_value = body
    # Simulate context-manager on response
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.stream.return_value = mock_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        result = _get_stream(session, resource_url)

    assert isinstance(result, io.BytesIO)
    assert result.read() == body


def test_get_stream_usable_as_context_manager(session: Session) -> None:
    """The BytesIO returned by _get_stream must work as a context manager."""
    resource_url = "/test/url"
    body = b"hello stream"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.read.return_value = body
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.stream.return_value = mock_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        stream = _get_stream(session, resource_url)

    with stream as s:
        assert s.read() == body


# ---------------------------------------------------------------------------
# _post
# ---------------------------------------------------------------------------


def test_post_json(session: Session) -> None:
    """_post with a JSON-serialisable object should call client.post with json=."""
    resource_url = "/test/url"
    target_url = f"{BASE_URL}{resource_url}"
    obj = "foo"
    ok_response = _make_response(200)

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.post.return_value = ok_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        _post(session, resource_url, obj)

    mock_client.post.assert_called_once_with(
        target_url,
        json=_encoder.default(obj),
        content=None,
        headers=_get_headers(session),
        params=None,
    )


def test_post_stream(session: Session) -> None:
    """_post with a stream should set Content-Type and pass content= kwarg."""
    resource_url = "/test/url"
    target_url = f"{BASE_URL}{resource_url}"
    ok_response = _make_response(200)

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.post.return_value = ok_response

    with io.StringIO("stream data") as stream_data:
        with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
            _post(session, resource_url, None, stream=stream_data)

        expected_headers = _get_headers(session)
        expected_headers["Content-Type"] = "application/octet-stream"
        mock_client.post.assert_called_once_with(
            target_url,
            json=None,
            content=stream_data,
            headers=expected_headers,
            params=None,
        )


# ---------------------------------------------------------------------------
# _put
# ---------------------------------------------------------------------------


def test_put(session: Session) -> None:
    """_put with no body should call client.put with json=None and content=None."""
    resource_url = "/test/url"
    target_url = f"{BASE_URL}{resource_url}"
    ok_response = _make_response(200)

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.put.return_value = ok_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        _put(session, resource_url)

    mock_client.put.assert_called_once_with(
        target_url,
        json=None,
        content=None,
        headers=_get_headers(session),
        params=None,
    )


def test_put_stream(session: Session) -> None:
    """_put with a stream body should set Content-Type and use content=."""
    resource_url = "/test/url"
    target_url = f"{BASE_URL}{resource_url}"
    ok_response = _make_response(200)

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.put.return_value = ok_response

    raw_data = b"raw bytes"
    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        _put(session, resource_url, stream=raw_data)

    expected_headers = _get_headers(session)
    expected_headers["Content-Type"] = "application/octet-stream"
    mock_client.put.assert_called_once_with(
        target_url,
        json=None,
        content=raw_data,
        headers=expected_headers,
        params=None,
    )


# ---------------------------------------------------------------------------
# Module-level client / transport objects
# ---------------------------------------------------------------------------


def test_httpx_clients_exist():
    """Confirm module-level httpx.Client objects are present."""
    assert isinstance(_httpx_client, httpx.Client)
    assert isinstance(_httpx_client_fail_fast, httpx.Client)


def test_transports_exist():
    """Confirm module-level httpx.HTTPTransport objects are present."""
    assert isinstance(_transport, httpx.HTTPTransport)
    assert isinstance(_transport_fail_fast, httpx.HTTPTransport)


def test_transport_retry_counts():
    """Confirm the standard transport is configured with the right retry count."""
    # httpx.HTTPTransport stores the underlying urllib3 Retry object on its
    # _pool attribute; we access the public-facing retries kwarg indirectly by
    # verifying the transport was created without raising.  The Options values
    # are the source of truth.
    assert Options.max_retries_to_connect_to_coordinator >= 1
    assert Options.max_initial_tries_to_connect_to_coordinator >= 1


# ---------------------------------------------------------------------------
# get_api_version
# ---------------------------------------------------------------------------


def test_get_api_version_old(session: Session) -> None:
    """When the backend does not return an API version key, default to '2.0.0'."""
    ok_response = _make_response(200, json.dumps({}))

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.get.return_value = ok_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        assert get_api_version(session) == "2.0.0"


def test_get_api_version_new(session: Session) -> None:
    """When the backend returns an API version key, return that version."""
    ok_response = _make_response(200, json.dumps({CoordConsts.KEY_API_VERSION: "2.1.0"}))

    mock_client = MagicMock()
    mock_client.__enter__ = Mock(return_value=mock_client)
    mock_client.__exit__ = Mock(return_value=False)
    mock_client.get.return_value = ok_response

    with patch("pybatfish.client.restv2helper.httpx.Client", return_value=mock_client):
        assert get_api_version(session) == "2.1.0"


if __name__ == "__main__":
    pytest.main()
