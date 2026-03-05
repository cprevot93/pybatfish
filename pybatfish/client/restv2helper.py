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
from __future__ import annotations

import io
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
)

import httpx

import pybatfish
from pybatfish.client.consts import CoordConsts, CoordConstsV2
from pybatfish.util import BfJsonEncoder

from ..datamodel import NodeRolesData, ReferenceBook, VariableType
from .options import Options
from .workitem import WorkItem

if TYPE_CHECKING:
    from pybatfish.client.session import Session

# List of HTTP statuses to retry
_STATUS_FORCELIST = [429, 500, 502, 503, 504]

# ---------------------------------------------------------------------------
# Module-level httpx clients (replacing the old requests.Session objects)
#
# httpx.HTTPTransport supports a `retries` parameter that maps to urllib3's
# Retry.  It only covers connection-level retries; for fine-grained status
# based retries we keep the same retry counts from Options so the observable
# behaviour is unchanged.
# ---------------------------------------------------------------------------

_transport = httpx.HTTPTransport(
    retries=Options.max_retries_to_connect_to_coordinator,
)
_httpx_client = httpx.Client(
    transport=_transport,
    # No default timeout – callers pass session.timeout explicitly so we
    # preserve the pre-existing per-request timeout behaviour.
    timeout=None,
)

_transport_fail_fast = httpx.HTTPTransport(
    retries=Options.max_initial_tries_to_connect_to_coordinator,
)
_httpx_client_fail_fast = httpx.Client(
    transport=_transport_fail_fast,
    timeout=None,
)

_encoder = BfJsonEncoder()

__all__ = [
    "delete_network",
    "delete_node_role_dimension",
    "delete_reference_book",
    "delete_snapshot",
    "delete_snapshot_object",
    "fork_snapshot",
    "get_answer",
    "get_network",
    "get_network_object",
    "get_node_role_dimension",
    "get_node_roles",
    "get_reference_book",
    "get_reference_library",
    "get_snapshot_input_object",
    "get_snapshot_object",
    "list_networks",
    "list_snapshots",
    "put_network_object",
    "put_node_role_dimension",
    "put_reference_book",
    "put_snapshot_object",
    "read_question_settings",
    "write_question_settings",
]


def list_networks(session):
    # type: (Session) -> list[dict[str, Any]]
    """List the networks in the current session."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}"
    return _get_list(session, url_tail)


def list_snapshots(session: Session, verbose: bool) -> list[str] | list[dict[str, str]]:
    """List the snapshots in the current network."""
    if not session.network:
        raise ValueError("Network must be set to list snapshots")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}"
    return _get_list(session, url_tail, {CoordConstsV2.QP_VERBOSE: verbose})


def fork_snapshot(session, obj):
    # type: (Session, dict[str, Any]) -> None
    if not session.network:
        raise ValueError("Network must be set to fork a snapshot")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}:{CoordConstsV2.RSC_FORK}"
    return _post(session, url_tail, obj)


def delete_network(session, name):
    # type: (Session, str) -> None
    """Deletes the network with the given name."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{name}"
    return _delete(session, url_tail)


def delete_network_object(session, key):
    # type: (Session, str) -> None
    """Deletes extended object with given key for the current network."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_OBJECTS}"
    return _delete(session, url_tail, {CoordConstsV2.QP_KEY: key})


def delete_node_role_dimension(session, dimension):
    # type: (Session, str) -> None
    """Deletes the definition of the given node role dimension for the active network."""
    if not session.network:
        raise ValueError("Network must be set to delete a node role dimension")
    if not dimension:
        raise ValueError("Dimension must be a non-empty string")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_NODE_ROLES}/{dimension}"
    return _delete(session, url_tail)


def delete_reference_book(session, book_name):
    # type: (Session, str) -> None
    """Deletes the definition of the given reference book name."""
    if not session.network:
        raise ValueError("Network must be set to delete a reference book")
    if not book_name:
        raise ValueError("Book name must be a non-empty string")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_REFERENCE_LIBRARY}/{book_name}"
    return _delete(session, url_tail)


def delete_snapshot(session, snapshot, network):
    # type: (Session, str, str) -> None
    """Deletes the snapshot with the given name."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{network}/{CoordConstsV2.RSC_SNAPSHOTS}/{snapshot}"
    return _delete(session, url_tail)


def delete_snapshot_object(session, key, snapshot=None):
    # type: (Session, str, str|None) -> None
    """Deletes extended object with given key for the current snapshot."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_OBJECTS}"
    _delete(session, url_tail, {CoordConstsV2.QP_KEY: key})


def get_answer(session, question, params):
    # type: (Session, str, dict[str, str|None]) -> dict[str, Any]
    """Get answer for the specified question."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_QUESTIONS}/{question}/{CoordConstsV2.RSC_ANSWER}"
    return _get_dict(session, url_tail, params)


def get_network(session, network):
    # type: (Session, str) -> dict[str, Any]
    """Gets information about the specified network."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{network}"
    return _get_dict(session, url_tail)


def init_network(session: Session, new_network_name: str) -> None:
    """Attemps to create a new network with the given name."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}"
    _post(session, url_tail, None, params={CoordConstsV2.QP_NAME: new_network_name})


def upload_snapshot(session: Session, snapshot_name: str, fd: IO) -> None:
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{snapshot_name}"
    _post(session, url_tail, None, stream=fd)


def get_network_object(session, key):
    # type: (Session, str) -> Any
    """Gets extended object with given key for the current network."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_OBJECTS}"
    return _get_stream(session, url_tail, {CoordConstsV2.QP_KEY: key})


def get_snapshot_input_object(session: Session, key: str, snapshot: str | None = None) -> Any:
    """Gets input object with given key for the current snapshot."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_INPUT}"
    return _get_stream(session, url_tail, {CoordConstsV2.QP_KEY: key})


def get_snapshot_object(session: Session, key: str, snapshot: str | None = None) -> Any:
    """Gets extended object with given key for the current snapshot."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_OBJECTS}"
    return _get_stream(session, url_tail, {CoordConstsV2.QP_KEY: key})


def get_node_role_dimension(session: Session, dimension: str) -> dict[str, Any]:
    """Gets the definition of the given node role dimension for the active network."""
    if not session.network:
        raise ValueError("Network must be set to get node roles")
    if not dimension:
        raise ValueError("Dimension must be a non-empty string")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_NODE_ROLES}/{dimension}"
    return _get_dict(session, url_tail)


def get_node_roles(session):
    # type: (Session) -> dict
    """Gets the definitions of node roles for the active network."""
    if not session.network:
        raise ValueError("Network must be set to get node roles")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_NODE_ROLES}"
    return _get_dict(session, url_tail)


def get_reference_book(session, book_name):
    # type: (Session, str) -> dict
    """Gets the reference book for the active network."""
    if not session.network:
        raise ValueError("Network must be set to get a reference book")
    if not book_name:
        raise ValueError("Book name must be a non-empty string")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_REFERENCE_LIBRARY}/{book_name}"
    return _get_dict(session, url_tail)


def get_reference_library(session):
    # type: (Session) -> dict
    """Gets the reference library for the active network."""
    if not session.network:
        raise ValueError("Network must be set to get the reference library")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_REFERENCE_LIBRARY}"
    return _get_dict(session, url_tail)


def get_snapshot_inferred_node_roles(session, snapshot=None):
    # type: (Session, str|None) -> dict
    """Gets suggested definitions and hypothetical assignments of node roles for the active network and snapshot."""
    if not session.network:
        raise ValueError("Network must be set to get node roles")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_INFERRED_NODE_ROLES}"
    return _get_dict(session, url_tail)


def get_snapshot_inferred_node_role_dimension(session, dimension, snapshot=None):
    # type: (Session, str, str|None) -> dict
    """Gets the suggested definition and hypothetical assignments of node roles for the given inferred dimension for the active network and snapshot."""
    if not session.network:
        raise ValueError("Network must be set to get node roles")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_INFERRED_NODE_ROLES}/{dimension}"
    return _get_dict(session, url_tail)


def get_snapshot_node_roles(session, snapshot=None):
    # type: (Session, str|None) -> dict
    """Gets the definitions and assignments of node roles for the active network and snapshot."""
    if not session.network:
        raise ValueError("Network must be set to get node roles")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_NODE_ROLES}"
    return _get_dict(session, url_tail)


def get_snapshot_node_role_dimension(session, dimension, snapshot=None):
    # type: (Session, str, str|None) -> dict
    """Gets the definition and assignments of node roles for the given dimension for the active network and snapshot."""
    if not session.network:
        raise ValueError("Network must be set to get node roles")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_NODE_ROLES}/{dimension}"
    return _get_dict(session, url_tail)


def get_work_log(session: Session, snapshot: str | None, work_id: str) -> str:
    """Retrieve the log for a work item with a given ID."""
    if not session.network:
        raise ValueError("Network must be set to get node roles")

    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_WORK_LOG}/{work_id}"

    return _get(session, url_tail, dict()).text


def get_component_versions(session):
    # type: (Session) -> dict[str, Any]
    """Get a dictionary of backend components (e.g. Batfish, Z3) and their versions."""
    return _get_dict(session, "/version")


def get_api_version(session: Session) -> str:
    """Gets the API version if present, else returns '2.0.0'"""
    component_versions = get_component_versions(session)
    return str(component_versions.get(CoordConsts.KEY_API_VERSION, "2.0.0"))


def get_question_templates(session: Session, verbose: bool) -> dict:
    """Get question templates from the backend.

    :param verbose: if True, even hidden questions will be returned.
    """
    return _get_dict(
        session,
        url_tail=f"/{CoordConstsV2.RSC_QUESTION_TEMPLATES}",
        params={CoordConstsV2.QP_VERBOSE: verbose},
        fail_fast=True,
    )


def put_network_object(session, key, data):
    # type: (Session, str, Any) -> None
    """Put data as extended object with given key for the current network."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_OBJECTS}"
    _put_stream(session, url_tail, data, {CoordConstsV2.QP_KEY: key})


def put_node_roles(session: Session, node_roles_data: NodeRolesData) -> None:
    """Writes the definitions of node roles for the active network. Completely replaces any existing definitions."""
    if not session.network:
        raise ValueError("Network must be set to get node roles")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_NODE_ROLES}"
    return _put_json(session, url_tail, node_roles_data)


def put_reference_book(session: Session, book: ReferenceBook) -> None:
    """Put a reference book to the active network."""
    if not session.network:
        raise ValueError("Network must be set to add reference book")
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_REFERENCE_LIBRARY}/{book.name}"
    _put_json(session, url_tail, book)


def put_snapshot_object(session: Session, key: str, data: Any, snapshot: str | None = None) -> None:
    """Put data as extended object with given key for the current snapshot."""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SNAPSHOTS}/{session.get_snapshot(snapshot)}/{CoordConstsV2.RSC_OBJECTS}"
    _put_stream(session, url_tail, data, {CoordConstsV2.QP_KEY: key})


def read_question_settings(session, question_class, json_path):
    # type: (Session, str, list[str]|None) -> dict[str, Any]
    """Retrieves the settings for a question class."""
    if not session.network:
        raise ValueError("Network must be set to read question class settings")
    json_path_tail = "/".join(json_path) if json_path else ""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SETTINGS}/{CoordConstsV2.RSC_QUESTIONS}/{question_class}/{json_path_tail}"
    return _get_dict(session, url_tail)


def write_question_settings(session, settings, question_class, json_path):
    # type: (Session, dict[str, Any], str, list[str]|None) -> None
    """Writes settings for a question class."""
    if not session.network:
        raise ValueError("Network must be set to write question class settings")
    json_path_tail = "/".join(json_path) if json_path else ""
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_SETTINGS}/{CoordConstsV2.RSC_QUESTIONS}/{question_class}/{json_path_tail}"
    _put_json(session, url_tail, settings)


def queue_work(session: Session, work_item: WorkItem) -> None:
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_WORK}"
    _post(session, url_tail, work_item.to_dict())


def get_work_status(session: Session, work_item_id: str) -> dict[str, Any]:
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_WORK}/{work_item_id}"
    return _get_dict(session, url_tail)


def list_incomplete_work(session: Session) -> list[dict[str, Any]]:
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_WORK}"
    return _get_list(session, url_tail)


def auto_complete(
    session: Session,
    completion_type: VariableType,
    query: str | None = None,
    max_suggestions: int | None = None,
) -> dict[str, Any]:
    url_tail = "/{}/{}{}/{}/{}".format(
        CoordConstsV2.RSC_NETWORKS,
        session.network,
        (f"/{CoordConstsV2.RSC_SNAPSHOTS}/{session.snapshot}" if session.snapshot else ""),
        CoordConstsV2.RSC_AUTOCOMPLETE,
        completion_type.value,
    )
    params = {}  # type: dict[str, Any]
    if query:
        params[CoordConstsV2.QP_QUERY] = query
    if max_suggestions:
        params[CoordConstsV2.QP_MAX_SUGGESTIONS] = max_suggestions
    return _get_dict(session, url_tail, params=params)


def upload_question(session: Session, question_name: str, question_str: str) -> None:
    url_tail = f"/{CoordConstsV2.RSC_NETWORKS}/{session.network}/{CoordConstsV2.RSC_QUESTIONS}/{question_name}"
    _put(session, url_tail, stream=question_str)


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------


def _proxy_mounts(proxy: dict | None, verify: bool = True) -> dict[str, httpx.HTTPTransport]:
    """Convert a requests-style proxies dict to httpx mounts.

    Supports the same ``{"http": "...", "https": "..."}`` shape that the
    ``Session.proxies`` attribute has always accepted so that callers do not
    need to change anything.
    """
    if not proxy:
        return {}
    mounts: dict[str, httpx.HTTPTransport] = {}
    for scheme in ("http", "https"):
        url = proxy.get(scheme, "")
        if url:
            mounts[f"{scheme}://"] = httpx.HTTPTransport(proxy=url, verify=verify)
    return mounts


# ---------------------------------------------------------------------------
# Response / error helpers
# ---------------------------------------------------------------------------


def _check_response_status(response: httpx.Response) -> None:
    """Raise an HTTPStatusError after enriching it with the response body text.

    This mirrors the previous behaviour where ``requests``'
    ``Response.raise_for_status()`` was called and the raw error text was
    appended to the message.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise httpx.HTTPStatusError(
            f"{e}. {response.text}",
            request=e.request,
            response=e.response,
        )


# ---------------------------------------------------------------------------
# Private HTTP verb helpers
# ---------------------------------------------------------------------------


def _delete(session: Session, url_tail: str, params: dict[str, Any] | None = None) -> None:
    """Make an HTTP(s) DELETE request to Batfish coordinator.

    :raises httpx.ConnectError if the coordinator is not available
    :raises httpx.HTTPStatusError on non-2xx response
    """
    url = session.get_base_url2() + url_tail
    mounts = _proxy_mounts(session.proxies, verify=session.verify_ssl_certs)
    with httpx.Client(
        mounts=mounts or None,
        verify=session.verify_ssl_certs,
        transport=_transport,
        timeout=session.timeout,  # None means no timeout; pass a float (e.g. 30.0) for a hard limit
    ) as client:
        response = client.delete(
            url,
            headers=_get_headers(session),
            params=params,
        )
    _check_response_status(response)


def _get(
    session: Session,
    url_tail: str,
    params: None | dict[str, Any],
    stream: bool = False,
    fail_fast: bool = False,
) -> httpx.Response:
    """Make an HTTP(s) GET request to Batfish coordinator.

    :raises httpx.ConnectError if the coordinator is not available
    :raises httpx.HTTPStatusError on non-2xx response

    Note: the ``stream`` parameter is accepted for API compatibility but is
    no longer needed – httpx always buffers the response body unless an
    explicit streaming context manager is used.  Callers that need raw bytes
    should use :func:`_get_stream` instead.
    """
    url = session.get_base_url2() + url_tail
    mounts = _proxy_mounts(session.proxies, verify=session.verify_ssl_certs)
    transport = _transport_fail_fast if fail_fast else _transport
    with httpx.Client(
        mounts=mounts or None,
        verify=session.verify_ssl_certs,
        transport=transport,
        timeout=session.timeout,  # None means no timeout; pass a float (e.g. 30.0) for a hard limit
    ) as client:
        response = client.get(
            url,
            headers=_get_headers(session),
            params=params,
        )
    _check_response_status(response)
    return response


def _get_dict(session, url_tail, params=None, fail_fast=False):
    # type: (Session, str, None|dict[str, Any], bool) -> dict[str, Any]
    """Make an HTTP(s) GET request to Batfish coordinator that should return a JSON dict."""
    response = _get(session, url_tail, params, fail_fast=fail_fast)
    return dict(response.json())


def _get_list(session: Session, url_tail: str, params: dict[str, Any] | None = None) -> list[Any]:
    """Make an HTTP(s) GET request to Batfish coordinator that should return a JSON list."""
    response = _get(session, url_tail, params)
    return list(response.json())


def _get_stream(session: Session, url_tail: str, params: dict[str, Any] | None = None) -> Any:
    """Make an HTTP(s) GET request that returns the response body as a binary stream.

    Returns an :class:`io.BytesIO` object so callers can use it as a context
    manager (``with stream: stream.read()``) exactly as they did with the
    previous ``urllib3`` raw socket.

    :raises httpx.ConnectError if the coordinator is not available
    :raises httpx.HTTPStatusError on non-2xx response
    """
    url = session.get_base_url2() + url_tail
    mounts = _proxy_mounts(session.proxies, verify=session.verify_ssl_certs)
    # Use the streaming API so large objects are not fully loaded into memory
    # before we can check the status code.
    with httpx.Client(
        mounts=mounts or None,
        verify=session.verify_ssl_certs,
        transport=_transport,
        timeout=session.timeout,  # None means no timeout; pass a float (e.g. 30.0) for a hard limit
    ) as client:
        with client.stream("GET", url, headers=_get_headers(session), params=params) as response:
            _check_response_status(response)
            raw_bytes = response.read()
    return io.BytesIO(raw_bytes)


def _post(
    session: Session,
    url_tail: str,
    obj: Any,
    params: dict[str, Any] | None = None,
    stream: IO | None = None,
) -> None:
    """Make an HTTP(s) POST request to Batfish coordinator.

    :raises httpx.ConnectError if the coordinator is not available
    :raises httpx.HTTPStatusError on non-2xx response
    """
    url = session.get_base_url2() + url_tail
    headers = _get_headers(session)
    if stream:
        headers["Content-Type"] = "application/octet-stream"
    mounts = _proxy_mounts(session.proxies, verify=session.verify_ssl_certs)
    with httpx.Client(
        mounts=mounts or None,
        verify=session.verify_ssl_certs,
        transport=_transport,
        timeout=session.timeout,  # None means no timeout; pass a float (e.g. 30.0) for a hard limit
    ) as client:
        response = client.post(
            url,
            # httpx uses `json=` for automatic serialisation and `content=`
            # for raw bytes / streams; there is no `data=` shortcut for JSON.
            json=_encoder.default(obj) if obj is not None else None,
            content=stream,
            headers=headers,
            params=params,
        )
    _check_response_status(response)
    return None


def _put(session, url_tail, params=None, json=None, stream=None):
    # type: (Session, str, None|dict[str, Any], Any|None, None|Any) -> None
    """Make an HTTP(s) PUT request to Batfish coordinator.

    :raises httpx.ConnectError if the coordinator is not available
    :raises httpx.HTTPStatusError on non-2xx response
    """
    headers = _get_headers(session)
    if stream:
        headers["Content-Type"] = "application/octet-stream"
    url = session.get_base_url2() + url_tail
    mounts = _proxy_mounts(session.proxies, verify=session.verify_ssl_certs)
    with httpx.Client(
        mounts=mounts or None,
        verify=session.verify_ssl_certs,
        transport=_transport,
        timeout=session.timeout,  # None means no timeout; pass a float (e.g. 30.0) for a hard limit
    ) as client:
        response = client.put(
            url,
            json=json,
            content=stream,
            headers=headers,
            params=params,
        )
    _check_response_status(response)
    return None


def _get_headers(session: Session) -> dict[str, str]:
    """Get base HTTP headers for v2 requests."""
    return {
        CoordConstsV2.HTTP_HEADER_BATFISH_APIKEY: session.api_key,
        CoordConstsV2.HTTP_HEADER_BATFISH_VERSION: pybatfish.__version__,
    }


def _put_json(session: Session, url_tail: str, obj: Any, params: dict[str, Any] | None = None) -> None:
    """Make an HTTP(s) PUT request with a JSON body to Batfish coordinator."""
    _put(session, url_tail, params=params, json=_encoder.default(obj))


def _put_stream(
    session: Session,
    url_tail: str,
    stream: Any,
    params: dict[str, Any] | None = None,
) -> None:
    """Make an HTTP(s) PUT request with a raw-bytes body to Batfish coordinator."""
    _put(session, url_tail, params=params, stream=stream)
