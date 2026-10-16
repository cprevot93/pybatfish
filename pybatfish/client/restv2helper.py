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

import email.utils
import io
import sys
import time
from datetime import UTC, datetime
from typing import IO, TYPE_CHECKING, Any

import httpx2

# Re-export the native exception for callers that historically imported the
# transport's error type from this module.
HTTPStatusError = httpx2.HTTPStatusError

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
_IDEMPOTENT_METHODS = frozenset({"DELETE", "GET", "HEAD", "OPTIONS", "PUT", "TRACE"})

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
    # Sending a snapshot takes as long as it takes; a session that did not ask
    # for a timeout gets none here, since timing out mid-upload only leads to
    # uploading the snapshot again.
    _post(
        session,
        url_tail,
        None,
        stream=fd,
        request_kwargs=session._get_request_kwargs(default_timeout=None),
    )


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

    response = _get(session, url_tail, dict())
    try:
        return response.text
    finally:
        response.close()


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


def _check_response_status(response: httpx2.Response) -> None:
    """Raise HTTPX2's status error with the Batfish response text included."""
    try:
        response.raise_for_status()
    except RuntimeError:
        # Responses assembled by custom transports/tests may not carry a
        # Request instance.  HTTPX2 normally always sets one, but preserve the
        # public exception contract for these lightweight response objects.
        if response.status_code < 400:
            raise
        try:
            detail = response.text
        except RuntimeError:
            response.read()
            detail = response.text
        request = httpx2.Request("GET", "http://localhost")
        response.close()
        raise httpx2.HTTPStatusError(
            f"HTTP status {response.status_code}. {detail}", request=request, response=response
        )
    except httpx2.HTTPStatusError as exc:
        # A streamed error response may not have loaded ``text`` yet.  Read it
        # before constructing the enriched exception, then release the socket.
        try:
            detail = response.text
        except RuntimeError:
            response.read()
            detail = response.text
        response.close()
        raise httpx2.HTTPStatusError(f"{exc}. {detail}", request=exc.request, response=exc.response) from exc


def _proxy_mounts(proxies: dict[str, str] | str | None, verify: bool = True) -> dict[str, Any]:
    """Translate proxy mappings into HTTPX2 transport mounts."""
    if not proxies:
        return {}
    if isinstance(proxies, str):
        proxies = {"all": proxies}
    mounts: dict[str, Any] = {}
    for key, proxy in proxies.items():
        if not proxy:
            continue
        scheme = key.lower().rstrip(":/")
        if scheme == "all":
            mount_key = "all://"
        elif scheme in {"http", "https"}:
            mount_key = f"{scheme}://"
        else:
            # Ignore auxiliary entries such as ``no_proxy``. HTTP(S) targets
            # are selected by the mapping key; SOCKS is expressed in the URL.
            continue
        mounts[mount_key] = httpx2.HTTPTransport(proxy=proxy, retries=0, verify=verify)
    return mounts


def _proxy_key(proxies: dict[str, str] | str | None) -> tuple[tuple[str, str], ...]:
    if isinstance(proxies, str):
        return (("all", proxies),)
    return tuple(sorted((str(k), str(v)) for k, v in (proxies or {}).items()))


# Options accepted by HTTPX2.Client and Client.request/Client.stream.  The
# Session API historically accepted arbitrary transport keyword arguments, so
# unknown options are intentionally ignored instead of being passed through to
# HTTPX2 and raising a surprising TypeError.
_CLIENT_OPTION_NAMES = {
    "auth",
    "params",
    "headers",
    "cookies",
    "cert",
    "trust_env",
    "http1",
    "http2",
    "proxy",
    "mounts",
    "follow_redirects",
    "limits",
    "max_redirects",
    "event_hooks",
    "base_url",
    "transport",
    "default_encoding",
}
_REQUEST_OPTION_NAMES = {
    "content",
    "data",
    "files",
    "json",
    "params",
    "headers",
    "cookies",
    "auth",
    "follow_redirects",
    "extensions",
}


def _session_request_kwargs(session: Session) -> dict[str, Any]:
    """Return a defensive copy of optional transport kwargs on ``session``."""
    getter = getattr(session, "_get_request_kwargs", None)
    if callable(getter):
        kwargs = getter()
    else:
        kwargs = getattr(session, "request_kwargs", {})
    return dict(kwargs) if isinstance(kwargs, dict) else {}


def _effective_verify(session: Session, request_kwargs: dict[str, Any]) -> Any:
    """Resolve the HTTPX2 TLS verification setting.

    ``verify_ssl_certs`` is the explicit Session setting.  For backwards
    compatibility with the master branch's ``request_kwargs`` API, a generic
    ``verify`` option is honored when the Session is otherwise using its
    default verification setting.
    """
    session_verify = getattr(session, "verify_ssl_certs", True)
    generic_verify = request_kwargs.get("verify")
    if generic_verify is not None and session_verify == Options.verify_ssl_certs:
        return generic_verify
    return session_verify


def _new_client(session: Session, fail_fast: bool) -> httpx2.Client:
    request_kwargs = _session_request_kwargs(session)
    verify = _effective_verify(session, request_kwargs)
    session_proxies = getattr(session, "proxies", None)
    proxies = session_proxies if session_proxies is not None else request_kwargs.get("proxies")

    client_kwargs = {key: value for key, value in request_kwargs.items() if key in _CLIENT_OPTION_NAMES}
    # Session-level proxy and TLS settings are represented using HTTPX2's
    # native client configuration. They take precedence over generic options.
    proxy_mounts = _proxy_mounts(proxies, verify=verify) if proxies else {}
    if proxy_mounts:
        client_kwargs["mounts"] = proxy_mounts
        client_kwargs.pop("proxy", None)
        client_kwargs.pop("transport", None)
    elif session_proxies is not None:
        client_kwargs.pop("mounts", None)
        client_kwargs.pop("proxy", None)
    client_kwargs["verify"] = verify
    # Timeout is deliberately supplied on each request so changing a
    # Session's timeout does not require rebuilding its connection pool.
    client_kwargs["timeout"] = None
    client_kwargs.setdefault("trust_env", True)
    return httpx2.Client(**client_kwargs)


def _get_client(session: Session, fail_fast: bool = False) -> httpx2.Client:
    """Return the Session's reusable client, rebuilding mutable TLS/proxy state."""
    # Fail-fast changes the number of attempts in _request; it does not need a
    # second pool because HTTPX2's transport retries are disabled and all
    # retries are controlled by that request layer.
    client_attr = "_httpx_client"
    config_attr = "_httpx_client_config"
    config = (session.verify_ssl_certs, _proxy_key(session.proxies))
    client = getattr(session, client_attr, None)
    if client is not None and getattr(session, config_attr, None) != config:
        client.close()
        client = None
    if client is None:
        client = _new_client(session, fail_fast)
        setattr(session, client_attr, client)
        setattr(session, config_attr, config)
    return client


def close_session(session: Session) -> None:
    """Close all HTTPX2 clients owned by a Session."""
    client = getattr(session, "_httpx_client", None)
    if client is not None:
        client.close()
        setattr(session, "_httpx_client", None)
    setattr(session, "_httpx_client_config", None)


def _body_position(body: Any) -> tuple[bool, int | None]:
    """Return whether a request body can be replayed and its starting offset."""
    if body is None or isinstance(body, (bytes, bytearray, memoryview, str)):
        return True, None
    try:
        if hasattr(body, "seekable") and not body.seekable():
            return False, None
        position = body.tell()
        return True, position
    except (AttributeError, OSError, ValueError):
        return False, None


def _rewind_body(body: Any, position: int | None) -> bool:
    if position is None or body is None or isinstance(body, (bytes, bytearray, memoryview, str)):
        return True
    try:
        body.seek(position)
        return True
    except (AttributeError, OSError, ValueError):
        return False


def _retry_delay(response: httpx2.Response | None, retry_number: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(max(0.0, float(retry_after)))
            except ValueError:
                try:
                    date = email.utils.parsedate_to_datetime(retry_after)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=UTC)
                    return float(max(0.0, (date - datetime.now(UTC)).total_seconds()))
                except (TypeError, ValueError, OverflowError):
                    pass
    return float(min(Options.request_backoff_factor * (2**retry_number), 120.0))


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Match the old transport's handling of ``None`` query values."""
    if params is None:
        return None
    return {key: value for key, value in params.items() if value is not None}


def _request_options(
    session: Session,
    params: dict[str, Any] | None,
    headers: dict[str, str],
    json: Any,
    content: Any,
    request_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build HTTPX2 request options while protecting transport invariants."""
    generic = _session_request_kwargs(session) if request_kwargs is None else request_kwargs
    options = {key: value for key, value in generic.items() if key in _REQUEST_OPTION_NAMES}

    # Preserve additional caller headers while ensuring Batfish's required
    # headers win if a generic option supplied the same key.
    generic_headers = options.get("headers")
    if isinstance(generic_headers, dict):
        merged_headers = dict(generic_headers)
        merged_headers.update(headers)
        options["headers"] = merged_headers
    else:
        options["headers"] = headers

    # The REST helper owns query/body/timeout values. These internal fields
    # always override generic request options, including an explicit None
    # timeout which disables request timeouts.
    options["params"] = _clean_params(params)
    options["json"] = json
    options["content"] = content
    # An absent timeout in the resolved options means no timeout. HTTPX2
    # requires an explicit None to disable its own default.
    options["timeout"] = generic.get("timeout")
    return options


def _request(
    session: Session,
    method: str,
    url_tail: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    json: Any = None,
    content: Any = None,
    fail_fast: bool = False,
    request_kwargs: dict[str, Any] | None = None,
) -> httpx2.Response:
    """Retry connections for all methods, and other failures for idempotent methods."""
    client = _get_client(session, fail_fast)
    url = session.get_base_url2() + url_tail
    max_retries = (
        Options.max_initial_tries_to_connect_to_coordinator
        if fail_fast
        else Options.max_retries_to_connect_to_coordinator
    )
    # The former transport's total=N behavior means N retries in addition to
    # the initial request.
    max_attempts = max(1, int(max_retries) + 1)
    replayable, position = _body_position(content)
    request_headers = headers if headers is not None else _get_headers(session)

    for attempt in range(max_attempts):
        if attempt and not _rewind_body(content, position):
            raise ValueError("Request body cannot be rewound for retry")
        try:
            response = client.request(
                method, url, **_request_options(session, params, request_headers, json, content, request_kwargs)
            )
        except httpx2.TransportError as exc:
            safe_to_retry = method in _IDEMPOTENT_METHODS or isinstance(
                exc, (httpx2.ConnectError, httpx2.ConnectTimeout)
            )
            if not safe_to_retry or attempt + 1 >= max_attempts or not replayable:
                raise
            time.sleep(_retry_delay(None, attempt))
            continue

        if (
            method in _IDEMPOTENT_METHODS
            and response.status_code in _STATUS_FORCELIST
            and attempt + 1 < max_attempts
            and replayable
        ):
            delay = _retry_delay(response, attempt)
            response.close()
            time.sleep(delay)
            continue
        _check_response_status(response)
        return response

    raise AssertionError("unreachable")


def _delete(session: Session, url_tail: str, params: dict[str, Any] | None = None) -> None:
    """Issue an HTTPX2 DELETE request to the coordinator."""
    _request(session, "DELETE", url_tail, params=params).close()


def _get(
    session: Session, url_tail: str, params: None | dict[str, Any], stream: bool = False, fail_fast: bool = False
) -> httpx2.Response:
    """Issue an HTTPX2 GET request, optionally selecting fail-fast retries."""
    # ``stream`` remains accepted for compatibility; callers needing a file
    # object use _get_stream, which keeps the response socket open.
    return _request(session, "GET", url_tail, params=params, fail_fast=fail_fast)


def _get_dict(
    session: Session,
    url_tail: str,
    params: dict[str, Any] | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    response = _get(session, url_tail, params, fail_fast=fail_fast)
    try:
        return dict(response.json())
    finally:
        response.close()


def _get_list(session: Session, url_tail: str, params: dict[str, Any] | None = None) -> list[Any]:
    response = _get(session, url_tail, params)
    try:
        return list(response.json())
    finally:
        response.close()


class _HTTPXStream(io.RawIOBase):
    """A decoded, incremental, file-like adapter around an HTTPX2 response."""

    def __init__(self, response: httpx2.Response, context: Any):
        self._response = response
        self._context = context
        self._iterator = iter(response.iter_bytes())
        self._buffer = bytearray()
        self._closed = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            return b""
        if size is None or size < 0:
            chunks = [bytes(self._buffer)]
            self._buffer.clear()
            chunks.extend(self._iterator)
            return b"".join(chunks)
        while len(self._buffer) < size:
            try:
                self._buffer.extend(next(self._iterator))
            except StopIteration:
                break
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                # The stream context normally closes the response itself.
                # Fall back to response.close() for lightweight test/adaptor
                # context managers that do not implement that guarantee.
                self._context.__exit__(None, None, None)
            finally:
                if getattr(self._response, "is_closed", False) is not True:
                    self._response.close()
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def _get_stream(session: Session, url_tail: str, params: dict[str, Any] | None = None) -> Any:
    """Return a decoded, incremental binary stream for a coordinator GET."""
    client = _get_client(session)
    url = session.get_base_url2() + url_tail
    max_attempts = max(1, int(Options.max_retries_to_connect_to_coordinator) + 1)
    for attempt in range(max_attempts):
        stream_options = _request_options(session, params, _get_headers(session), None, None)
        context = client.stream("GET", url, **stream_options)
        try:
            response = context.__enter__()
        except httpx2.TransportError:
            context.__exit__(*sys.exc_info())
            if attempt + 1 >= max_attempts:
                raise
            time.sleep(_retry_delay(None, attempt))
            continue
        if response.status_code in _STATUS_FORCELIST and attempt + 1 < max_attempts:
            delay = _retry_delay(response, attempt)
            context.__exit__(None, None, None)
            if getattr(response, "is_closed", False) is not True:
                response.close()
            time.sleep(delay)
            continue
        try:
            _check_response_status(response)
        except BaseException:
            context.__exit__(*sys.exc_info())
            raise
        return _HTTPXStream(response, context)
    raise AssertionError("unreachable")


def _post(
    session: Session,
    url_tail: str,
    obj: Any,
    params: dict[str, Any] | None = None,
    stream: IO | None = None,
    request_kwargs: dict[str, Any] | None = None,
) -> None:
    """Issue an HTTPX2 POST request, optionally with JSON or raw content."""
    headers = _get_headers(session)
    if stream is not None:
        headers["Content-Type"] = "application/octet-stream"
    _request(
        session,
        "POST",
        url_tail,
        params=params,
        headers=headers,
        json=_encoder.default(obj) if obj is not None else None,
        content=stream,
        request_kwargs=request_kwargs,
    ).close()


def _put(session, url_tail, params=None, json=None, stream=None):
    """Issue an HTTPX2 PUT request, optionally with JSON or raw content."""
    headers = _get_headers(session)
    if stream is not None:
        headers["Content-Type"] = "application/octet-stream"
    _request(session, "PUT", url_tail, params=params, headers=headers, json=json, content=stream).close()


def _get_headers(session: Session) -> dict[str, str]:
    """Get base HTTP headers for v2 requests."""
    return {
        CoordConstsV2.HTTP_HEADER_BATFISH_APIKEY: session.api_key,
        CoordConstsV2.HTTP_HEADER_BATFISH_VERSION: pybatfish.__version__,
    }


def _put_json(session: Session, url_tail: str, obj: Any, params: dict[str, Any] | None = None) -> None:
    """Make an HTTPX2 PUT request with a JSON body."""
    _put(session, url_tail, params=params, json=_encoder.default(obj))


def _put_stream(
    session: Session,
    url_tail: str,
    stream: Any,
    params: dict[str, Any] | None = None,
) -> None:
    """Make an HTTPX2 PUT request with a raw request body."""
    _put(session, url_tail, params=params, stream=stream)
