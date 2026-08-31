import pytest

from pybatfish.client.session import Session

session = Session(load_questions=False)


def test_network_validation():
    with pytest.raises(ValueError):
        session.set_network("foo/bar")
