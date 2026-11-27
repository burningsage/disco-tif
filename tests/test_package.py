"""Package-level checks that hold for every phase of the rewrite."""

import re

import disco_tif


def test_version_is_pep440():
    """The single-source version must be a valid PEP 440 string."""
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", disco_tif.__version__)
