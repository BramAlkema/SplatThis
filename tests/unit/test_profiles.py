"""Profile resolution: every name the CLI advertises must resolve.

``profiles.PROFILE_NAMES`` feeds the argparse ``choices`` for ``--profile``;
the dict of tuned defaults inside ``get_profile_defaults`` is maintained by
hand. This is the test the comment in ``profiles.py`` promises: each
advertised name resolves to a complete profile, and unknown names raise
instead of guessing.
"""

from __future__ import annotations

import pytest

from splatthis.cli import build_parser
from splatthis.profiles import PROFILE_NAMES, get_profile_defaults


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_every_advertised_profile_resolves(name: str) -> None:
    defaults = get_profile_defaults(name)
    for key in ("learning_rates", "loss_weights", "refinement", "schedule"):
        assert key in defaults, f"profile {name!r} is missing {key!r}"
    assert all(rate > 0 for rate in defaults["learning_rates"].values())


def test_unknown_profile_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="Unknown quality profile"):
        get_profile_defaults("definitely-not-a-profile")


def test_cli_profile_choices_are_exactly_the_advertised_names() -> None:
    action = next(a for a in build_parser()._actions if a.dest == "profile")
    assert tuple(action.choices) == PROFILE_NAMES
    assert action.default in PROFILE_NAMES
