import pytest

from minstrel_analyzer.clap import Clap


@pytest.fixture(scope="session")
def clap() -> Clap:
    """Load CLAP once for the whole test session — the checkpoint is ~1.5GB."""
    return Clap()
