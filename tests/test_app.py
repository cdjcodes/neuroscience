"""Smoke-test the Streamlit app headlessly (needs NeuroMorpho.org for the default view)."""
import pytest
from streamlit.testing.v1 import AppTest


def _app():
    at = AppTest.from_file("../streamlit_app.py", default_timeout=180)
    return at


@pytest.mark.network
def test_default_view_loads_and_simulates():
    at = _app().run()
    assert not at.exception and not at.error
    labels = [m.label for m in at.metric]
    assert "Total length" in labels and "Firing rate" in labels


def test_ball_and_stick_mode_offline():
    at = _app().run()
    at.sidebar.radio[0].set_value("Ball-and-stick model").run()
    assert not at.exception
    spikes = next(m for m in at.metric if m.label == "Spikes")
    assert int(spikes.value) > 0
