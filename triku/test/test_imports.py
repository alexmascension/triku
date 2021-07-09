import pytest


@pytest.mark.import_triku
def test_import_triku():
    import triku

    dir(triku)
