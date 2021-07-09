import pytest


@pytest.mark.import_triku
def import_triku():
    import triku

    dir(triku)
