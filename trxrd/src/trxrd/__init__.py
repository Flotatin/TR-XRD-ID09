"""TR-XRD v2: time-resolved X-ray diffraction analysis (ID09 dynamic DAC).

Version 2 of the program; version 1 is the legacy `cedapp` application.

Notebook entry point::

    from trxrd import XRDRun
    run = XRDRun("water02_scan0027.h5")
    run.summary
"""

__version__ = "2.0.0a1"


def __getattr__(name):
    # Lazy import keeps `import trxrd` light (no h5py/matplotlib until needed).
    if name == "XRDRun":
        from trxrd.notebook import XRDRun

        return XRDRun
    raise AttributeError(name)
