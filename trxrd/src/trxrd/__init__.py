"""TR-XRD: time-resolved X-ray diffraction analysis (ID09 dynamic DAC).

Notebook entry point::

    from trxrd import XRDRun
    run = XRDRun("water02_scan0027.h5")
    run.summary
"""

__version__ = "0.2.0"


def __getattr__(name):
    # Lazy import keeps `import trxrd` light (no h5py/matplotlib until needed).
    if name == "XRDRun":
        from trxrd.notebook import XRDRun

        return XRDRun
    raise AttributeError(name)
