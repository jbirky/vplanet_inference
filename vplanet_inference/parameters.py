import json
import re
import subprocess
import urllib.request

import astropy.units as u
import numpy as np

__all__ = ["VplanetParameters", "get_vplanet_units", "get_vplanet_units_source", "check_units"]


# Mapping from vplanet unit strings (as they appear in `vplanet -H`) to
# astropy units.  `None` means the unit is a custom VPLanet constant that
# has no standard astropy equivalent.
_VPLANET_UNIT_MAP = {
    # --- dimensionless / no unit ---
    "nd": u.dimensionless_unscaled,
    "no unit": u.dimensionless_unscaled,
    "null": u.dimensionless_unscaled,
    "None": None,
    "No negative behavior": None,
    # --- temperature ---
    "K": u.K,
    "Kelvin": u.K,
    "Celsius": u.deg_C,
    # --- time ---
    "sec": u.s,
    "Seconds": u.s,
    "days": u.day,
    "Days": u.day,
    "year": u.yr,
    "years": u.yr,
    "Years": u.yr,
    "Myr": u.Myr,
    "Gyr": u.Gyr,
    "1 Gyr": u.Gyr,
    # --- length / distance ---
    "m": u.m,
    "meters": u.m,
    "km": u.km,
    "au": u.AU,
    "AU": u.AU,
    "Rearth": u.R_earth,
    # --- mass ---
    "kg": u.kg,
    "amu": u.u,
    "Mearth": u.M_earth,
    # --- luminosity ---
    "LSUN": u.L_sun,
    "Solar Luminosity (LSUN)": u.L_sun,
    # --- power ---
    "TW": u.TW,
    "PW": u.PW,
    "ergs": u.erg,
    # --- pressure ---
    "bar": u.bar,
    "bars": u.bar,
    "Pa": u.Pa,
    "GPa": u.GPa,
    # --- angle ---
    "deg": u.deg,
    "Deg": u.deg,
    "Degrees": u.deg,
    "rad": u.rad,
    # --- compound units ---
    "W/m^2": u.W / u.m**2,
    "W/m/K": u.W / (u.m * u.K),
    "kg/m^2": u.kg / u.m**2,
    "kg/m^3": u.kg / u.m**3,
    "kg/s": u.kg / u.s,
    "kg/m^2/s": u.kg / (u.m**2 * u.s),
    "kg*m^2/s": u.kg * u.m**2 / u.s,
    "m/s": u.m / u.s,
    "m/s^2": u.m / u.s**2,
    "km/s": u.km / u.s,
    "m^2/s": u.m**2 / u.s,
    "m^2/s^3": u.m**2 / u.s**3,
    "m/K": u.m / u.K,
    "K/s": u.K / u.s,
    "J / K kg": u.J / (u.K * u.kg),
    "AU/Gyr": u.AU / u.Gyr,
    "deg/Gyr": u.deg / u.Gyr,
    "deg/year": u.deg / u.yr,
    "deg/yr": u.deg / u.yr,
    "days/Gyr": u.day / u.Gyr,
    "days/Myr": u.day / u.Myr,
    "Mearth/Myr": u.M_earth / u.Myr,
    "TW/Gyr": u.TW / u.Gyr,
    # --- rate units ---
    "1/day": 1 / u.day,
    "/day": 1 / u.day,
    "/Day": 1 / u.day,
    "1/year": 1 / u.yr,
    "/Year": 1 / u.yr,
    "1/Gyr": 1 / u.Gyr,
    "/Gyr": 1 / u.Gyr,
    "1/day 1/log10(erg)": 1 / u.day,   # approximate; log10(erg) is dimensionless
    # --- water content ---
    "TO": None,                          # Terrestrial Oceans — no astropy equivalent
    "Terrestrial Oceans (TO)": None,
    # --- VPLanet-specific Earth constants (no astropy equivalent) ---
    "EarthDensity": None,
    "Primordial Earth": None,
    "Primordial Earth Units": None,
    "Primoridal Earth Units": None,      # typo present in vplanet source
    "Primordial Earth 26Al Number": None,
    "Initial Primordial Earth Number": None,
    "EMAGMOM": None,
    "EMAGPAUSERAD": None,
    "EMASSIC": None,
    "EMASSIC_CHI": None,
    "EMASSOC": None,
    "EMASSOC_CHI": None,
    "EPRESSWIND": None,
    "m/orbit": None,
}

# Mapping from vplanet Dimension(s) strings to representative astropy units.
# Used as a fallback for input options that have no "Custom unit" row.
_VPLANET_DIMENSION_MAP = {
    "nd":                           u.dimensionless_unscaled,
    "angle":                        u.rad,
    "angle/time":                   u.rad / u.s,
    "energy":                       u.J,
    "energy/length^2":              u.J / u.m**2,
    "energy/temperature":           u.J / u.K,
    "energy/temperature/mass":      u.J / (u.K * u.kg),
    "energy/time":                  u.W,
    "energyflux":                   u.W / u.m**2,
    "length":                       u.m,
    "length/time":                  u.m / u.s,
    "length^2/time":                u.m**2 / u.s,
    "mass":                         u.kg,
    "mass/length^3":                u.kg / u.m**3,
    "mass/time":                    u.kg / u.s,
    "pressure":                     u.Pa,
    "temperature":                  u.K,
    "temperature/length":           u.K / u.m,
    "time":                         u.s,
    "time^-1":                      1 / u.s,
    "1/time":                       1 / u.s,
    "1/time/energy":                1 / (u.s * u.J),
    "time^3*ampere^2/mass/length":  u.s**3 * u.A**2 / (u.kg * u.m),  # electrical resistivity
}

# Per-parameter unit overrides keyed by *parameter name* (not unit string).
# Use these to correct known errors in vplanet's Dimension(s) documentation.
_VPLANET_PARAM_OVERRIDES = {
    # vplanet -H labels dActViscMan as "pressure" (Pa), but the module stores
    # and expects it in m^2/s (kinematic viscosity / diffusivity units).
    "dActViscMan": u.m**2 / u.s,
}

# Cache for get_vplanet_units_source() — populated on first call.
_VPLANET_SOURCE_CACHE = None


def get_vplanet_units():
    """Run ``vplanet -H`` and return a dictionary mapping every parameter name
    to its default astropy unit.

    The extended help output (``vplanet -H``) is parsed using the structured
    RST table format produced by VPLanet.  Each parameter block looks like::

        +------...------+
        | **ParamName** |
        +=======+=======+
        | Custom unit   || K  |
        +-------+-------+

    For **output parameters**, the ``Custom unit`` row gives the natural display
    unit (e.g. ``K``, ``TW``, ``Gyr``).

    For **input options**, VPLanet often omits ``Custom unit`` and instead
    provides a ``Dimension(s)`` row (e.g. ``temperature``, ``energy/time``,
    ``pressure``).  When ``Custom unit`` is absent, the dimension string is
    used as a fallback via ``_VPLANET_DIMENSION_MAP`` to produce a canonical
    SI unit for that physical dimension.

    Parameters that are dimensionless or whose unit is a VPLanet-specific
    constant (e.g. *Primordial Earth Units*, *EMAGMOM*) are mapped to ``None``.

    Returns
    -------
    dict
        ``{parameter_name: astropy_unit_or_None}``

    Examples
    --------
    >>> units = get_vplanet_units()
    >>> units["TJumpUMan"]    # output param — Custom unit row
    Unit("K")
    >>> units["dAge"]         # input param — Custom unit row
    Unit("Gyr")
    >>> units["dTMan"]        # input param — Dimension(s) fallback
    Unit("K")
    >>> units["dActViscMan"]  # input param — Dimension(s) fallback
    Unit("m2 / s")
    """
    proc = subprocess.Popen(
        ["vplanet", "-H"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = proc.communicate()
    lines = stdout.decode("utf-8", errors="replace").split("\n")

    result = {}
    param_name = None
    pending_dim = None   # Dimension(s) value seen before any Custom unit row

    for line in lines:
        # Blank line terminates a parameter block; commit pending dimension.
        if line.strip() == "":
            if param_name is not None and param_name not in result and pending_dim is not None:
                result[param_name] = _VPLANET_DIMENSION_MAP.get(pending_dim)
            param_name = None
            pending_dim = None
            continue

        # Detect the start of a parameter block: | **Name** ... |
        name_match = re.match(r"\|\s+\*\*(\S+)\*\*", line)
        if name_match:
            # Commit any pending dimension from the previous block (no blank line between)
            if param_name is not None and param_name not in result and pending_dim is not None:
                result[param_name] = _VPLANET_DIMENSION_MAP.get(pending_dim)
            param_name = name_match.group(1)
            pending_dim = None
            continue

        if param_name is None:
            continue

        # Prefer "Custom unit" row — most specific.
        unit_match = re.match(r"\|\s+Custom unit\s+\|\|\s+(.*?)\s*\|", line)
        if unit_match:
            unit_str = unit_match.group(1).strip()
            result[param_name] = _VPLANET_UNIT_MAP.get(unit_str, None)
            pending_dim = None   # Custom unit takes precedence; discard dimension
            continue

        # Fallback: store the Dimension(s) value for later commitment.
        dim_match = re.match(r"\|\s+Dimension\(s\)\s+\|\|\s+(.*?)\s*\|", line)
        if dim_match:
            pending_dim = dim_match.group(1).strip()

    # Handle last block if file doesn't end with a blank line
    if param_name is not None and param_name not in result and pending_dim is not None:
        result[param_name] = _VPLANET_DIMENSION_MAP.get(pending_dim)

    # Apply per-parameter overrides (correct known vplanet documentation errors)
    result.update(_VPLANET_PARAM_OVERRIDES)

    return result


# --------------------------------------------------------------------------- #
# Source-based unit lookup                                                     #
# --------------------------------------------------------------------------- #

def _fetch_source_units():
    """Fetch and parse all vplanet C source files from GitHub to build a
    complete ``{parameter_name: astropy_unit}`` mapping.

    The function searches two patterns inside every ``*.c`` file under
    ``src/`` in the VirtualPlanetaryLaboratory/vplanet GitHub repository:

    * **Input options** — the ``fvInitializeOptions*`` functions set
      ``options[OPT_X].cName`` (e.g. ``"dTMan"``) and
      ``options[OPT_X].cDimension`` (e.g. ``"temperature"``).
      The dimension string is looked up in ``_VPLANET_DIMENSION_MAP``.

    * **Output parameters** — the ``fvInitializeOutput*`` functions set
      ``output[OUT_X].cName`` (e.g. ``"TMan"``) and
      ``output[OUT_X].cNeg`` (e.g. ``"K"``).
      The unit string is looked up in ``_VPLANET_UNIT_MAP``; any string not
      found there is attempted via ``astropy.units.Unit()``.

    The version tag is derived from ``vplanet.__version__`` so that the
    source matches the installed binary.  Falls back to ``"main"`` if the
    version tag does not exist on GitHub.

    Returns
    -------
    dict
        ``{parameter_name: astropy_unit_or_None}`` for every parameter found
        across all source files.
    """
    # Determine GitHub ref from installed vplanet version
    try:
        import vplanet as _vp
        version = _vp.__version__
        tags_to_try = [f"v{version}", version, "main"]
    except Exception:
        tags_to_try = ["main"]

    c_file_entries = None
    for tag in tags_to_try:
        try:
            api_url = (
                f"https://api.github.com/repos/"
                f"VirtualPlanetaryLaboratory/vplanet/contents/src?ref={tag}"
            )
            req = urllib.request.Request(
                api_url,
                headers={
                    "User-Agent": "vplanet-python-inference",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    contents = json.loads(resp.read())
                    c_file_entries = [
                        (f["name"], f["download_url"])
                        for f in contents
                        if f["name"].endswith(".c")
                    ]
                    break
        except Exception:
            continue

    if not c_file_entries:
        return {}

    # Regex patterns — all single-line; vplanet never wraps these calls
    _opt_name_re = re.compile(
        r'fvFormattedString\s*\(\s*&options\[(\w+)\]\.cName\s*,\s*"([^"]+)"'
    )
    _opt_dim_re = re.compile(
        r'fvFormattedString\s*\(\s*&options\[(\w+)\]\.cDimension\s*,\s*"([^"]+)"'
    )
    _out_name_re = re.compile(
        r'fvFormattedString\s*\(\s*&output\[(\w+)\]\.cName\s*,\s*"([^"]+)"'
    )
    _out_neg_re = re.compile(
        r'fvFormattedString\s*\(\s*&output\[(\w+)\]\.cNeg\s*,\s*"([^"]+)"'
    )

    opt_names: dict = {}   # OPT_X  → parameter name (with leading 'd')
    opt_dims:  dict = {}   # OPT_X  → cDimension string
    out_names: dict = {}   # OUT_X  → output name (no leading 'd')
    out_negs:  dict = {}   # OUT_X  → cNeg unit string

    for _fname, url in c_file_entries:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                source = resp.read().decode("utf-8", errors="replace")
        except Exception:
            continue

        for key, name in _opt_name_re.findall(source):
            opt_names[key] = name
        for key, dim in _opt_dim_re.findall(source):
            opt_dims[key] = dim
        for key, name in _out_name_re.findall(source):
            out_names[key] = name
        for key, neg in _out_neg_re.findall(source):
            out_negs[key] = neg

    # Sentinel to distinguish "not in map" from "explicitly None"
    _MISSING = object()

    def _resolve_unit(unit_str):
        """Map a cNeg string to an astropy unit."""
        val = _VPLANET_UNIT_MAP.get(unit_str, _MISSING)
        if val is not _MISSING:
            return val
        try:
            return u.Unit(unit_str)
        except Exception:
            return None

    result = {}

    # Input options — use cDimension
    for opt_key, dim_str in opt_dims.items():
        name = opt_names.get(opt_key)
        if name:
            result[name] = _VPLANET_DIMENSION_MAP.get(dim_str)

    # Output parameters — use cNeg (preferred, more specific)
    for out_key, neg_str in out_negs.items():
        name = out_names.get(out_key)
        if name:
            result[name] = _resolve_unit(neg_str)

    # Apply per-parameter overrides
    for name, unit in _VPLANET_PARAM_OVERRIDES.items():
        if name in result:
            result[name] = unit

    return result


def get_vplanet_units_source(parameter):
    """Look up the unit for a vplanet parameter by parsing the C source code
    on GitHub.

    On the first call the entire vplanet ``src/*.c`` tree is fetched from
    GitHub and cached in memory.  Subsequent calls use the in-process cache,
    so only one round-trip of network requests is ever made per interpreter
    session.

    The lookup searches two complementary patterns in the C source:

    * **Input options** (``fvInitializeOptions*`` functions): uses the
      ``cDimension`` field, which contains a qualitative dimension description
      like ``"temperature"`` or ``"energy/time"``.
      Mapped via ``_VPLANET_DIMENSION_MAP`` to a canonical SI astropy unit.

    * **Output parameters** (``fvInitializeOutput*`` functions): uses the
      ``cNeg`` field, which contains the display unit string such as ``"K"``
      or ``"TW"``.
      Mapped via ``_VPLANET_UNIT_MAP`` (or ``astropy.units.Unit()`` as a
      fallback).

    The GitHub tag is matched to the installed ``vplanet.__version__``,
    falling back to ``main``.

    Parameters
    ----------
    parameter : str
        VPLanet parameter name.  Both the option form with a leading
        ``"d"`` (e.g. ``"dTMan"``) and the output form without
        (e.g. ``"TMan"``) are accepted.  The function tries the exact name
        first, then with/without the ``"d"`` prefix.

    Returns
    -------
    astropy.units.Unit or None
        The associated unit, or ``None`` for dimensionless / VPLanet-specific
        constants.

    Raises
    ------
    ValueError
        If the parameter is not found in any source file.

    Examples
    --------
    >>> get_vplanet_units_source("TJumpUMan")   # output form
    Unit("K")
    >>> get_vplanet_units_source("dTMan")       # input option form
    Unit("K")
    >>> get_vplanet_units_source("TMan")        # output form without 'd'
    Unit("K")
    >>> get_vplanet_units_source("HflowUMan")   # heat flow output
    Unit("TW")
    """
    global _VPLANET_SOURCE_CACHE

    if _VPLANET_SOURCE_CACHE is None:
        _VPLANET_SOURCE_CACHE = _fetch_source_units()

    cache = _VPLANET_SOURCE_CACHE

    # Try exact name first (handles both "TMan" output and "dTMan" option form)
    if parameter in cache:
        return cache[parameter]

    # Try without leading 'd' (output names never have it)
    if parameter.startswith("d"):
        no_d = parameter[1:]
        if no_d in cache:
            return cache[no_d]

    # Try with leading 'd' added (option names always have it)
    with_d = "d" + parameter
    if with_d in cache:
        return cache[with_d]

    raise ValueError(
        f"Parameter {parameter!r} was not found in the vplanet source code. "
        "Check that the parameter name is correct (case-sensitive)."
    )


class VplanetParameters(object):

    def __init__(self, names, units, bounds=None, true=None, data=None, labels=None, uncertainty=None):

        # required
        self.names = names
        self.units = units

        # optional
        self.bounds = bounds
        self.data = data
        self.true = true
        self.uncertainty = uncertainty

        if (labels is None) or (labels[0] is None):
            self.labels = self.names
        else:
            self.labels = labels

        self.num = len(self.names)

        if self.units is not None:
            self.dict_units = dict(zip(self.names, self.units))
        if self.bounds is not None:
            self.dict_bounds = dict(zip(self.names, self.bounds))
        if self.true is not None:
            self.dict_true = dict(zip(self.names, self.true))
        if self.data is not None:
            self.dict_data = dict(zip(self.names, self.data))
        if self.labels is not None:
            self.dict_labels = dict(zip(self.names, self.labels))


    def set_data(self, true):

        self.true = true
        self.data = np.array([self.true, self.uncertainty]).T

        self.dict_true = dict(zip(self.names, self.true))
        self.dict_data = dict(zip(self.names, self.data))


    def get_true_units(self, param_name):

        return self.true[param_name] * self.units[param_name]


def _get_physical_unit(unit):
    """Return the underlying `~astropy.units.UnitBase` for *unit*, unwrapping
    logarithmic (e.g. ``u.dex(...)``) and negated units so that
    :py:meth:`~astropy.units.UnitBase.is_equivalent` works reliably."""
    if unit is None:
        return None
    # Logarithmic wrapper (e.g. u.dex(u.erg/u.s))
    if isinstance(unit, u.LogUnit):
        return unit.physical_unit
    # Negated / scaled unit (e.g. -u.dimensionless_unscaled) — strip the scale
    if isinstance(unit, u.CompositeUnit):
        return u.CompositeUnit(1, unit.bases, unit.powers)
    return unit


def check_units(inparams, outparams=None, source=False):
    """Check that the astropy units supplied in *inparams* and *outparams* are
    physically consistent with the default units for those parameters.

    "Consistent" means the two units describe the *same physical dimension*,
    i.e. they are interconvertible (e.g. ``u.Msun`` and ``u.M_earth`` are
    both mass).  The actual numeric scale does not matter — VPLanet performs
    unit conversions internally.

    Parameters
    ----------
    inparams : dict
        Input-parameter dict passed to :class:`VplanetModel`, of the form
        ``{"body.ParamName": astropy_unit, ...}``.
    outparams : dict, optional
        Output-parameter dict passed to :class:`VplanetModel`, of the form
        ``{"final.body.ParamName": astropy_unit, ...}``.
    source : bool, optional
        If ``False`` (default), look up units by parsing ``vplanet -H``
        runtime output via :func:`get_vplanet_units`.  If ``True``, look up
        units by parsing the vplanet C source code on GitHub via
        :func:`get_vplanet_units_source`.  The source-based lookup is more
        precise for input options (uses the exact ``cDimension`` field) and
        caches results after the first call.

    Returns
    -------
    dict
        A summary dict with three keys:

        ``"consistent"``
            List of ``(key, user_unit, vplanet_unit)`` tuples that passed.
        ``"inconsistent"``
            List of ``(key, user_unit, vplanet_unit)`` tuples that failed —
            the two units have different physical dimensions.
        ``"unknown"``
            List of ``(key, user_unit)`` tuples for parameters that were not
            found in the lookup source, or where either unit is ``None`` /
            unmapped.

    Raises
    ------
    ValueError
        If any inconsistencies are found, a ``ValueError`` is raised listing
        every problematic parameter so all issues are visible at once.

    Examples
    --------
    >>> import astropy.units as u
    >>> inparams  = {"star.dMass": u.Msun, "star.dAge": u.yr}
    >>> outparams = {"final.star.Luminosity": u.Lsun}
    >>> check_units(inparams, outparams)              # uses vplanet -H
    >>> check_units(inparams, outparams, source=True) # uses GitHub C source
    """
    # Build the reference unit lookup depending on the chosen backend.
    if source:
        # Ensure the source cache is populated once, then use it as a dict.
        global _VPLANET_SOURCE_CACHE
        if _VPLANET_SOURCE_CACHE is None:
            _VPLANET_SOURCE_CACHE = _fetch_source_units()
        _lookup = _VPLANET_SOURCE_CACHE
        _not_found_label = "(not in source)"
    else:
        _lookup = get_vplanet_units()
        _not_found_label = "(not in vplanet -H)"

    consistent   = []   # (key, user_unit, vp_unit)
    inconsistent = []   # (key, user_unit, vp_unit)
    # _unknown_ext tracks (key, user_unit, vp_unit_or_None, reason) internally
    # so the report can give a precise label per entry.
    # reason values:
    #   "not_found"  — param absent from the lookup entirely
    #   "skipped"    — param found but user_unit=None (user opted out of check)
    #   "vp_none"    — param found but vplanet has no standard unit for it
    _unknown_ext = []

    # Combine both dicts
    all_params = {}
    if inparams is not None:
        all_params.update(inparams)
    if outparams is not None:
        all_params.update(outparams)

    for key, user_unit in all_params.items():
        # The VPLanet parameter name is always the last component of the key
        # e.g. "star.dMass" → "dMass",  "final.star.Luminosity" → "Luminosity"
        param_name = key.split(".")[-1]

        # Resolve vplanet unit from the chosen backend.
        if source:
            # Source dict uses exact names; try with/without 'd' prefix to
            # match both input option names and output parameter names.
            _candidates = [param_name]
            if param_name.startswith("d"):
                _candidates.append(param_name[1:])
            else:
                _candidates.append("d" + param_name)

            _found_key = next((c for c in _candidates if c in _lookup), None)
            if _found_key is None:
                _unknown_ext.append((key, user_unit, None, "not_found"))
                continue
            vp_unit = _lookup[_found_key]
        else:
            if param_name not in _lookup:
                _unknown_ext.append((key, user_unit, None, "not_found"))
                continue
            vp_unit = _lookup[param_name]

        # User opted out of unit checking for this parameter (e.g. dimensionless).
        if user_unit is None:
            _unknown_ext.append((key, user_unit, vp_unit, "skipped"))
            continue

        # VPLanet maps this to a custom/non-standard constant — can't compare.
        if vp_unit is None:
            _unknown_ext.append((key, user_unit, vp_unit, "vp_none"))
            continue

        user_phys = _get_physical_unit(user_unit)
        vp_phys   = _get_physical_unit(vp_unit)

        if user_phys is None or vp_phys is None:
            _unknown_ext.append((key, user_unit, vp_unit, "vp_none"))
            continue

        if user_phys.is_equivalent(vp_phys):
            consistent.append((key, user_unit, vp_unit))
        else:
            inconsistent.append((key, user_unit, vp_unit))

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #
    all_keys = (
        [k for k, *_ in consistent]
        + [k for k, *_ in inconsistent]
        + [k for k, *_ in _unknown_ext]
    )
    col_w = max((len(k) for k in all_keys), default=20)

    print(f"\n{'Parameter':<{col_w}}  {'User unit':<20}  {'VPLanet unit':<20}  Status")
    print("-" * (col_w + 64))

    for key, u_unit, vp_unit in consistent:
        print(f"{key:<{col_w}}  {str(u_unit):<20}  {str(vp_unit):<20}  OK")

    for key, u_unit, vp_unit in inconsistent:
        print(f"{key:<{col_w}}  {str(u_unit):<20}  {str(vp_unit):<20}  INCONSISTENT ✗")

    for key, u_unit, vp_unit, reason in _unknown_ext:
        if reason == "not_found":
            vp_label = _not_found_label
            status   = "unknown"
        elif reason == "skipped":
            # Show what vplanet expects so the user can decide whether to set a unit.
            vp_label = str(vp_unit) if vp_unit is not None else "(dimensionless)"
            status   = "skipped"
        else:  # vp_none
            vp_label = "(no standard unit)"
            status   = "skipped"
        print(f"{key:<{col_w}}  {str(u_unit):<20}  {vp_label:<20}  {status}")

    print()

    if inconsistent:
        lines = ["The following parameters have incompatible units:\n"]
        for key, u_unit, vp_unit in inconsistent:
            lines.append(
                f"  {key}: user supplied {u_unit!r}, "
                f"vplanet default unit has dimension '{vp_unit.physical_type}' "
                f"(e.g. {vp_unit})"
            )
        raise ValueError("\n".join(lines))

    # Return value: keep unknown as (key, user_unit) tuples for backward compat.
    unknown = [(key, u_unit) for key, u_unit, _, _reason in _unknown_ext]
    return {"consistent": consistent, "inconsistent": inconsistent, "unknown": unknown}