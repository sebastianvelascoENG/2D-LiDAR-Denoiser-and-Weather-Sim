"""weather - the per-effect weather switches (rain / fog / dust) and run labels.

Pure Python (no numpy, no ROS), so launch files and shell scripts can use it.

Each effect is enabled and disabled on its own:

    rain  calm | light | heavy          (0 / 6 / 25 mm/h, rain_models.CONDITION_RATES)
    fog   off  | light | heavy          (visibility 100 / 50 m; see FOG_VISIBILITY_M)
    dust  off  | light | heavy          (visibility 20 / 5 m, inside the dust clouds)

A run is labelled by the effects that are on, joined with '+':

    rain only          'calm' | 'light' | 'heavy'     (unchanged from rain-only sweeps,
                                                       so old CSVs stay comparable)
    anything else      'fog_heavy', 'rain_light+dust_heavy', 'fog_light+dust_light', ...

`calm` with fog and dust off is the clear-weather baseline. The labels contain
no commas or slashes, so they are safe in CSV cells and file names.

CLI (used by run_sweep.sh):
    python3 -m thesis_rain_sim.weather parse rain_heavy+fog_light   ->  heavy light off
    python3 -m thesis_rain_sim.weather label heavy light off         ->  rain_heavy+fog_light
"""
import sys

RAIN_LEVELS = ('calm', 'light', 'heavy')            # rates: rain_models.CONDITION_RATES
FOG_VISIBILITY_M = {'off': None, 'light': 100.0, 'heavy': 50.0}
DUST_VISIBILITY_M = {'off': None, 'light': 20.0, 'heavy': 5.0}
FOG_LEVELS = tuple(FOG_VISIBILITY_M)
DUST_LEVELS = tuple(DUST_VISIBILITY_M)


def _check(kind, level, allowed):
    level = str(level).strip().lower()
    if level not in allowed:
        raise ValueError(f"{kind} must be one of {' | '.join(allowed)}, got '{level}'")
    return level


def resolve_rain(rain='', condition=''):
    """The rain level from `rain`, or from the deprecated alias `condition`.

    Either may be empty. Both set to different levels is an error.
    """
    rain = str(rain or '').strip().lower()
    condition = str(condition or '').strip().lower()
    if rain and condition and rain != condition:
        raise ValueError(f"rain:='{rain}' and condition:='{condition}' disagree; condition is a "
                         'deprecated alias for rain, set only rain')
    return _check('rain', rain or condition or 'calm', RAIN_LEVELS)


def visibility_for(kind, level, explicit_m=-1.0):
    """Visibility [m] of fog or dust, or None when the effect is off.

    `explicit_m` > 0 replaces the level's preset (it does not switch an 'off' effect on).
    """
    table = FOG_VISIBILITY_M if kind == 'fog' else DUST_VISIBILITY_M
    level = _check(kind, level, tuple(table))
    if table[level] is None:
        return None
    explicit_m = float(explicit_m)
    return explicit_m if explicit_m > 0.0 else table[level]


def weather_label(rain='calm', fog='off', dust='off'):
    rain = _check('rain', rain, RAIN_LEVELS)
    fog = _check('fog', fog, FOG_LEVELS)
    dust = _check('dust', dust, DUST_LEVELS)
    if fog == 'off' and dust == 'off':
        return rain
    parts = [] if rain == 'calm' else [f'rain_{rain}']
    parts += [] if fog == 'off' else [f'fog_{fog}']
    parts += [] if dust == 'off' else [f'dust_{dust}']
    return '+'.join(parts)


def parse_weather_label(label):
    """Inverse of weather_label: (rain, fog, dust). Bare calm/light/heavy is rain."""
    rain, fog, dust = 'calm', 'off', 'off'
    seen = set()
    for part in str(label).strip().lower().split('+'):
        if part in RAIN_LEVELS:
            kind, level = 'rain', part
        elif '_' in part:
            kind, level = part.split('_', 1)
        else:
            kind, level = part, ''
        if kind not in ('rain', 'fog', 'dust'):
            raise ValueError(f"bad weather label part '{part}' in '{label}' (expected e.g. "
                             "calm, heavy, rain_light, fog_heavy, dust_light)")
        if kind in seen:
            raise ValueError(f"'{label}' sets {kind} twice")
        seen.add(kind)
        if kind == 'rain':
            rain = _check('rain', level, RAIN_LEVELS)
        elif kind == 'fog':
            fog = _check('fog', level, FOG_LEVELS)
        else:
            dust = _check('dust', level, DUST_LEVELS)
    return rain, fog, dust


def main(argv):
    try:
        if len(argv) == 2 and argv[0] == 'parse':
            print(*parse_weather_label(argv[1]))
            return 0
        if len(argv) == 4 and argv[0] == 'label':
            print(weather_label(*argv[1:]))
            return 0
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    print(__doc__.split('CLI (used by run_sweep.sh):')[1], file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
