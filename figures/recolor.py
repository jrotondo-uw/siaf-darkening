"""Reserve blue/orange for the SIAF components paper-wide.

blue  #3A6BAF = area loss      (matches f02/f04/f06 COL_EXT)
orange#D95F1A = darkening      (matches f02/f04/f06 COL_DRK)
red   #b5000a = observations   (matches f06 COL_OBS_EDGE)

Everything that previously borrowed the Okabe-Ito blue/orange for a
different categorical variable gets its own hue.
"""
import shutil
import subprocess
import sys

AREA, DRK, OBS = "#3A6BAF", "#D95F1A", "#b5000a"
MODEL_NEUTRAL = "#444444"   # f05: CMIP6 family means
TEAL = "#009E73"            # f08 common-variance, f09 APP-x
PURPLE = "#6A3FA5"          # f09 CLARA-A3

edits = {
    "f03.py": [("'Area loss','#0072B2'", "'Area loss','" + AREA + "'"),
               ("'Darkening','#D55E00'", "'Darkening','" + DRK + "'")],
    "s01.py": [("'Area loss','#0072B2'", "'Area loss','" + AREA + "'"),
               ("'Darkening','#D55E00'", "'Darkening','" + DRK + "'")],
    "f05.py": [("ecolor='#0072B2'", "ecolor='" + MODEL_NEUTRAL + "'"),
               ("c='#0072B2'", "c='" + MODEL_NEUTRAL + "'"),
               ("color='#0072B2'", "color='" + MODEL_NEUTRAL + "'"),
               ("color='#D55E00'", "color='" + OBS + "'")],
    "f08.py": [("color='#0072B2',label='Common variance'",
                "color='" + TEAL + "',label='Common variance'"),
               ("color='#D55E00'", "color='" + OBS + "'")],
    "f09.py": [("('CLARA-A3','#0072B2'", "('CLARA-A3','" + PURPLE + "'"),
               ("('APP-x','#D55E00'", "('APP-x','" + TEAL + "'")],
}

for fn, subs in edits.items():
    shutil.copy2(fn, fn + ".precolor_bak")
    t = open(fn).read()
    for old, new in subs:
        n = t.count(old)
        if n == 0:
            print("  !! NOT FOUND in %s: %r" % (fn, old))
            sys.exit(1)
        t = t.replace(old, new)
        print("  %-8s %-46s -> %-24s x%d" % (fn, old[:46], new[:24], n))
    open(fn, "w").write(t)

print("\nremaining #0072B2 / #D55E00 in these files:")
r = subprocess.run(["grep", "-n", "0072B2\\|D55E00"] + list(edits),
                   capture_output=True, text=True)
print(r.stdout.rstrip() or "  none")
