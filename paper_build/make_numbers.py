# -*- coding: utf-8 -*-
"""results.json -> numbers.tex : LaTeX macros for every number quoted in the JBHI manuscript.
Run build_results.py first. Multi-seed macros fall back to the seed-42 confirmation values
(and \\seedpending expands to a visible marker) until outputs/e1_multiseed_corrected is complete."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "results.json"))
L = []
def m(name, val, fmt="{:.3f}"):
    if isinstance(val, str):
        s = val
    else:
        s = fmt.format(val)
    L.append(f"\\newcommand{{\\{name}}}{{{s}}}")

# ---- headline: prefer multi-seed if all 4 configs have >= 3 seeds on all folds
def headline(tag):
    ms = R.get(f"multiseed_{tag}")
    need_folds = 6 if tag == "gl" else 10
    ok = ms and all(v.get("seeds", 0) >= 3 and v.get("folds", 0) >= need_folds for v in ms.values()) and len(ms) >= 4
    if ok:
        return {k: dict(r2=v["R2"], sd=v["R2_sd_seeds"], ml=v["ML"], ap=v["AP"], vert=v.get("VERT", float("nan"))) for k, v in ms.items()}, True
    h = R[f"headline_{tag}"]
    return {k: dict(r2=v["r2"], sd=v["sd"], ml=v["ml"], ap=v["ap"], vert=float("nan")) for k, v in h.items()}, False

names = {"ICF-hpo": "icf", "ICF-grid128": "icfg", "CNN-LSTM-hpo": "cnnlstm", "CNN-hpo": "cnn"}
allseeded = True
HEAD = {}
for tag in ["gl", "ph"]:
    H, seeded = headline(tag)
    HEAD[tag] = H
    allseeded &= seeded
    for cfg, short in names.items():
        if cfg in H:
            m(f"h{tag}{short}", H[cfg]["r2"]); m(f"h{tag}{short}sd", H[cfg]["sd"]); m(f"h{tag}{short}ml", H[cfg]["ml"]); m(f"h{tag}{short}ap", H[cfg]["ap"])
            if H[cfg]["vert"] == H[cfg]["vert"]: m(f"h{tag}{short}vert", H[cfg]["vert"])
    for k, v in R[f"headline_{tag}_pairs"].items():
        a, b = k.split(" vs ")
        key = f"p{tag}{names[a]}{names[b]}"
        m(key, v["p"]); m(key + "d", v["delta"], "{:+.3f}")
m("seedpending", "" if allseeded else "\\textcolor{red}{[IN PROGRESS: multi-seed run; single-seed values shown]}")

# ---- trial-level split on the corrected pipeline, same configs; inflation = trial split - LOSO (headline value)
tsc = R.get("trial_split_corrected", {})
tspend = False
for tag in ["gl", "ph"]:
    for cfg, short in names.items():
        if short == "icfg":
            continue
        v = tsc.get(tag, {}).get(cfg)
        if v is None or cfg not in HEAD[tag]:
            m(f"ts{tag}{short}", "\\todo{running}"); m(f"infl{tag}{short}", "\\todo{running}"); tspend = True
        else:
            m(f"ts{tag}{short}", v["r2"]); m(f"infl{tag}{short}", v["r2"] - HEAD[tag][cfg]["r2"], "{:+.3f}")
m("trialpending", "" if not tspend else "\\textcolor{red}{[IN PROGRESS: trial-split runs on the current pipeline; missing cells show as running]}")
m("seedsd", "SD across three seeds of the fold mean" if allseeded else "SD across held-out subjects (seed 42; multi-seed in progress)")

# ---- ablation (2x2), ICF paper HP
for k, v in R["ablation"].items():
    ds, arm = k.split("_")
    for ch in ["R2", "ML", "VERT", "AP", "PX", "PZ"]:
        m(f"abl{ds.lower()}{arm}{'r' if ch == 'R2' else ch.lower()}", v[ch])
for k, v in R.get("ablation_models", {}).items():          # \ablcnnlstmglbaselineap etc. (CNN-LSTM, CNN at tuned configs)
    mdl, ds, arm = k.split("_")
    for ch in ["R2", "ML", "VERT", "AP", "PX", "PZ"]:
        m(f"abl{mdl.replace('-', '')}{ds.lower()}{arm}{'r' if ch == 'R2' else ch.lower()}", v[ch], "{:.3f}")
if "ablation_models" in R:                                  # co-rotation-only AP gain on GL / ML change on PH, per model
    for mdl in ["cnn-lstm", "cnn"]:
        a = R["ablation_models"]; s = mdl.replace("-", "")
        m(f"abl{s}glapgain", a[f"{mdl}_GL_corot"]["AP"] - a[f"{mdl}_GL_baseline"]["AP"], "{:.2f}")
        m(f"abl{s}phmldrop", a[f"{mdl}_PH_corot"]["ML"] - a[f"{mdl}_PH_baseline"]["ML"], "{:+.2f}")
    m("ablphmldrop", R["ablation"]["PH_corot"]["ML"] - R["ablation"]["PH_baseline"]["ML"], "{:+.2f}")
for tag in ["gl", "ph"]:
    p = R[f"ablation_paired_{tag}"]
    m(f"ablpaired{tag}delta", p["delta_r2"], "{:+.3f}"); m(f"ablpaired{tag}wins", f"{p['wins']}/{p['n']}"); m(f"ablpaired{tag}p", p["p"])
    m(f"ablpaired{tag}apbefore", p["ap_before"]); m(f"ablpaired{tag}apafter", p["ap_after"])

# ---- m6 paper HP corrected
for k, v in R["m6_paperHP_corrected"].items():
    ds, model = k.split("_", 1); s = {"ICF": "icf", "CNN-LSTM": "cnnlstm", "CNN": "cnn"}[model]
    for ch in ["R2", "ML", "VERT", "AP", "PX", "PZ"]:
        m(f"msix{ds.lower()}{s}{'r' if ch == 'R2' else ch.lower()}", v[ch])

# ---- cross-lab corrected
def clip(x): return "$<-10$" if x < -10 else f"{x:.2f}"
for k, v in R["crosslab_corrected"].items():
    d, model = k.split("_", 1); s = {"ICF": "icf", "CNN-LSTM": "cnnlstm", "CNN": "cnn"}[model]
    dk = "glph" if "GL -> PH" in d else "phgl"
    m(f"xl{dk}{s}", clip(v["R2"])); m(f"xl{dk}{s}vert", clip(v["VERT"])); m(f"xl{dk}{s}ap", clip(v["AP"]))
# submitted-pipeline cross-lab (Feb runs; equal to the submitted paper's Table)
sub = {"glph": {"icf": -0.447, "cnnlstm": -0.784, "cnn": -0.720}, "phgl": {"icf": -1.397, "cnnlstm": -1.489, "cnn": -1.099}}
for dk, dd in sub.items():
    for s, v in dd.items(): m(f"xlsub{dk}{s}", v, "{:.2f}")

# ---- combined + matched
for model, v in R["combined_corrected"].items():
    s = {"ICF": "icf", "CNN-LSTM": "cnnlstm", "CNN": "cnn"}[model]
    m(f"comb{s}gl", v["GL_R2"]); m(f"comb{s}glsd", v["GL_sd"]); m(f"comb{s}ph", v["PH_R2"]); m(f"comb{s}phsd", v["PH_sd"])
for r in R["matched_corrected"]:
    s = {"ICF": "icf", "CNN-LSTM": "cnnlstm", "CNN": "cnn"}[r["model"]]; t = r["dataset"].lower()
    m(f"mat{s}{t}single", r["single_matched"]); m(f"mat{s}{t}comb", r["combined"]); m(f"mat{s}{t}delta", r["delta"], "{:+.3f}"); m(f"mat{s}{t}wins", f"{r['wins']}/{r['n']}"); m(f"mat{s}{t}p", r["p"])
for r in R["matched_submitted"]:
    t = r["dataset"].lower()
    m(f"matsub{t}single", r["single_matched"]); m(f"matsub{t}comb", r["combined"]); m(f"matsub{t}delta", r["delta"], "{:+.3f}"); m(f"matsub{t}wins", f"{r['wins']}/{r['n']}")

# ---- ridge, recover, trial split
for ds, v in R["ridge"].items():
    m(f"ridge{ds.lower()}", v["R2"], "{:.2f}"); m(f"ridge{ds.lower()}sd", v["R2_sd"], "{:.2f}"); m(f"ridge{ds.lower()}vert", v["VERT"], "{:.2f}")
for ds, v in R["recover"].items():
    t = ds.lower()
    m(f"rec{t}all", v["R2_all"]); m(f"rec{t}base", v["R2_base"]); m(f"rec{t}nall", str(v["n_all"])); m(f"rec{t}subj", v["re_included"].replace("_", "\\_")); m(f"rec{t}vals", v["R2_re_included"]); m(f"rec{t}excl", v["R2_all_excluding_reincluded"])
ts = {r["dataset"]: r["test_R2"] for r in R["trial_split"]}
m("trialgl", ts["GL"]); m("trialph", ts["PH"])
# inflation vs the submitted-pipeline LOSO ICF (0.655 / 0.561, the numbers in the frozen Feb runs)
m("lososubgl", 0.655); m("lososubph", 0.561)
m("inflgl", ts["GL"] - 0.655, "{:+.3f}"); m("inflph", ts["PH"] - 0.561, "{:+.3f}")
# channel hierarchy
for r in R["hierarchy"]:
    m(f"hier{r['dataset'].lower()}{r['channel']}sub", r["submitted"]); m(f"hier{r['dataset'].lower()}{r['channel']}cor", r["corrected"])

# ---- AP gain from co-rotation (abstract), masked rescoring, biomechanical units
ab = R["ablation"]; m("ablglapgain", ab["GL_corot"]["AP"] - ab["GL_baseline"]["AP"], "{:.2f}")
if "masked" in R:
    for t in ("gl", "ph"):
        g = [v["masked"] - v["orig"] for k, v in R["masked"].items() if k.startswith(f"{t}_loso_")]
        m(f"mskgain{t}lo", min(g), "{:.2f}"); m(f"mskgain{t}hi", max(g), "{:.2f}")
        lo_ = {k[len(f"{t}_loso_"):]: v["masked"] for k, v in R["masked"].items() if k.startswith(f"{t}_loso_")}
        tr_ = {k[len(f"{t}_trial_split_"):]: v["masked"] for k, v in R["masked"].items() if k.startswith(f"{t}_trial_split_")}
        inf = [tr_[c] - lo_[c] for c in lo_ if c in tr_]
        m(f"mskinfl{t}lo", min(inf), "{:+.2f}"); m(f"mskinfl{t}hi", max(inf), "{:+.2f}")
        for cfg, short in names.items():
            if cfg in lo_: m(f"msk{t}{short}", lo_[cfg])
        lf = [v["low_load_frac"] for k, v in R["masked"].items() if k.startswith(f"{t}_loso_")]
        m(f"lowload{t}", 100 * sum(lf) / len(lf), "{:.0f}")
        va = {k[len(f"{t}_loso_"):]: (v["vert_all"], v["vert_loaded"]) for k, v in R["masked"].items() if k.startswith(f"{t}_loso_")}
        m(f"vertall{t}cnnlstm", va["CNN-LSTM-hpo"][0], "{:.1f}"); m(f"vertloaded{t}cnnlstm", va["CNN-LSTM-hpo"][1], "{:.1f}")
if "units" in R:
    for t in ("gl", "ph"):
        uu = [v for k, v in R["units"].items() if k.startswith(f"{t}_loso_")]
        for q in ("vert", "peak", "cop"):
            m(f"u{q}{t}lo", min(x[q] for x in uu), "{:.1f}"); m(f"u{q}{t}hi", max(x[q] for x in uu), "{:.1f}")
        uu = [v for k, v in R["units"].items() if k.startswith(f"{t}_trial_split_")]
        m(f"utsvert{t}lo", min(x["vert"] for x in uu), "{:.1f}"); m(f"utsvert{t}hi", max(x["vert"] for x in uu), "{:.1f}")

(HERE / "numbers.tex").write_text("% AUTO-GENERATED by make_numbers.py from results.json. Do not edit.\n" + "\n".join(L) + "\n", encoding="utf-8")
print("wrote numbers.tex with", len(L), "macros; multi-seed complete:", allseeded)
