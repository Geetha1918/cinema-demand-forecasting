"""Fills reports/TECHNICAL_REPORT.md tables from reports/metrics.json (so numbers are never hand-typed)."""
import json
import pandas as pd

m = json.load(open("reports/metrics.json"))
imp = pd.read_csv("reports/feature_importance.csv")


def tbl(d, cols=("MAE_tickets", "RMSE_tickets", "WAPE", "MAE_occupancy", "staffing_band_accuracy")):
    names = {"MAE_tickets": "MAE (tickets)", "RMSE_tickets": "RMSE (tickets)", "WAPE": "WAPE",
             "MAE_occupancy": "MAE (occupancy)", "staffing_band_accuracy": "Staffing-band acc."}
    out = "| Model | " + " | ".join(names[c] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for k, v in d.items():
        out += f"| {k} | " + " | ".join(f"{v[c]:.3f}" if c != "MAE_tickets" and c != "RMSE_tickets" else f"{v[c]:.1f}" for c in cols) + " |\n"
    return out


def grp():
    b, best = m["strongest_baseline"], m["best_model"]
    out = f"| Test group | n | Baseline MAE | {best} MAE | Baseline WAPE | {best} WAPE | {best} band-acc. |\n|---|---|---|---|---|---|---|\n"
    for g, v in m["test_by_group"].items():
        out += (f"| {g} | {v['baseline']['n']} | {v['baseline']['MAE_tickets']:.1f} | {v[best]['MAE_tickets']:.1f} | "
                f"{v['baseline']['WAPE']:.3f} | {v[best]['WAPE']:.3f} | {v[best]['staffing_band_accuracy']:.3f} |\n")
    return out


s, dq, c = m["split"], m["data_quality"], m["interval_coverage_test_80pct"]
best, base = m["best_model"], m["strongest_baseline"]
tex = open("reports/REPORT_TEMPLATE.md").read()
rep = {"{{SPLIT_TRAIN}}": f"{s['train'][0][:10]} to {s['train'][1][:10]} ({s['train'][2]} shows)",
       "{{SPLIT_VAL}}": f"{s['validation'][0][:10]} to {s['validation'][1][:10]} ({s['validation'][2]} shows)",
       "{{SPLIT_TEST}}": f"{s['test'][0][:10]} to {s['test'][1][:10]} ({s['test'][2]} shows)",
       "{{DQ}}": "\n".join(f"- {k.replace('_', ' ')}: **{v}**" for k, v in dq.items()),
       "{{VAL_TABLE}}": tbl(m["metrics"]["val"]), "{{TEST_TABLE}}": tbl(m["metrics"]["test"]),
       "{{GROUP_TABLE}}": grp(), "{{BEST}}": best, "{{BASE}}": base,
       "{{HP}}": json.dumps(m["chosen_hyperparameters"]),
       "{{COVER}}": f"cold-start {c['cold']:.0%}, low-data {c['low']:.0%}, established {c['est']:.0%}",
       "{{IMP}}": "| Feature | MAE increase when shuffled |\n|---|---|\n" + "\n".join(
           f"| {r.feature} | {r.mae_increase:.4f} |" for r in imp.head(8).itertuples()),
       "{{TW}}": f"{m['metrics']['test'][best]['WAPE']:.3f}", "{{BW}}": f"{m['metrics']['test'][base]['WAPE']:.3f}",
       "{{TM}}": f"{m['metrics']['test'][best]['MAE_tickets']:.1f}", "{{BM}}": f"{m['metrics']['test'][base]['MAE_tickets']:.1f}"}
for k, v in rep.items():
    tex = tex.replace(k, v)
open("reports/TECHNICAL_REPORT.md", "w").write(tex)
print("report written")
