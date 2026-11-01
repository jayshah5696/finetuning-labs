"""Render measured scores and a few exact error examples without external dependencies."""

from html import escape

NAMES = {
    "230m_base": "230M base",
    "230m_lora": "230M + LoRA",
    "350m_base": "350M base",
    "350m_lora": "350M + LoRA",
    "luna6": "Luna 6",
}


def render(report, assessments, generations, bundle):
    summaries, modes, bars, exceptions = [], [], [], []
    for name, label in NAMES.items():
        arm = report["arms"][name]
        metrics = arm["metrics"]
        total = metrics["all_141"]["pooled"]
        old, new = metrics["original_81"]["pooled"], metrics["expansion_60"]["pooled"]
        rate = total["verified_pass_rate_pct"]
        critical = total["critical_error_rate_pct"]
        if total["semantic_reviewed"] != total["scheduled"]:
            exceptions.append(
                f"{label}: {total['semantic_reviewed']}/{total['scheduled']} outputs received "
                f"semantic review; {total['counts']['truncated']} truncated outputs, "
                f"{total['counts']['generation_error']} generation errors and "
                f"{total['counts']['judge_error']} judge errors."
            )
        summaries.append(
            f"<tr><th scope='row'>{label}</th><td>{total['counts']['pass']}/141 "
            f"({rate:.1f}%)</td><td>{old['counts']['pass']}/81</td>"
            f"<td>{new['counts']['pass']}/60</td><td>{total['critical_error_count']}/141 "
            f"({critical:.1f}%)</td></tr>"
        )
        modes.append(
            "<tr><th scope='row'>"
            + label
            + "</th>"
            + "".join(
                f"<td>{m['counts']['pass']}/{m['scheduled']} "
                f"({m['verified_pass_rate_pct']:.1f}%)</td>"
                for m in metrics["all_141"]["per_mode"].values()
            )
            + "</tr>"
        )
        bars.append(
            f"<div class='chartrow'><span>{label}</span><div class='track'>"
            f"<div class='passbar' style='width:{rate}%'>&nbsp;</div></div>"
            f"<strong>{rate:.1f}%</strong><div class='track'>"
            f"<div class='criticalbar' style='width:{critical}%'>&nbsp;</div></div>"
            f"<strong>{critical:.1f}%</strong></div>"
        )
    by_id = {c.id: c for c in bundle.cases}
    examples = []
    for name in ["230m_lora", "350m_lora", "luna6"]:
        candidates = [r for r in assessments[name].values() if r["critical_error"]]
        candidates.sort(key=lambda r: len(by_id[r["case_id"]].input))
        for row in candidates[:2]:
            case = by_id[row["case_id"]]
            text = generations[name][case.id]["text"]
            critique = row["judgment"]["critique"]
            examples.append(
                f"<details><summary>{NAMES[name]} · {case.id} · {case.mode}</summary>"
                f"<h4>Input</h4><pre>{escape(case.input)}</pre>"
                f"<h4>Model output</h4><pre>{escape(text)}</pre>"
                f"<h4>Sol review</h4><p>{escape(critique)}</p></details>"
            )
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>English rewriter evaluation, 1 November 2026</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#000;color:#f5f7fa}
body{max-width:1100px;margin:0 auto;padding:28px 20px;font-size:15px;line-height:1.55}
h1{font-size:25px;line-height:1.2}h2{font-size:19px;margin-top:30px}h4{margin-bottom:5px}
p{max-width:94ch}a{color:#60a5fa}.muted{color:#a9adb5}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #2a2a2a}
thead{background:#111}caption{text-align:left;color:#a9adb5;margin-bottom:8px}th[scope=row]{font-weight:500}
.chartrow{display:grid;grid-template-columns:135px 1fr 58px 1fr 58px;gap:12px;align-items:center;margin:15px 0}
.track{height:16px;background:#181818}.passbar,.criticalbar{height:16px}.passbar{background:#60a5fa}.criticalbar{background:#fb7185}
.axis{text-align:center;color:#a9adb5}.axishead{display:grid;grid-template-columns:135px 1fr 58px 1fr 58px;gap:12px}
details{border-top:1px solid #2a2a2a;padding:13px 0}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111;padding:12px;font:13px/1.6 ui-monospace,monospace}
@media(max-width:650px){body{padding:18px 12px}.chartrow,.axishead{grid-template-columns:95px 1fr 44px 1fr 44px;gap:6px;font-size:12px}.chartrow strong{font-size:12px}table{min-width:650px}}
@media print{body{max-width:none}details{break-inside:avoid}summary{font-weight:600}}
</style></head><body><main><h1>English rewriter evaluation</h1>
<p>All five models ran on the same 141 English development-validation cases. The original 81 cases and the 60 new cases are reported separately. These are model-reviewed diagnostic results, not an unseen final test or human gold labels.</p>
<h2>Overall comparison</h2><p class="muted">Both axes run from 0 to 100%. Higher pass rate is better. Lower critical-error rate is better.</p>
<div class="axishead"><span></span><span class="axis">Strict pass</span><span></span><span class="axis">Critical error</span><span></span></div>
"""
        + "".join(bars)
        + """
<div class="scroll"><table><caption>One generation per model and case; complete outputs reviewed once</caption>
<thead><tr><th scope="col">Model</th><th scope="col">All 141</th><th scope="col">Original 81</th><th scope="col">New 60</th><th scope="col">Critical errors</th></tr></thead><tbody>
"""
        + "".join(summaries)
        + """</tbody></table></div>
<p>A strict pass requires the deterministic case rules and all five Sol criteria to pass. Critical errors count cases where Sol identified changed, omitted or invented meaning. Exact identity, file tags and list formatting can cause a strict failure without a critical meaning error.</p>
<p>Truncated or erroneous outputs count against the strict pass rate. Their missing meaning reviews do not establish absence of critical errors.</p>
"""
        + ("<p>" + " ".join(exceptions) + "</p>" if exceptions else "")
        + """
<h2>Pass rate by mode</h2><div class="scroll"><table>
<thead><tr><th scope="col">Model</th><th scope="col">/slack</th><th scope="col">/transcribe</th><th scope="col">/prompt</th><th scope="col">/polish</th></tr></thead><tbody>
"""
        + "".join(modes)
        + """</tbody></table></div>
<h2>Inspect critical errors</h2><p>These are the two shortest input cases with flagged critical errors from each tuned model and Luna. They are examples, not an estimate of the error distribution. The complete comparison is in <a href="case-results.csv">case-results.csv</a>.</p>
"""
        + "".join(examples)
        + """
<h2>Run conditions</h2><p>Liquid generation used Modal A10 GPUs, BF16 base weights, greedy decoding, a 4,096-token context limit and a 2,048-token generation limit. Both size comparisons use matched base and adapter settings. Luna 6 used the same behavioral prompts with reasoning disabled and a 2,048-token output limit. GPT-6.1 Sol judged all models with the same rubric and high reasoning.</p>
<p>Dataset revision: <code>36a75c009d008415fad557d49005ba4e5c73f4d7</code>.<br>230M adapter: <code>664f541045c0148e7bec66b267b35238716d4688</code>.<br>350M adapter: <code>b660a2046092b07e46061ee25b738da3f4ef0688</code>.</p>
<p>Exact settings, coverage, uncertainty and error counts, per-family results and saved-call usage are in <a href="comparison.json">comparison.json</a>. The <a href="README.md">run notes</a> explain the method and reproduction commands.</p>
</main></body></html>"""
    )
