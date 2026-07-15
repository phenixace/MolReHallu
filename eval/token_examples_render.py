"""Render token_examples.json -> a flat CSV (edit in Excel/pandas) + a self-contained HTML
heatmap (open in a browser, no assets). Per-token background = normalized gradient saliency
(toggle to answer->token attention); bottom-border color = token category. Answer region shown
plain (its self-saliency is not meaningful). Run: python eval/token_examples_render.py
"""
import csv, html, json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "paper", "ChemR_Hallucination_ICLR", "data", "token_examples")
CATCOL = {"SMILES_frag": "#2c7fb8", "FG_word": "#e6550d", "position_digit": "#31a354"}
REGION_LABEL = {"input": "INPUT (task prompt)", "trace": "REASONING TRACE (&lt;think&gt;)", "answer": "ANSWER"}


def main():
    ex = json.load(open(os.path.join(OUT, "token_examples.json")))
    order = {"cap2mol/er0": 0, "cap2mol/erpos": 1, "retrosynthesis/er0": 2, "mol2cap/er0": 3}
    ex.sort(key=lambda x: (x["model"], order.get(x.get("spec", ""), 9)))

    # ---- CSV (flat, editable) ----
    with open(os.path.join(OUT, "token_examples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "spec", "task", "id", "er", "exact",
                    "token_index", "region", "cat", "text", "sal", "sal_norm", "attn", "attn_norm"])
        for e in ex:
            for t in e["tokens"]:
                w.writerow([e["model"], e.get("spec", ""), e["task"], e["id"], e["er"], e["exact"],
                            t["i"], t["region"], t["cat"], t["text"], t["sal"], t["sal_norm"], t["attn"], t["attn_norm"]])

    # ---- HTML heatmap ----
    def render_tokens(e):
        blocks, cur, parts = [], None, []
        for t in e["tokens"]:
            if t["region"] in ("other",):
                continue
            if t["region"] != cur:
                if parts:
                    blocks.append((cur, parts)); parts = []
                cur = t["region"]
            txt = html.escape(t["text"]).replace("\n", "↵\n")
            bd = CATCOL.get(t["cat"], "transparent")
            if t["region"] == "answer":
                parts.append(f'<span class="tk" style="background:#eef;border-bottom:2px solid {bd}">{txt}</span>')
            else:
                parts.append(f'<span class="tk" data-g="{t["sal_norm"]}" data-a="{t["attn_norm"]}" '
                             f'style="border-bottom:2px solid {bd}">{txt}</span>')
        if parts:
            blocks.append((cur, parts))
        html_blocks = []
        for reg, ps in blocks:
            html_blocks.append(f'<div class="reglab">{REGION_LABEL.get(reg, reg)}</div>'
                               f'<div class="reg">{"".join(ps)}</div>')
        return "".join(html_blocks)

    def minis(e):
        tr = [t for t in e["tokens"] if t["region"] == "trace"]
        tot = sum(t["sal"] for t in tr) or 1.0
        sm = 100 * sum(t["sal"] for t in tr if t["cat"] == "SMILES_frag") / tot
        fg = 100 * sum(t["sal"] for t in tr if t["cat"] == "FG_word") / tot
        return f"trace-saliency: SMILES {sm:.0f}% &middot; FG {fg:.0f}%"

    cards = []
    for e in ex:
        tag = "ER&gt;0 (fabricates)" if e["er"] > 0 else "ER=0 (clean)"
        ok = "correct" if e["exact"] else "wrong"
        cards.append(
            f'<div class="card"><div class="hd"><b>{html.escape(e["model"])}</b> '
            f'&middot; {html.escape(e["task"])} &middot; <span class="badge">{tag}</span> '
            f'&middot; {ok} &middot; <span class="mini">{minis(e)}</span></div>'
            f'<div class="ans">predicted answer: <code>{html.escape(e["answer_pred"])}</code></div>'
            f'{render_tokens(e)}</div>')

    page = """<!doctype html><html><head><meta charset="utf-8">
<title>R5 token-level saliency examples</title>
<style>
 body{font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#111;background:#fff;max-width:1100px}
 h1{font-size:20px} .sub{color:#555;margin-bottom:12px}
 .bar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ddd;z-index:5}
 button{font-size:14px;padding:6px 12px;margin-right:6px;border:1px solid #888;border-radius:6px;background:#f6f6f6;cursor:pointer}
 button.on{background:#111;color:#fff;border-color:#111}
 .legend span{margin-right:14px} .sw{display:inline-block;width:12px;height:12px;border-bottom:3px solid;vertical-align:middle;margin-right:4px}
 .card{border:1px solid #e2e2e2;border-radius:10px;padding:14px 16px;margin:16px 0}
 .hd{margin-bottom:6px} .badge{background:#fde;border-radius:4px;padding:1px 6px;font-size:12px}
 .mini{color:#555;font-size:12px} .ans{font-size:12px;color:#333;margin-bottom:10px}
 .reglab{font-size:11px;letter-spacing:.05em;color:#777;margin:10px 0 3px;text-transform:uppercase}
 .reg{white-space:pre-wrap;word-break:break-word;background:#fafafa;border-radius:6px;padding:8px}
 .tk{border-radius:2px;padding:0 .5px}
</style></head><body>
<h1>Chemical CoT &mdash; per-token answer saliency (R5)</h1>
<div class="sub">Background intensity = how much the final answer depends on that token
(<b id="mode">gradient&times;input saliency</b>), normalized within INPUT+TRACE per example.
Bottom-border color = token type.
Story: INPUT lights up far more than TRACE; within the trace, Chem-R/Chem-R-Faithful key on
<span style="color:#2c7fb8">SMILES fragments</span>, ChemDFM does not (few/no SMILES).</div>
<div class="bar">
 <button id="bg" class="on" onclick="setMode('g')">Gradient saliency</button>
 <button id="ba" onclick="setMode('a')">Answer&rarr;token attention</button>
 <span class="legend" style="margin-left:12px">
  <span><i class="sw" style="border-color:#2c7fb8"></i>SMILES fragment</span>
  <span><i class="sw" style="border-color:#e6550d"></i>functional-group word</span>
  <span><i class="sw" style="border-color:#31a354"></i>position/ring digit</span>
 </span>
</div>
__CARDS__
<script>
function paint(m){document.querySelectorAll('.tk[data-'+m+']').forEach(function(s){
  var v=parseFloat(s.getAttribute('data-'+m))||0; s.style.background='rgba(214,40,40,'+(v*0.85)+')';});}
function setMode(m){paint(m);
  document.getElementById('bg').className=(m=='g'?'on':'');
  document.getElementById('ba').className=(m=='a'?'on':'');
  document.getElementById('mode').textContent=(m=='g'?'gradient×input saliency':'answer→token attention');}
setMode('g');
</script></body></html>"""
    page = page.replace("__CARDS__", "\n".join(cards))
    open(os.path.join(OUT, "token_examples.html"), "w").write(page)
    print(f"wrote token_examples.csv ({sum(len(e['tokens']) for e in ex)} token rows) + token_examples.html ({len(ex)} examples)")


if __name__ == "__main__":
    main()
