"""Build a CLAIM-LEVEL reliability set for the hallucination detector.

The arena (arena.py) validates the detector at the trace/ranking level. Reviewers
also want per-CLAIM precision: when the detector flags that "the model claims the
answer molecule has <functional group>", did the model REALLY make that claim?

Structural presence is verified reliably by RDKit/SMARTS, so the error-prone step is
CLAIM EXTRACTION from free text (negation, hypotheticals, mentions of a different
molecule). This samples individual extracted FG claims (balanced across the detector's
fabricated / verified verdicts and across tasks/models), shows a WIDE CoT context
snippet around the mention (the answer structure is rendered only to identify which
molecule is the answer, NOT to re-check presence), and emits a SELF-CONTAINED
annotate_claims.html. The annotator reads the model's wording and judges
claims / no_claim / unsure, BLIND to the detector's verdict. score_claims.py then
computes the detector's claim-extraction precision.

Grounding (matches the detector's input-grounding ER rule):
  cap2mol : claim checked against the model's molecule (pred) OR the target (gt) OR
            named in the description  -> we render pred + gt and show the caption.
  mol2cap : the molecule IS the input SMILES -> render it; ask presence.

Run:  python human_eval/build_claim_set.py
"""
import base64
import glob
import html
import io
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from diagnose_hallucination import FUNCTIONAL_GROUP_DB  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402
RDLogger.DisableLog("rdApp.*")
random.seed(0)

SYN = {fg: kws for fg, (_s, kws) in FUNCTIONAL_GROUP_DB.items()}
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_DELIMS = (("<|think_start|>", "<think>"), ("<|think_end|>", "</think>"),
           ("<|answer_start|>", "<answer>"), ("<|answer_end|>", "</answer>"))

# models + tasks to sample claims from
MODELS = ["Chem-R", "ChemDFM-R", "+process", "Chem-R-Faithful",
          "DeepSeek-R1", "ether-0"]
TASKS = ["cap2mol", "mol2cap", "retrosynthesis"]
TARGET_PER_LABEL = 150       # aim ~150 fabricated + ~150 verified = ~300 claims
# per-label caps (kept separate so the abundant "verified" claims don't crowd out
# the rarer, precision-critical "fabricated" ones within a single model/task)
MAX_FAB_PER_MT = 60
MAX_VER_PER_MT = 15


def _norm(t):
    for a, b in _DELIMS:
        if a in t:
            t = t.replace(a, b)
    return t


def svg_datauri(smiles):
    """Render a SMILES to an inline SVG data-URI (crisp, small). None if unparsable."""
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    d = rdMolDraw2D.MolDraw2DSVG(320, 240)
    d.DrawMolecule(m)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def context_snippet(think, fg, width=280):
    """A short window of the reasoning around the first mention of the group."""
    low = think.lower()
    for syn in sorted(SYN.get(fg, [fg.replace('_', ' ')]), key=len, reverse=True):
        pat = re.compile(r"(?<![a-z])" + re.escape(syn.lower()) + r"(?:e?s)?(?![a-z])")
        m = pat.search(low)
        if m:
            a = max(0, m.start() - width)
            b = min(len(think), m.end() + width)
            seg = think[a:b]
            # bold the matched surface form
            seg = seg[:m.start() - a] + "【" + seg[m.start() - a:m.end() - a] + "】" + seg[m.end() - a:]
            return ("…" if a > 0 else "") + seg.strip() + ("…" if b < len(think) else "")
    return None


def load(model, task):
    o = (glob.glob(f"{ROOT}/se_results/{model}/{task}/output.json")
         or glob.glob(f"{ROOT}/data/results/{model}/{task}/output.json"))
    d = glob.glob(f"{ROOT}/data/results/{model}/{task}/*hallucination_details.jsonl")
    if not o or not d:
        return None, None
    O = {str(x["id"]): x for x in json.load(open(o[0]))}
    D = {str(json.loads(l)["id"]): json.loads(l) for l in open(d[0])}
    return O, D


def main():
    pools = {"fabricated": [], "verified": []}
    for model in MODELS:
        for task in TASKS:
            O, D = load(model, task)
            if not O:
                continue
            pf = pv = 0                      # picked fabricated / verified for this (model,task)
            ids = list(O.keys())
            random.shuffle(ids)
            for i in ids:
                if pf >= MAX_FAB_PER_MT and pv >= MAX_VER_PER_MT:
                    break
                d = D.get(i)
                o = O.get(i)
                if not d or not o:
                    continue
                er = d.get("details", {}).get("ER", {})
                think = _norm(o.get("answer", ""))
                mt = THINK_RE.search(think)
                think = (mt.group(1) if mt else think).strip()
                # grounding molecule image(s)
                if task == "mol2cap":
                    imgs = [("molecule", svg_datauri(o.get("question", "")))]
                    caption = None
                elif task == "retrosynthesis":
                    imgs = [("product (given)", svg_datauri(o.get("question", ""))),
                            ("reactants (reference)", svg_datauri(o.get("gt", "")))]
                    caption = None
                else:  # cap2mol
                    imgs = [("model's molecule", svg_datauri(d.get("pred_smiles", ""))),
                            ("reference (target)", svg_datauri(o.get("gt", "")))]
                    caption = o.get("question", "")
                imgs = [(lbl, u) for lbl, u in imgs if u]
                if not imgs:
                    continue
                for label, fgs, cap_ok in (
                        ("fabricated", er.get("fabricated_fgs", []), pf < MAX_FAB_PER_MT),
                        ("verified", er.get("verified_fgs", []), pv < MAX_VER_PER_MT)):
                    if not cap_ok:
                        continue
                    for fg in fgs:
                        ctx = context_snippet(think, fg)
                        if not ctx:
                            continue
                        pools[label].append({
                            "cid": f"{model}|{task}|{i}|{label}|{fg}",
                            "task": task, "group": fg.replace("_", " "),
                            "context": ctx, "caption": caption, "images": imgs,
                            "_label": label,   # HIDDEN from annotator (stripped into key)
                        })
                        if label == "fabricated":
                            pf += 1
                        else:
                            pv += 1
    fab = random.sample(pools["fabricated"], min(TARGET_PER_LABEL, len(pools["fabricated"])))
    ver = random.sample(pools["verified"], min(TARGET_PER_LABEL, len(pools["verified"])))
    claims = fab + ver
    random.shuffle(claims)

    # split visible claims (no label) from the answer key
    key = {c["cid"]: c["_label"] for c in claims}
    for c in claims:
        c.pop("_label")
    json.dump(key, open(os.path.join(os.path.dirname(__file__), "claims_key.json"), "w"))
    write_html(claims, key)
    print(f"pools: {len(pools['fabricated'])} fabricated, {len(pools['verified'])} verified available")
    print(f"sampled {len(fab)} fabricated + {len(ver)} verified = {len(claims)} claims")


def write_html(claims, key):
    out = os.path.join(os.path.dirname(__file__), "annotate_claims.html")
    data = json.dumps(claims, ensure_ascii=False)
    # the answer key is embedded but used ONLY on the results screen (after all items
    # are annotated); it is never shown per-card, so annotation stays blind.
    page = (_HTML.replace("/*__CLAIMS__*/", data)
                 .replace("/*__KEY__*/", json.dumps(key)))
    open(out, "w").write(page)
    print(f"wrote {out}  ({round(os.path.getsize(out)/1024)} KB) -- double-click to annotate")


_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Claim reliability annotation</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:780px;margin:24px auto;color:#222;padding:0 14px}
 #bar{height:6px;background:#eee;border-radius:3px;margin:10px 0}
 #fill{height:6px;background:#4a7;border-radius:3px;width:0;transition:width .2s}
 .card{border:1px solid #ddd;border-radius:10px;padding:18px;box-shadow:0 1px 4px #0001}
 .chip{display:inline-block;font-size:11px;background:#eef;color:#446;border-radius:10px;padding:2px 9px;margin-bottom:6px}
 .q{font-size:20px;margin:6px 0 4px}
 .q b{color:#b00}
 .hint{font-size:12px;color:#888;margin-bottom:10px}
 .imgs{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}
 .imgs figure{margin:0;text-align:center}
 .imgs figcaption{font-size:12px;color:#666}
 .imgs img{border:1px solid #eee;border-radius:6px;background:#fff}
 .ctx{background:#f7f7f9;border-left:3px solid #bbb;padding:8px 12px;font-size:13px;color:#444;border-radius:4px}
 .cap{font-size:13px;color:#555;margin:8px 0}
 #note{width:100%;box-sizing:border-box;margin-top:10px;padding:7px;font-size:13px;border:1px solid #ddd;border-radius:6px;resize:vertical}
 .btns{display:flex;gap:10px;margin-top:14px}
 button{flex:1;padding:12px;font-size:15px;border:1px solid #ccc;border-radius:8px;cursor:pointer;background:#fafafa}
 button:hover{background:#eef}
 .yes{border-color:#4a7}.no{border-color:#b55}.un{border-color:#aaa}
 #intro,#done{text-align:left;padding:6px 2px}
 #done{display:none;text-align:center}
 #card{display:none}
 #meta{font-size:13px;color:#888;display:flex;justify-content:space-between}
 .kbd{font-size:11px;color:#999}
 .ex{border:1px solid #eee;border-radius:8px;padding:10px 12px;background:#fafcff;font-size:13px;margin:8px 0}
 #toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#333;color:#fff;padding:8px 16px;border-radius:20px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none}
</style></head><body>
<h2>Functional-group claim check</h2>

<div id="intro">
 <p style="font-size:14px;color:#444">A chemical reasoning model, while solving a molecule task, wrote the
 reasoning shown below. We want to know whether the model <b>actually asserts, in its own words</b>, that the
 answer molecule <b>contains</b> a particular functional group (shown in <b style="color:#b00">red</b>). Your job,
 for each of the <b id="ntot"></b> items: <b>read the model's text — does it claim the answer molecule HAS that
 group?</b></p>
 <ul style="font-size:13px;color:#555;line-height:1.6">
  <li><b>Claims it</b> — the model states / asserts that the answer molecule contains the group.</li>
  <li><b>No claim</b> — the model does NOT assert this: it <i>negates or rejects</i> it ("no …", "not a …",
      considered then dropped), OR the mention is about a <i>different</i> molecule (the input, a reactant, a
      hypothetical), OR it is only mentioned in passing without asserting presence in the answer.</li>
  <li><b>Unsure</b> — genuinely ambiguous; add a note.</li>
 </ul>
 <p style="font-size:13px;color:#555">Judge <b>from the model's wording</b>, not from the structure — an automatic
 checker already verifies structural presence reliably; here we only check whether the model truly <i>made the
 claim</i>. The structure(s) are shown only to identify which molecule is the answer.</p>
 <div class="ex"><b>Example.</b> Group: <b style="color:#b00">carboxylic acid</b>. If the model wrote "the target
 contains a carboxylic acid", choose <b>Claims it</b>. If it wrote "there is no carboxylic acid here", or "I first
 considered a carboxylic acid but used an ester instead", choose <b>No claim</b>.</div>
 <p style="font-size:13px;color:#777">Progress auto-saves in this browser — you can close and resume anytime.
 Keys: <span class="kbd">[Y] Present &nbsp; [N] Absent &nbsp; [U] Unsure &nbsp; [←] back</span></p>
 <button style="max-width:280px" onclick="start()">Start &nbsp;▶</button>
</div>

<div id="meta"><span id="who"></span><span id="count"></span></div>
<div id="bar"><div id="fill"></div></div>
<div class="card" id="card">
  <div class="chip" id="chip"></div>
  <div class="q" id="q"></div>
  <div class="hint" id="hint"></div>
  <div class="imgs" id="imgs"></div>
  <div class="cap" id="cap"></div>
  <div class="ctx" id="ctx"></div>
  <textarea id="note" rows="1" placeholder="optional note (e.g. why unsure)…"></textarea>
  <div class="btns">
    <button class="yes" onclick="ans('claims')">Claims it (Y)</button>
    <button class="no"  onclick="ans('no_claim')">No claim (N)</button>
    <button class="un"  onclick="ans('unsure')">Unsure (U)</button>
  </div>
</div>
<div id="done">
  <h3>All done — thank you! 🎉</h3>
  <div id="result"></div>
  <button style="max-width:280px;margin:auto" onclick="save()">⬇ Download annotations JSON</button>
  <p style="font-size:13px;color:#888">The score above is computed in your browser — no need to send anything back.
  Download is optional (keep for your records or an inter-annotator check via score_claims.py).</p>
</div>
<div id="toast"></div>
<script>
const CLAIMS = /*__CLAIMS__*/;
const VERDICT = /*__KEY__*/;   // detector verdict per claim; used ONLY on the results screen, never shown while annotating
const TASKLBL = {cap2mol:"molecule design (text → molecule)", mol2cap:"molecule captioning (molecule → text)",
  retrosynthesis:"retrosynthesis (product → reactants)"};
let who = localStorage.getItem('claim_annotator') || prompt('Your name/initials:') || 'anon';
localStorage.setItem('claim_annotator', who);
const KEY='claim_ann_'+who, NKEY='claim_note_'+who;
let ann = JSON.parse(localStorage.getItem(KEY) || '{}');
let notes = JSON.parse(localStorage.getItem(NKEY) || '{}');
let i = CLAIMS.findIndex(c => !(c.cid in ann));
if (i < 0) i = CLAIMS.length;
document.getElementById('who').textContent='annotator: '+who;
document.getElementById('ntot').textContent=CLAIMS.length;
let started=false;
function start(){ started=true; document.getElementById('intro').style.display='none';
  document.getElementById('card').style.display='block'; render(); }
function toast(m){ const t=document.getElementById('toast'); t.textContent=m; t.style.opacity=1;
  setTimeout(()=>t.style.opacity=0,1500); }
function render(){
  if (i >= CLAIMS.length){ document.getElementById('card').style.display='none';
     document.getElementById('done').style.display='block';
     document.getElementById('fill').style.width='100%';
     document.getElementById('count').textContent=CLAIMS.length+' / '+CLAIMS.length;
     showResult(); return; }
  const c = CLAIMS[i];
  const t = (c.task||'').replace(/^s2_/,'s2 ').replace(/_/g,' ');
  document.getElementById('chip').textContent = TASKLBL[c.task] || t;
  document.getElementById('q').innerHTML = 'Does the model claim the answer molecule contains a <b>'+c.group+'</b>?';
  document.getElementById('hint').textContent =
     'Judge from the model text below; the structure(s) only show which molecule is the answer.';
  document.getElementById('imgs').innerHTML = c.images.map(
    ([lbl,u])=>'<figure><img src="'+u+'" width="300"><figcaption>'+lbl+'</figcaption></figure>').join('');
  document.getElementById('cap').innerHTML = c.caption ? '<i>caption:</i> '+c.caption : '';
  document.getElementById('ctx').innerHTML = '<i>model wrote:</i> “'+c.context+'”';
  document.getElementById('note').value = notes[c.cid] || '';
  document.getElementById('count').textContent=(i)+' / '+CLAIMS.length;
  document.getElementById('fill').style.width=(100*i/CLAIMS.length)+'%';
}
function showResult(){
  let fc=0,fn=0,vc=0,vn=0;   // fab claims/no_claim, verified claims/no_claim
  for(const cid in ann){ const det=VERDICT[cid], h=ann[cid];
    if(h==='unsure'||!det) continue;
    if(det==='fabricated'){ if(h==='claims')fc++; else if(h==='no_claim')fn++; }
    else if(det==='verified'){ if(h==='claims')vc++; else if(h==='no_claim')vn++; } }
  const pct=(a,b)=> (a+b)? (100*a/(a+b)).toFixed(1)+'%' : '—';
  document.getElementById('result').innerHTML =
    '<div class="ex" style="text-align:left">'
    + '<b>Detector fabrication precision = '+pct(fc,fn)+'</b> '
    + '<span style="color:#888">(of the claims the detector flagged fabricated, the share where you confirmed the model really makes the claim; n='+(fc+fn)+')</span><br>'
    + 'Verified-side extraction agreement = '+pct(vc,vn)+' (n='+(vc+vn)+')<br>'
    + 'Overall extraction precision = '+pct(fc+vc,fn+vn)
    + '<br><span style="color:#888">Structural absence is trusted from SMARTS; recall is not estimable from this set.</span></div>';
}
function ans(v){ const c=CLAIMS[i]; ann[c.cid]=v;
  const nt=document.getElementById('note').value.trim(); if(nt) notes[c.cid]=nt; else delete notes[c.cid];
  localStorage.setItem(KEY,JSON.stringify(ann)); localStorage.setItem(NKEY,JSON.stringify(notes));
  const done=Object.keys(ann).length;
  if(done%50===0 && done<CLAIMS.length) toast(done+' / '+CLAIMS.length+' done — nice, take a break if you like');
  i++; render(); }
function back(){ if(i>0){i--; render();} }
function save(){ const blob=new Blob([JSON.stringify({annotator:who,annotations:ann,notes:notes},null,1)],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='claim_annotations_'+who+'.json'; a.click(); }
document.addEventListener('keydown',e=>{ if(!started) return;
  if(document.activeElement && document.activeElement.id==='note' && e.key!=='Enter') return;
  const k=e.key.toLowerCase();
  if(k==='y')ans('claims'); else if(k==='n')ans('no_claim'); else if(k==='u')ans('unsure');
  else if(e.key==='ArrowLeft')back();});
</script></body></html>"""


if __name__ == "__main__":
    main()
