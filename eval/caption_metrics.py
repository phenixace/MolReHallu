import json,glob,re,sys
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.tokenize import wordpunct_tokenize
from rouge_score import rouge_scorer
sm=SmoothingFunction().method1
rs=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
ans_re=re.compile(r"<answer>(.*?)</answer>",re.DOTALL)
def predcap(full):
    m=ans_re.search(full)
    if m: return m.group(1).strip()
    if "</think>" in full: return full.split("</think>",1)[1].strip()
    return full.strip()
for model in sys.argv[1:]:
    p=glob.glob(f"se_results/{model}/mol2cap/output.json")
    if not p: print(f"{model}: 无"); continue
    d=json.load(open(p[0])); items=d if isinstance(d,list) else d.get("results",[])
    b2=b4=r1=r2=rl=n=0
    for s in items:
        gt=(s.get("gt") or "").strip(); pr=predcap(s.get("answer") or "")
        if not gt: continue
        rt=wordpunct_tokenize(gt); pt=wordpunct_tokenize(pr); n+=1
        b2+=sentence_bleu([rt],pt,weights=(.5,.5),smoothing_function=sm)
        b4+=sentence_bleu([rt],pt,weights=(.25,.25,.25,.25),smoothing_function=sm)
        sc=rs.score(gt,pr); r1+=sc['rouge1'].fmeasure; r2+=sc['rouge2'].fmeasure; rl+=sc['rougeL'].fmeasure
    print(f"{model:22} BLEU-2={100*b2/n:.1f} BLEU-4={100*b4/n:.1f} ROUGE-1={100*r1/n:.1f} ROUGE-2={100*r2/n:.1f} ROUGE-L={100*rl/n:.1f} (n={n})")
