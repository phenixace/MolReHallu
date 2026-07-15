import json,glob,re,subprocess,sys,os,pandas as pd
SRC="data/s2-bench"
SE="se_results"
EVAL="S2-TOMG-Bench"; DD="s2eval"
os.makedirs(f"{DD}/full",exist_ok=True)
SUBS=[("MolCustom","AtomNum"),("MolCustom","BondNum"),("MolCustom","FunctionalGroup"),
      ("MolEdit","AddComponent"),("MolEdit","DelComponent"),("MolEdit","SubComponent"),
      ("MolOpt","LogP"),("MolOpt","MR"),("MolOpt","QED")]
ans_re=re.compile(r"<answer>(.*?)</answer>",re.DOTALL)
# 建 GT (前500)
for task,sub in SUBS:
    cfg=f"{task}_{sub}"; g=f"{DD}/full/{cfg}.csv"
    if not os.path.exists(g): pd.read_csv(f"{SRC}/{cfg}.csv").head(500).to_csv(g,index=False)
def run_model(model):
    res={}
    for task,sub in SUBS:
        sdir=f"s2_{task}_{sub}"
        o=glob.glob(f"{SE}/{model}/{sdir}/output.json")
        if not o: res[f"{task}_{sub}"]=None; continue
        d=json.load(open(o[0])); items=d if isinstance(d,list) else d.get("results",[])
        outs=[]
        for s in items[:500]:
            full=s.get("answer","") or ""; m=ans_re.search(full); outs.append(m.group(1).strip() if m else full.strip())
        pf=f"{DD}/pred_{model}_{task}_{sub}.csv"; pd.DataFrame({"outputs":outs}).to_csv(pf,index=False)
        out=subprocess.run([sys.executable,"evaluate.py","--task",task,"--subtask",sub,
            "--predictions",pf,"--data_dir",DD,"--no_use_hf","--correct"],cwd=EVAL,
            capture_output=True,text=True).stdout
        acc=re.search(r"(Accuracy|Success Rate):\s*([\d.]+)",out)
        val=re.search(r"Validt?y:\s*([\d.]+)",out)
        res[f"{task}_{sub}"]=(float(acc.group(2))*100 if acc else None, float(val.group(1))*100 if val else None)
    return res
for model in sys.argv[1:]:
    r=run_model(model)
    print(f"\n===== {model} (官方 success% / validity%) =====")
    for k,v in r.items():
        print(f"  {k:24} {('%.1f / %.1f'%(v[0],v[1])) if v and v[0] is not None else 'NA'}")
