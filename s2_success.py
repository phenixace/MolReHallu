"""Official-criterion S2-TOMG success (0/1), ported from S2-TOMG-Bench/evaluate.py.
MolCustom: exact count match (constraints in metadata). MolEdit: group +/-1 (group parsed
from instruction, validated 100% vs official columns). MolOpt: property moved in the
instructed direction. Uses the official mol_prop."""
import re
from s2_official_eval import mol_prop

_FG=['benzene_ring','hydroxyl','anhydride','aldehyde','ketone','carboxyl','ester','amide',
     'amine','nitro','halo','nitrile','thiol','sulfide','disulfide','sulfoxide','sulfone','borane']
_ATOM=['carbon','oxygen','nitrogen','sulfur','fluorine','chlorine','bromine','iodine','phosphorus',
       'boron','silicon','selenium','tellurium','arsenic','antimony','bismuth','polonium']
_BOND=['single','double','triple','rotatable','aromatic']

def _norm(g):
    g=(g or "").strip().lower().replace("_"," ").strip(); g=re.sub(r'\s+',' ',g)
    if 'benzene' in g: return 'benzene_ring'
    return g.rstrip('s').replace(' ','_')
def _add(i):
    m=re.search(r'add(?:ing)? (?:an?\s+) (.+?)(?: to the molecule| in the molecule|\.|$)',i,re.I)
    if not m: m=re.search(r'add(?:ing)? (?:an?\s+)(.+?)(?: to the molecule| in the molecule|\.|$)',i,re.I)
    return _norm(m.group(1)) if m else None
def _del(i):
    m=re.search(r'(?:remov\w+|delet\w+) (?:an?\s+)(.+?)(?: from the molecule| in the molecule|\.|$)',i,re.I)
    return _norm(m.group(1)) if m else None
def _sub(i):
    rem=re.search(r'(?:substitut\w*|replac\w*)\s+(?:an?\s+)?(.+?)\s+(?:in the molecule|by|with|into|in)\b',i,re.I)
    add=re.search(r'.*\b(?:with|by|into)\s+(?:an?\s+)?([a-zA-Z][a-zA-Z ]*?)\.?\s*$',i,re.I)
    return (_norm(rem.group(1)) if rem else None,_norm(add.group(1)) if add else None)

def s2_success(subtask, pred, metadata, instruction):
    if not pred or mol_prop(pred,"validity") is None: return 0.0
    cons=(metadata or {}).get("constraints",{}) or {}
    src=(metadata or {}).get("source_molecule",""); instr=instruction or ""
    try:
        if "MolCustom_AtomNum" in subtask:
            for a in _ATOM:
                tgt=int(cons.get(a,0) or 0)
                if tgt>0 and mol_prop(pred,"num_"+a)!=tgt: return 0.0
            return 1.0
        if "MolCustom_BondNum" in subtask:
            for b in _BOND:
                tgt=int(cons.get(b,0) or 0)
                if tgt<=0: continue
                if mol_prop(pred,"rot_bonds" if b=="rotatable" else "num_"+b+"_bonds")!=tgt: return 0.0
            return 1.0
        if "MolCustom_FunctionalGroup" in subtask:
            for g in _FG:
                tgt=int(cons.get(g,0) or 0)
                if tgt>0 and mol_prop(pred,"num_"+g)!=tgt: return 0.0
            return 1.0
        if "MolEdit_AddComponent" in subtask:
            g=_add(instr)
            if not g or not src: return 0.0
            return 1.0 if mol_prop(pred,"num_"+g)==(mol_prop(src,"num_"+g) or 0)+1 else 0.0
        if "MolEdit_DelComponent" in subtask:
            g=_del(instr)
            if not g or not src: return 0.0
            return 1.0 if mol_prop(pred,"num_"+g)==(mol_prop(src,"num_"+g) or 0)-1 else 0.0
        if "MolEdit_SubComponent" in subtask:
            rem,add=_sub(instr)
            if not rem or not add or not src: return 0.0
            return 1.0 if (mol_prop(pred,"num_"+rem)==(mol_prop(src,"num_"+rem) or 0)-1 and
                           mol_prop(pred,"num_"+add)==(mol_prop(src,"num_"+add) or 0)+1) else 0.0
        if "MolOpt" in subtask:
            prop={"LogP":"logP","MR":"MR","QED":"qed"}.get(subtask.split("_")[-1])
            if not prop or not src: return 0.0
            lower=bool(re.search(r'\b(lower|decreas|reduc)\w*',instr,re.I))
            pv,sv=mol_prop(pred,prop),mol_prop(src,prop)
            if pv is None or sv is None: return 0.0
            return 1.0 if ((lower and pv<sv) or (not lower and pv>sv)) else 0.0
    except Exception:
        return 0.0
    return 0.0
