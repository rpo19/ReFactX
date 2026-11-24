"""Finer, mutually exclusive error classes for eval logs (first match wins).

Rules, not hand labels. near_match mixes real errors with acceptable answers.

Usage: python -m utils.classify_errors_fine LOG.out.0 [...] --out errors_fine.jsonl
"""
import re,json,collections,unicodedata,os,argparse
from pathlib import Path
from utils.analyze_eval100_errors import load,parse_triples,answer_of
from utils.answer_normalization import parse_date,dates_match
def aslist(x): return x if isinstance(x,list) else [x]
def loose(s):
    s=re.sub(r'\([^()]*\)','',str(s)); s=unicodedata.normalize('NFKD',s); s=''.join(c for c in s if not unicodedata.combining(c)).lower()
    s=re.sub(r'\blanguage\b','',s); s=re.sub(r'[^\w ]',' ',s); s=re.sub(r'^\s*(the|a|an)\s+','',s); return ' '.join(s.split())
MID=re.compile(r'^[mg][/.][0-9a-z_]+$')
NUM=re.compile(r'^[+-]?[\d,]+(\.\d+)?$')
def same(a,b):
    if loose(a)==loose(b) and loose(a): return True
    try:
        if not NUM.match(str(a).strip()) and not NUM.match(str(b).strip()) and parse_date(str(a)) and parse_date(str(b)) and dates_match(str(a),str(b)): return True
    except Exception: pass
    return False
def ents(s):
    o=[]
    q=s.get('q_entity')
    if isinstance(q,str):
        q=[q]
    o+=q or []
    qe=s.get('questionEntity')
    if isinstance(qe,list): o+=[e.get('label') or e.get('mention') for e in qe if e.get('entityType')=='entity']
    if s.get('evidences'):
        ev=s['evidences']; ev=json.loads(ev) if isinstance(ev,str) else ev; heads={e[0] for e in ev};tails={e[2] for e in ev}; o+=[h for h in heads-tails]
    for p in (s.get('program') or []):
        if p.get('function')=='Find': o+=p['inputs'][:1]
    return [e for e in o if e]
def classify(r,bud):
    s=r['input_sample']; pred=r['prediction']; g=[str(x) for x in aslist(r['gt_answer'])]; T=parse_triples(r['triples'])
    if r['evaluation']['correct']: return 'correct'
    if not pred or all(loose(p) in('','i don t know') for p in aslist(pred)): return 'no_answer'
    pred=[str(p) for p in pred]
    hitg=[x for x in set(g) if any(same(x,p) for p in pred)]; hitp=[p for p in pred if any(same(x,p) for x in g)]
    if len(hitg)==len(set(g)) and len(hitp)==len(pred): return 'surface_form'
    if hitg: return 'partial_list'
    if any(MID.match(p) for p in pred): return 'raw_mid_answer'
    yn={'yes','no','true','false'}
    if {loose(x) for x in g}<=yn: return 'yesno_wrong'
    if len(g)==1 and NUM.match(g[0].strip()) and not(len(g[0].strip())==4 and g[0][:2] in('18','19','20')):
        return 'number_wrong_count_used' if '<count>' in r['full_prediction'] else 'number_wrong_no_count'
    lg,lp=[loose(x) for x in g],[loose(p) for p in pred]
    if any(min(len(a),len(b))>=4 and (re.search(rf'\b{re.escape(a)}\b',b) or re.search(rf'\b{re.escape(b)}\b',a)) for a in lg for b in lp): return 'near_match'
    nodes={loose(x) for t in T for x in (t[0],t[2])}
    if any(x in nodes for x in lg): return 'retrieved_not_selected'
    E=ents(s); subj={loose(t[0]) for t in T}; rawsubj=' | '.join(subj)
    if E and not any(loose(e) in nodes or (len(loose(e))>4 and loose(e) in rawsubj) for e in E): return 'wrong_start_entity'
    ev=s.get('evidences')
    if ev:
        ev=json.loads(ev) if isinstance(ev,str) else ev; br={loose(e[2]) for e in ev}&{loose(e[0]) for e in ev}
        if s.get('type') in('comparison','bridge_comparison'): return 'comparison_wrong'
        if br and not br&nodes: return 'missed_bridge'
        if br: return 'wrong_last_hop'
    return 'right_start_wrong_path' if E else 'other_missing'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('logs',nargs='+');ap.add_argument('--out',default='errors_fine.jsonl');a=ap.parse_args()
    out=open(a.out,'w');C={}
    for f in sorted(a.logs):
        n=os.path.basename(f)[:-6]; R=load(Path(f))
        if not R: continue
        m=re.search(r'_b(\d+)',n);bud=int(m[1]) if m else None
        c=C[n]=collections.Counter()
        for r in R:
            k=classify(r,bud);c[k]+=1
            out.write(json.dumps(dict(run=n,question=r['question'],cls=k,gold=r['gt_answer'],pred=r['prediction'],ntool=len(r['triples']),at_budget=bool(bud and len(r['triples'])+r['full_prediction'].count('</count>')>=bud),sent=sum('<no further' in x for x in re.findall(r'<fact>.*',r['full_prediction'])),facts=r['full_prediction'].count('<fact>')))+'\n')
    ks=[k for k,_ in sum(C.values(),collections.Counter()).most_common()]
    print('run',*ks)
    for n,c in C.items(): print(n,*[c[k] or '.' for k in ks])

if __name__=='__main__':
    main()
