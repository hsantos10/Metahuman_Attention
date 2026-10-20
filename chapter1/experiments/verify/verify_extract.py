"""Verification tab, step 1: training curves from the e1 logs, per-subject unit checks and trial picks.
Run from the repo root after `sync.sh pull`; writes outputs/verification/verify.json. Steps 2-3: verify_extract2.py, verify_csv_build.py."""
import re, glob, json, numpy as np, pandas as pd, os
R=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'runs' / 'hprc')
S=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'outputs' / 'verification'); os.makedirs(S, exist_ok=True)
out={}
# A. training curves: GL CNN-LSTM s42 (v2 task 6), PH CNN-LSTM s42 (3196981 task 18)
ep=re.compile(r"Epoch \[(\d+)/(\d+)\], Train Loss: ([\d.]+), Val Loss: ([\d.]+), Train MAE: ([\d.]+), Val MAE: ([\d.]+), Train R2: ([-\d.]+), Val R2: ([-\d.]+)")
ho=re.compile(r"LOSO: held-out subject='(\w+)'")
res=re.compile(r"\] (\w+): R2=([-\d.]+)")
def curves(f):
    rows=[]; summ=[]; subj=None; es=None
    for line in open(f,encoding='utf-8',errors='ignore'):
        m=ho.search(line)
        if m: subj=m.group(1); continue
        m=ep.search(line)
        if m and subj: rows.append(dict(subject=subj,epoch=int(m.group(1)),tl=float(m.group(3)),vl=float(m.group(4)),tr2=float(m.group(7)),vr2=float(m.group(8))))
        if 'Early stopping triggered at epoch' in line: es=int(re.search(r'epoch (\d+)',line).group(1))
        m=res.search(line)
        if m and subj: summ.append(dict(subject=subj,test_r2=float(m.group(2)),stop=es)); es=None
    return pd.DataFrame(rows), pd.DataFrame(summ)
for tag,f in [('gl',R+'/runs/slurm/ch1-e1_pred_20067543_6.out'),('ph',R+'/runs/slurm/ch1-e1_pred_3196981_18.out')]:
    c,s=curves(f)
    # best epoch by val loss
    best=c.loc[c.groupby('subject').vl.idxmin()][['subject','epoch','vr2','tr2']].rename(columns={'epoch':'best_epoch','vr2':'best_val_r2','tr2':'train_r2_at_best'})
    s=s.merge(best,on='subject',how='left'); s['last_epoch']=s.subject.map(c.groupby('subject').epoch.max())
    out['curves_'+tag]=c.round(4).to_dict('records'); out['curvesum_'+tag]=s.round(3).to_dict('records')
    print(tag, len(c), s.to_string())
# B/C. units + inference from npz
def load(pat):
    fs=sorted(glob.glob(pat)); return [np.load(f,allow_pickle=True) for f in fs]
def den(z): return z['preds']*z['std']+z['mean'], z['acts']*z['std']+z['mean']
def trial_stats(zs):
    rows=[]
    for z in zs:
        p,a=den(z); cols=list(z['columns']); iv1,iv2=cols.index('ground_force_1_vert'),cols.index('ground_force_2_vert')
        for i in range(len(z['trials'])):
            m=z['mask'][i]>0; n=int(m.sum())
            at=a[i,m][:,[iv1,iv2]].sum(1); pt=p[i,m][:,[iv1,iv2]].sum(1)
            # per-trial R2 over all 10 channels (mean of channel R2)
            r2=[]
            for j in range(10):
                y=a[i,m,j]; yh=p[i,m,j]; ss=((y-y.mean())**2).sum()
                r2.append(1-((y-yh)**2).sum()/ss if ss>1e-9 else np.nan)
            rows.append(dict(subject=str(z['subjects'][i]),trial=str(z['trials'][i]),frames=n,mass=float(z['meta'][i,0]),height=float(z['meta'][i,1]),
                             act_total_mean=float(at.mean()),pred_total_mean=float(pt.mean()),r2=float(np.nanmean(r2)),
                             r2_vert=float(np.nanmean([r2[iv1],r2[iv2]]))))
    return pd.DataFrame(rows)
new=trial_stats(load(R+'/outputs/v2/e1/pred/gl_loso_CNN-LSTM-hpo_s42_*.npz'))
old=trial_stats(load(R+'/outputs/e1_pred/pred/gl_loso_CNN-LSTM-hpo_s42_*.npz'))
ph=trial_stats(load(R+'/outputs/e1_pred/pred/ph_loso_CNN-LSTM-hpo_s42_*.npz'))
def subj_units(d): return d.groupby('subject').agg(mass=('mass','first'),height=('height','first'),trials=('trial','size'),act=('act_total_mean','median'),pred=('pred_total_mean','median'),r2=('r2','mean')).reset_index()
u=subj_units(new).merge(subj_units(old)[['subject','act','pred']],on='subject',suffixes=('','_old'))
print(u.round(3).to_string()); print(subj_units(ph).round(3).to_string())
out['units_gl']=u.round(3).to_dict('records'); out['units_ph']=subj_units(ph).round(3).to_dict('records')
# pick traces: median and worst trial (by r2) among trials with >=150 frames
def pick(d):
    q=d[d.frames>=150].sort_values('r2'); med=q.iloc[len(q)//2]; worst=q.iloc[0]; best=q.iloc[-1]; return med,worst,best
traces={}
for tag,d,pat in [('gl',new,R+'/outputs/v2/e1/pred/gl_loso_CNN-LSTM-hpo_s42_*.npz'),('ph',ph,R+'/outputs/e1_pred/pred/ph_loso_CNN-LSTM-hpo_s42_*.npz')]:
    zs=load(pat); med,worst,best=pick(d)
    print(tag,'median',med.trial,round(med.r2,3),'worst',worst.trial,round(worst.r2,3),'best',best.trial,round(best.r2,3))
    for kind,row in [('median',med),('worst',worst),('best',best)]:
        for z in zs:
            tl=list(z['trials'])
            if row.trial in tl and str(z['subjects'][tl.index(row.trial)])==row.subject:
                i=tl.index(row.trial); p,a=den(z); m=z['mask'][i]>0; cols=list(z['columns'])
                pi=p[i,m]; ai=a[i,m]; n=len(pi); step=max(1,int(np.ceil(n/240)))
                h=float(z['meta'][i,1])
                recs=[]
                for t in range(0,n,step):
                    rec=dict(t=round(t/50,2))
                    for j,c in enumerate(cols):
                        k=100 if not c.endswith(('_px','_pz')) else h/10   # %BW or cm
                        rec['a_'+c]=round(float(ai[t,j]*k),2); rec['p_'+c]=round(float(pi[t,j]*k),2)
                    recs.append(rec)
                traces[f'{tag}_{kind}']=dict(trial=row.trial,subject=row.subject,r2=round(row.r2,3),r2_vert=round(row.r2_vert,3),frames=int(n),rows=recs)
out['traces']=traces
# per-trial r2 distribution
out['trial_r2_gl']=new[['subject','trial','r2','r2_vert','frames']].round(3).to_dict('records')
out['trial_r2_ph']=ph[['subject','trial','r2','r2_vert','frames']].round(3).to_dict('records')
json.dump(out,open(os.path.join(S,'verify.json'),'w'))
print('ok')
