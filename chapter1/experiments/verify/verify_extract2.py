import glob, json, numpy as np, pandas as pd, os, re
R=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'runs' / 'hprc'); S=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'outputs' / 'verification'); os.makedirs(S, exist_ok=True)
out=json.load(open(os.path.join(S,'verify.json')))
def den(z): return z['preds']*z['std']+z['mean'], z['acts']*z['std']+z['mean']
# 1. recompute fold R2 from npz vs CSV, all configs/seeds, both datasets
rows=[]
for ds,pdir,cdir in [('gl',R+'/outputs/v2/e1',R+'/outputs/v2/e1'),('ph',R+'/outputs/e1_pred',R+'/outputs/e1_pred')]:
    csv=pd.concat([pd.read_csv(f) for f in glob.glob(cdir+f'/{ds}_*_multiseed.csv')])
    for f in sorted(glob.glob(pdir+f'/pred/{ds}_loso_*.npz')):
        z=np.load(f,allow_pickle=True); m=z['mask']>0
        r2=[]
        for j in range(10):
            y=z['acts'][...,j][m]; yh=z['preds'][...,j][m]; r2.append(1-((y-yh)**2).sum()/((y-y.mean())**2).sum())
        c=csv[(csv.config==str(z['config']))&(csv.seed==int(z['seed']))&(csv.subject==str(z['held_out']))]
        rows.append(dict(dataset=ds,config=str(z['config']),seed=int(z['seed']),subject=str(z['held_out']),recomputed=float(np.mean(r2)),logged=float(c.r2.iloc[0]) if len(c) else np.nan))
rc=pd.DataFrame(rows); rc['diff']=(rc.recomputed-rc.logged).abs()
print(len(rc), 'max diff', rc['diff'].max(), 'missing', rc.logged.isna().sum())
out['recompute']=dict(n=len(rc),max_diff=float(rc['diff'].max()),missing=int(rc.logged.isna().sum()))
# 2. traces by vertical RMSE %BW
def tstats(pat):
    L=[]
    for f in sorted(glob.glob(pat)):
        z=np.load(f,allow_pickle=True); p,a=den(z); cols=list(z['columns'])
        iv=[cols.index('ground_force_1_vert'),cols.index('ground_force_2_vert')]
        for i in range(len(z['trials'])):
            m=z['mask'][i]>0
            e=np.sqrt(np.mean((p[i,m][:,iv]-a[i,m][:,iv])**2))*100
            L.append(dict(f=f,i=i,subject=str(z['subjects'][i]),trial=str(z['trials'][i]),frames=int(m.sum()),rmse=float(e)))
    return pd.DataFrame(L)
traces={}; dist={}
for ds,pat in [('gl',R+'/outputs/v2/e1/pred/gl_loso_CNN-LSTM-hpo_s42_*.npz'),('ph',R+'/outputs/e1_pred/pred/ph_loso_CNN-LSTM-hpo_s42_*.npz')]:
    d=tstats(pat); q=d[d.frames>=150].sort_values('rmse').reset_index(drop=True)
    dist[ds]=d[['subject','trial','frames','rmse']].round(2).to_dict('records')
    picks={'typical':q.iloc[len(q)//2],'poor':q.iloc[int(len(q)*0.9)],'good':q.iloc[int(len(q)*0.1)]}
    for k,r in picks.items():
        z=np.load(r.f,allow_pickle=True); p,a=den(z); m=z['mask'][r.i]>0; cols=list(z['columns']); h=float(z['meta'][r.i,1])
        pi,ai=p[r.i,m],a[r.i,m]; n=len(pi); step=max(1,int(np.ceil(n/200)))
        recs=[]
        for t in range(0,n,step):
            rec=dict(t=round(t/50,2))
            for j,c in enumerate(cols):
                kk=100 if not c.endswith(('_px','_pz')) else h/10
                key=c.replace('ground_force_','g')
                rec['a'+key]=round(float(ai[t,j]*kk),1); rec['p'+key]=round(float(pi[t,j]*kk),1)
            recs.append(rec)
        traces[f'{ds}_{k}']=dict(trial=r.trial,subject=r.subject,rmse=round(r.rmse,1),frames=int(n),pct={'good':10,'typical':50,'poor':90}[k],rows=recs)
        print(ds,k,r.subject,r.trial,round(r.rmse,1),n)
    print(ds,'trial vert RMSE pct 10/50/90:',np.percentile(d.rmse,[10,50,90]).round(1), 'n',len(d))
out['traces']=traces; out['trial_rmse']=dist
# 3. data accounting from logs
acc={}
for ds,f in [('gl',R+'/runs/slurm/ch1-e1_pred_20067543_6.out'),('ph',R+'/runs/slurm/ch1-e1_pred_3196981_18.out')]:
    txt=open(f,encoding='utf-8',errors='ignore').read()
    acc[ds]=dict(excluded=re.findall(r"Excluded (\d+) trials from subjects (\[.*?\])",txt)[:1],
                 paired=re.findall(r"Paired files: (\d+)",txt)[:1],
                 reasons=len(re.findall(r"Reason: ",txt)),
                 folds=re.findall(r"held-out subject='(\w+)' → (\d+) test trials, (\d+) train\+val trials",txt),
                 missing_markers=sorted(set(re.findall(r"header: \['(\w+)'\]",txt))))
    print(ds, acc[ds])
out['accounting']=acc
json.dump(out,open(os.path.join(S,'verify.json'),'w')); print('ok')
