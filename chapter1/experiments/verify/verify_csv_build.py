import json, numpy as np, pandas as pd, os, glob
S=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'outputs' / 'verification'); os.makedirs(S, exist_ok=True); D=os.path.join(S,'verify_csv'); os.makedirs(D, exist_ok=True); R=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'runs' / 'hprc')
o=json.load(open(os.path.join(S,'verify.json')))
# traces: rebuild from npz at chosen trials (long format)
def den(z): return z['preds']*z['std']+z['mean'], z['acts']*z['std']+z['mean']
pats={'gl':R+'/outputs/v2/e1/pred/gl_loso_CNN-LSTM-hpo_s42_*.npz','ph':R+'/outputs/e1_pred/pred/ph_loso_CNN-LSTM-hpo_s42_*.npz'}
chname={'ml':'Mediolateral shear (%BW)','vert':'Vertical force (%BW)','ap':'Anteroposterior shear (%BW)','px':'CoP, x (cm)','pz':'CoP, z (cm)'}
order={'vert':1,'ap':2,'ml':3,'px':4,'pz':5}
def rows_for(ds,pick,npts,chans):
    tr=o['traces'][f'{ds}_{pick}']
    for f in sorted(glob.glob(pats[ds])):
        z=np.load(f,allow_pickle=True); tl=list(z['trials']); sl=list(z['subjects'])
        idx=[i for i in range(len(tl)) if tl[i]==tr['trial'] and sl[i]==tr['subject']]
        if not idx: continue
        i=idx[0]; p,a=den(z); m=z['mask'][i]>0; cols=list(z['columns']); h=float(z['meta'][i,1])
        pi,ai=p[i,m],a[i,m]; n=len(pi); sel=np.unique(np.linspace(0,n-1,npts).round().astype(int))
        out=[]
        for j,c in enumerate(cols):
            plate=int(c.split('_')[2]); ch=c.split('_')[3]
            if ch not in chans: continue
            k=100 if ch not in ('px','pz') else h/10
            for t in sel: out.append(dict(ds=ds,pick=pick,subject=tr['subject'],trial=tr['trial'],t=round(t/50,2),plate=plate,ch=ch,chan=chname[ch],chorder=order[ch],act=round(float(ai[t,j]*k),2),pred=round(float(pi[t,j]*k),2)))
        return out
for ds in ['gl','ph']:
    pd.DataFrame(rows_for(ds,'typical',180,['ml','vert','ap','px','pz'])).to_csv(os.path.join(D,f'traces_{ds}_typical_all.csv'),index=False)
v=[]
for ds in ['gl','ph']:
    for pk in ['good','typical','poor']: v+=rows_for(ds,pk,160,['vert'])
pd.DataFrame(v).to_csv(os.path.join(D,'traces_vertical_picks.csv'),index=False)
c=[]
for ds in ['gl','ph']:
    for r in o['curves_'+ds]: c.append(dict(ds=ds,subject=r['subject'],epoch=r['epoch'],train_r2=r['tr2'],val_r2=r['vr2'],train_loss=r['tl'],val_loss=r['vl']))
pd.DataFrame(c).to_csv(os.path.join(D,'training_curves.csv'),index=False)
t=[]
for ds in ['gl','ph']:
    for r in o['trial_rmse'][ds]: t.append(dict(ds=ds,subject=r['subject'],trial=r['trial'],frames=r['frames'],rmse=r['rmse']))
pd.DataFrame(t).to_csv(os.path.join(D,'trial_vertical_rmse.csv'),index=False)
for f in sorted(os.listdir(D)):
    d=pd.read_csv(os.path.join(D,f)); print(f, d.shape, list(d.columns))
