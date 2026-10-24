import json, numpy as np, os
os.chdir('/home/claude/evaluacion-cambio-canal/simulacion')
pred=json.load(open('predicciones.json'))
L=lambda f: json.load(open(f))
col=lambda R,k: np.array([r[k] for r in R])
def z(x,l): return (x.mean()-l)/(x.std(ddof=1)/np.sqrt(len(x)))
print('--- exact z, main table')
for nm in ['base','con_C','confunde_sin_C','confunde_con_C','solap_medio','solap_bajo','confusion_U','n_pequeno']:
    R=L(f'res_{nm}_200.json'); th=pred[nm]['theta']; lim=pred[nm]['lim']
    dl={'con_C':0.009575447347237531,'confunde_con_C':0.013139808831507744}.get(nm,lim)
    h,d,se=col(R,'hajek'),col(R,'dr'),col(R,'dr_se')
    print(nm, f"H {h.mean()*100:.4f} z={z(h,lim):+.3f}  DR {d.mean()*100:.4f} z={z(d,dl):+.3f} SD {d.std(ddof=1)*100:.4f} cov {np.mean(abs(d-th)<=1.96*se)*100:.1f} se/sd {se.mean()/d.std(ddof=1):.3f} n0 {col(R,'n0').mean():.0f} n1 {col(R,'n1').mean():.0f}")
print('--- trimming DR, RMSE & SE')
for nm in ['base','solap_bajo','solap_medio','n_10000','n_pequeno','confusion_U','confunde_sin_C']:
    R=L(f'res_{nm}_200.json'); th=pred[nm]['theta']; d=col(R,'dr'); se=col(R,'dr_se')
    out=[f'{nm}: rmse0 {np.sqrt(np.mean((d-th)**2))*100:.4f}']
    for q in (99,95,90):
        dt=col(R,f'dr_trim{q}'); st=col(R,f'dr_trim{q}_se')
        out.append(f'p{q} rmse {np.sqrt(np.mean((dt-th)**2))*100:.4f} se_t/sd {st.mean()/dt.std(ddof=1):.3f} cov_t {np.mean(abs(dt-th)<=1.96*st)*100:.1f} cov_untrimSE {np.mean(abs(dt-th)<=1.96*se)*100:.1f}')
    print('  '.join(out))
print('--- known cutoff, low overlap trimmed DR')
for nm in ['solap_bajo','base','solap_medio']:
    R=L(f'res_corte_{nm}_200.json'); th=pred[nm]['theta']
    for m in ('std','corte'):
        c=lambda k: np.array([r[m][k] for r in R])
        d=c('dr'); se=c('dr_se'); dt=c('dr_trim90'); 
        print(nm,m,f"SD {d.std(ddof=1)*100:.4f} rmse0 {np.sqrt(np.mean((d-th)**2))*100:.4f} rmse90 {np.sqrt(np.mean((dt-th)**2))*100:.4f} cov90 untrimSE {np.mean(abs(dt-th)<=1.96*se)*100:.1f} cov90 trimSE {np.mean(abs(dt-th)<=1.96*c('dr_trim90_se'))*100:.1f}")
print('--- SD reduction known cutoff')
for nm in ['base','n_10000','n_pequeno']:
    R=L(f'res_corte_{nm}_200.json')
    s=np.std([r['std']['dr'] for r in R],ddof=1); c=np.std([r['corte']['dr'] for r in R],ddof=1)
    print(nm, f'{s*100:.4f} {c*100:.4f} red {1-c/s:.4f}')
sb=np.std([r['corte']['dr'] for r in L('res_corte_base_200.json')],ddof=1)
s10c=np.std([r['corte']['dr'] for r in L('res_corte_n_10000_200.json')],ddof=1)
s10s=np.std([r['std']['dr'] for r in L('res_corte_n_10000_200.json')],ddof=1)
sbs=np.std([r['std']['dr'] for r in L('res_corte_base_200.json')],ddof=1)
print('sqrt n check corte', s10c/(sb*np.sqrt(3)), 'std', s10s/(sbs*np.sqrt(3)))
print('--- decomposition')
for nm in ['base','n_10000','n_pequeno']:
    R=L(f'res_corte_descomp_{nm}_200.json'); c=lambda k: np.array([r['std_con_corte'][k] for r in R])
    print(nm, f"SD {c('dr').std(ddof=1)*100:.4f}", 'trim g:', [f"{c(f'eff_g{q}').mean()*1e5:+.2f}" for q in (99,95,90)], f"neff {c('ess_ratio').mean():.3f}")
print('--- medium overlap replication')
a=L('res_solap_medio_200.json'); b=L('res_solap_medio_rep7000.json'); lim=pred['solap_medio']['lim']
for nm,R in (('orig',a),('rep',b),('all',a+b)):
    print(nm,len(R), f"H z {z(col(R,'hajek'),lim):+.2f} DR z {z(col(R,'dr'),lim):+.2f}")
print('identical ORIGINAL?', L('res_solap_medio_ORIGINAL.json')==a)
print('--- size slope')
n=np.array([3000,5000,10000,15000,30000]); sd=np.array([0.912,0.508,0.294,0.183,0.112])
print('endpoint slope', np.log(sd[0]/sd[-1])/np.log(10), 'OLS', np.polyfit(np.log(n),np.log(sd),1)[0])
print('--- sensitivity coverage')
for nm in ['confunde_sin_C','confusion_U']:
    R=L(f'res_{nm}_200.json'); th=pred[nm]['theta']; d=col(R,'dr'); se=col(R,'dr_se'); lo,hi=d-1.96*se,d+1.96*se
    print(nm, 'theta/theta0', th/pred[nm]['lim'], [ (Lm, np.mean((lo/Lm<=th)&(th<=hi*Lm))*100) for Lm in (1.25,1.5,2,2.5)], 'corr(d,se)', np.corrcoef(d,se)[0,1])
