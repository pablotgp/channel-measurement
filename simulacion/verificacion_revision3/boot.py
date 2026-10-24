import json, numpy as np, os
os.chdir('/home/claude/evaluacion-cambio-canal/simulacion')
th=json.load(open('predicciones.json'))['base']['theta']
def summ(res,name):
    est=np.array([r['est'] for r in res]); se=np.array([r['se'] for r in res]); b=np.array([r['boot'] for r in res]); sb=b.std(axis=1,ddof=1)
    print(f"{name:28s} R={len(res)} SD {est.std(ddof=1)*100:.3f} Wald {se.mean()*100:.3f} ({np.mean(abs(est-th)<=1.96*se)*100:.1f}) boot {sb.mean()*100:.3f} ({np.mean(abs(est-th)<=1.96*sb)*100:.1f}) ratio {sb.mean()/est.std(ddof=1):.3f}")
    return est
P='resultados_pc/'
b80=json.load(open(P+'base/boot_base_parte_local.json')); bb=json.load(open(P+'base_bayes/boot_base_bayes_parte_local.json'))
n80=json.load(open(P+'n_10000/boot_n_10000_parte_local.json')); nb=json.load(open(P+'n_10000_bayes/boot_n_10000_bayes_parte_local.json'))
n40=json.load(open('boot_n_10000_parte1.json'))
e1=summ(b80,'base replacement 80'); e2=summ(b80[:40],'base replacement first40'); e3=summ(bb,'base bayes 40')
print(' same est first40 vs bayes:', np.allclose(e2,e3), np.max(abs(e2-e3)))
f1=summ(n80,'n10000 replacement 80'); f2=summ(n80[:40],'n10000 repl first40'); f3=summ(nb,'n10000 bayes 40'); f4=summ(n40,'n10000 local parte1 40')
print(' same est first40 vs bayes:', np.allclose(f2,f3), np.max(abs(f2-f3)), ' parte1 vs first40', np.allclose(f4,f2))
# order check: match est sets
print(' set match base', len(set(np.round(e3,12))&set(np.round(e1,12))), ' n10000', len(set(np.round(f3,12))&set(np.round(f1,12))))
cb=json.load(open(P+'corte_bayes_base/boot_corte_bayes_base_parte_local.json')); cn=json.load(open(P+'corte_bayes_n_10000/boot_corte_bayes_n_10000_parte_local.json'))
summ(cb,'corte bayes base'); summ(cn,'corte bayes n10000')
for d in ['n_5000','n_15000']:
    R=[json.load(open(P+d+'/'+f)) for f in sorted(os.listdir(P+d)) if f.startswith('rep_')]
    print(d, list(R[0].keys())[:6])
    dr=np.array([r['dr'] for r in R]); se=np.array([r['dr_se'] for r in R])
    print(d, len(R), f"SD {dr.std(ddof=1)*100:.3f} cov {np.mean(abs(dr-th)<=1.96*se)*100:.1f} neff {np.mean([r['ess_ratio'] for r in R]):.3f}")
