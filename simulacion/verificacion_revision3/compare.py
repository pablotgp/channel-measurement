import re
from collections import Counter
base='/home/claude/evaluacion-cambio-canal/paper/'
pairs=[('s0_abstract','s0_resumen'),('s1_introduction','s1_introduccion'),('s2_setup','s2_planteamiento'),('s3_identification','s3_identificacion'),('s4_classifier','s4_clasificador'),('s5_cutoff','s5_corte'),('s6_channel','s6_canal'),('s7_inference','s7_inferencia'),('s8_sensitivity','s8_sensibilidad'),('s9_design','s9_diseno'),('s10_results','s10_resultados'),('s11_discussion','s11_discusion'),('sA_proof','sA_prueba')]
def nums(t):
    t=re.sub(r'\\(label|ref|eqref|cite[pt]?|citealp|input|begin|end)\{[^}]*\}','',t)
    t=re.sub(r'\[[^\]]*Theorem[^\]]*\]|\[[^\]]*Teorema[^\]]*\]','',t)
    t=t.replace('{,}','.').replace('{.}','').replace('\\,',' ')
    t=re.sub(r'(\d)\{\}(\d)',r'\1\2',t)
    t=re.sub(r'(\d),(\d{3})(?!\d)',r'\1\2',t)
    t=re.sub(r'(\d)[ ~](\d{3})(?!\d)',r'\1\2',t)
    return re.findall(r'\d+(?:\.\d+)?',t)
for en,es in pairs:
    a=open(base+'en/'+en+'.tex').read(); b=open(base+'es/'+es+'.tex').read()
    na,nb=nums(a),nums(b)
    ca,cb=Counter(na),Counter(nb)
    d1=ca-cb; d2=cb-ca
    print(f'{en}: en {len(na)} es {len(nb)}', 'only EN:',dict(d1) if d1 else '-', 'only ES:',dict(d2) if d2 else '-')
