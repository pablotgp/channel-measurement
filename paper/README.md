# Paper 1 en LaTeX

- `en/`: versión oficial en inglés (`paper_en.tex`).
- `es/`: la misma versión en español (`paper_es.tex`).
- `preambulo_comun.tex`: preámbulo compartido.

Cada sección está en su propio fichero, con el mismo orden y los mismos números de resultados en las dos versiones; cualquier cambio debe hacerse en ambas.

Compilar con XeLaTeX (dos pasadas para las referencias cruzadas):

```bash
cd paper/en && xelatex paper_en.tex && xelatex paper_en.tex
cd paper/es && xelatex paper_es.tex && xelatex paper_es.tex
```

La estructura sigue `borradores/estructura_propuesta_paper1.md`. Correspondencia de la numeración con el borrador en Claude Docs: Proposición 8 → 2; Proposición 2 → 3; Proposición 3 → 4; Proposición 4 → 5; Proposición 5 → 6; Teorema 5 → 4; Teorema 4 → 5; la antigua Proposición 6 (descomposición de la pérdida) es ahora la Observación 2. Se han quitado del manuscrito el estado del borrador y la lista de pendientes; los problemas abiertos están en la discusión.
