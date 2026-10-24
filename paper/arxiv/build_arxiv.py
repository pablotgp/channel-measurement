"""Construye el paquete para arXiv a partir de paper/en: un único main.tex compatible con pdflatex.

Uso: python paper/arxiv/build_arxiv.py  -> paper/arxiv/main.tex, main.pdf y arxiv_submission.tar.gz
"""
import os, re, subprocess, tarfile
HERE = os.path.dirname(os.path.abspath(__file__))
EN = os.path.join(HERE, "..", "en")

def flatten(path, base):
    out = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*\\input\{([^}]+)\}", line)
        if m:
            f = m.group(1)
            f = f if f.endswith(".tex") else f + ".tex"
            p = os.path.normpath(os.path.join(base, f))
            out.append(flatten(p, os.path.dirname(p)))
        else:
            out.append(line)
    return "".join(out)

tex = flatten(os.path.join(EN, "paper_en.tex"), EN)
# fontspec (XeLaTeX) -> pdflatex: Palatino vía mathpazo con codificación T1
tex = re.sub(r"\\usepackage\[no-math\]\{fontspec\}\n\\setmainfont\{[^\n]*\n[^\n]*\n", "\\\\usepackage[T1]{fontenc}\n\\\\usepackage[utf8]{inputenc}\n\\\\renewcommand{\\\\ttdefault}{pcr}\n", tex)
assert "fontspec" not in tex and "setmainfont" not in tex
tex = re.sub(r"(?m)^%.*\n", "", tex)  # quita líneas de comentario
open(os.path.join(HERE, "main.tex"), "w", encoding="utf-8").write(tex)
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=HERE, stdout=subprocess.DEVNULL)
log = open(os.path.join(HERE, "main.log"), encoding="latin-1").read()
print("errores:", log.count("\n!"), "| indefinidas:", log.count("undefined"), "| overfull:", log.count("Overfull"))
with tarfile.open(os.path.join(HERE, "arxiv_submission.tar.gz"), "w:gz") as t:
    t.add(os.path.join(HERE, "main.tex"), arcname="main.tex")
print("paquete:", os.path.join(HERE, "arxiv_submission.tar.gz"))
