"""Construye el preprint (Zenodo, versión 2) con el estilo de Observational Studies a partir de paper/en.

Uso: python paper/preprint/build_preprint.py  -> paper/preprint/main.tex y main.pdf
Cabecera neutra (preprint), sin nombre de la revista: el estilo solo se usa por su aspecto.

paper/en sigue siendo la única fuente del contenido. Este script solo cambia la forma:
- plantilla de la revista (obs_study_style.sty, sin cambios), fuente Computer Modern y pdflatex;
- numeración propia de cada tipo de resultado (Teorema 1-5, Proposición 1-7...), como en en/;
- palabras clave tras el resumen;
- «Code and data» y «AI assistance» salen de la discusión y pasan, con las demás
  declaraciones (declaraciones.tex), a una sección sin número antes de los apéndices;
- bibliografía en orden alfabético, con la lista completa de autores en la primera cita
  para obras de 3 a 6 autores (natbib, longnamesfirst), como piden las normas;
- citas múltiples ordenadas por apellido y año.
"""
import os, re, subprocess, unicodedata, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
EN = os.path.join(HERE, "..", "en")
STY = "obs_study_style.sty"


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


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


tex = flatten(os.path.join(EN, "paper_en.tex"), EN)

# --- Preámbulo -------------------------------------------------------------
body = tex[tex.index("\\begin{document}"):]
pre = open(os.path.join(HERE, "..", "preambulo_comun.tex"), encoding="utf-8").read()
notacion = pre[pre.index("% Notación"):]
teoremas = re.search(r"\\theoremstyle\{plain\}.*?\\newtheorem\{remark\}\{Remark\}\n", tex, re.S).group(0)

preamble = r"""\documentclass[twoside,11pt]{article}
\PassOptionsToPackage{longnamesfirst}{natbib}
\usepackage{obs_study_style}

% The journal style numbers all results with one counter and defines proof;
% these environments are released so that each type keeps its own numbering and amsthm is used.
\makeatletter
\let\proof\@undefined \let\endproof\@undefined
\@for\obs@env:=theorem,lemma,proposition,remark,corollary,definition,conjecture,axiom,example\do{%
  \expandafter\let\csname\obs@env\endcsname\@undefined
  \expandafter\let\csname end\obs@env\endcsname\@undefined
  \expandafter\let\csname c@\obs@env\endcsname\@undefined
  \expandafter\let\csname the\obs@env\endcsname\@undefined}
\makeatother

\usepackage{amsmath,amsthm,mathtools}
\usepackage[utf8]{inputenc}
\usepackage{booktabs,array}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\setlist{itemsep=0.2em,topsep=0.3em}
% Computer Modern glyphs (avoids bitmap TS1 fonts when cm-super is not installed)
\renewcommand{\labelitemi}{$\bullet$}
\DeclareRobustCommand{\copyright}{{\ooalign{\hfil\raise.07ex\hbox{c}\hfil\crcr\mathhexbox20D}}}
\setlength{\emergencystretch}{3em}

""" + notacion + "\n" + teoremas + r"""
% Preprint: neutral first-page header and footer (the style is used only for its layout)
\makeatletter
\def\ps@jmlrtps{\let\@mkboth\@gobbletwo
\def\@oddhead{\normalsize Preprint, version 2 \hfill October 2026}%
\def\@oddfoot{\scriptsize \copyright\ 2026 Pablo Antonio Garc\'ia Pastor. \hfill}%
\def\@evenhead{}\def\@evenfoot{}}
\makeatother
\AtBeginDocument{\thispagestyle{jmlrtps}}
\ShortHeadings{Invisible bias: when channels measure covariates differently}{Garc\'ia Pastor}
\firstpageno{1}

"""

title = r"""\title{Invisible bias: when channels measure covariates differently}

\author{\name Pablo Antonio Garc\'ia Pastor \email pablotgp2002@gmail.com \\
       \addr MSc student, ICADE, Universidad Pontificia Comillas\\
       Madrid, Spain}

\maketitle
"""
body = re.sub(r"\\begin\{document\}\n\\maketitle\n", lambda m: "\\begin{document}\n\n" + title, body, count=1)
assert "\\name Pablo" in body

# --- Palabras clave ----------------------------------------------------------
claves = open(os.path.join(HERE, "palabras_clave.tex"), encoding="utf-8").read().strip()
body = body.replace("\\end{abstract}\n", "\\end{abstract}\n\n\\begin{keywords}\n" + claves + "\n\\end{keywords}\n", 1)

# --- Declaraciones -------------------------------------------------------------
for par in ("Code and data", "AI assistance"):
    pat = r"\\paragraph\{" + par + r"\.\}.*?\n"
    assert len(re.findall(pat, body)) == 1, par
    body = re.sub(pat, "", body)
decl = open(os.path.join(HERE, "declaraciones.tex"), encoding="utf-8").read()
body = body.replace("\\appendix\n", decl.rstrip() + "\n\n\\newpage\n\\appendix\n", 1)

# --- Demostraciones al apéndice ---------------------------------------------------------
# Las normas piden llevar las demostraciones a un apéndice cuando cortan el hilo, y el límite
# de 25 páginas no cuenta los apéndices. Cada demostración sigue a su resultado; se mueve,
# en el mismo orden, a un apéndice nuevo con el título «Proof of <Resultado> n».
PRUEBAS_AL_APENDICE = True
NOMBRES = {"theorem": "Theorem", "proposition": "Proposition", "lemma": "Lemma", "corollary": "Corollary"}
if PRUEBAS_AL_APENDICE:
    ini, fin = body.index("\\section{Introduction}"), body.index("\\section*{Statements")
    texto = body[ini:fin]
    apendice = ["\\section{Proofs of the results in the main text}\\label{app:proofs}\n"]
    pos = 0
    for mp in list(re.finditer(r"\\begin\{proof\}.*?\\end\{proof\}\n?", texto, re.S)):
        previos = list(re.finditer(r"\\begin\{(theorem|proposition|lemma|corollary)\}(?:\[[^\]]*\])?\s*\\label\{([^}]+)\}",
                                   texto[:mp.start()]))
        tipo, etiqueta = previos[-1].group(1), previos[-1].group(2)
        # entre el resultado y su demostración solo puede haber un comentario breve, que se queda
        entre = texto[previos[-1].end():mp.start()]
        assert len(entre.split("\\end{" + tipo + "}", 1)[1]) < 300, (etiqueta, entre[-200:])
        apendice.append("\\paragraph{Proof of " + NOMBRES[tipo] + "~\\ref{" + etiqueta + "}.}\n"
                        + re.sub(r"^\\begin\{proof\}\s*", "", mp.group(0)).replace("\\end{proof}", "\\hfill$\\square$") + "\n")
    texto = re.sub(r"\\begin\{proof\}.*?\\end\{proof\}\n?", "", texto, flags=re.S)
    aviso = "Section~\\ref{sec:discussion} concludes."
    assert texto.count(aviso) == 1
    texto = texto.replace(aviso, aviso + " Proofs are in the appendices.")
    body = body[:ini] + texto + body[fin:]
    body = body.replace("\\newpage\n\\appendix\n", "\\newpage\n\\appendix\n" + "\n".join(apendice) + "\n", 1)

# --- Bibliografía ------------------------------------------------------------------
m = re.search(r"(\\begin\{thebibliography\}\{99\}\\small\n)(.*?)(\\end\{thebibliography\})", body, re.S)
items = [it.strip() for it in re.split(r"\n(?=\\bibitem)", m.group(2).strip()) if it.strip()]
INI = re.compile(r"^(?:[A-Z][a-z]?\.(?:-[A-Z]\.)?\s?)+$")
entradas = []
for it in items:
    b = re.match(r"\\bibitem\[(.*?)\((\d{4})\)\]\{([^}]+)\}\s*(.*?)\s\(\2\)", it, re.S)
    assert b, it[:60]
    corto, anio, clave, autores = b.groups()
    tokens = [t.strip() for t in re.split(r",\s|\s(?:and)\s", autores) if t.strip()]
    apellidos = [t for t in tokens if not INI.match(t) and t not in ("Jr.",)]
    largo = ""
    if 3 <= len(apellidos) <= 6:
        largo = ", ".join(apellidos[:-1]) + ", and " + apellidos[-1]
        largo = largo.replace("~", " ")
    nuevo = re.sub(r"^\\bibitem\[.*?\]", lambda _: "\\bibitem[" + corto + "(" + anio + ")" + largo + "]", it, count=1)
    entradas.append((norm(apellidos[0]) if apellidos else norm(corto), anio, clave, apellidos, nuevo))
entradas.sort(key=lambda e: (e[0], e[1]))
orden = {e[2]: (e[0], e[1]) for e in entradas}
body = body[:m.start(2)] + "\n\n".join(e[4] for e in entradas) + "\n\n" + body[m.end(2):]

# Citas múltiples por apellido y año
def ordenar(mc):
    claves_c = [k.strip() for k in mc.group(2).split(",")]
    return mc.group(1) + "{" + ",".join(sorted(claves_c, key=lambda k: orden[k])) + "}"
body = re.sub(r"(\\cite[a-z]*\*?(?:\[[^\]]*\])*)\{([^}]*,[^}]*)\}", ordenar, body)

body = re.sub(r"(?m)^%.*\n", "", body)  # quita líneas de comentario del texto
tex = preamble + body
assert "fontspec" not in tex and "geometry" not in tex
open(os.path.join(HERE, "main.tex"), "w", encoding="utf-8").write(tex)

for _ in range(3):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=HERE, stdout=subprocess.DEVNULL)
log = open(os.path.join(HERE, "main.log"), encoding="latin-1").read()
print("errores:", log.count("\n!"), "| indefinidas:", len(re.findall(r"undefined", log)),
      "| overfull:", log.count("Overfull"))
for marca in ("PENDIENTE",):
    if marca in tex:
        print("AVISO: quedan marcas", marca)

print("pdf:", os.path.join(HERE, "main.pdf"))
