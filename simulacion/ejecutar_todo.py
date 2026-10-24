"""Ejecuta el bootstrap pesado en tu ordenador, con un solo comando.

    python ejecutar_todo.py                 # bootstrap de 30 000 clientes (tarea "base")
    python ejecutar_todo.py n_10000         # repetición del bootstrap de 10 000 clientes
    python ejecutar_todo.py base n_10000    # las dos, una detrás de otra
    python ejecutar_todo.py n_10000 base_bayes n_10000_bayes   # bootstrap bayesiano
    python ejecutar_todo.py base --nucleos 6

Qué hace:
  1. Comprueba Python y las librerías; si faltan, las instala desde requirements.txt.
  2. Usa casi todos los núcleos (deja 2 libres para que puedas seguir usando el ordenador).
  3. Guarda cada réplica en su propio fichero en resultados_local/<tarea>/ en cuanto termina.
     Si cierras la ventana, se va la luz o el ordenador se suspende, vuelve a lanzar el mismo
     comando: las réplicas ya hechas se saltan y sigue donde lo dejó.
  4. Al acabar, analiza los resultados y crea resultados_<tarea>.zip para enviármelo.

Solo usa datos simulados: no lee ni escribe nada fuera de esta carpeta.
"""
import os, sys, subprocess

# En Windows la consola no usa UTF-8 por defecto: los acentos y la θ romperían los scripts.
if os.environ.get("PYTHONUTF8") != "1":
    sys.exit(subprocess.call([sys.executable] + sys.argv, env={**os.environ, "PYTHONUTF8": "1"}))

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI); sys.path.insert(0, AQUI)

# Mismo diseño y semillas que las predicciones registradas: no cambiar.
TAREAS = {
    "base":    dict(R=80, B=60, semilla=1000, pred="prediccion_bootstrap_base.json"),
    "n_10000": dict(R=80, B=60, semilla=3000, pred="prediccion_bootstrap.json"),
    "base_bayes": dict(R=40, B=60, semilla=1000, pred="prediccion_bootstrap_bayesiano.json"),
    "n_10000_bayes": dict(R=40, B=60, semilla=3000, pred="prediccion_bootstrap_bayesiano.json"),
}


def comprobar_entorno():
    if sys.version_info < (3, 11):
        sys.exit(f"Necesitas Python 3.11 o posterior (tienes {sys.version.split()[0]}). "
                 "Instálalo desde python.org (recomendado 3.13) y vuelve a lanzar.")
    try:
        import numpy, scipy, sklearn, lightgbm, joblib  # noqa: F401
    except ImportError:
        print("Instalando librerías (solo la primera vez)...", flush=True)
        req = os.path.join(AQUI, "..", "requirements.txt")
        r = subprocess.call([sys.executable, "-m", "pip", "install", "-r", req])
        if r != 0:
            print("Falló la instalación con versiones exactas; pruebo sin fijar versiones.", flush=True)
            r = subprocess.call([sys.executable, "-m", "pip", "install",
                                 "numpy", "scipy", "scikit-learn", "lightgbm", "joblib"])
            if r != 0:
                sys.exit("No se pudieron instalar las librerías. Mándame el mensaje de error.")
    import numpy, scipy, sklearn, lightgbm, joblib
    versiones = {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
                 "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}
    print("Versiones: " + ", ".join(f"{k} {v}" for k, v in versiones.items()))
    return versiones


def una_replica(nombre, seed, B, cs, carpeta):
    """Una réplica completa; escribe su fichero al terminar (así se puede reanudar)."""
    import json
    from bootstrap import one, one_bayes
    if nombre.endswith("_bayes"):
        r = one_bayes(nombre[:-6], seed, B, cs)
    else:
        r = one(nombre, seed, B, cs)
    r["semilla"] = seed
    tmp = os.path.join(carpeta, f"rep_{seed}.json.tmp")
    with open(tmp, "w") as f:
        json.dump(r, f)
    os.replace(tmp, os.path.join(carpeta, f"rep_{seed}.json"))
    return seed


def ejecutar(nombre, nucleos, versiones):
    import json, time, shutil, zipfile, platform
    from joblib import Parallel, delayed
    t = TAREAS[nombre]
    cs = json.load(open("c_star.json"))[nombre.replace("_bayes", "")]
    carpeta = os.path.join("resultados_local", nombre); os.makedirs(carpeta, exist_ok=True)
    semillas = [t["semilla"] + r for r in range(t["R"])]
    hechas = {s for s in semillas if os.path.exists(os.path.join(carpeta, f"rep_{s}.json"))}
    faltan = [s for s in semillas if s not in hechas]
    print(f"\n=== Tarea {nombre}: {t['R']} réplicas × {t['B']} remuestreos (semillas {semillas[0]}-{semillas[-1]})")
    print(f"Ya hechas: {len(hechas)}; faltan: {len(faltan)}; núcleos: {nucleos}", flush=True)
    if faltan:
        t0 = time.time(); n = 0
        gen = Parallel(n_jobs=nucleos, return_as="generator_unordered")(
            delayed(una_replica)(nombre, s, t["B"], cs, carpeta) for s in faltan)
        for s in gen:
            n += 1; dt = time.time() - t0
            resto = dt / n * (len(faltan) - n)
            print(f"  réplica {s} lista ({n}/{len(faltan)}) · llevas {dt/60:.0f} min · "
                  f"quedan unos {resto/60:.0f} min", flush=True)
        # Las primeras réplicas tardan lo mismo que todas las de su tanda, así que la estimación
        # de tiempo restante es pesimista al principio y se ajusta tras la primera tanda.
    # Unir en el formato que espera analyze_bootstrap.py
    res = [json.load(open(os.path.join(carpeta, f"rep_{s}.json"))) for s in semillas]
    union = os.path.join(carpeta, f"boot_{nombre}_parte_local.json")
    json.dump(res, open(union, "w"))
    salida = subprocess.run([sys.executable, "analyze_bootstrap.py", nombre, carpeta],
                            capture_output=True, text=True)
    informe = salida.stdout + salida.stderr
    print("\n" + informe)
    info = dict(versiones, sistema=platform.platform(), procesador=platform.processor(),
                nucleos_usados=nucleos, tarea=nombre, **t)
    json.dump(info, open(os.path.join(carpeta, "entorno.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(carpeta, "analisis.txt"), "w").write(informe)
    zip_path = f"resultados_{nombre}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(carpeta)):
            if not f.endswith(".tmp"):
                z.write(os.path.join(carpeta, f), f"{nombre}/{f}")
    print(f"Listo. Envíame este fichero: {os.path.join(AQUI, zip_path)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    nucleos = None
    if "--nucleos" in args:
        i = args.index("--nucleos"); nucleos = int(args[i + 1]); del args[i:i + 2]
    tareas = args or ["base"]
    for x in tareas:
        if x not in TAREAS:
            sys.exit(f"Tarea desconocida: {x}. Opciones: {', '.join(TAREAS)}")
    versiones = comprobar_entorno()
    if nucleos is None:
        nucleos = max(1, (os.cpu_count() or 2) - 2)
    print(f"Tu ordenador tiene {os.cpu_count()} hilos; uso {nucleos}. "
          "Cada uno necesita unos 200 MB de memoria.")
    print("Consejo: deja el portátil enchufado y desactiva la suspensión mientras trabaja.")
    for x in tareas:
        ejecutar(x, nucleos, versiones)
