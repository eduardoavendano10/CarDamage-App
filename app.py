"""cardamage.ia — aplicación Gradio de clasificación de daños vehiculares.

Interfaz cartoon retro compacta. Mantiene inferencia.py (EfficientNet-B0)
y recomendaciones.py (Gemini). No necesita un archivo ui.html.
"""

from functools import lru_cache
from html import escape
import logging
from pathlib import Path

import gradio as gr

from inferencia import cargar_modelo, predecir_imagen
from recomendaciones import generar_recomendacion


BASE_DIR = Path(__file__).resolve().parent
RUTA_PESOS = BASE_DIR / "model" / "model.pt"
RUTA_CLASES = BASE_DIR / "model" / "classes.json"

ETIQUETAS = {
    "crack": "Grieta",
    "dent": "Abolladura",
    "glass_shatter": "Vidrio roto",
    "lamp_broken": "Foco dañado",
    "scratch": "Rayadura",
    "tire_flat": "Neumático desinflado",
}

HEADER_HTML = """
<header class="cd-hero">
  <div>
    <h1>cardamage<span>.ia</span></h1>
    <p>Una foto cercana. Seis tipos de daño. Orientación inteligente.</p>
  </div>
</header>
"""

# Estética caricaturesca con una sola sombra por tarjeta.
# Sin sol, efectos de fondo, contenedores grises ni decoración repetida.
CUSTOM_CSS = r"""
:root {
    --paper: #faf1e2;
    --surface: #fffaf2;
    --ink: #272321;
    --red: #e44e3d;
    --yellow: #ffd558;
    --blue: #89cce2;
    --muted: #655f58;
}
body, .gradio-container {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: "Trebuchet MS", "Segoe UI", sans-serif !important;
}
.gradio-container {
    max-width: 1160px !important;
    padding: 18px 18px 30px !important;
}
.gradio-container > footer { display: none !important; }

.cd-hero {
    background: var(--red);
    color: #fff8ef;
    border: 3px solid var(--ink);
    border-radius: 21px;
    box-shadow: 6px 6px 0 var(--ink);
    padding: 19px 27px;
    margin: 0 6px 19px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 136px;
}
.cd-hero h1 {
    color: #fff8ef !important;
    margin: 0 0 5px !important;
    font-size: clamp(2.7rem, 6vw, 4.8rem);
    letter-spacing: -3px;
    font-weight: 1000;
    line-height: 1.05;
    text-shadow: 4px 4px 0 var(--ink);
}
.cd-hero h1 span { color: var(--yellow); }
.cd-hero p {
    color: #fff8ef;
    font-size: .96rem;
    font-weight: 700;
    margin: 0;
}
.cd-panel {
    border: 3px solid var(--ink) !important;
    background: var(--surface) !important;
    border-radius: 20px !important;
    box-shadow: 5px 5px 0 var(--ink) !important;
    padding: 19px !important;
    margin: 0 6px 10px 0 !important;
    gap: 9px !important;
}
.cd-panel .block { box-shadow: none !important; }
.cd-heading {
    display: inline-block;
    background: var(--yellow);
    border: 2px solid var(--ink);
    border-radius: 9px;
    box-shadow: 3px 3px 0 var(--ink);
    padding: 5px 11px;
    font-size: .9rem;
    font-weight: 1000;
    color: var(--ink);
    margin-bottom: 8px;
}
.cd-heading.blue { background: var(--blue); }
.cd-helper { color: var(--muted); font-size: .87rem; margin: 0 0 8px; }
#cd-upload {
    background: #fff4df !important;
    border: 2px dashed #8d775d !important;
    border-radius: 14px !important;
}
#cd-submit, #cd-reset {
    border: 3px solid var(--ink) !important;
    border-radius: 12px !important;
    box-shadow: 3px 3px 0 var(--ink) !important;
    font-weight: 900 !important;
    min-height: 48px;
}
#cd-submit { background: var(--red) !important; color: white !important; }
#cd-reset { background: var(--yellow) !important; color: var(--ink) !important; }
#cd-submit:hover, #cd-reset:hover { transform: translate(-1px, -1px); }
.cd-state {
    display: inline-flex;
    gap: 7px;
    align-items: center;
    background: var(--yellow);
    border: 2px solid var(--ink);
    border-radius: 30px;
    padding: 5px 12px;
    color: var(--ink);
    font-size: .82rem;
    font-weight: 900;
}
.cd-state.ok { background: #bfe6b7; }
.cd-state.error { background: #ffb9ad; }
.cd-result {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 13px;
    margin-top: 16px;
}
.cd-result-item {
    border: 2px solid #e9ddcd;
    background: #fffcf7;
    border-radius: 12px;
    padding: 14px;
    min-height: 104px;
}
.cd-result-item small {
    font-size: .79rem;
    color: var(--muted);
    display: block;
    font-weight: 700;
    margin-bottom: 8px;
}
.cd-result-item strong {
    color: var(--ink);
    font-size: clamp(1.2rem, 2.3vw, 1.8rem);
    overflow-wrap: anywhere;
    line-height: 1.2;
}
.cd-result-item strong.cd-accent { color: #b43b2d; }
.cd-inline-note {
    color: var(--muted);
    margin-top: 11px;
    font-size: .82rem;
    line-height: 1.5;
}
.cd-ai { padding-top: 2px; }
.cd-ai p { line-height: 1.6; }
.cd-mini-title {
    margin: 13px 0 9px;
    font-weight: 900;
    font-size: .85rem;
    color: var(--ink);
}
.cd-disclaimer {
    margin: 12px 4px 0;
    color: var(--muted);
    font-size: .76rem;
    line-height: 1.55;
    text-align: center;
}
@media (max-width: 760px) {
    .gradio-container { padding: 12px 10px 20px !important; }
    .cd-hero { padding: 19px 16px; min-height: auto; }
    .cd-hero h1 { letter-spacing: -2px; }
    .cd-panel { padding: 12px !important; }
    .cd-result { gap: 7px; }
    .cd-result-item { padding: 10px; min-height: 94px; }
}
"""


def estado_modelo():
    """No presentar una simulación si hay una instalación incompleta."""
    pesos = RUTA_PESOS.is_file()
    clases = RUTA_CLASES.is_file()
    if pesos and clases:
        return "real"
    if pesos or clases:
        return "incompleto"
    return "demo"


@lru_cache(maxsize=1)
def obtener_modelo():
    """Si se reemplaza model.pt, reiniciar la aplicación."""
    return cargar_modelo(ruta_pesos=RUTA_PESOS, ruta_clases=RUTA_CLASES)


def tarjeta_resultado(estado, clase="—", puntuacion="—", nota="", tipo="pending"):
    """Construye una salida compacta y escapa datos del modelo."""
    clase, puntuacion, estado, nota = [escape(str(v)) for v in (clase, puntuacion, estado, nota)]
    clase_css = "cd-accent" if tipo == "ok" else ""
    return f"""
    <div class="cd-output">
      <span class="cd-state {tipo}">{estado}</span>
      <div class="cd-result">
        <div class="cd-result-item">
          <small>Tipo de daño</small>
          <strong class="{clase_css}">{clase}</strong>
        </div>
        <div class="cd-result-item">
          <small>Puntuación Softmax</small>
          <strong>{puntuacion}</strong>
        </div>
      </div>
      {'<p class="cd-inline-note">' + nota + '</p>' if nota else ''}
    </div>
    """


RESULTADO_INICIAL = tarjeta_resultado("En espera")
INFORME_INICIAL = "Selecciona una fotografía y presiona **Analizar daño**."


def analizar(imagen):
    """Devuelve (tarjeta de resultado, orientación de IA)."""
    if imagen is None:
        return (tarjeta_resultado("Falta fotografía", nota="Carga una imagen para continuar."), INFORME_INICIAL)

    modo = estado_modelo()
    if modo == "demo":
        return (
            tarjeta_resultado(
                "Sin modelo entrenado",
                clase="Pendiente",
                nota="Demostración de interfaz: no se ha clasificado esta fotografía.",
            ),
            "La interfaz funciona. **La clasificación real y Gemini estarán disponibles "
            "cuando se instalen los archivos del modelo de Martín.**",
        )

    if modo == "incompleto":
        return (
            tarjeta_resultado(
                "Modelo incompleto",
                clase="No disponible",
                nota="Se necesitan model/model.pt y model/classes.json.",
                tipo="error",
            ),
            "No se ejecutó ninguna clasificación ni se consultó a Gemini.",
        )

    try:
        modelo, clases = obtener_modelo()
        resultado = predecir_imagen(imagen, modelo, clases)
        clase = resultado["clase"]
        score = float(resultado["confianza"])
        puntuaciones = resultado["probabilidades"]
    except Exception:
        logging.exception("Falló la inferencia EfficientNet-B0")
        return (
            tarjeta_resultado(
                "Error de inferencia",
                clase="No disponible",
                nota="Revisa la terminal. No se generaron datos simulados.",
                tipo="error",
            ),
            "No se consultó a Gemini porque falló la clasificación.",
        )

    try:
        recomendacion = generar_recomendacion(
            clase=clase,
            confianza=score,
            probabilidades=puntuaciones,
            modo_prueba=False,
        )
    except Exception:
        logging.exception("Falló la recomendación de Gemini")
        recomendacion = (
            "Clasificación completada, pero no se pudo generar la orientación. "
            "Comprueba `GEMINI_API_KEY` y revisa la terminal."
        )

    return (
        tarjeta_resultado(
            "Modelo activo",
            clase=ETIQUETAS.get(clase, clase),
            puntuacion=f"{score:.1%}",
            nota="La puntuación Softmax no equivale a precisión global.",
            tipo="ok",
        ),
        recomendacion,
    )


def limpiar():
    return None, RESULTADO_INICIAL, INFORME_INICIAL


with gr.Blocks(title="cardamage.ia") as app:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, min_width=320, elem_classes=["cd-panel"]):
            gr.HTML('<div class="cd-heading">01 / Fotografía</div>')
            gr.HTML('<p class="cd-helper">Carga un acercamiento de la zona dañada.</p>')
            imagen = gr.Image(
                label="Fotografía del daño",
                sources=["upload"],
                type="pil",
                height=300,
                elem_id="cd-upload",
            )
            with gr.Row():
                boton_analizar = gr.Button("⚡ Analizar daño", variant="primary", elem_id="cd-submit")
                boton_limpiar = gr.Button("↺ Limpiar", variant="secondary", elem_id="cd-reset")

            rutas_ejemplos = [
                BASE_DIR / "ejemplos" / "dent.jpg.jpg",
                BASE_DIR / "ejemplos" / "glass_shatter.jpg.jpg",
                BASE_DIR / "ejemplos" / "lamp_broken.jpg.jpg",
                BASE_DIR / "ejemplos" / "tire_flat.jpg.jpg",
            ]
            rutas_ejemplos = [str(r) for r in rutas_ejemplos if r.is_file()]
            if rutas_ejemplos:
                gr.Examples(
                    examples=[[ruta] for ruta in rutas_ejemplos],
                    inputs=[imagen],
                    label="Ejemplos",
                )

        with gr.Column(scale=5, min_width=320, elem_classes=["cd-panel"]):
            gr.HTML('<div class="cd-heading blue">02 / Resultado</div>')
            resultado = gr.HTML(value=RESULTADO_INICIAL)
            gr.HTML('<div class="cd-mini-title">Orientación inteligente</div>')
            informe = gr.Markdown(INFORME_INICIAL, elem_classes=["cd-ai"])

    gr.HTML(
        '<div class="cd-disclaimer">Prototipo académico. Clasifica recortes de daños visibles; '
        'no localiza daños en imágenes completas ni sustituye una inspección profesional. '
        'No reconoce vehículos sin daños.</div>'
    )

    boton_analizar.click(fn=analizar, inputs=[imagen], outputs=[resultado, informe])
    boton_limpiar.click(fn=limpiar, inputs=[], outputs=[imagen, resultado, informe])


if __name__ == "__main__":
    # En Gradio 6 el CSS personalizado se entrega al lanzar la aplicación.
    app.launch(inbrowser=True, share=False, footer_links=[], css=CUSTOM_CSS)
