

from pathlib import Path
from functools import lru_cache

from inferencia import cargar_modelo, predecir_imagen

import gradio as gr


# ==========================================
# 1. CONFIGURACIÓN DE CLASES
# ==========================================

# Las seis categorías originales de CarDD.

CLASES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat"
]


# ==========================================
# 2. PREDICCIÓN SIMULADA
# ==========================================

# Estos valores son ficticios.
# NO provienen de EfficientNet-B0.
# Solo permiten probar la interfaz.

PROBABILIDADES_DEMO = {
    "dent": 0.15,
    "scratch": 0.55,
    "crack": 0.10,
    "glass shatter": 0.08,
    "lamp broken": 0.07,
    "tire flat": 0.05
}


def simular_prediccion(imagen):
    """
    Simula la clasificación de una fotografía.

    Posteriormente reemplazaremos esta lógica
    por la inferencia real de EfficientNet-B0.
    """

    # Validar que existe una imagen.
    if imagen is None:

        return (
            "Debes cargar una fotografía antes de analizar.",
            "",
            "",
            None
        )

    # Obtener las puntuaciones simuladas.
    probabilidades = PROBABILIDADES_DEMO.copy()

    # Identificar la clase con mayor puntuación.
    clase_predicha = max(
        probabilidades,
        key=probabilidades.get
    )

    # Obtener la puntuación de esa clase.
    confianza = probabilidades[clase_predicha]

    # Mensaje de demostración.
    estado = (
        "SIMULACIÓN: análisis completado. "
        "Estos resultados no corresponden "
        "a una predicción real del modelo."
    )

    return (
        estado,
        clase_predicha,
        f"{confianza:.1%}",
        probabilidades
    )


# ==========================================
# CONEXIÓN GRADIO + EFFICIENTNET-B0
# ==========================================

CARPETA_MODELO = Path(__file__).resolve().parent / "model"


@lru_cache(maxsize=1)
def obtener_modelo_real():
    """Carga una sola vez el modelo entrenado."""

    return cargar_modelo(
        CARPETA_MODELO / "model.pt",
        CARPETA_MODELO / "classes.json"
    )


def analizar_fotografia(imagen):

    if imagen is None:
        return simular_prediccion(imagen)

    ruta_pesos = CARPETA_MODELO / "model.pt"
    ruta_clases = CARPETA_MODELO / "classes.json"

    # Todavía no recibimos el modelo de Martín.
    if not ruta_pesos.exists() and not ruta_clases.exists():
        return simular_prediccion(imagen)

    # No permitir una instalación incompleta.
    if not ruta_pesos.is_file() or not ruta_clases.is_file():
        return (
            "ERROR: faltan archivos del modelo entrenado.",
            "",
            "",
            None
        )

    try:
        modelo, clases = obtener_modelo_real()

        resultado = predecir_imagen(
            imagen,
            modelo,
            clases
        )

        return (
            "Análisis completado con EfficientNet-B0.",
            resultado["clase"],
            f'{resultado["confianza"]:.1%}',
            resultado["probabilidades"]
        )

    except Exception as error:
        print("Error de inferencia:", error)

        return (
            "ERROR: no se pudo ejecutar EfficientNet-B0.",
            "",
            "",
            None
        )


# ==========================================
# 3. CONSTRUCCIÓN DE LA INTERFAZ
# ==========================================

with gr.Blocks(title="CarDamage AI") as app:

    # --------------------------------------
    # ENCABEZADO
    # --------------------------------------

    gr.Markdown("# CarDamage AI")

    gr.Markdown(
        "Sistema de clasificación de daños "
        "visibles en vehículos."
    )

    gr.Markdown(
        "**Versión 1.3 — Modo demostración**"
    )

    gr.Markdown(
        "Actualmente utilizamos resultados simulados. "
        "El modelo EfficientNet-B0 todavía no está conectado."
    )

    # --------------------------------------
    # SECCIÓN 1: CARGAR IMAGEN
    # --------------------------------------

    gr.Markdown("### 1. Cargar fotografía")

    imagen = gr.Image(
        label="Fotografía del daño",
        type="pil",
        sources=["upload"],
        height=350
    )

    gr.Markdown(
        "Sube una fotografía cercana de una "
        "zona dañada del vehículo."
    )

    # --------------------------------------
    # SECCIÓN 2: ANALIZAR
    # --------------------------------------

    gr.Markdown("### 2. Analizar fotografía")

    boton_analizar = gr.Button(
        "Analizar daño",
        variant="primary"
    )

    estado = gr.Textbox(
        label="Estado del análisis",
        interactive=False
    )

    # --------------------------------------
    # SECCIÓN 3: RESULTADOS
    # --------------------------------------

    gr.Markdown("### 3. Resultado de clasificación")

    gr.Markdown(
        "**ATENCIÓN: los siguientes resultados "
        "son simulados, no predicciones reales.**"
    )

    with gr.Row():

        clase_resultado = gr.Textbox(
            label="Clase predicha (simulación)",
            interactive=False
        )

        confianza_resultado = gr.Textbox(
            label="Confianza simulada",
            interactive=False
        )

    # --------------------------------------
    # SECCIÓN 4: PUNTUACIONES
    # --------------------------------------

    gr.Markdown(
        "### 4. Puntuaciones de las seis clases"
    )

    probabilidades_resultado = gr.Label(
        label="Distribución simulada",
        num_top_classes=6
    )

    # --------------------------------------
    # CONEXIÓN DEL BOTÓN
    # --------------------------------------

    boton_analizar.click(
        fn=analizar_fotografia,
        inputs=imagen,
        outputs=[
            estado,
            clase_resultado,
            confianza_resultado,
            probabilidades_resultado
        ]
    )

    # --------------------------------------
    # INFORMACIÓN DEL PROYECTO
    # --------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        "### Información del proyecto"
    )

    gr.Markdown(
        "**Dataset:** CarDD — Car Damage Detection\n\n"
        "**Modelo principal:** EfficientNet-B0 "
        "(pendiente de integración)\n\n"
        "**Categorías:** dent, scratch, crack, "
        "glass shatter, lamp broken y tire flat.\n\n"
        "**Limitación:** el sistema supone que "
        "la fotografía contiene una región "
        "con daño visible.\n\n"
        "**Advertencia:** las puntuaciones actuales "
        "son ficticias y no deben utilizarse "
        "para diagnosticar daños."
    )


# ==========================================
# 4. EJECUTAR APLICACIÓN
# ==========================================

if __name__ == "__main__":
    app.launch()
