
# ==========================================================
# CarDamage AI - Version 1.5
# Gradio + EfficientNet-B0 + Gemini
# ==========================================================

from functools import lru_cache
from pathlib import Path

import gradio as gr

from inferencia import cargar_modelo, predecir_imagen
from recomendaciones import generar_recomendacion


# ==========================================================
# 1. CONFIGURACION
# ==========================================================

VERSION = "1.5"

BASE_DIR = Path(__file__).resolve().parent

# Ruta acordada para recibir el modelo de Martin.
CARPETA_MODELO = BASE_DIR / "model"

RUTA_PESOS = CARPETA_MODELO / "model.pt"
RUTA_CLASES = CARPETA_MODELO / "classes.json"


# ==========================================================
# 2. DATOS DE DEMOSTRACION
# ==========================================================

# Resultados ficticios.
# No provienen del modelo entrenado.

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
    Devuelve resultados ficticios para comprobar
    el funcionamiento de la interfaz.
    """

    if imagen is None:
        return (
            "Debes cargar una fotografia antes de analizar.",
            "",
            "",
            None
        )

    probabilidades = PROBABILIDADES_DEMO.copy()

    clase = max(
        probabilidades,
        key=probabilidades.get
    )

    confianza = probabilidades[clase]

    return (
        "SIMULACION: analisis completado. "
        "Los resultados son ficticios.",
        clase,
        f"{confianza:.1%}",
        probabilidades
    )


# ==========================================================
# 3. VERIFICACION DEL MODELO
# ==========================================================

def verificar_archivos_modelo():
    """
    Determina si existe el modelo entrenado.

    Posibles resultados:
    - simulacion
    - incompleto
    - disponible
    """

    tiene_pesos = RUTA_PESOS.is_file()
    tiene_clases = RUTA_CLASES.is_file()

    if tiene_pesos and tiene_clases:
        return "disponible"

    if tiene_pesos or tiene_clases:
        return "incompleto"

    return "simulacion"


# ==========================================================
# 4. CARGA DE EFFICIENTNET-B0
# ==========================================================

@lru_cache(maxsize=1)
def obtener_modelo_real():
    """
    Carga el modelo una sola vez.

    Cuando se reemplacen los pesos,
    se debe reiniciar Gradio.
    """

    return cargar_modelo(
        ruta_pesos=RUTA_PESOS,
        ruta_clases=RUTA_CLASES
    )


# ==========================================================
# 5. CLASIFICACION DE LA IMAGEN
# ==========================================================

def analizar_fotografia(imagen):
    """
    Ejecuta simulacion o inferencia real.

    Devuelve cinco resultados:
    1. Estado
    2. Clase
    3. Confianza
    4. Puntuaciones
    5. Informacion del modo
    """

    if imagen is None:
        return (
            "Debes cargar una fotografia antes de analizar.",
            "",
            "",
            None,
            "**Estado:** esperando una fotografia."
        )

    modo = verificar_archivos_modelo()

    # ------------------------------------------------------
    # SIMULACION
    # ------------------------------------------------------

    if modo == "simulacion":

        estado, clase, confianza, probabilidades = (
            simular_prediccion(imagen)
        )

        return (
            estado,
            clase,
            confianza,
            probabilidades,
            "**Modo activo: SIMULACION.** "
            "Las puntuaciones son ficticias "
            "y no dependen de la fotografia."
        )

    # ------------------------------------------------------
    # ARCHIVOS INCOMPLETOS
    # ------------------------------------------------------

    if modo == "incompleto":
        return (
            "ERROR: instalacion incompleta del modelo.",
            "",
            "",
            None,
            "**Modelo incompleto:** deben existir "
            "model/model.pt y model/classes.json."
        )

    # ------------------------------------------------------
    # INFERENCIA EFFICIENTNET-B0
    # ------------------------------------------------------

    try:

        modelo, clases = obtener_modelo_real()

        resultado = predecir_imagen(
            imagen,
            modelo,
            clases
        )

        return (
            "Analisis completado con EfficientNet-B0 "
            "y pesos cargados.",
            resultado["clase"],
            f'{resultado["confianza"]:.1%}',
            resultado["probabilidades"],
            "**Modo activo: EFFICIENTNET-B0.** "
            "Las puntuaciones Softmax no equivalen "
            "a la precision global del modelo."
        )

    except Exception as error:

        print("ERROR DE INFERENCIA:", repr(error))

        return (
            "ERROR: no se pudo ejecutar EfficientNet-B0.",
            "",
            "",
            None,
            "**Error del modelo:** revisa la terminal. "
            "No se sustituyen errores por simulaciones."
        )


# ==========================================================
# 6. CLASIFICACION + RECOMENDACION CON GEMINI
# ==========================================================

def analizar_con_recomendacion(imagen):
    """
    Combina los resultados del clasificador
    con una recomendacion orientativa.

    Gemini solamente se consulta cuando existe
    una clasificacion obtenida del modelo cargado.
    """

    resultado = analizar_fotografia(imagen)

    estado, clase, confianza, probabilidades, informacion = (
        resultado
    )

    # ------------------------------------------------------
    # SIN IMAGEN
    # ------------------------------------------------------

    if imagen is None:

        recomendacion = (
            "Carga una fotografia para iniciar el analisis."
        )

    # ------------------------------------------------------
    # MODO SIMULACION
    # ------------------------------------------------------

    elif verificar_archivos_modelo() == "simulacion":

        recomendacion = (
            "**MODO SIMULACION:** "
            "La fotografia aun no ha sido clasificada "
            "por un modelo entrenado. "
            "Los resultados son ficticios. "
            "No se consulto a Gemini."
        )

    # ------------------------------------------------------
    # ERROR DE CLASIFICACION
    # ------------------------------------------------------

    elif not clase or not isinstance(probabilidades, dict):

        recomendacion = (
            "No es posible generar una recomendacion "
            "porque no se obtuvo una clasificacion valida."
        )

    # ------------------------------------------------------
    # RESULTADO DEL MODELO ENTRENADO
    # ------------------------------------------------------

    else:

        try:

            confianza_numerica = probabilidades[clase]

            recomendacion = generar_recomendacion(
                clase=clase,
                confianza=confianza_numerica,
                probabilidades=probabilidades,
                modo_prueba=False
            )

        except Exception as error:

            print(
                "ERROR EN RECOMENDACIONES:",
                repr(error)
            )

            recomendacion = (
                "No fue posible obtener una recomendacion "
                "de IA en este momento. "
                "Revisa la conexion o intenta nuevamente."
            )

    # Cinco resultados anteriores + recomendacion.
    return (*resultado, recomendacion)


# ==========================================================
# 7. INTERFAZ GRADIO
# ==========================================================

with gr.Blocks(title="CarDamage AI") as app:

    # ------------------------------------------------------
    # ENCABEZADO
    # ------------------------------------------------------

    gr.Markdown("# CarDamage AI")

    gr.Markdown(
        "Sistema de clasificacion de danos "
        "visibles en vehiculos."
    )

    gr.Markdown(
        f"**Version {VERSION} — "
        "EfficientNet-B0 + Gemini + Gradio**"
    )

    gr.Markdown(
        "La aplicacion trabaja en simulacion mientras "
        "no existan los archivos del modelo entrenado. "
        "El estado de cada analisis indica el modo utilizado."
    )

    # ------------------------------------------------------
    # SECCION 1 - FOTOGRAFIA
    # ------------------------------------------------------

    gr.Markdown("### 1. Cargar fotografia")

    imagen = gr.Image(
        label="Fotografia del dano",
        type="pil",
        sources=["upload"],
        height=350
    )

    gr.Markdown(
        "Sube una fotografia cercana de una zona "
        "con dano visible. "
        "El sistema clasifica la region danada; "
        "no localiza automaticamente danos "
        "en una fotografia completa del vehiculo."
    )

    # ------------------------------------------------------
    # SECCION 2 - ANALISIS
    # ------------------------------------------------------

    gr.Markdown("### 2. Analizar fotografia")

    boton_analizar = gr.Button(
        "Analizar dano",
        variant="primary"
    )

    estado = gr.Textbox(
        label="Estado del analisis",
        interactive=False
    )

    # ------------------------------------------------------
    # SECCION 3 - CLASIFICACION
    # ------------------------------------------------------

    gr.Markdown("### 3. Resultado de clasificacion")

    informacion_modo = gr.Markdown(
        "**Estado:** esperando una fotografia."
    )

    with gr.Row():

        clase_resultado = gr.Textbox(
            label="Clase predicha",
            interactive=False
        )

        confianza_resultado = gr.Textbox(
            label="Confianza de la prediccion",
            interactive=False
        )

    # ------------------------------------------------------
    # SECCION 4 - PUNTUACIONES
    # ------------------------------------------------------

    gr.Markdown(
        "### 4. Puntuaciones por categoria"
    )

    probabilidades_resultado = gr.Label(
        label="Distribucion de puntuaciones",
        num_top_classes=6
    )

    # ------------------------------------------------------
    # SECCION 5 - RECOMENDACIONES
    # ------------------------------------------------------

    gr.Markdown(
        "### 5. Recomendacion orientativa con IA"
    )

    recomendacion_resultado = gr.Markdown(
        "Carga una fotografia y ejecuta el analisis "
        "para consultar una recomendacion orientativa."
    )

    # ------------------------------------------------------
    # CONEXION DEL BOTON
    # ------------------------------------------------------

    boton_analizar.click(
        fn=analizar_con_recomendacion,
        inputs=[imagen],
        outputs=[
            estado,
            clase_resultado,
            confianza_resultado,
            probabilidades_resultado,
            informacion_modo,
            recomendacion_resultado
        ]
    )

    # ------------------------------------------------------
    # INFORMACION DEL PROYECTO
    # ------------------------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        "### Informacion del proyecto"
    )

    gr.Markdown(
        "**Proyecto:** CarDamage AI\n\n"
        "**Dataset:** CarDD - Car Damage Detection\n\n"
        "**Modelo principal:** EfficientNet-B0 "
        "(PyTorch / Torchvision)\n\n"
        "**IA generativa:** Google Gemini, "
        "para recomendaciones orientativas\n\n"
        "**Interfaz:** Gradio\n\n"
        "**Categorias iniciales:** dent, scratch, crack, "
        "glass shatter, lamp broken y tire flat.\n\n"
        "**Clases del modelo:** se obtienen desde "
        "classes.json cuando existen pesos disponibles.\n\n"
        "**Limitacion:** requiere una fotografia de "
        "una region con dano visible. "
        "No existe una clase sin dano.\n\n"
        "**Interpretacion:** las puntuaciones Softmax "
        "no garantizan una prediccion correcta ni "
        "representan la precision global en Test.\n\n"
        "**Advertencia:** herramienta academica, "
        "no diagnostico mecanico definitivo."
    )


# ==========================================================
# 8. EJECUTAR
# ==========================================================

if __name__ == "__main__":
    app.launch()
