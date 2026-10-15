
# ==========================================================
# CARDAMAGE AI - VERSION 1.4
# Clasificacion de danos visibles en vehiculos
# Gradio + EfficientNet-B0
# ==========================================================

import gradio as gr

from pathlib import Path
from functools import lru_cache

from inferencia import cargar_modelo, predecir_imagen


# ==========================================================
# 1. CONFIGURACION
# ==========================================================

VERSION = "1.4"

BASE_DIR = Path(__file__).resolve().parent
CARPETA_MODELO = BASE_DIR / "model"

RUTA_PESOS = CARPETA_MODELO / "model.pt"
RUTA_CLASES = CARPETA_MODELO / "classes.json"


# ==========================================================
# 2. DATOS DE DEMOSTRACION
# ==========================================================

# Estos valores son ficticios.
# Solo se utilizan cuando no existe un modelo instalado.

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
    Genera resultados ficticios para probar la interfaz.
    NO representa una prediccion de EfficientNet-B0.
    """

    if imagen is None:
        return (
            "Debes cargar una fotografia antes de analizar.",
            "",
            "",
            None
        )

    probabilidades = PROBABILIDADES_DEMO.copy()

    clase_predicha = max(
        probabilidades,
        key=probabilidades.get
    )

    confianza = probabilidades[clase_predicha]

    estado = (
        "SIMULACION: analisis completado. "
        "Los resultados son ficticios."
    )

    return (
        estado,
        clase_predicha,
        f"{confianza:.1%}",
        probabilidades
    )


# ==========================================================
# 3. DETECCION DEL MODELO
# ==========================================================

def verificar_archivos_modelo():
    """
    Comprueba si existen los archivos necesarios.

    Devuelve:
    - "disponible"
    - "incompleto"
    - "simulacion"
    """

    existe_modelo = RUTA_PESOS.is_file()
    existen_clases = RUTA_CLASES.is_file()

    if existe_modelo and existen_clases:
        return "disponible"

    if existe_modelo or existen_clases:
        return "incompleto"

    return "simulacion"


# ==========================================================
# 4. CARGAR EFFICIENTNET-B0
# ==========================================================

@lru_cache(maxsize=1)
def obtener_modelo_real():
    """
    Carga EfficientNet-B0 y sus clases.

    El modelo se carga una sola vez.
    """

    modelo, clases = cargar_modelo(
        ruta_pesos=RUTA_PESOS,
        ruta_clases=RUTA_CLASES
    )

    return modelo, clases


# ==========================================================
# 5. FUNCION PRINCIPAL DE ANALISIS
# ==========================================================

def analizar_fotografia(imagen):
    """
    Selecciona automaticamente entre:

    1. Simulacion si no existe un modelo.
    2. Inferencia EfficientNet si hay pesos.
    3. Error si la instalacion es incompleta.

    Devuelve cinco resultados para Gradio.
    """

    # ------------------------------------------------------
    # VALIDACION DE IMAGEN
    # ------------------------------------------------------

    if imagen is None:

        return (
            "Debes cargar una fotografia antes de analizar.",
            "",
            "",
            None,
            "**Estado:** esperando una fotografia."
        )

    # ------------------------------------------------------
    # IDENTIFICAR MODO
    # ------------------------------------------------------

    modo = verificar_archivos_modelo()

    # ------------------------------------------------------
    # MODO SIMULACION
    # ------------------------------------------------------

    if modo == "simulacion":

        estado, clase, confianza, probabilidades = (
            simular_prediccion(imagen)
        )

        informacion = (
            "**Modo activo: SIMULACION.** "
            "La clasificacion y las puntuaciones "
            "son ficticias. No dependen del contenido "
            "de la fotografia."
        )

        return (
            estado,
            clase,
            confianza,
            probabilidades,
            informacion
        )

    # ------------------------------------------------------
    # MODELO INCOMPLETO
    # ------------------------------------------------------

    if modo == "incompleto":

        return (
            "ERROR: instalacion incompleta del modelo.",
            "",
            "",
            None,
            "**Modelo incompleto:** deben existir ambos "
            "archivos: model/model.pt y model/classes.json."
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

        clase = resultado["clase"]
        confianza = resultado["confianza"]
        probabilidades = resultado["probabilidades"]

        estado = (
            "Analisis completado utilizando "
            "EfficientNet-B0 con pesos cargados."
        )

        informacion = (
            "**Modo activo: EFFICIENTNET-B0.** "
            "Se utilizo el modelo cargado desde "
            "model/model.pt. La confianza corresponde "
            "a una puntuacion Softmax y no equivale "
            "a la precision global del modelo."
        )

        return (
            estado,
            clase,
            f"{confianza:.1%}",
            probabilidades,
            informacion
        )

    except Exception as error:

        print("ERROR DE INFERENCIA:", repr(error))

        return (
            "ERROR: no se pudo ejecutar EfficientNet-B0.",
            "",
            "",
            None,
            "**Error del modelo:** revisa la terminal "
            "de VS Code. No se muestran resultados "
            "simulados cuando falla un modelo instalado."
        )


# ==========================================================
# 6. CONSTRUCCION DE LA INTERFAZ
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
        f"**Version {VERSION} — EfficientNet-B0 + Gradio**"
    )

    gr.Markdown(
        "La aplicacion utiliza resultados simulados "
        "mientras no esten disponibles los archivos "
        "del modelo entrenado. El modo utilizado "
        "se indica en cada analisis."
    )

    # ------------------------------------------------------
    # SECCION 1 - IMAGEN
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
        "con dano visible. El modelo clasifica "
        "la region danada, no detecta automaticamente "
        "danos en una fotografia completa del vehiculo."
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
    # SECCION 3 - RESULTADOS
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
    # CONEXION DEL BOTON
    # ------------------------------------------------------

    boton_analizar.click(
        fn=analizar_fotografia,
        inputs=[imagen],
        outputs=[
            estado,
            clase_resultado,
            confianza_resultado,
            probabilidades_resultado,
            informacion_modo
        ]
    )

    # ------------------------------------------------------
    # INFORMACION DEL PROYECTO
    # ------------------------------------------------------

    gr.Markdown("---")

    gr.Markdown("### Informacion del proyecto")

    gr.Markdown(
        "**Proyecto:** CarDamage AI\n\n"
        "**Dataset:** CarDD - Car Damage Detection\n\n"
        "**Modelo principal:** EfficientNet-B0\n\n"
        "**Arquitectura:** PyTorch / Torchvision\n\n"
        "**Interfaz:** Gradio\n\n"
        "**Categorias iniciales:** dent, scratch, "
        "crack, glass shatter, lamp broken y tire flat.\n\n"
        "**Clases definitivas:** se leen desde "
        "classes.json cuando hay un modelo instalado.\n\n"
        "**Limitacion:** se requiere una fotografia "
        "de una region con dano visible. No existe "
        "una categoria sin dano.\n\n"
        "**Interpretacion:** las puntuaciones Softmax "
        "no garantizan que la prediccion sea correcta. "
        "La precision global del modelo debe evaluarse "
        "separadamente sobre el conjunto Test.\n\n"
        "**Advertencia:** el sistema es una herramienta "
        "academica de clasificacion visual, no un "
        "diagnostico mecanico definitivo."
    )


# ==========================================================
# 7. EJECUTAR
# ==========================================================

if __name__ == "__main__":
    app.launch()
