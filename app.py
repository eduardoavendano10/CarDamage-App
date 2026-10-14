
import gradio as gr


# ==========================================
# 1. FUNCIONES
# ==========================================

def verificar_imagen(imagen):
    """
    Comprueba si el usuario cargó una fotografía.
    Posteriormente conectaremos aquí el modelo.
    """

    if imagen is None:
        return "Debes cargar una fotografía antes de analizar."

    return (
        "Imagen recibida correctamente. "
        "El modelo de inteligencia artificial aún no está conectado."
    )


# ==========================================
# 2. INTERFAZ WEB
# ==========================================

with gr.Blocks(title="CarDamage AI") as app:

    # Encabezado principal
    gr.Markdown("# CarDamage AI")

    gr.Markdown(
        "Sistema de clasificación de daños visibles en vehículos."
    )

    # --------------------------------------
    # SECCIÓN 1: CARGAR FOTOGRAFÍA
    # --------------------------------------

    gr.Markdown("### 1. Cargar fotografía")

    imagen = gr.Image(
        label="Fotografía del daño",
        type="pil",
        sources=["upload"],
        height=350
    )

    gr.Markdown(
        "Sube una fotografía cercana de una zona dañada del vehículo."
    )

    # --------------------------------------
    # SECCIÓN 2: ANALIZAR FOTOGRAFÍA
    # --------------------------------------

    gr.Markdown("### 2. Analizar fotografía")

    boton_analizar = gr.Button(
        "Analizar daño",
        variant="primary"
    )

    resultado = gr.Textbox(
        label="Estado del análisis",
        interactive=False
    )

    # Conectar el botón con la función
    boton_analizar.click(
        fn=verificar_imagen,
        inputs=imagen,
        outputs=resultado
    )

    # --------------------------------------
    # INFORMACIÓN DEL PROYECTO
    # --------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        "**Versión 1.2 — Prototipo inicial**\n\n"
        "Actualmente la aplicación permite cargar fotografías "
        "y verificar que fueron recibidas correctamente.\n\n"
        "La clasificación mediante EfficientNet-B0 "
        "y las recomendaciones generadas por IA "
        "se incorporarán en las próximas etapas."
    )


# ==========================================
# 3. EJECUTAR APLICACIÓN
# ==========================================

if __name__ == "__main__":
    app.launch()
