
# ==============================================
# CarDamage AI - Recomendaciones con Gemini
# ==============================================

import os
from google import genai

MODELO_LLM = "gemini-3.5-flash-lite"

# Umbral provisional, pendiente de validacion
UMBRAL_CONFIANZA = 0.60


def generar_recomendacion(
    clase,
    confianza,
    probabilidades,
    modo_prueba=False
):
    """
    Genera una recomendacion orientativa
    a partir de la salida de EfficientNet.
    """

    # Caso de baja confianza
    if confianza < UMBRAL_CONFIANZA:

        if modo_prueba:
            return (
            "MODO SIMULACION: los resultados son ficticios. "
            "No se ha evaluado realmente la fotografia "
            "ni se puede determinar el tipo de dano."
        )

        return (
        "El clasificador obtuvo una puntuacion "
        "Softmax inferior al umbral provisional. "
        "Se recomienda tomar otra fotografia "
        "de la zona danada o solicitar una "
        "revision profesional."
     )

    # Verificar la clave sin mostrarla
    clave = os.getenv("GEMINI_API_KEY")

    if not clave:
        raise RuntimeError(
            "No se encontro GEMINI_API_KEY"
        )

    # Preparar las puntuaciones para Gemini
    resumen = "\n".join(
        f"- {nombre}: {valor:.1%}"
        for nombre, valor in probabilidades.items()
    )

    tipo_datos = (
        "DATOS SIMULADOS PARA PRUEBAS"
        if modo_prueba
        else "RESULTADO DE UN CLASIFICADOR AUTOMATICO"
    )

    prompt = f"""
Eres un asistente de orientacion sobre
danos visibles en vehiculos.

Contexto: {tipo_datos}

Un clasificador de imagenes produjo:

Clase principal: {clase}
Confianza Softmax: {confianza:.1%}

Puntuaciones:
{resumen}

Redacta una recomendacion en espanol,
de un maximo de 100 palabras.

Reglas:
- No afirmes haber examinado la fotografia.
- Presenta la clasificacion como tentativa.
- No inventes costos ni tiempos.
- No asegures que una reparacion sea suficiente.
- No emitas un diagnostico mecanico definitivo.
- Sugiere una revision profesional cuando corresponda.
- Si los datos son simulados, dejalo claro.
- Utiliza lenguaje sencillo y profesional.
"""

    cliente = genai.Client(api_key=clave)

    respuesta = cliente.models.generate_content(
        model=MODELO_LLM,
        contents=prompt
    )

    if not respuesta.text:
        raise RuntimeError(
            "Gemini no devolvio una recomendacion"
        )

    return respuesta.text.strip()
