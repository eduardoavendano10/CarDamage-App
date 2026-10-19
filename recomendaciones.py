"""cardamage.ia — recomendaciones preventivas con Gemini.

Se integra con app.py mediante:
    generar_recomendacion(clase, confianza, probabilidades, modo_prueba=False)

La clave se almacena en clave_local.py (archivo excluido de Git)
o se lee de la variable de entorno GEMINI_API_KEY.
"""

import os

from google import genai


# Conservamos el modelo que ya funciona en la aplicación del equipo.
MODELO_LLM = "gemini-3.5-flash-lite"

# Umbral provisional: no representa una probabilidad calibrada de acierto.
UMBRAL_CONFIANZA = 0.60

# Información de seguridad orientativa, no diagnósticos de la fotografía.
RIESGOS_POR_CLASE = {
    "lamp_broken": (
        "Un foco roto puede dejar de iluminar correctamente o hacer que otros "
        "conductores no perciban bien el vehículo. De noche, si las luces fallan, "
        "un obstáculo o peatón podría detectarse demasiado tarde. Indicar que "
        "se compruebe el funcionamiento de las luces y que no se circule de "
        "noche si la iluminación necesaria está comprometida."
    ),
    "tire_flat": (
        "Un neumático desinflado puede reducir la estabilidad y capacidad de "
        "control. Continuar la marcha puede agravar el daño y ocasionar una "
        "situación peligrosa. Recomendar detenerse en un lugar seguro y no "
        "continuar hasta solucionar el problema."
    ),
    "glass_shatter": (
        "Un cristal quebrado puede causar cortes, disminuir la visibilidad "
        "y dejar a los ocupantes expuestos. Recomendar no tocar los fragmentos "
        "y acudir a un profesional. Si la visibilidad o la protección de los "
        "ocupantes está comprometida, evitar circular."
    ),
    "crack": (
        "Una grieta puede extenderse y debilitar la pieza afectada según su "
        "material y ubicación. No asumir que existe una falla estructural "
        "solo por la imagen. Recomendar una inspección de la grieta."
    ),
    "dent": (
        "Una abolladura puede ser estética, pero en ciertos lugares puede "
        "ocultar daños en elementos cercanos o dificultar el funcionamiento "
        "de una puerta u otro mecanismo. Recomendar revisión si existen "
        "señales de funcionamiento anormal."
    ),
    "scratch": (
        "Una rayadura superficial suele ser estética. Si atraviesa la capa "
        "protectora y deja metal expuesto, puede favorecer la corrosión. "
        "Recomendar comprobar la profundidad y proteger la superficie "
        "si corresponde."
    ),
}


def obtener_clave_gemini():
    """Lee la clave sin mostrarla ni incluirla en mensajes de error."""
    try:
        from clave_local import GEMINI_API_KEY
    except ModuleNotFoundError as error:
        if error.name != "clave_local":
            raise
        clave = os.getenv("GEMINI_API_KEY", "")
    else:
        clave = GEMINI_API_KEY

    if not isinstance(clave, str):
        raise RuntimeError("La clave de Gemini debe ser una cadena de texto.")

    clave = clave.strip()
    if not clave or clave == "PEGA_AQUI_TU_CLAVE_COMPLETA":
        raise RuntimeError(
            "Configura GEMINI_API_KEY en clave_local.py "
            "o como variable de entorno."
        )

    return clave


def generar_recomendacion(
    clase,
    confianza,
    probabilidades,
    modo_prueba=False,
):
    """Genera una advertencia concreta y una acción preventiva apropiada."""
    # No llamar a Gemini ni generar conclusiones con datos ficticios.
    if modo_prueba:
        return (
            "**MODO DEMOSTRACIÓN:** no existe una clasificación real "
            "de esta fotografía. No se generó una recomendación."
        )

    if not isinstance(probabilidades, dict) or clase not in probabilidades:
        raise ValueError("La predicción no contiene puntuaciones válidas.")

    confianza = float(confianza)
    if not 0.0 <= confianza <= 1.0:
        raise ValueError("La puntuación Softmax debe estar entre 0 y 1.")

    # Con baja puntuación, priorizar una inspección humana.
    if confianza < UMBRAL_CONFIANZA:
        return (
            "**ADVERTENCIA:** La imagen no permite orientar con suficiente "
            "claridad según el umbral provisional del sistema.\n\n"
            "**QUÉ HACER AHORA:** Toma otra fotografía cercana, bien enfocada "
            "e iluminada. Si sospechas un problema de seguridad, pide "
            "una inspección profesional antes de circular."
        )

    resumen = "\n".join(
        f"- {nombre}: {float(valor):.1%}"
        for nombre, valor in probabilidades.items()
    )
    riesgo = RIESGOS_POR_CLASE.get(
        clase,
        "Explica las posibles consecuencias sin atribuir daños no observados. "
        "Sugiere una revisión apropiada.",
    )

    prompt = f"""Eres el asistente de seguridad preventiva de cardamage.ia.

Un clasificador automático analizó un RECORTE de una zona dañada,
no necesariamente la fotografía completa del vehículo.

Clase predicha: {clase}
Puntuación Softmax: {confianza:.1%}
Puntuaciones de las clases:\n{resumen}

Riesgo orientativo asociado a esa clase:\n{riesgo}

Escribe en español una recomendación de MÁXIMO 110 palabras,
con esta estructura y estos encabezados EXACTOS:

**⚠️ ADVERTENCIA**
Una frase directa, memorable y seria sobre el posible riesgo.

**¿QUÉ PODRÍA PASAR?**
Describe en lenguaje sencillo una cadena concreta de causa y efecto.
Usa condicionales si la consecuencia depende del funcionamiento real.

**QUÉ HACER AHORA**
Da una acción inmediata, específica y segura. Si afecta la seguridad
para circular, aconseja detenerse en lugar seguro o evitar conducir
hasta verificar y reparar lo necesario.

Reglas obligatorias:
- Tono contundente y preventivo, no sensacionalista.
- No inventes probabilidades de choque, lesiones ni daños ocultos.
- No afirmes que inspeccionaste la foto: solo recibes la predicción.
- La clasificación es tentativa; Softmax no garantiza acierto.
- Una rayadura o abolladura estética no justifica alarma extrema.
- No afirmes que la luz está apagada solo porque hay un foco roto.
- No sugieras reparaciones improvisadas como sustituto de una revisión.
- No inventes costos, plazos ni diagnósticos definitivos.
- Evita introducciones, despedidas y texto fuera de esas 3 secciones.
"""

    cliente = genai.Client(api_key=obtener_clave_gemini())
    respuesta = cliente.models.generate_content(
        model=MODELO_LLM,
        contents=prompt,
    )
    if not respuesta.text:
        raise RuntimeError("Gemini no devolvió una recomendación.")

    return respuesta.text.strip()
