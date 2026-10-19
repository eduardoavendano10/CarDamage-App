
# inferencia.py
# CarDamage AI - Modulo de inferencia

from torch import nn
from torchvision.models import efficientnet_b0


def construir_modelo(num_clases):
    """
    Construye EfficientNet-B0 con una capa
    clasificadora adaptada al numero de clases.
    """

    if not isinstance(num_clases, int) or num_clases < 2:
        raise ValueError("El numero de clases debe ser >= 2")

    # Construir la arquitectura del modelo
    modelo = efficientnet_b0(weights=None)

    # Obtener cantidad de caracteristicas de entrada
    entradas = modelo.classifier[1].in_features

    # Reemplazar ultima capa clasificadora
    modelo.classifier[1] = nn.Linear(
        entradas,
        num_clases
    )

    return modelo


from torchvision import transforms


def preprocesar_imagen(imagen):
    """
    Prepara una imagen para EfficientNet-B0.
    Respeta el contrato de entrenamiento de Martín.
    """

    transformacion = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    imagen = imagen.convert("RGB")

    tensor = transformacion(imagen)

    # Agregar dimensión de batch
    tensor = tensor.unsqueeze(0)

    return tensor

import json
from pathlib import Path


def cargar_clases(ruta="model/classes.json"):
    """
    Lee las clases en el orden exacto
    utilizado durante el entrenamiento.
    """

    with open(ruta, "r", encoding="utf-8") as archivo:
        clases = json.load(archivo)

    if (
        not isinstance(clases, list)
        or len(clases) < 2
        or any(
            not isinstance(c, str) or not c.strip()
            for c in clases
        )
        or len(set(clases)) != len(clases)
    ):
        raise ValueError(
            "classes.json debe contener una lista "
            "de clases únicas y válidas."
        )

    return clases

import torch


def cargar_modelo(
    ruta_pesos="model/model.pt",
    ruta_clases="model/classes.json"
):
    """
    Carga EfficientNet-B0 y sus clases.

    Los pesos deben ser un state_dict
    exportado por el entrenamiento de Martin.
    """

    # Leer el orden oficial de las clases
    clases = cargar_clases(ruta_clases)

    # Comprobar que existe el archivo de pesos
    ruta_pesos = Path(ruta_pesos)

    if not ruta_pesos.is_file():
        raise FileNotFoundError(
            f"No se encontro el modelo: {ruta_pesos}"
        )

    # Construir arquitectura con N salidas
    modelo = construir_modelo(len(clases))

    # Cargar pesos en CPU
    pesos = torch.load(
        ruta_pesos,
        map_location="cpu",
        weights_only=True
    )

    modelo.load_state_dict(pesos, strict=True)

    # Desactivar comportamiento de entrenamiento
    modelo.eval()

    return modelo, clases

def predecir_imagen(imagen, modelo, clases):
    """
    Clasifica una imagen utilizando EfficientNet-B0.

    Devuelve:
    - clase predicha
    - confianza (score Softmax)
    - puntuaciones de todas las clases
    """

    if imagen is None:
        raise ValueError("Debes proporcionar una imagen")

    # Preparar la imagen
    tensor = preprocesar_imagen(imagen)

    # Inferencia sin calcular gradientes
    with torch.inference_mode():
        salidas = modelo(tensor)

        if salidas.shape != (1, len(clases)):
            raise ValueError(
                "Las salidas del modelo no coinciden "
                "con el numero de clases"
            )

        probabilidades = torch.softmax(salidas, dim=1)[0]

    indice = probabilidades.argmax().item()

    return {
        "clase": clases[indice],
        "confianza": probabilidades[indice].item(),
        "probabilidades": {
            clase: probabilidades[i].item()
            for i, clase in enumerate(clases)
        }
    }
