# cardamage.ia — Clasificación de daños vehiculares

**Proyecto académico · IA Generativa para la Gestión Industrial · Octubre de 2026**

**Equipo:** Eduardo Avendaño (datos y CLIP), Martín Rentería (EfficientNet-B0), Jorge Moscoso (aplicación e integración).

> Prototipo educativo. No reemplaza la inspección de un profesional ni evalúa automáticamente la seguridad mecánica de un vehículo.

## 1. Resumen del proyecto

**cardamage.ia** clasifica fotografías **cercanas de una zona dañada** en una de seis categorías: grieta, abolladura, vidrio roto, foco dañado, rayadura o neumático desinflado. El clasificador EfficientNet-B0 entrega la clase y su puntuación Softmax. Después, Gemini redacta una advertencia, describe posibles consecuencias y sugiere una acción preventiva. Las recomendaciones son orientativas.

**Problema:** agilizar una primera clasificación visual y llamar la atención sobre el riesgo de subestimar daños potencialmente relevantes para la seguridad. El proyecto compara un modelo general sin ajuste al dataset (CLIP) frente a un modelo especializado (EfficientNet-B0).

## 2. Datos y fuentes

- **Dataset:** CarDD (*A New Dataset for Vision-based Car Damage Detection*), Wang y colaboradores (2023). Fuente: https://cardd-ustc.github.io/
- **Formato original:** fotografías con cajas delimitadoras de daños (anotaciones COCO).
- **Conversión:** recortes por caja con 10 % de margen; se omitieron cajas con lado menor de 32 píxeles.
- **División:** particiones originales por fotografía, para evitar que una misma foto aparezca en train, val y test.
- **Volumen:** **8.303 recortes:** entrenamiento **5.878**, validación **1.667**, prueba **758**.
- **Clases:** `crack`, `dent`, `glass_shatter`, `lamp_broken`, `scratch`, `tire_flat`.
- **Restricción:** el dataset original no se distribuye dentro de este repositorio. Consultar las condiciones de CarDD y de las imágenes originales antes de reutilizarlo.

**Código de referencia:** el proceso de exploración y CLIP figura en `notebooks/Parte_Edu_C1_IA.ipynb`; el entrenamiento especializado figura en `notebooks/entrenamiento_efficientnet.ipynb`.

## 3. Enfoque técnico

| Etapa | Herramienta | Propósito |
|---|---|---|
| Línea base | CLIP zero-shot (`openai/clip-vit-base-patch32`) | Clasificación sin reentrenamiento sobre CarDD |
| Modelo final | EfficientNet-B0 de `torchvision`, ajustado a seis clases | Mejorar la precisión de clasificación visual |
| Aplicación | Gradio + PyTorch | Cargar fotografía, ejecutar inferencia y mostrar resultado |
| Recomendaciones | Gemini | Generar una advertencia y acción sugerida a partir de la predicción |

El modelo final parte de pesos preentrenados en ImageNet. Usa recortes RGB redimensionados a **224 × 224** y normalización ImageNet. Se eligió el mejor checkpoint con **F1 macro de validación** y se evaluó al final sobre el test.

## 4. Resultados del conjunto de prueba

**Evaluación de 758 recortes.** Métricas registradas en el trabajo del equipo:

| Métrica | CLIP zero-shot | EfficientNet-B0 |
|---|---:|---:|
| Accuracy | 53,0 % | **94,7 %** |
| Accuracy balanceada | 73,5 % | **95,6 %** |
| F1 macro | 61,0 % | **95,9 %** |
| Daños graves clasificados como leves | 14/226 (6,2 %) | **7/226 (3,1 %)** |

**Daños graves (métrica del equipo):** `crack`, `glass_shatter`, `lamp_broken` y `tire_flat`. **Leves:** `dent` y `scratch`. Esta agrupación sirve para analizar errores del clasificador; no es un diagnóstico de seguridad de cada vehículo.

**Interpretación:** EfficientNet mejora notablemente a la línea base CLIP en las fotografías recortadas de CarDD. La cifra no debe extrapolarse sin pruebas a fotos completas, imágenes nocturnas o daños fuera de esas seis categorías.

## 5. Demostración en Windows

**Inicio recomendado:** hacer doble clic en `ABRIR_CARDAMAGE.bat`, seleccionar **Presentación para el profesor** y luego **Abrir aplicación**.

Para ejecutar manualmente la app, desde la carpeta del proyecto:

```powershell
python app.py
```

Dirección local habitual: http://127.0.0.1:7860

Para una predicción real se necesitan `model/model.pt` y `model/classes.json`. Los pesos **no están incluidos en GitHub**: deben entregarse por separado por el equipo y copiarse a `model/`. También se requieren dependencias de `requirements.txt`. La recomendación de Gemini requiere acceso a Internet y una **clave API propia**, configurada localmente; nunca debe compartirse ni publicarse.

Si faltan los pesos, la app informa que el modelo está pendiente: no simula una predicción como si fuera real.

## 6. Evidencia y límites

- **Eduardo:** `notebooks/Parte_Edu_C1_IA.ipynb` — línea base CLIP, métricas y matriz de confusión.
- **Martín:** `notebooks/entrenamiento_efficientnet.ipynb` — entrenamiento y evaluación de EfficientNet-B0.
- **Jorge:** `app.py`, `inferencia.py`, `recomendaciones.py` — interfaz, inferencia e IA generativa.

**Limitaciones importantes:**

- El modelo **clasifica recortes**, no detecta automáticamente el daño en una imagen completa.
- No hay clase «sin daño»; incluso una foto sin daños recibiría alguna etiqueta.
- Softmax **no es** la probabilidad garantizada de acertar. Martín observó errores con puntuaciones elevadas.
- La evaluación procede de un conjunto limitado de fotos: no garantiza desempeño en vehículos reales de otros entornos.
- Gemini genera orientaciones preventivas, **no** inspecciones técnicas definitivas.

## 7. Repositorio y referencias

- Código del equipo (rama de integración): https://github.com/eduardoavendano10/CarDamage-App/tree/jorge-app
- Dataset CarDD: https://cardd-ustc.github.io/
- Modelo CLIP utilizado: https://huggingface.co/openai/clip-vit-base-patch32
- EfficientNet-B0 (torchvision): https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html

**Uso de IA generativa en el desarrollo:** el equipo empleó asistentes de IA para apoyar la redacción, revisión de código y elaboración de la interfaz y las recomendaciones. Los resultados cuantitativos proceden de las evaluaciones registradas en los notebooks, no de respuestas generadas por el asistente.
