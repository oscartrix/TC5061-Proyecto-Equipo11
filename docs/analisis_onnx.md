# Análisis: ONNX Runtime como servidor de inferencia — Equipo 11

Repo: https://github.com/oscartrix/TC5061-Proyecto-Equipo11
PR: #36 `feat/inferencia-onnx` (independiente del contenedor, #34)
Modelo: champion de P1 (`mibici-equipo11-p1-demanda` v1, HistGradientBoostingRegressor con `StandardScaler` y `OneHotEncoder`)

## 1. Qué se integró

- `scripts/export_onnx.py` convierte el pipeline de sklearn a `model.onnx` (skl2onnx 1.20, opset 17) y compara sus predicciones con las de sklearn.
- `mi_bici/api/onnx_model.py` expone la misma interfaz que el modelo actual (`name`, `version`, `run_id`, `predict`), por lo que `main.py` no cambió.
- `INFERENCE_BACKEND=onnx` activa ONNX Runtime; por defecto la API sigue con sklearn.
- OpenVINO no se integró: parte de un modelo ONNX, solo acelera en CPU/GPU Intel y para un árbol de decisión sobre pocas columnas no aporta más que ONNX Runtime.

## 2. Latencia (CPU local, 200 repeticiones, solo inferencia del modelo)

| Filas | Motor | p50 ms | p95 ms | Filas/s |
|---:|---|---:|---:|---:|
| 1 | sklearn | 5.09 | 6.88 | 196 |
| 1 | ONNX Runtime | 0.11 | 0.16 | 8 850 |
| 100 | sklearn | 5.27 | 6.44 | 18 977 |
| 100 | ONNX Runtime | 0.39 | 0.46 | 257 693 |
| 500 | sklearn | 7.56 | 11.67 | 66 116 |
| 500 | ONNX Runtime | 1.10 | 1.45 | 454 663 |

ONNX Runtime es entre 7 y 46 veces más rápido según el tamaño del lote. La mayor ganancia está en peticiones de una fila, que es el caso típico de la API.

## 3. Ventajas de cambiar

| Ventaja | Detalle |
|---|---|
| Latencia | Menos de 1 ms por fila contra unos 5 ms de sklearn; el lote de 500 filas baja de 7.6 a 1.1 ms |
| Portabilidad | El `.onnx` no depende de la versión de sklearn ni de pickle; se puede cargar desde otros lenguajes |
| Seguridad | No se deserializa un pickle al arrancar; el `.onnx` es un grafo de operadores |
| Archivo más pequeño | `model.onnx` 450 KB contra `model.pkl` 725 KB |
| Dependencias | Con una API sin MLflow ni sklearn bastaría `onnxruntime` y `numpy`; hoy `model.py` aún importa MLflow, por lo que la imagen no se redujo en este PR |

## 4. Costos y riesgos

| Riesgo | Detalle |
|---|---|
| Diferencias numéricas | ONNX Runtime ejecuta los árboles en float32. En 20 000 filas de prueba reales, la diferencia media contra sklearn fue 0.088 viajes, p99 1.11 y máxima 16.9; 25.7 % de las filas cambian más de 0.01 |
| Calidad global sin cambio | MAE contra los viajes reales: sklearn 5.698 y ONNX 5.693 |
| Conversión por modelo | Cada familia (Ridge, RF, HGB) y cada reentrenamiento exige reconvertir y revisar la paridad |
| Parche necesario | skl2onnx 1.20 con onnx 1.23 rechaza booleanos en los árboles de HGB; `export_onnx.py` los convierte a enteros antes de serializar |
| Doble mantenimiento | Dos caminos de inferencia en la API y una dependencia opcional más (`requirements-onnx.txt`) |
| Doble precisión | El intento de usar `double` falló: ONNX Runtime no carga `TreeEnsembleRegressor` con salida double en esta versión |

## 5. Recomendación

1. Mantener sklearn como motor por defecto: la demanda por estación no tiene un requisito de latencia que 5 ms no cumpla.
2. Usar ONNX para los modelos lineales (Ridge) o cuando haya un volumen alto de peticiones; ahí la paridad es exacta salvo redondeo.
3. Antes de activarlo con el HGB, definir una tolerancia aceptada para los 25 % de filas que cambian, o reentrenar con características que no queden sobre los umbrales de los árboles (por ejemplo, sin valores enteros pegados al umbral).
4. Si se adopta, refactorizar `model.py` para importar MLflow solo cuando se use y construir una segunda imagen con `onnxruntime`.
