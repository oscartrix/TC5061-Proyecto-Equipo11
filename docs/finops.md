# FinOps: costo por inferencia de la API de MiBici (P1)

Los números salen de las pruebas de carga de `reports/carga/` (Locust y `docker stats`) y se recalculan con:

```bash
make load-test INFERENCE_BACKEND=onnx
make load-test INFERENCE_BACKEND=sklearn
make load-report
python scripts/finops.py
```

`finops.py` deja `reports/carga/figuras/finops.csv` y `finops_escenarios.csv`.

## 1. Supuestos

| Concepto | Valor | Fuente |
|---|---|---|
| Réplica de la API | 1 vCPU y 1 GB, los mismos límites de `docker-compose.yml` | Prueba de carga |
| Fargate (Linux x86, us-east-1) | 0.04048 USD por vCPU-hora y 0.004445 USD por GB-hora | AWS Price List API, vigente desde 2026-11-01 |
| EC2 on-demand (us-east-1) | t3.small 0.0208, t3.medium 0.0416 y t4g.small 0.0168 USD/h | AWS Price List API, vigente desde 2026-11-01 |
| GitHub Actions | Linux 2 núcleos 0.006 USD/min y macOS 0.062 USD/min; gratis en repos públicos | Tabla de precios de GitHub, 2026 |
| Horas por mes | 730 | |
| Unidad de inferencia | Una fila predicha (una estación en un día) | |
| Nivel de servicio | p95 de `/p1/predict` ≤ 300 ms y 0 % de errores | |
| Pico | 5 veces la demanda promedio | |

Con estos precios, una réplica de 1 vCPU y 1 GB en Fargate cuesta 0.0449 USD/h, o 32.80 USD al mes.

## 2. Capacidad medida por réplica

Prueba por escalones de 20 a 400 usuarios, 60 s por escalón, sin errores en ninguna corrida.

| Motor | Escalón más alto dentro del nivel de servicio | Peticiones de predicción/s | Filas/s | p95 | CPU | Memoria máx. | USD por millón de filas (réplica al 100 %) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ONNX Runtime (1 hilo) | 400 usuarios | 189 | 616 | 43 ms | 78 % | 133 MB | 0.020 |
| sklearn (MLflow pyfunc) | 50 usuarios | 24 | 83 | 150 ms | 58 % | 233 MB | 0.150 |
| ONNX Runtime con 8 hilos | 20 usuarios | 9 | 31 | 88 ms | 92 % | 108 MB | 0.401 |

- ONNX no llegó a saturarse con 400 usuarios: el límite de la prueba fue Locust, no la API. Los 616 filas/s son un piso de su capacidad.
- sklearn pasa el p95 de 300 ms antes de 100 usuarios y desde 200 usuarios está en 100 % de CPU; de ahí en adelante solo crece la fila de espera (p95 de 3.3 s con 400 usuarios).
- La primera versión con ONNX era la más cara. ONNX Runtime veía los 8 núcleos del host y creaba un hilo por núcleo, y la espera activa de esos hilos consumía la cuota de 1 CPU del contenedor (92 % de CPU con solo 20 usuarios). Con `intra_op_num_threads=1` y sin espera activa, la capacidad máxima subió de 34 a 189 peticiones de predicción por segundo (5.6 veces) con la misma réplica. Sin la medición de CPU con `docker stats`, este problema no se habría visto.

## 3. Costo por escenario de demanda

El costo por inferencia depende de cuánto tiempo pasa la réplica sin trabajo. MiBici tiene 470 estaciones con predicción diaria.

| Escenario | Filas/día | ONNX: réplicas y USD/mes | sklearn: réplicas y USD/mes | USD por millón de filas (ONNX / sklearn) |
|---|---:|---|---|---|
| Lote diario de las 470 estaciones | 470 | 1 · 32.80 | 1 · 32.80 | 2 326 / 2 326 |
| Replaneación cada hora | 11 280 | 1 · 32.80 | 1 · 32.80 | 97 / 97 |
| App pública | 1 M | 1 · 32.80 | 1 · 32.80 | 1.09 / 1.09 |
| Alto volumen | 50 M | 5 · 163.98 | 35 · 1 147.83 | 0.11 / 0.77 |

Los resultados se leen así:

1. Hasta 1 M de filas al día, una réplica de cualquiera de los dos motores sobra, y el costo es el de tenerla encendida. Con el lote diario, cada predicción cuesta unos 2.3 milésimos de dólar, aunque el modelo tarda milisegundos.
2. La diferencia entre motores aparece con alto volumen: ONNX necesita 5 réplicas y sklearn 35, unos 984 USD al mes menos (−86 %).
3. Escalar es lineal en réplicas, porque la API no guarda estado y el modelo va dentro de la imagen. El límite es el costo, no la arquitectura.

## 4. Consumo del stack completo

Promedios y máximos de `docker stats` durante la prueba de ONNX (reports/carga/onnx/docker_stats.csv):

| Contenedor | CPU prom. | CPU máx. | Memoria máx. |
|---|---:|---:|---:|
| mibici-api (ONNX) | 44 % | 108 % | 133 MB |
| mibici-worker (RQ) | 16 % | 54 % | 108 MB |
| mibici-redis | 3 % | 10 % | 50 MB |
| mibici-equipo11 (MLflow) | 5 % | 63 % | 2 058 MB |

El servidor de MLflow ocupa el 87 % de la memoria del stack y no participa en la inferencia. El compose completo (~2.4 GB) no cabe en una t3.small de 2 GB y necesita una t3.medium (30.37 USD/mes). Sin MLflow, la API, el worker y Redis usan menos de 300 MB.

## 5. Optimizaciones

| # | Optimización | Dato que la justifica | Ahorro estimado |
|---|---|---|---|
| 1 | ONNX Runtime con 1 hilo como motor por defecto (ya aplicado) | 616 contra 83 filas/s por réplica dentro del nivel de servicio | −86 % con alto volumen (164 contra 1 148 USD/mes) |
| 2 | Ajustar el tamaño de la réplica: 0.25 vCPU y 0.5 GB mientras la demanda sea menor a 1 M de filas al día | La API con ONNX usa ~130 MB y menos del 15 % de CPU con 50 usuarios | 32.80 → 9.01 USD/mes (−73 %) |
| 3 | Lote diario como tarea programada, no como servicio encendido: el worker de RQ o una tarea de Fargate que se apaga al terminar | 470 filas caben en una sola llamada de pocos milisegundos | ~2 min al día: 0.05 USD/mes contra 9–33 USD/mes |
| 4 | Sacar MLflow del despliegue de inferencia y levantarlo solo para entrenar y registrar | 2 GB de 2.3 GB del stack; el modelo ya va en la imagen | t3.medium → t4g.small: 30.37 → 12.26 USD/mes (−60 %); la imagen ya se publica para arm64 |
| 5 | Si MLflow tiene que quedar encendido, usar `--workers 1` | 1.4 GB con 1 worker contra ~1.8–2.0 GB con los 4 por defecto | ~0.5 GB de memoria |
| 6 | Mandar lotes en vez de filas sueltas | El lote de 10 filas tarda casi lo mismo que 1 fila (p50 de 14 contra 13 ms con ONNX) | Hasta 10 veces menos peticiones para el mismo número de filas |
| 7 | En el CI, correr la matriz de macOS solo en PR hacia `main` y en la corrida semanal | En repo privado, macOS sería el 61 % del costo de un push a `main` con 4 de 31 minutos | ~0.25 de 0.41 USD por push (hoy es gratis por ser repo público) |
| 8 | Agregar alertas de presupuesto y etiquetas por servicio (`api`, `worker`, `mlflow`) | El costo por inferencia cambia más de 20 000 veces entre escenarios (de 2 326 a 0.11 USD por millón de filas) | Hace visible el gasto cuando cambia la demanda |

## 6. Riesgos

- Las mediciones son de una laptop con 8 núcleos y la API limitada a 1 CPU. En Fargate el CPU puede rendir distinto, así que conviene repetir `make load-test` en la instancia elegida antes de fijar el número de réplicas.
- El escalado a 0.25 vCPU supone que la capacidad baja en proporción al CPU. Con sklearn hay que medirlo, porque la carga de MLflow pyfunc no es lineal.
- ONNX trabaja en float32. Con el champion actual, la diferencia media contra sklearn es de 0.14 viajes (p99 de 1.19). `export_onnx.py` corta el pipeline si la diferencia media pasa de 0.25.
