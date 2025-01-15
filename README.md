
# Predicción de variables oceanográficas basada en métodos de aprendizaje profundo

¡Bienvenid@ al repositorio de mi Trabajo de Fin de Grado (TFG)!  
Este proyecto se centra en **adaptar y entrenar el modelo Aurora en el ámbito oceánico**, utilizando datos de temperatura oceánica (thetao) en distintas profundidades y una máscara de tierra (lsm). A continuación, se describe la estructura del repositorio, los requisitos para ejecutar el código, así como las instrucciones para **entrenar e inferir** el modelo, y la forma de ejecutar los tests y notebooks de experimentos.

---

## Índice
1. [Descripción del proyecto](#descripción-del-proyecto)  
2. [Estructura del repositorio](#estructura-del-repositorio)  
3. [Requerimientos e instalación](#requerimientos-e-instalación)  
4. [Uso del código](#uso-del-código)  
    - [Ejecutar el entrenamiento e inferencia (main.py)](#ejecutar-el-entrenamiento-e-inferencia-mainpy)  
    - [Entrenar e inferir de forma modular (src/)](#entrenar-e-inferir-de-forma-modular-src)  
    - [Experimentos Jupyter (Experimentos/)](#experimentos-jupyter-experimentos)  
5. [Descripción de los principales scripts](#descripción-de-los-principales-scripts)  
6. [Ejecutar los tests](#ejecutar-los-tests)  
7. [Detalles adicionales para configuraciones y experimentos](#detalles-adicionales-para-configuraciones-y-experimentos)  
8. [Contacto](#contacto)  

---

## Descripción del proyecto
Este TFG tiene como objetivo **adaptar el modelo Aurora**, originalmente preentrenado con datos atmosféricos, al dominio oceánico. Para ello, se utilizan datos de temperatura (thetao) a diferentes profundidades, junto con una máscara de tierra (lsm), abarcando distintos períodos y regiones oceánicas.  
El proyecto:
- Incorpora un **generador de lotes** (batch) que facilita el manejo y la iteración de datos temporales de gran volumen.  
- Aplica técnicas de **finetuning** para congelar y descongelar partes del modelo (encoder, backbone, decoder, etc.).  
- Muestra **métricas de evaluación** (RMSE, Bias, ACC), tomando en cuenta la geometría esférica y el coseno de la latitud para mediciones globales.  
- Incluye ejemplos de **predicción a largo plazo** (rollout) de hasta 10 días.  

---

## Estructura del repositorio

```bash
TFG_victor/
├── requirements.txt
├── README.md
├── .gitignore
├── DatosCop.ipynb
├── main.py
├── tests/
│   ├── test_model.py
│   ├── conftest.py
│   ├── __init__.py
│   ├── test_checkpoint_adaptation.py
│   ├── test_rollout.py
│   ├── test_batch.py
│   └── test_headers.py
├── aurora/
│   ├── _version.py
│   ├── batch.py
│   ├── __init__.py
│   ├── rollout.py
│   ├── normalisation.py
│   ├── area.py
│   └── model/
│       ├── posencoding.py
│       ├── film.py
│       ├── decoder.py
│       ├── swin3d.py
│       ├── perceiver.py
│       ├── util.py
│       ├── __init__.py
│       ├── encoder.py
│       ├── aurora.py
│       ├── lora.py
│       ├── patchembed.py
│       └── fourier.py
├── Experimentos/
│   ├── test_0.ipynb
│   ├── test_1.ipynb
│   ├── test_2.ipynb
│   └── ... (varios notebooks numerados)
├── src/
│   ├── training/
│   │   ├── metrics.py
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── inference.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── visualization.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── aurora/...
│   ├── data/
│       ├── data_preprocessing.py
│       ├── data_loader.py
│       ├── __init__.py
│       └── batch_generator.py
```

- **main.py**: Script principal con la lógica de entrenamiento e inferencia.
- **tests/**: Scripts para test unitarios e integración.
- **aurora/**: Código principal del modelo Aurora con sus subcomponentes.
- **Experimentos/**: Colección de notebooks (`.ipynb`) con diferentes configuraciones y pruebas.
- **src/**: Contiene módulos para el entrenamiento, inferencia, procesado de datos y utilidades:
  - `src/training/`: Funciones de entrenamiento, inferencia y métricas.  
  - `src/utils/`: Funciones auxiliares (visualizaciones).  
  - `src/models/aurora/`: Arquitectura base de Aurora y sus submódulos.  
  - `src/data/`: Scripts de carga y preprocesamiento de datos, y el generador de lotes (batch).  

---

## Requerimientos e instalación

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/<usuario>/TFG_victor.git
   cd TFG_victor
   ```

2. **Creación y activación de un entorno virtual** (opcional pero recomendado):
   ```bash
   python -m venv env
   source env/bin/activate   # Linux/Mac
   # o en Windows
   # .\env\Scripts\activate
   ```

3. **Instalación de dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
   Asegúrate de tener instalado [PyTorch](https://pytorch.org/) con la versión que corresponda a tu GPU/CPU y versión de CUDA o CPU.

---

## Uso del código

### Ejecutar el entrenamiento e inferencia (main.py)

Este proyecto está diseñado para que, **con un solo script**, se puedan entrenar e inferir resultados de Aurora sobre datos oceánicos.

1. **Revisar rutas de entrada en `main.py`**:  
   - Ajusta la variable `dataset_path` con tu ruta al archivo NetCDF de temperatura oceánica.
   - Ajusta `lsm_path` con la ruta a la máscara de tierra (lsm).
   - Configura las variables y `depth_slice` según necesites.

2. **Ejecutar**:
   ```bash
   python main.py
   ```
   - El script cargará los datos, dividirá en entrenamiento, validación y test, y entrenará el modelo según los hiperparámetros establecidos.  
   - Guardará el modelo entrenado y, posteriormente, realizará inferencia en el conjunto de test.  

### Entrenar e inferir de forma modular (src/)
Si prefieres **ejecutar el entrenamiento e inferencia por separado**, en la carpeta `src/training/` encontrarás:

- `train.py`: Función `train(...)` para entrenar el modelo Aurora.  
- `inference.py`: Función `evaluate_model(...)` para evaluar uno o varios modelos, reportando métricas y visualizaciones.

Ejemplo de uso en un script personalizado:
```python
from src.training.train import train
from src.training.inference import evaluate_model

# Crear tu modelo, generadores de lotes, etc.
train_losses, val_losses, val_rmses = train(
    model=model,
    train_generator=train_generator,
    val_generator=val_generator,
    ...
)

model_paths = ['ruta/al/modelo_1.pth', 'ruta/al/modelo_2.pth']
evaluate_model(model, model_paths, test_generator, device, latitudes, batch_size=8)
```

### Experimentos Jupyter (Experimentos/)
Los notebooks en **`Experimentos/`** contienen distintos pasos y **resultados de pruebas**. Aquí se detalla qué se hace en cada uno:

- **`test_0.ipynb`**: Reducción de resolución para inferencia con Aurora  
- **`test_1.ipynb`**: Adaptación de datos específicos e inferencia con Aurora  
- **`test_2.ipynb`**: Impacto de datos sin conversión a Kelvin en predicciones Aurora  
- **`test_3.ipynb`**: Reducción de sobreestimación y mejora del patrón global  
- **`test_4.ipynb`**: Creación del Generador de Lotes (paso a paso)  
- **`test_5.ipynb`**: Documentar implementación y pruebas del Batch Generator  
- **`test_6.ipynb`**: Normalización en Kelvin y ajuste de predicción HRES en Aurora  
- **`test_7.ipynb`**: Prueba con ventana deslizante de dos días  
- **`test_8.ipynb`**: División del conjunto de datos  
- **`test_11.ipynb`**: Implementación de las métricas  
- **`test_12.ipynb`**: Estimación del tiempo por época de entrenamiento  
- **`test_13.ipynb`**: Entreno con los meses de verano y lr=1e-5  
- **`test_14.ipynb`**: Entreno con meses de verano y lr=1e-4  
- **`test_15.ipynb`**: Pruebas de función de entreno  
- **`test_17.ipynb`**: Entreno con meses de verano y lr=1e-5 (segunda variante)  
- **`test_18.ipynb`**: Entreno Fine Tuning completo y todos los meses (bs=3, lr=1e-5)  
- **`test_19.ipynb`**: Pruebas para congelar y descongelar partes de Aurora  
- **`test_20.ipynb`**: Comprobaciones de inferencia y visualizaciones  
- **`test_21.ipynb`**: Entreno del Decoder (bs=3, lr=1e-4)  
- **`test_22.ipynb`**: Entreno con toda la red descongelada y el nuevo decoder (1e-5, bs=3)  
- **`test_24.ipynb`**: Inferencia y visualizaciones con los modelos entrenados  
- **`test_25.ipynb`**: Entreno Fine Tuning completo y todos los meses (bs=8, lr=1e-5)  
- **`test_26.ipynb`**: Entreno del Decoder (bs=8, lr=1e-4)  
- **`test_27.ipynb`**: Entreno con toda la red descongelada y el nuevo decoder (bs=8, lr=1e-5)  
- **`test_29.ipynb`**: Predicción a largo plazo  
   
Para lanzar un notebook, por ejemplo:
```bash
jupyter notebook Experimentos/test_0.ipynb
```
(ajusta el nombre según el notebook que desees abrir)

---

## Descripción de los principales scripts

- **`main.py`**  
  - Carga y preprocesa datos (load_dataset,load_lsm).  
  - Define hiperparámetros (batch size, épocas, profundidad).  
  - Inicializa el modelo Aurora y aplica congelación/descongelación.  
  - Ejecuta el entrenamiento (`train`) y guarda el modelo.  
  - Realiza la inferencia y evaluación final (`evaluate_model`).  

- **`src/training/train.py`**  
  - Función principal de entrenamiento: recibe el modelo, generadores de lotes (train/val), función de pérdida, optimizador, etc.  
  - Registra y retorna las curvas de pérdidas y métricas.  

- **`src/training/inference.py`**  
  - Función principal de evaluación: Para distintos .pth guardados, realiza inferencia y computa métricas (RMSE, Bias, ACC).  

- **`src/data/data_loader.py`**  
  - Carga los ficheros NetCDF y variables oceánicas, usando xarray.  
  - Ajusta dimensiones, latitudes y longitudes según el modelo.  

- **`src/data/data_preprocessing.py`**  
  - Clase DataPreprocessor para normalizar, dividir en train/val/test, rellenar huecos, etc.

- **`src/data/batch_generator.py`**  
  - Generador de lotes (BatchGenerator) con ventanas deslizantes, padding y shuffle.  

- **`src/training/metrics.py`**  
  - Implementa RMSE, Bias y ACC con pesos basados en el coseno de la latitud.  

- **`tests/*.py`**  
  - Scripts de test unitarios e integración, para comprobar consistencia del modelo, checkpoints, batch generation, etc.  

---

## Ejecutar los tests
La carpeta `tests/` incluye **tests unitarios** para verificar la consistencia del proyecto. Para ejecutarlos, corre:

```bash
pytest tests/
```
o
```bash
python -m pytest tests/
```

Estos tests validan:
- **Generación de lotes** (dimensiones y estructura).  
- **Checkpoint** y carga de pesos preentrenados.  
- **Rollout** para predicciones multi-paso.  
- **Headers** y configuraciones de ficheros NetCDF.  

---

## Detalles adicionales para configuraciones y experimentos

- **Congelar y descongelar capas**: Modifica `param.requires_grad = False/True` en `main.py` o en los notebooks (p.ej. `test_19.ipynb`).  
- **Experimentar con distintos batch_size**: Ajusta la variable `batch_size` en `main.py` o en notebooks. Ten en cuenta la memoria disponible de tu GPU.  
- **Predicción a largo plazo (rollout)**: Requiere incrementar `sample_size` a 12 para predecir 10 días (2 días de entrada + 10 futuros). Ajusta también la carga de datos a `slice(2,12)` en lugar de `slice(2,None)`.  
- **Métricas personalizadas**: En `metrics.py` están RMSE, Bias, ACC con pesos en latitud, evitando sesgos en zonas polares.  

Para más detalle, revisa los *notebooks* en la carpeta `Experimentos/` que documentan la configuración de cada prueba, tiempo por época, normalización, etc.

---

## Contacto
Para más información, dudas o sugerencias:  
- **Correo**: victormedina2157@gmail.com  

¡Gracias por visitar este repositorio!  
