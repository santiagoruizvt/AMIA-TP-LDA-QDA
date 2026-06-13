# AMIA-TP-LDA-QDA
Este repositorio corresponde al trabajo grupal de la asignatura Análisis Matemático para IA de la Carrera de Especialización en IA

## Estructura del repositorio

```
├── AMIA_2025_TP1.ipynb          # Notebook principal con desarrollo, análisis y respuestas
├── AMIA_2025_TP1_gh.ipynb       # Versión renderizable en GitHub (sin metadata ni outputs innecesarios)
├── tp.ipynb                     # Trabajo práctico (referencia)
├── base/                        # Implementaciones base de clasificadores
│   ├── bayesian.py
│   ├── cholesky.py
│   └── qda.py
├── utils/                       # Utilidades para datasets y benchmarking
│   ├── bench.py
│   └── datasets.py
├── pyproject.toml               # Configuración del proyecto
└── LICENSE
```

### Notebooks

- **`AMIA_2025_TP1.ipynb`**: Versión de trabajo con metadata, outputs y ejecuciones.
- **`AMIA_2025_TP1_gh.ipynb`**: Versión limpia optimizada para visualización en GitHub (sin outputs intermedios, metadata reducida).

### Módulos

- **`base/`**: Implementaciones de QDA y sus variantes optimizadas (Tensorizado, Cholesky, Efficient).
- **`utils/`**: Funciones para cargar datasets y ejecutar benchmarks de rendimiento.
