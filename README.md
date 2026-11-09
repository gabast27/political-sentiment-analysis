# Comparación de modelos transformer para sentimiento en español

Dos experimentos sobre **las mismas particiones** de tu dataset (`id, text, label`):

1. **Sin entrenar:** los modelos originales (ya entrenados en sentimiento) se evalúan sobre tu test.
2. **Con fine-tuning:** los mismos modelos (más algunos modelos base) se afinan con tu train y se evalúan en el mismo test.

## Uso

```bash
pip install -r requirements.txt
pip install pysentimiento --no-deps      # solo para su preprocesado (RoBERTuito / BETO-sentiment)
pytest -q                                # prueba el código con modelos diminutos locales (sin internet)

# 1. edita config.yaml: ruta del CSV y data.label_map (tus etiquetas -> NEG/NEU/POS)
python 01_make_splits.py                 # particiones (una sola vez)
python 02_zero_shot.py                   # experimento 1
python 03_finetune.py                    # experimento 2 (reanudable)
python 04_compare.py                     # tablas, IC, tests y figuras
```

Opciones útiles: `--models robertuito_sentiment xlmt_sentiment`, `--seeds 42` (prueba rápida),
`--splits test val` (en `02`), `--overwrite`.

## Decisiones de diseño

**Particiones (70/15/15, estratificadas, `seed` fija).** Train entrena, val elige el mejor epoch y
la parada temprana, **test se usa una sola vez** al final. Se guardan en `splits/` junto a
`split_report.json` (distribución de clases, hash del CSV, textos duplicados y textos con etiquetas
contradictorias). Los textos idénticos siempre caen en la misma partición. Si tienes una columna de
vídeo, ponla en `data.group_col`: la partición pasa a ser por grupo y evita que comentarios del mismo
vídeo (mismo tema y vocabulario) aparezcan en train y test.

**Experimento 1: solo modelos con cabeza de sentimiento.** Un modelo base (BETO, RoBERTa-BNE...)
tiene la cabeza de clasificación sin entrenar: evaluarlo "sin entrenar" daría resultados aleatorios. Por eso
llevan `zero_shot: false` y solo participan en el experimento 2.

**Etiquetas.** Todo se traduce a NEG/NEU/POS. Si un modelo tiene más clases (1-5 estrellas), se suman sus
probabilidades por etiqueta (`label_map`). Si tu dataset no tiene neutral, la clase neutral de los modelos se
descarta y se renormaliza. Si no sabe traducir una clase, falla con un mensaje claro en vez de adivinar.

**Preprocesado por modelo.** Cada familia espera sus marcadores (`@usuario`/`url` en RoBERTuito y
BETO-sentiment; `@user`/`http` en XLM-T). Los `replacements` de `config.yaml` traducen tus `<USUARIO>` y `<URL>`.
Con el formato equivocado, el modelo original rinde menos de lo que puede y el experimento 1 queda sesgado
contra él.

**Fine-tuning.** Bucle propio en PyTorch (AdamW, calentamiento + decaimiento lineal, recorte de gradiente,
precisión mixta, mejor epoch por macro-F1 de validación, parada temprana). 3 semillas por modelo.
Si el modelo ya traía cabeza de sentimiento con las mismas clases, se **reutiliza reordenada** a tu orden de
etiquetas (test incluido); si no, se crea una nueva.

**Métrica principal: macro-F1** (no la domina la clase mayoritaria). También accuracy, accuracy balanceada,
kappa, MCC, métricas por clase y matrices de confusión.

## Resultados (`results/`)

```
results/zero_shot/<modelo>/                  metrics_test.json, predictions_test.csv, run_info.json
results/finetune/<modelo>/seed_<n>/          + history.json (curvas por epoch)
results/comparison/
    summary.md / summary.csv                 tabla principal (media ± desv. entre semillas, IC95% bootstrap)
    finetuning_gain.csv                      mejora de macro-F1 por fine-tuning
    significance.csv                         bootstrap pareado entre modelos, con corrección de Holm
    per_class.csv, all_runs.csv, macro_f1.png, confusion_matrices.png
```

Cada `run_info.json` guarda el commit de git, versiones de librerías, el commit exacto del modelo en
Hugging Face y la configuración. Para fijar un modelo, añade `revision: <hash>` en su entrada de `config.yaml`.

## Cómo leer la comparación

- Si los IC95% de dos modelos se solapan mucho, no afirmes que uno es mejor: mira `significance.csv`
  (`p_holm`; el p-valor bootstrap es aproximado y 0.0 significa "< 1/n_boot").
- Para fine-tuning, compara la media ± desviación entre semillas; la diferencia debe ser mayor que esa variación.
- Describe en el TFM que el experimento 1 mide la **transferencia de dominio** (tweets de TASS → comentarios
  de YouTube políticos), no la calidad intrínseca del modelo.

## Advertencias

- **No he podido ejecutarlo con los modelos reales** (el entorno donde lo preparé no accede a Hugging Face).
  Probé la lógica completa con modelos transformer diminutos locales: particiones, mapeo de etiquetas,
  reordenación de la cabeza, ambos experimentos y la comparación (18 tests). Los nombres de modelo del
  `config.yaml` conviene comprobarlos antes de lanzar; el autor de `beto-sentiment-analysis` avisa de que
  puede eliminarse.
- Los modelos de `pysentimiento` están sujetos a licencias de uso científico y no comercial.
- Tiempo: 7 modelos × 3 semillas son muchas ejecuciones. Con GPU conviene empezar con `--seeds 42` y
  pocos modelos para estimar el tiempo (depende de la GPU y de `max_length`). En CPU no es práctico.
- Si el dataset está muy desbalanceado, prueba `class_weights: balanced` (y repite para todos los modelos).
- `max_length: 128` trunca comentarios largos; comprueba qué porcentaje de tus comentarios supera ese tamaño.
