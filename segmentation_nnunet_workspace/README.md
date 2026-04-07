# nnU-Net Segmentation Workspace

Esta carpeta deja aislado el flujo de segmentacion de CoW con nnU-Net.

## Estructura esperada

- `code/segmentation/`: scripts del proyecto para manifest y export de datasets.
- `nnUNet/`: aqui puedes clonar la copia original de `MIC-DKFZ/nnUNet`.
- `topcow-2024-nnunet/`: copia local previa. Solo se usa como fallback si `nnUNet/` no existe.
- `nnUNet_raw/`: datasets exportados para nnU-Net.
- `nnUNet_preprocessed/`: salida de `plan_and_preprocess`.
- `nnUNet_results/`: checkpoints y resultados de entrenamiento.
- `logs/`: logs de export, preprocess y train.

## Recomendacion de entorno

No se instala nada desde aqui. Cuando crees tu entorno, usa una version de PyTorch recomendada por el repo de nnU-Net para evitar la regresion de rendimiento con 3D conv + AMP:

- `torch <= 2.8.x`

## Paso a paso

1. Crear y activar tu nuevo ambiente.
2. Clonar aqui tu nnU-Net original en `segmentation_nnunet_workspace/nnUNet`.
3. Instalar nnU-Net y dependencias dentro de ese ambiente.
4. Cargar variables del workspace:

```bash
source /Users/alejo/Documents/Internship/NeuroFlow/segmentation_nnunet_workspace/env_nnunet.sh
```

5. Exportar datasets 301, 302 y 303:

```bash
bash /Users/alejo/Documents/Internship/NeuroFlow/segmentation_nnunet_workspace/run_export_datasets_301_303.sh
```

6. Planificar y preprocesar:

```bash
bash /Users/alejo/Documents/Internship/NeuroFlow/segmentation_nnunet_workspace/run_plan_and_preprocess_301_303.sh
```

7. Entrenar un fold:

```bash
DATASET_ID=301 FOLD=0 GPU=0 bash /Users/alejo/Documents/Internship/NeuroFlow/segmentation_nnunet_workspace/run_train_one.sh
```

## Datasets definidos

- `301`: `CoW3TMagProj`
- `302`: `CoW3TMagVelProj`
- `303`: `CoW3TAngioMagSpeed`

## Clone recomendado

```bash
cd /Users/alejo/Documents/Internship/NeuroFlow/segmentation_nnunet_workspace
git clone https://github.com/MIC-DKFZ/nnUNet.git
```

Los wrappers de esta carpeta detectan `nnUNet/` automaticamente.

## Notas

- El export usa como default `mag projection = max`.
- El `velocity projection` de `302` usa `abs_max` por componente.
- El script de export lee el manifest generado en `data/segmentation_nnunet/manifest_master_3t_gt7t.csv`.
