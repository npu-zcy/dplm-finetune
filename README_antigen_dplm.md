# HI-distance DPO + DPLM-2 Antigen Map Pipeline

This folder contains `train_antigen_dplm.py`, a complete training pipeline for:

1. Reading HA metadata/sequence rows.
2. Reading HI pair rows.
3. Sampling preference triplets from HI-derived distance.
4. Fine-tuning a multimodal dplm-2 encoder with DPO + optional LoRA.
5. Freezing most of the encoder for downstream two-dimensional antigen map training.
6. Exporting final coordinates.

## Inputs

HA CSV columns:

```text
index,name,location,id,year,seq
```

HI CSV columns:

```text
at_index,sr_index,max_year,min_year,distance,class
```

Structure files are searched recursively by `id` first, then by sanitized `name`, under `--structure-dir`.
Supported feature formats are `.pt`, `.npy`, `.npz`. Raw `.pdb` and `.cif` files are accepted but currently mapped to zero features unless you replace the parsing branch in `StructureLoader`.

By default, viruses with exactly the same HA sequence share one structure file. The script writes the selected representative structure to:

```text
outputs/structure_sequence_dedup_mapping.csv
```

Disable this behavior with `--no-dedupe-structure-by-seq`.

## Quick pipeline test

Use the built-in simple encoder first to verify the data pipeline:

```powershell
python .\train_antigen_dplm.py `
  --ha .\HA.csv `
  --hi .\HI.csv `
  --structure-dir /home/zhouchunyan/postgraduate/influenza-virus_LLM/research2_geometric_graph_learning/myResearch/data/features/Structures/H5N1 `
  --encoder simple `
  --output-dir .\outputs_simple
```

Use all valid HI triplets instead of random sampling:

```powershell
python .\train_antigen_dplm.py `
  --ha .\HA.csv `
  --hi .\HI.csv `
  --structure-dir /home/zhouchunyan/postgraduate/influenza-virus_LLM/research2_geometric_graph_learning/myResearch/data/features/Structures/H5N1 `
  --encoder simple `
  --hi-triplet-mode all `
  --output-dir .\outputs_all_triplets
```

## DPLM-2 run

Wire your local dplm-2 class through the adapter arguments:

```powershell
python .\train_antigen_dplm.py `
  --ha .\HA.csv `
  --hi .\HI.csv `
  --structure-dir .\structures `
  --encoder dplm2 `
  --dplm-module byprot.models.dplm2 `
  --dplm-class MultimodalDiffusionProteinLanguageModel `
  --dplm-model /home/zhouchunyan/postgraduate/influenza-virus_LLM/research2_dplm/my_dplm/airkingbd/dplm2_150m `
  --dplm-struct-tokenizer-dir /home/zhouchunyan/postgraduate/influenza-virus_LLM/research2_dplm/my_dplm/airkingbd/struct_tokenizer `
  --use-lora `
  --lora-targets q_proj,k_proj,v_proj,o_proj `
  --output-dir .\outputs_dplm2
```

If your dplm-2 forward API is different, edit only `DPLM2Backbone` in `train_antigen_dplm.py`.

## Main outputs

```text
outputs/hi_dpo_triplets.csv
outputs/sequence_triplets.csv
outputs/stage1_dplm2_lora/
outputs/stage2_antigen_map.pt
outputs/antigen_map_coordinates.csv
```
