# Rate-Accounted Cross-Modality Structure and Timestep-Routed Priors for RGB–Thermal Image Compression


## Abstract

This repository provides the implementation of **DuP-CMDC**, a dual-prior diffusion-based framework for RGB–Thermal joint image compression at extremely low bitrates.

DuP-CMDC combines transmitted image-specific structural information with transmission-free stage-specific restoration priors to improve cross-modal compression and generative reconstruction. The framework consists of three key components:

- **Thermal-side Structured Prompt (TSP):** Encodes scene geometry and target attributes into compact structural side information shared by both RGB and Thermal modalities.
- **Stage Prompt Prior (SPP):** Provides timestep-adaptive restoration guidance derived from multistage reconstruction errors without additional per-image transmission overhead.
- **Dual-Prior Modality Adapter (DPMA):** Integrates TSP and SPP into a frozen SANA Diffusion Transformer through Cross-Attention and FiLM, with a modality-specific LowRank path for modality adaptation.



## Setup

```
git clone https://github.com/gmx-9752/DuP-CMDC.git
cd DuP-CMDC
conda env create -f environment.yml
conda activate dup_cmdc
```

## Train

```
bash RUN_TRAIN.sh
```

Before starting training, download the pretrained [SANA](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers) model into ./weight.

## Test

For quick evaluation, use

```
bash RUN_EVAL.sh
```

To save the compressed representation of an image as a `diffc` file, use

```
python compress.py --config configs/SanaDiT-600M-1024-mydata.yaml --checkpoint outputs/best_val_macro.pt --rgb_list /path/to/your/RGB/test/images.list --thermal_list /path/to/your/thermal/test/images.list --recon_timestep 170 --output_dir results/bitstreams
```

To reconstruct an image/images from their compressed representations, use

```
python decompress.py --config configs/SanaDiT-600M-1024-mydata.yaml --checkpoint outputs/best_val_macro.pt --input_dir results/bitstreams --output_dir results/bitstreams/recon
```

Note that currently, compress and decompress.py only work with `SanaDiT-600M-1024-mydata.yaml`. To make them work with the other configs, you would need to specify `manual_dkl_per_step` in the config file.


## Acknowledgments

We sincerely thank the authors of [diffc](https://github.com/JeremyIV/diffc) for making their valuable research and implementations publicly available.

Our implementation builds upon their codebase. We acknowledge their contributions and refer users to the corresponding repositories for the original methods, implementation details, and licensing information.

## License and Attribution

The original implementation of diffc remains subject to their respective licenses and copyright notices. Any redistributed or modified third-party components must comply with the applicable license terms.

Please refer to the original repositories and dataset providers for their licensing conditions.
