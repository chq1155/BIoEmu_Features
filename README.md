
<h1>
<p align="center">
    <img src="assets/emu.png" alt="BioEmu logo" width="300"/>
</p>
</h1>

[![DOI:10.1101/2024.12.05.626885](https://zenodo.org/badge/DOI/10.1101/2024.12.05.626885.svg)](https://doi.org/10.1101/2024.12.05.626885)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)


# Biomolecular Emulator (BioEmu)

Biomolecular Emulator (BioEmu for short) is a model that samples from the approximated equilibrium distribution of structures for a protein monomer, given its amino acid sequence.

For more information see our <a href="assets/bioemu_paper.pdf" target="_blank">paper</a>, [citation below](#citation).

This repository contains inference code and model weights.

## Table of Contents
- [Installation](#installation)
- [Sampling structures](#sampling-structures)
- [Steering to avoid chain breaks and clashes](#steering-to-avoid-chain-breaks-and-clashes)
- [Azure AI Foundry](#azure-ai-foundry)
- [Training data](#training-data)
- [Get in touch](#get-in-touch)
- [Citation](#citation)

## Installation
bioemu is provided as a Linux-only pip-installable package. We support Python 3.10 and above:

```bash
pip install bioemu
```

To install with CUDA support:

```bash
pip install bioemu[cuda]
```

> [!NOTE]
> BioEmu uses an inlined version of [ColabFold](https://github.com/sokrypton/ColabFold) and [AlphaFold2](https://github.com/google-deepmind/alphafold) for MSA retrieval and embedding generation. These are bundled with the package — no separate environment or installation is needed. On first use, AlphaFold2 model weights (~3.5 GB) will be automatically downloaded to `~/.cache/colabfold/`.


## Sampling structures
You can sample structures for a given protein sequence using the `sample` module. To run a tiny test using the default model parameters and denoising settings:
```
python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10 --output_dir ~/test-chignolin
```

Alternatively, you can use the Python API:

```python
from bioemu.sample import main as sample
sample(sequence='GYDPETGTWG', num_samples=10, output_dir='~/test_chignolin')
```

The model parameters will be automatically downloaded from [huggingface](https://huggingface.co/microsoft/bioemu). A path to a single-sequence FASTA file can also be passed to the `sequence` argument.

Sampling times will depend on sequence length and available infrastructure. The following table gives times for collecting 1000 samples measured on an A100 GPU with 80 GB VRAM for sequences of different lengths (using a `batch_size_100=20` setting in `sample.py`):
 | sequence length | time / min |
 | --------------: | ---------: |
 |             100 |          4 |
 |             300 |         40 |
 |             600 |        150 |

By default, unphysical structures (steric clashes or chain discontinuities) will be filtered out, so you will typically get fewer samples in the output than requested. The difference can be very large if your protein has large disordered regions which are very likely to produce clashes. If you want to get all generated samples in the output, irrespective of whether they are physically valid, use the `--filter_samples=False` argument.


> [!NOTE]
> If you wish to use your own generated MSA instead of the ones retrieved via the ColabFold MMseqs2 server, you can pass an A3M file containing the query sequence as the first row to the `sequence` argument. Additionally, the `msa_host_url` argument can be used to override the default MSA query server. See [sample.py](./src/bioemu/sample.py) for more options.

This code only supports sampling structures of monomers. You can try to sample multimers using the [linker trick](https://x.com/ag_smith/status/1417063635000598528), but in our limited experiments, this has not worked well.

## Steering to avoid chain breaks and clashes

BioEmu includes a [steering system](https://arxiv.org/abs/2501.06848) that guides the diffusion process toward more physically plausible protein structures.
Steering applies potential energy functions during denoising to favor conformations that satisfy physical constraints.
It uses **SMC (Sequential Monte Carlo)** sampling, which simulates multiple *candidate samples* (particles) per desired output sample and resamples between them according to the favorability of the provided potentials.

Empirically, using three (or up to 10) steering particles per output sample greatly reduces the number of unphysical samples (steric clashes or chain breaks) produced by the model.

### Quick start with steering

Steering is configured via a single YAML file passed as `denoiser_config`. This file specifies the denoiser, potentials, and steering parameters together.

Enable steering with physical constraints using the CLI:

```bash
python -m bioemu.sample \
    --sequence GYDPETGTWG \
    --num_samples 100 \
    --output_dir ~/steered-samples \
    --denoiser_config src/bioemu/config/steering/physical_steering.yaml
```

Or using the Python API:

```python
from bioemu.sample import main as sample

sample(
    sequence='GYDPETGTWG',
    num_samples=100,
    output_dir='~/steered-samples',
    denoiser_config="src/bioemu/config/steering/physical_steering.yaml",
)
```

### Key steering parameters

Inside the steering YAML config (e.g., [`physical_steering.yaml`](./src/bioemu/config/steering/physical_steering.yaml)):

- `num_particles`: Number of particles per sample (higher = stronger steering, more compute)
- `ess_threshold`: Effective sample size threshold for resampling (0.0–1.0)
- `start`: Diffusion time to start steering (0.0–1.0, default: 0.1; reverse process goes 1→0)
- `end`: Diffusion time to stop steering (0.0–1.0, default: 0.0)
- `fk_potentials`: List of potential energy functions to apply (Hydra-instantiated)

### Available potentials

The [`physical_steering.yaml`](./src/bioemu/config/steering/physical_steering.yaml) configuration provides potentials for physical realism:
- **CaCaDistance** + **UmbrellaPotential**: Prevents backbone discontinuities by penalizing large Cα–Cα distances
- **PairwiseClash** + **UmbrellaPotential**: Avoids steric clashes between non-neighboring residues

## Azure AI Foundry
BioEmu is also available on [Azure AI Foundry](https://ai.azure.com/). See [How to run BioEmu on Azure AI Foundry](AZURE_AI_FOUNDRY.md) for more details.

## Training data
The molecular dynamics training data used for BioEmu is available on Zenodo:
- [CATH](https://doi.org/10.5281/zenodo.15629740)
- [Octapeptides](https://doi.org/10.5281/zenodo.15641199)
- [MegaSim](https://doi.org/10.5281/zenodo.15641184)

For a full description of these, see the <a href="assets/bioemu_paper.pdf" target="_blank">paper</a>.

## Reproducing results from the paper
You can use this code together with code from [bioemu-benchmarks](https://github.com/microsoft/bioemu-benchmarks) to approximately reproduce results from our [paper].

- The `bioemu-v1.0` checkpoint contains the model weights used to produce the results in the preprint. Due to simplifications made in the embedding computation and a more efficient sampler, the results obtained with this code are not identical but consistent with the preprint statistics, i.e., mode coverage and free energy errors averaged over the proteins in a test set. Results for individual proteins may differ. 
- [Default] The `bioemu-v1.1` checkpoint contains the model weights used to produce the results in the published Science [paper]. 
- The `bioemu-v1.2` checkpoint contains the model weights trained from an extended set of MD simulations and experimental measurements of folding free energies. 

For more details, please check the [BIOEMU_RESULTS.md](https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/BIOEMU_RESULTS.md) document on the bioemu-benchmarks repository.

To use a specific checkpoint, you can specify the `model_name` in the `bioemu.sample` args, for example, `--model_name="bioemu-v1.1"`.


## Side-chain reconstruction and MD-relaxation
BioEmu outputs structures in backbone frame representation. To reconstruct the side-chains, several tools are available. As an example, we interface with [HPacker](https://github.com/gvisani/hpacker) to conduct side-chain reconstruction, and also provide basic tooling for running a short molecular dynamics (MD) equilibration.

> [!WARNING]
> Side-chain reconstruction relies on [HPacker](https://github.com/gvisani/hpacker) which requires a [conda-based package manager](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Make sure that `conda` is in your `PATH` and that you have CUDA12-compatible drivers before running the following code. Note that `conda` is **not** required for BioEmu's core sampling functionality.

Install optional dependencies:

```bash
pip install bioemu[md]
```

You can compute side-chain reconstructions via the `bioemu.sidechains_relax` module:
```bash
python -m bioemu.sidechain_relax --pdb-path path/to/topology.pdb --xtc-path path/to/samples.xtc
```


> [!NOTE]
> The first time this module is invoked, it will attempt to install `hpacker` and its dependencies into a separate virtualenv using a bundled setup script. If the automatic setup fails, you can install hpacker manually by following the instructions at [hpacker's repository](https://github.com/gvisani/hpacker) and setting the `HPACKER_PYTHONBIN` environment variable to the path of the python executable where hpacker is installed.

By default, side-chain reconstruction and local energy minimization are performed (no full MD integration for efficiency reasons).
Note that the runtime of this code scales with the size of the system.
We suggest running this code on a selection of samples rather than the full set.

There are two other options:
- To only run side-chain reconstruction without MD equilibration, add `--no-md-equil`.
- To run a short NVT equilibration (0.1 ns), add `--md-protocol nvt_equil`

To see the full list of options, call `python -m bioemu.sidechain_relax --help`.

The script saves reconstructed all-heavy-atom structures in `samples_sidechain_rec.{pdb,xtc}` and MD-equilibrated structures in `samples_md_equil.{pdb,xtc}` (filename to be altered with `--outname other_name`).

## Extracting MD features
Given a trajectory (the `samples_md_equil.{pdb,xtc}` produced above, or any mdtraj-loadable trajectory), the `bioemu.extract_md_features` module extracts per-frame featurizations for analysis or machine learning. The topology is auto-detected, so it works on both backbone-only samples and side-chain-reconstructed structures.

```bash
python -m bioemu.extract_md_features --traj samples_md_equil.xtc --top samples_md_equil.pdb --out-dir features/
```

This computes dihedrals (backbone + side-chain), Cα–Cα distances and contacts, radius of gyration, RMSD, DSSP, SASA and H-bonds, and writes them to `features/`:
- `*_features.npz` — raw per-frame feature matrices for TICA/MSM analysis.
- `*_features_ml.pt` — z-scored `torch` tensors with normalisation stats for ML training.
- `*_summary.csv`, `*_per_residue.csv`, `*_metadata.json`.

For long trajectories, stream the file in chunks and/or subsample frames:

```bash
python -m bioemu.extract_md_features --traj traj.xtc --top top.pdb --out-dir features/ --chunk 200 --stride 10
```

To see the full list of options, call `python -m bioemu.extract_md_features --help`.

## Extracting BioEmu internal representations
Distinct from the geometric MD features above, the `bioemu.extract_internal_features` module extracts BioEmu's *learned internal activations* — the encoder's per-residue representation `x1d` (width 512) tapped from the diffusion score model via forward hooks. These are not system-size-dependent geometric descriptors but the model's own latent features, captured at chosen encoder layers and diffusion times. Two output forms are produced per `(layer, time)`: `per_residue` of shape `[L, 512]` (one vector per residue, so size-dependent) and `pooled` of shape `[512]` (residue-pooled, fixed width and comparable across proteins of different lengths).

```python
from bioemu.extract_internal_features import extract_bioemu_internal_features

feats = extract_bioemu_internal_features(
    sequence="GYDPETGTWG",        # protein sequence
    layers=(-1, 3, 7),            # -1 = pre-encoder x1d; 0..7 = encoder layer outputs
    diffusion_times=(0.99, 0.5, 0.1),  # in [0,1]; snapped to nearest denoiser timestep
    pooling="mean",               # "mean" (no params) | "attention" | "learned"
    model_name="bioemu-v1.1",
    batch_size=1,
    num_denoiser_steps=50,
    denoiser_type="dpm",          # or "heun"
    seed=0,
)
# returns dict keyed by (layer, time) -> InternalFeature with:
#   .per_residue  Tensor [L, 512]   (system-size dependent)
#   .pooled       Tensor [512]      (fixed width, comparable across proteins)
#   .snapped_time float             (nearest denoiser timestep actually used)
```

Or using the CLI (runnable as a module):

```bash
python -m bioemu.extract_internal_features --sequence GYDPETGTWG \
    --layers -1,3,7 --times 0.99,0.5,0.1 --pooling mean --out feats.pt
```

- **Layer indices**: `-1` is the pre-encoder `x1d` (input to layer 0); `0..7` are the outputs of encoder layers 0 through 7 (8 encoder layers total).
- **`diffusion_times`**: values in `[0,1]` along the reverse process; each is snapped to the nearest denoiser timestep actually visited, reported back as `.snapped_time`.
- **Pooling**: `mean` (default, parameter-free) pools over residues; `attention` and `learned` are trainable modules exposed for later fine-tuning.
- **Output shapes**: `per_residue` is `[L, 512]` (varies with sequence length `L`); `pooled` is `[512]` (fixed width).

> [!NOTE]
> This feature requires a working `torch_geometric` + BioEmu runtime (the same environment used to run BioEmu sampling). On first use it downloads the pretrained `bioemu-v1.1` checkpoint and queries the ColabFold MSA server (or pass a cached `msa_file`). It is verified against `bioemu-v1.1` (repo version `1.4.0`) and asserts the architecture at runtime (8 encoder layers, `d_model` 512), so a future BioEmu architecture change fails loudly rather than silently returning wrong features.

To see the full list of options, call `python -m bioemu.extract_internal_features --help`. A runnable demo is in [`examples/extract_internal_features_demo.py`](examples/extract_internal_features_demo.py).

## Third-party code
- The code in `src/bioemu/openfold/` is copied from [OpenFold](https://github.com/aqlaboratory/openfold) (Apache 2.0) with minor modifications described in the relevant source files.
- The code in `src/_vendor/alphafold/` is a vendored, patched subset of [AlphaFold2](https://github.com/google-deepmind/alphafold) v2.3.2 (Apache 2.0). See [src/_vendor/alphafold/README.md](src/_vendor/alphafold/README.md) for details on the modifications.
- The code in `src/bioemu/colabfold_inline/` contains functions derived from [ColabFold](https://github.com/sokrypton/ColabFold) v1.5.4 (MIT). See the license headers in each file for details.
## Get in touch
If you have any questions not covered here, please create an issue or contact the BioEmu team by writing to the corresponding author on our [paper].

## Citation
If you are using our code or model, please cite the following paper:
```bibtex
@article{bioemu2025,
  title={Scalable emulation of protein equilibrium ensembles with generative deep learning},
  author={Lewis, Sarah and Hempel, Tim and Jim{\'e}nez-Luna, Jos{\'e} and Gastegger, Michael and Xie, Yu and Foong, Andrew YK and Satorras, Victor Garc{\'\i}a and Abdin, Osama and Veeling, Bastiaan S and Zaporozhets, Iryna and Chen, Yaoyi and Yang, Soojung and Foster, Adam E. and Schneuing, Arne and Nigam, Jigyasa and Barbero, Federico and Stimper Vincent and  Campbell, Andrew and Yim, Jason and Lienen, Marten and Shi, Yu and Zheng, Shuxin and Schulz, Hannes and Munir, Usman and Sordillo, Roberto and Tomioka, Ryota and Clementi, Cecilia and No{\'e},  Frank},
  journal={Science},
  pages={eadv9817},
  year={2025},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.adv9817}
}
```
[paper]: https://www.science.org/doi/10.1126/science.adv9817