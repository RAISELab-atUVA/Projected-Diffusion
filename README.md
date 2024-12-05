<div align="center"><h1>&nbsp;Projected Diffusion Models</h1></div>

<!-- <p align="center">
| <a href="https://arxiv.org/pdf/2402.03559.pdf"><b>Paper</b></a> | 
<a href="https://neurips.cc/virtual/2024/poster/95942"><b>Poster</b></a> |
</p> -->

<p align="center">
<a href="https://arxiv.org/abs/2402.03559" target="_blank">
  <img src="https://img.shields.io/badge/arXiv-2402.03559-red.svg" alt="arXiv">
</a>
<a href="https://neurips.cc/virtual/2024/poster/95942" target="_blank">
  <img src="https://img.shields.io/badge/NeurIPS%20Poster-2024-blue.svg" alt="NeurIPS Poster">
</a>
<a href="">
    <img src="https://img.shields.io/badge/Version-v1.0.0-orange.svg" alt="Version">
  </a>
<!--   <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a> -->

</p>

<p align="center">
  <img src="./figs/constraint_violations.png" alt="benchmark" width="790">
</p>

**Projected Diffusion Models (PDM)** alter the sampling step in diffusion models to generate outputs that satisfy desired constraints and physical principles. The approach endows generative diffusion processes the ability to *satisfy and certify compliance with constraints and physical principles.* The proposed method recasts the traditional sampling process of generative diffusion models as a constrained optimization problem, steering the generated data distribution to remain within a specified region to ensure adherence to the given constraints. These capabilities are validated on applications featuring both convex and challenging, non-convex, constraints as well as ordinary differential equations, in domains spanning from synthesizing new materials with precise morphometric properties, generating physics-informed motion, optimizing paths in planning scenarios, and human motion synthesis.

Explored Settings:
- [x] **Constrained Microstructure Generation:** Behavior in low data regimes and with original distribution violating constraints, as part of a real-world material science experiment.
- [x] **Physics-Informed Motion:** Behavior on ODEs and under constraints outside the training distribution.
- [ ] **Constrained Trajectory Optimization:** Behavior on complex non-convex constraints.
- [ ] **3D Human Motion:** Behavior on 3-dimensional sequence generation with physical constraints.

---

## Training

For a given experiment, the model can be trained by setting the `--mode` flag to **'train'**.


```
python main.py --mode 'train' --experiment [experimental setting] --train_set_path [/path/to/train/data] --val_set_path [/path/to/val/data] --run_name [run name]
```


## Inference

Constrained sampling for a given experimental setting can be done by setting the `--mode` flag to **'sample'**.

```
python main.py --mode 'sample' --n_samples 1 --experiment [experimental setting] --model_path [path/to/model/checkpoints]
```  


## Example: Physics-Informed Motion Experiment

Begin by setting up your environment.

Now, we can train the model:

```
python main.py --mode 'train' --experiment 'motion' --run_name 'example_training'
```

When the training has converged, we can run the inference step:

```
python main.py --mode 'sample' --n_samples 5 --experiment 'motion' --model_path 'examples/motion/models/example_training/ckpt.pt'
```  

Now, five consecutive frames have been generated in the `outputs` directory. By default, this experiment generates frames with the Earth's gravitational pull, but we can change to use the moon's gravity by rerunning the inference step with a different ODE.

```
python main.py --mode 'sample' --n_samples 5 --experiment 'motion' --model_path 'examples/motion/models/example_training/ckpt.pt' --gravity 1.6
```  

The `outputs` directory will now contain frames with the altered dynamics for the governing equations.




## Reference
For technical details and full experimental results, please check [our paper](https://arxiv.org/pdf/2402.03559.pdf).
```
@inproceedings{christopher2024constrained, 
	author = {Jacob K Christopher, Stephen Baek, and Ferdinando Fioretto}, 
	title = {Constrained Synthesis with Projected Diffusion Models}, 
	booktitle = {Neural Information Processing Systems},
	year = {2024}
}
```

## Acknowledgements

This research is partially supported by NSF grants 2334936, 2334448, and NSF CAREER Award 2401285. Fioretto is also supported by an Amazon Research Award and a Google Research Scholar Award. The authors acknowledge Research Computing at the University of Virginia for providing computational resources that have contributed to the results reported within this paper. The views and conclusions of this work are those of the authors only
