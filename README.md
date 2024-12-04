<div align="center"><h1>&nbsp;Projected Diffusion Models</h1></div>

<p align="center">
| <a href="https://arxiv.org/pdf/2402.03559.pdf"><b>Paper</b></a> | 
<a href="https://neurips.cc/virtual/2024/poster/95942"><b>Poster</b></a> |
</p>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/Version-v1.0.0-orange.svg" alt="Version">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">
  </a>

</p>

---

## Training

```
python main.py --mode 'train' --experiment [experimental setting] --train_set_path [/path/to/train/data] --val_set_path [/path/to/val/data] --run_name [run name]
```


## Inference

```
python main.py --mode 'sample' --n_samples 1 --experiment [experimental setting] 
```


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