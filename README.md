# Feature attribution for robust Optimal Transport Neural Nets and single-cell RNA sequenced data

This repository contains code, figures and the thesis text of Ulli Steindl's master's thesis project. 

The project investigates a scRNA data set using robust Neural Networks, and a number of XAI methods including feature attribution methods. 

## Motivation
This code was written to investigate features and variation in a scRNA data set involving HGSOC (high-grade Serous Ovatian carcinoma) in different culturing contexts. The data is not included in this repository but is available upon request from the drugsens team [^drugsens].

The pipeline includes several major components: 
- The training and benchmarking of Optimal Transport Neural Nets[^otnn] on this data set, usign the `torchlip` libary [^torchlip]. 
- The visualization the relative importance of the input features using feature attributions from Explainable AI (XAI) usign the `Xplique` library [^xplique]


## Execution
The environment specs are given in `environment.yml`

To run hyperparameter tuning and model training, run these scripts. There were three different experimental setups, based on the input parameters used (small, diffex, full), and two further parameters, based on whether or not Optimal Transport Neural Nets (OTNNs) are used, or neural nets (NN). The condition is specified in parenthesis.

Hyperparameter tuning optimizes the $alpha$, and hinge margin value as well as learning rate, batch size, and network size of OTNNs, and the learning rate and batch size, optimizing for the model loss, for at most 3 epochs. The subsequent optimal parameter set is then trained for 200 maximum epochs, with early stopping and a patience of 5. 

```
src/train_baseline_full_diffex.sh (diffex, baseline)
src/train_full_baseline.sh (full, baseline)
src/train_small_baseline.sh (small, baseline)
src/train_models_full_diffex.sh (diffex, OTNN)
src/train_full_models.sh (full, OTNN)
src/train_small_models.sh (small, OTNN)
```

Generate attributions, test statistics, and attribution metrics are calculated in the following scripts.  
```
src/run_attributions.sh
src/run_attributions_diffex.sh
src/run_attributions_full.sh
```


## Discussion
An in-depth discussion of the preliminary training setup is given in `docs/Masters_thesis_Artificial_Intelligence_2026_Ulrike_Steindl.pdf`.

## Plots

Some plots [available here](https://ullisteindl.github.io/xai-for-hgsoc/)

## Acknowledgments

This work was done while I was part of the group [Machine Learning in Cancer Genetics](https://www.mlcg.rwth-aachen.de/cms/~bjstxw/mlcg/) at RWTH Aachen, and also a student in the Artificial Intelligence Master's program at Maastricht University.

## References
[^otnn]: M. Serrurier, F. Mamalet, T. Fel, L. Béthune, and T. Boissin, *On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective,* Feb. 02, 2024, arXiv: arXiv:2206.06854. doi: 10.48550/arXiv.2206.06854.

[^drugsens]: K. B. Labrosse et al., *Ex vivo drug sensitivity testing predicts treatment outcomes in advanced ovarian cancer,* npj Precis. Onc., vol. 10, no. 1, p. 273, Feb. 2026, doi: 10.1038/s41698-026-01321-4.

[^torchlip]: M. Serrurier, F. Mamalet, A. González-Sanz, T. Boissin, J.-M. Loubes, and E. del Barrio, *Achieving robustness in classification using optimal transport with hinge regularization,* in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2021, pp. 505–514. doi: 10.1109/CVPR46437.2021.00057. GitHub. https://github.com/deel-ai/deel-torchlip

[^xplique]: T. Fel et al., *Xplique: A Deep Learning Explainability Toolbox,* Jun. 09, 2022, arXiv: arXiv:2206.04394. doi: 10.48550/arXiv.2206.04394. GitHub. https://github.com/deel-ai/xplique 

