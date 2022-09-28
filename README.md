# Optimistic Posterior Sampling for Reinforcement Learning with Few Samples and Tight Guarantees

Official implementation of `OPSRL` algorithm and baselines from the paper D.Tiapkin et al. "Optimistic Posterior Sampling for Reinforcement Learning with Few Samples and Tight Guarantees". The algorithms are implemented in the folder `algorithms/`, the parameters are contained in the folder `config\`.

Requirements:
* Python 3.8
* rlberry 0.2.1

Running experiment `opsrl_vs_baselines` and generate the plots
```
    python run.py config/experiments/opsrl_vs_baselines.yaml
    python plot_opsrl_vs_baselines.py
```

Running experiment `opsrl_samples` and generate the plots
```
    python run.py config/experiments/opsrl_samples.yaml
    python plot_opsrl_samples.py
```

Running experiment `opsrl_prior` and generate the plots
```
    python run.py config/experiments/opsrl_prior.yaml
    python plot_opsrl_prior.py
```
