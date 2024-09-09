### Lightweight model for the heavy dataset RAISE-LPBF

This repo is dedicated to my research on finding a lightweight model to reconstruct the laser parameters of the RAISE-LBPF dataset.
Most of the architectures/models I worked on are omitted as well as the Neural Architecture Search (NAS) code.

More information on [Makebench](https://www.makebench.eu/benchmark/The%20RAISE-LPBF-Laser%20benchmark).

In brief, I tested many architectures to determine which one is the more appropriate for our case. One architecture was outperforming the other one, the 3D-CNN based models. I then managed to use Neural Architecture Search (NAS) to found the best 3D-CNN based model.

Below are the NAS results.
![NAS results](https://github.com/abadiet/RAISE-LPBF/blob/main/NAS_results.png)
