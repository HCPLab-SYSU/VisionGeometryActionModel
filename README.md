<!-- [![Website Badge](https://raw.githubusercontent.com/referit3d/referit3d/eccv/images/project_website_badge.svg)]() -->
[![arXiv](https://img.shields.io/badge/arXiv-2604.12908-b31b1b.svg?style=plastic)]()
# Robotic Manipulation is Vision-to-Geometry Mapping ($f(v) \rightarrow G$): Vision-Geometry Backbones over Language and Video Models
This is the official PyTorch implementation of **Robotic Manipulation is Vision-to-Geometry Mapping ($f(v) \rightarrow G$): Vision-Geometry Backbones over Language and Video Models**


[📘paper](https://arxiv.org/abs/2604.12908) |
<!-- [🚀Project Page](https://hcplab-sysu.github.io/PhysicalAutoregressiveModel/) | -->

## Abstract
At its core, robotic manipulation is a problem of vision-to-geometry mapping ($f(v) \rightarrow G$). Physical actions-such as reaching, grasping, and orienting-are fundamentally defined by geometric properties like 3D positions, rotations, and spatial relationships. Consequently, we argue that the foundation for generalizable robotic control should be a vision-geometry backbone, rather than the widely adopted vision-language or video models. Conventional Vision-Language-Action (VLA) and video-predictive models rely on backbones pretrained on large-scale 2D image-text or temporal pixel data. While effective, their representations are largely shaped by semantic concepts or 2D priors, which do not intrinsically align with the precise 3D geometric nature required for physical manipulation. Driven by this insight, we propose the Vision-Geometry-Action (VGA) model, which directly conditions action generation on pretrained native 3D representations. Specifically, VGA replaces conventional language or video backbones with a pretrained 3D world model, establishing a seamless vision-to-geometry mapping that translates visual inputs directly into physical actions. To further enhance geometric consistency, we introduce a Progressive Volumetric Modulation module and adopt a joint training strategy that simultaneously predicts actions and 3D properties, improving both representation fidelity and cross-modal interaction. Extensive experiments validate the effectiveness of our approach. In simulation benchmarks, VGA outperforms top-tier VLA baselines including $\pi_{0.5}$, SpatialVLA, and GeoVLA, demonstrating its superiority in precise spatial manipulation. More importantly, VGA exhibits remarkable zero-shot generalization to unseen viewpoints in real-world deployments, consistently outperforming $\pi_{0.5}$ in terms of success rate. These results highlight that operating on native 3D representations-rather than translating through language or 2D video priors-is a highly promising direction for achieving generalizable physical intelligence.

---

<p align="center">
  <img src="docs/assets/img/overview/figure2.png" alt="" width="800">
</p>

<p align="center">
  <em>Fig 1: The illustration of Vision Geometry Action Model.</em>
</p>


---


## Citation 
```
@article{song2026robotic,
        title={Robotic manipulation is vision-to-geometry mapping (f (v)→ g): Vision-geometry backbones over language and video models},
        author={Song, Zijian and Li, Qichang and Zhou, Jiawei and Yuan, Zhenlong and Chen, Tianshui and Lin, Liang and Wang, Guangrun},
        journal={arXiv preprint arXiv: https://arXiv. org/abs/2604.12908},
        year={2026}
}
```