# TopoNet
Topology Reasoning for Driving Scenes

Tianyu Li,
Li Chen,
Xiangwei Geng,
Huijie Wang,
Yang Li,
Zhenbo Liu,
Shengyin Jiang,
Yuting Wang,
Hang Xu,
Chunjing Xu,
Feng Wen,
Ping Luo,
Junchi Yan,
Wei Zhang,
Xiaogang Wang,
Yu Qiao,
Hongyang Li

![method](figs/pipeline.png "Model Architecture")

---

## Abstract
Understanding the road genome is essential to realize autonomous driving.
This highly intelligent problem contains two aspects - the connection relationship of lanes, and the assignment relationship between lanes and traffic elements, where a comprehensive topology reasoning method is vacant.
On one hand, previous map learning techniques struggle in deriving lane connectivity with segmentation or laneline paradigms; or prior lane topology-oriented approaches focus on centerline detection and neglect the interaction modeling. 
On the other hand, the traffic element to lane assignment problem is limited in the image domain, leaving how to construct the correspondence from two views an unexplored challenge.
To address these issues, we present ***TopoNet***, the first end-to-end framework capable of abstracting traffic knowledge beyond conventional perception tasks.
To capture the driving scene topology, we introduce three key designs: (1) an embedding module to incorporate semantic knowledge from 2D elements into a unified feature space; (2) a curated scene graph neural network to model relationships and enable feature interaction inside the network; (3) instead of transmitting messages arbitrarily, a scene knowledge graph is devised to differentiate prior knowledge from various types of the road genome.
We evaluate TopoNet on the challenging scene understanding benchmark, **OpenLane-V2**, where our approach outperforms all previous works by a great margin on all perceptual and topological metrics.
The code would be released soon.


## News

- Code & model with be released soon. Please stay tuned!
- [2023/4/11] TopoNet [paper]() is available on arXiv!

## Main Results

| Method       | backbone  | epoch | DET$_l$ | TOP$_{ll}$ | DET$_t$ | TOP$_{lt}$ | OLS  |
|--------------|-----------|-------|---------|------------|---------|------------|------|
| TopoNet      | ResNet-50 | 24    | 22.1    | 2.7        | 59.1    | 14.9       | 34.0 |
| TopoNet-swin | Swin-t    | 24    | 22.5    | 2.6        | 71.7    | 17.8       | 38.2 |

## License

All assets and code are under the [Apache 2.0 license](https://github.com/OpenDriveLab/TopoNet/blob/master/LICENSE) unless specified otherwise.

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@article{li2023topology,
  title={Topology Reasoning for Driving Scenes},
  author={Li, Tianyu and Chen, Li and Geng, Xiangwei and Wang, Huijie and Li, Yang and Liu, Zhenbo and Jiang, Shengyin and Wang, Yuting and Xu, Hang and Xu, Chunjing and Wen, Feng and Luo, Ping and Yan, Junchi and Zhang, Wei and Wang, Xiaogang and Qiao, Yu and Li, Hongyang}
  journal={arXiv preprint arXiv:2304.},
  year={2023}
}
```

## Related resources

- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) (:rocket:Ours!)
- [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2) (:rocket:Ours!)
- [MapTR](https://github.com/hustvl/MapTR)
