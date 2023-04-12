# TopoNet: A New Baseline for Scene Topology Reasoning

This reporsitory will contain the source code of **TopoNet** from the paper, [Topology Reasoning for Driving Scenes](https://arxiv.org/abs/2304.05277).

TopoNet is the first end-to-end framework capable of abstracting traffic knowledge beyond conventional perception tasks, ie., **reasoning connections between centerlines and traffic elements** from sensor inputs. It unifies heterogeneous feature
learning and enhances feature interactions via the graph neural network architecture and the knowledge graph design.

![method](figs/pipeline.png "Model Architecture")

> **Topology Reasoning for Driving Scenes**
> 
> Tianyu Li*, Li Chen*†, Xiangwei Geng, Huijie Wang, Yang Li, Zhenbo Liu, Shengyin Jiang, Yuting Wang, Hang Xu, Chunjing Xu, Feng Wen, Ping Luo, Junchi Yan, Wei Zhang, Xiaogang Wang, Yu Qiao, Hongyang Li†.
>
> Paper: [Full paper on arXiv](https://arxiv.org/abs/2304.05277)


## News

- Code & model will be released **around June**. Please stay tuned!
- [2023/4/11] TopoNet [paper](https://arxiv.org/abs/2304.05277) is available on arXiv.

## Main Results

We provide results on [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2). Models will be released together with codes.

|    Method    |  Backbone | Epoch | DET<sub>l</sub> | DET<sub>l,chamfer</sub> | TOP<sub>ll</sub> | DET<sub>t</sub> | TOP<sub>lt</sub> |  OLS | Model
|:------------:|:---------:|:-----:|:-------:|:-------:|:----------:|:-------:|:----------:|:----:|:------:|
|     STSU     | ResNet-50 |   24  |   12.0  |  11.5  |     0.3    |   **62.3**  |    10.1    | 27.9 |    -    |
| VectorMapNet | ResNet-50 |   24  |   11.3  |  13.4  |     0.1    |   58.5  |    6.2     | 24.5 |    -    |
|     MapTR    | ResNet-50 |   24  |   8.3   |  17.7  |     0.2    |   60.7  |    5.8     | 24.3 |    -    |
|     MapTR*   | ResNet-50 |   24  |   8.3   |  17.7  |     1.1    |   60.7  |    10.1    | 30.2 |    -    |
|    **TopoNet**   | ResNet-50 |   24  |   **22.1**  |  **20.2**  |     **2.7**    |   59.1  |    **14.9**    | **34.0** |    -    |
<!-- | TopoNet-swin |   Swin-t  |   24  |   22.5  |  21.7  |     2.6    |   71.7  |    17.8    | 38.2 |    -    | -->

> $*$: evaluation based on matching results on Chamfer distance.

## License

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@article{li2023toponet,
  title={Topology Reasoning for Driving Scenes},
  author={Li, Tianyu and Chen, Li and Geng, Xiangwei and Wang, Huijie and Li, Yang and Liu, Zhenbo and Jiang, Shengyin and Wang, Yuting and Xu, Hang and Xu, Chunjing and Wen, Feng and Luo, Ping and Yan, Junchi and Zhang, Wei and Wang, Xiaogang and Qiao, Yu and Li, Hongyang},
  journal={arXiv preprint arXiv:2304.05277},
  year={2023}
}
```

## Related resources

We acknowledge all the open source contributors for the following projects to make this work possible:

- [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [MapTR](https://github.com/hustvl/MapTR)

