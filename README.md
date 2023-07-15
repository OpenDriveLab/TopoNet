# TopoNet: A New Baseline for Scene Topology Reasoning

This reporsitory contains the source code of **TopoNet**, [Topology Reasoning for Driving Scenes](https://arxiv.org/abs/2304.05277).

TopoNet is the first end-to-end framework capable of abstracting traffic knowledge beyond conventional perception tasks, ie., **reasoning connections between centerlines and traffic elements** from sensor inputs. It unifies heterogeneous feature
learning and enhances feature interactions via the graph neural network architecture and the knowledge graph design. 

We believe instead of recognizing lanes, modelling the laneline topology is the right thing to construct components within perception framework, to facilitate the ultimate driving comfort. This is in accordance with the [UniAD philosophy](https://github.com/OpenDriveLab/UniAD).

> **Topology Reasoning for Driving Scenes**
> 
> Tianyu Li*, Li Chen*, etc., Hongyang Li
>
> Paper: [Full paper on arXiv](https://arxiv.org/abs/2304.05277)

![method](figs/pipeline.png "Model Architecture")



## News

- Code & model will be released **around Late July**. Please stay tuned!
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

@article{wang2023openlanev2,
  title={OpenLane-V2: A Topology Reasoning Benchmark for Scene Understanding in Autonomous Driving}, 
  author={Wang, Huijie and Li, Tianyu and Li, Yang and Chen, Li and Sima, Chonghao and Liu, Zhenbo and Wang, Yuting and Jiang, Shengyin and Jia, Peijin and Wang, Bangjun and Wen, Feng and Xu, Hang and Luo, Ping and Yan, Junchi and Zhang, Wei and Li, Hongyang},
  journal={arXiv preprint arXiv:2304.10440},
  year={2023}
}
```

## Related resources

We acknowledge all the open source contributors for the following projects to make this work possible:

- [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)

