# GAF-Net for Multi-Disease Corn Leaf Detection

A reproduction of **GAF-Net** (Cao et al., *IEEE Transactions on AgriFood Electronics*, 2025) on a public corn leaf disease dataset, plus a controlled four-model ablation testing two modifications of our own.

GAF-Net modifies the neck of YOLOv8 in two places:

- **GSConv** — a lightweight convolution that splits work between a standard path and a depthwise path, then shuffles the channels so the two actually mix.
- **FASFF** — adaptive spatial feature fusion, which learns per-pixel weights for the three feature-pyramid levels instead of combining them uniformly.

The paper reports 80.8% mAP@0.5, 4.1 points above its own YOLOv8 baseline. This repo asks whether the same modules transfer to a different, larger, public dataset.

Short answer: **on this dataset GAF-Net is an efficiency result, not an accuracy result.** It matches the baseline using 36% fewer parameters.

---

## Headline findings

**1. GAF-Net matches the baseline at two-thirds the size.**
Equal mAP@0.5 (0.9836 vs 0.9833), +0.73 points on mAP@0.5:0.95, with 1.93M parameters against 3.01M and 6.5 GFLOPs against 8.1. That parameter reduction is the only effect here large enough to sit outside run-to-run noise.

**2. Our first comparison was invalid, and finding out why was the most useful part of the project.**
The initial run showed GAF-Net *below* the baseline, with most of the deficit in the gray spot class. We spent a while hunting for an architectural cause. The actual cause was that the two runs had been launched with different optimizer settings — the baseline on Ultralytics' `auto`, GAF-Net with an explicit `SGD` and learning rate. The two `results.csv` files give it away: the baseline logged eight learning-rate columns, GAF-Net logged three.

Retraining under one matched recipe, **with zero change to the architecture**, lifted gray spot AP from 0.9400 to 0.9630 and mAP@0.5 from 0.9769 to 0.9836.

**3. Our proposed fusion module did not work.**
CG-FASFF scored lowest of the four. We report it as a negative result rather than dropping it — details below.

---

## Results (test split: 417 images, 419 instances, single seed)

All four models trained under an identical recipe: 100 epochs, 640px, batch 16, seed 0, deterministic kernels, optimizer and learning rate left at Ultralytics defaults, single Tesla T4.

| Model | Neck conv | P3 block | Fusion | mAP@0.5 | mAP@0.5:0.95 | Params | GFLOPs |
|---|---|---|---|---:|---:|---:|---:|
| **A** YOLOv8n | Conv | C2f | — | 0.9833 | 0.9042 | 3.01 M | 8.1 |
| **B** GAF-Net | GSConv | GSConv | FASFF | 0.9836 | **0.9115** | **1.93 M** | **6.5** |
| **C** + GSConvX | GSConvX | C2f | FASFF | **0.9854** | 0.9053 | 2.73 M | 8.2 |
| **D** + CG-FASFF | GSConvX | C2f | CG-FASFF | 0.9794 | 0.9047 | 2.75 M | 8.3 |

### Per-class AP@0.5

| Model | Blight | Common rust | Gray spot | Healthy |
|---|---:|---:|---:|---:|
| **A** YOLOv8n | 0.9862 | 0.9891 | 0.9630 | 0.9950 |
| **B** GAF-Net | 0.9850 | 0.9915 | 0.9630 | 0.9950 |
| **C** + GSConvX | **0.9872** | **0.9921** | **0.9674** | 0.9950 |
| **D** + CG-FASFF | 0.9833 | 0.9889 | 0.9504 | 0.9950 |

Precision and recall for model D are worth noting on their own: **P 0.9837 (highest of the four), R 0.9290 (lowest)**. It did not get worse at seeing — it got more conservative.

---

## The two modifications

### GSConvX — corrected downsampling

Our GSConv implementation performed its strided convolution with a **1×1 kernel**, so at stride 2 it read one pixel in every four with no spatial support at all. That is textbook aliasing, and it costs fine texture more than anything else. It also departs from the GSConv paper, whose first equation specifies a k×k kernel.

```python
# before
self.cv1 = nn.Conv2d(c1, c_, 1, s)              # k ignored

# after
self.cv1 = nn.Conv2d(c1, c_, k, s, padding=k//2, bias=False)
self.bn1 = nn.BatchNorm2d(c_)
self.act1 = nn.SiLU()
```

Alongside this we restored a single **C2f** block at the P3 branch. Our GAF-Net neck was stacked GSConv throughout, which left it with no cross-stage path anywhere.

### CG-FASFF — cross-level gated, channel-grouped fusion

FASFF has two properties we wanted to change. Each pyramid level scores *itself* with its own 1×1 convolution, with no sight of the other two. And that score is a single number per pixel, shared across all 64 channels.

CG-FASFF concatenates the three reduced feature maps and passes them through **one shared gate**, so every weight is computed with the other levels visible. The gate emits **24 channels instead of 3** — read as three levels by eight groups, softmaxed across levels within each group, then broadcast over each group's share of the channels. The idea was to let texture-sensitive channels draw from P3 while colour-sensitive channels draw from P5, at the same pixel.

```python
w = self.gate(torch.cat(f, dim=1))              # B, 3G, H, W
w = w.view(B, 3, self.g, H, W).softmax(dim=1)   # softmax across levels
w = w.repeat_interleave(self.c2 // self.g, dim=2)
return (w * torch.stack(f, dim=1)).sum(dim=1)
```

Cost: roughly 20K parameters in total, under 1% of the model.

The mechanism is not new in kind. Joint-descriptor gating and per-channel softmax across branches both come from **SKNet** (Li et al., CVPR 2019), which applies them across kernel branches; we apply the same two properties across pyramid levels on top of ASFF.

### Why it failed

Three reasons, in descending order of confidence.

1. **There is no multi-scale problem to solve on this data.** The test split has 419 instances across 417 images — essentially one object per image, with the box covering most of the leaf. All three pyramid levels are describing the same thing. A module that lets channel groups specialise by scale needs objects at different scales to exist.
2. **Sharper gating throws away free ensembling.** Twenty-four weights per pixel instead of three gives the gate room to commit hard to one level per group. When the levels carry redundant information, the uniform averaging it replaced was quietly acting as a useful ensemble. The precision-up, recall-down signature is what that looks like.
3. **Capacity with nothing left to learn.** The extra parameters went into a model already at the ceiling of what this dataset can measure.

---

## An important caveat on all of these numbers

The baseline already scores **98.3% mAP@0.5**, leaving 1.7 points of headroom. All four models fall within **0.6 points** of each other, on a single seed.

| | GAF-Net paper | This repo |
|---|---:|---:|
| Images | 1,219 | 10,884 |
| Labelled instances | 7,460 | ~10,884 |
| Instances per image | 6.1 | **1.0** |
| Baseline mAP@0.5 | 76.7% | **98.3%** |
| Headroom above baseline | 23.3 pts | **1.7 pts** |

A gain the size of the paper's is arithmetically unavailable here. We do not claim model C's 0.21-point lead is real — it cannot be separated from seed variance. The 36% parameter reduction can.

---

## Dataset

[Corn Leaf Disease v17](https://universe.roboflow.com/aziz-jda2k/corn-leaf-disease-hgosu) on Roboflow Universe, YOLOv8 export. Four classes: northern leaf blight, common rust, gray leaf spot, healthy.

| Split | Images | Instances |
|---|---:|---:|
| train | 10,048 | 10,048 |
| valid | 419 | 420 |
| test | 417 | 419 |

Gray spot carries only **53 test instances**, so one missed detection moves its AP by roughly 1.9 points. We use it as a diagnostic signal, not a headline number.

---

## Repository contents

- **`multiple-corn-disease.ipynb`** — the first reproduction: baseline YOLOv8n and GAF-Net. This is the run with the optimizer mismatch described above. Kept for the record; **its reported numbers are superseded** by the controlled ablation in the tables on this page.
- **`results/`** — plots from that first reproduction: training and validation loss curves, mAP curves, precision–recall, per-class AP@0.5, and confusion matrices for both models.

The notebook for the controlled four-model ablation is not in this repo yet.

### Plots from the first reproduction

![Training dashboard](results/plot_full_dashboard.png)

![Loss curves](results/plot_loss_curves.png)

![mAP curves](results/plot_map_curves.png)

![Per-class AP50](results/plot_per_class_ap50.png)

---

## Running it

```bash
conda create -n gafnet python=3.10 -y
conda activate gafnet
pip install ultralytics roboflow opencv-python matplotlib pandas
```

Then open `multiple-corn-disease.ipynb`. It downloads the dataset through the Roboflow API, trains both models and writes the plots to `results/`. It runs end to end on a Kaggle session with a single T4; each 100-epoch run takes about 2.7 hours.

Custom modules cannot be registered with Ultralytics directly, so the notebook patches `parse_model`. Each custom layer is temporarily swapped for a standard `Conv` carrying the same arguments, Ultralytics performs its own channel and depth bookkeeping against the stand-in, and the real module is then built from the resolved channel counts and substituted back. The three fusion inputs are added to the `save` list so their outputs survive for later layers.

---

## Limitations

- **Single seed.** Differences below roughly half a point cannot be separated from initialisation variance. Three seeds per model with mean and standard deviation is the fix.
- **Saturated benchmark.** This dataset cannot rank these architectures. A multi-instance lesion dataset is needed to test the fusion modules for what they were designed to do.
- **No robustness evaluation.** The base paper names occlusion and lighting as open problems and never measures them. Neither did we. Corrupting the test set and re-running validation needs no retraining and is the obvious next experiment.
- **Partial transfer of pretrained weights.** Loading `yolov8n.pt` into a modified architecture only transfers layers whose shapes match, so B, C and D started with partly random necks while A did not. This favours A, which makes B's efficiency result stronger rather than weaker, but it is an asymmetry we should name.

---

## References

- Cao, K., Yang, Y., Xu, L., Wan, X., Yang, L., & Jia, W. (2025). GAF-Net: An enhanced multidisease for corn leaves detection model based on YOLOv8. *IEEE Transactions on AgriFood Electronics, 3*(2), 420–432. https://doi.org/10.1109/TAFE.2025.3561196
- Li, H., Li, J., Wei, H., Liu, Z., Zhan, Z., & Ren, Q. (2024). Slim-neck by GSConv: A lightweight-design for real-time detector architectures. *Journal of Real-Time Image Processing, 21*(3), Article 62.
- Liu, S., Huang, D., & Wang, Y. (2019). Learning spatial fusion for single-shot object detection. arXiv. https://arxiv.org/abs/1911.09516
- Li, X., Wang, W., Hu, X., & Yang, J. (2019). Selective kernel networks. *CVPR*, 510–519. https://arxiv.org/abs/1903.06586
- Dai, Y., Gieseke, F., Oehmcke, S., Wu, Y., & Barnard, K. (2021). Attentional feature fusion. *WACV*, 3560–3569. https://arxiv.org/abs/2009.14082
- Zhang, R. (2019). Making convolutional networks shift-invariant again. *ICML*, 7324–7334. https://arxiv.org/abs/1904.11486
- Wang, C.-Y., Liao, H.-Y. M., Wu, Y.-H., Chen, P.-Y., Hsieh, J.-W., & Yeh, I.-H. (2020). CSPNet: A new backbone that can enhance learning capability of CNN. *CVPRW*, 390–391.
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
