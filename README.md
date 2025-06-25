🚀 HiLoHSI: Efficient HSI Classification via High-Low Frequency Hybrid Quantization
📌 Introduction
HiLoHSI is a novel dual-path general framework for lightweight hyperspectral image (HSI) classification. It innovatively combines high-low frequency decoupling and hybrid quantization to reduce overhead while ensuring accuracy.

Framework Overview:
[https://media/image1.png](https://github.com/Cybersecurity-Hui/HiLowHSI/blob/main/Overall%20Framework.png)
*Fig. 1: Dual-path architecture of HiLoHSI*

📊 Experimental Results
HiLoHSI achieves state-of-the-art efficiency-accuracy trade-offs across multiple benchmarks:

Quantitative metrics (OA, AA, Kappa)

Visualization of classification maps

Speed vs. accuracy comparisons

Ablation studies on quantization

Results Preview:

https://media/image2.png	https://media/image3.png
https://media/image4.png	https://media/image5.png
✨ Core Advantages
Lightweight: INT8 quantization reduces model size by 4×.

Flexible: Plug-and-play support for mainstream backbones.

Effective: Maintains >99% accuracy after quantization.

💡 Citation
If this work aids your research, please cite:

latex
@article{hilohsi2024,  
  title={HiLoHSI: Efficient HSI Classification via High-Low Frequency Hybrid Quantization},  
  author={Anonymous},  
  journal={Submitted},  
  year={2024}  
}  
