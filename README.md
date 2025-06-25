## **🚀 HiLoHSI: Efficient HSI Classification via High-Low Frequency Hybrid Quantization**  
### **📌 Introduction**  
HiLoHSI is a novel dual-path general framework for lightweight hyperspectral image (HSI) classification. It innovatively combines high-low frequency decoupling and hybrid quantization to reduce overhead while ensuring accuracy.

#### Framework Overview:
![image](./Overall-Framework.png)
*Fig.1The overall architecture of HiLoHSI: a dual-path framework for efficient hyperspectral image classification*

#### MambaHSI Framework Embedding:

![image](./MambaHSi-Framework-Embedding.png)

*Fig.2Architectural overview of MambaHSI with high-low frequency decomposition: comparison of the original model (top) and the frequency-decomposed variant (bottom), showing the integration of frequency decomposition modules within the three-block structure*

#### GSCVIT Framework Embedding:
![image](./GSCVIT-Framework-Embedding.png)
*Fig.3Transformer model architecture modifications: Illustration of the original model (top) and its frequency-decomposed version (bottom), highlighting the replacement of the original module with a high-low frequency mixed quantization module*

### 📊 Experimental Results：
#### Mamba-Base:
![image](./Result-Mamba-Base.png)
#### Transformer-Base:
![image](./Result-Transformer-Base.png)
#### Other Quantitative Methods:
![image](./Result-Bin-Conv.png)
#### Ablation Experiment:
![image](./Result-Ablation-Experiment.png)

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
