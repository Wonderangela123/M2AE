# M<sup>2</sup>AE

We developed an innovative multi-task multi-view integrative approach (**M<sup>2</sup>AE, Multi-task Multi-View Attentive Encoders**), to impute blood SCFA levels using gut metagenomic sequencing (MGS) data, while taking into account the intricate interplay among the gut microbiome, dietary features, and host characteristics.

![image](https://github.com/user-attachments/assets/c95ad56b-2f83-4d74-aed4-487d1ec19906)



M<sup>2</sup>AE is a framework for prediction tasks with multi-view data as input. Each view corresponds to a distinct category of data input, i.e., gut microbiome compositions, dietary features, or host characteristics. The workflow of M2AE is shown in Figure 1 and can be summarized into two components. (1) View-specific representation learning via attentive encoders. For each view, an attentive encoder is designed in a symmetric auto-encoder fashion, where the encoder part is composited with one graph convolutional module and two fully-connected layers for view-specific representation learning. (2) Multi-view integration via the View Interactive Network (VIN). A cross-view interactive tensor is calculated using the latent representations from all the view-specific networks. A VIN is then trained with the cross-view discovery tensor to produce the final predictions. VIN can effectively learn the intra-view and inter-view correlations in the higher-level space for better prediction with multi-view data. M2AE is an end-to-end model, where both view-specific attentive encoders and VIN module are trained jointly.



# <sub><sup>Acknowledgement</sub></sup>

Anqi Liu<sup>#</sup>, Bo Tian<sup>#</sup>, Kuan-Jui Su, Lindong Jiang, Chen Zhao, Meng Song, Yong Liu, Gang Qu, Ziyu Zhou, Xiao Zhang, Chuan Qiu, Zhe Luo, Qing Tian, Hui Shen, Zhengming Ding*, Hong-Wen Deng*. **Interpretable Multi-View Integrative Approaches Imputing Serum Short-Chain Fatty Acids from Gut Microbiome.**

<sup>#</sup>Co-first authors

*Co-corresponding authors 
