# M2AE

We developed an innovative multi-task multi-view integrative approach (M2AE, Multi-task Multi-View Attentive Encoders) to predict the abundances of serum short chain fatty acids (SCFAs) by integrating three types of views, i.e., gut microbiomes, dietary habits, and host characteristics.

![image](https://github.com/Wonderangela123/M2AE/assets/140135188/37f7b95a-5971-4a65-8277-a9b4a5788811)


M2AE is a predictive framework with multi-view data. The model learns view-specific representation via attentive encoders (AEs) and integrates multi-view data using the View Correlation Discovery Network (VCDN). For each view, an attentive encoder was employed to leverage both view-specific features and their corresponding sample similarity network, employing Multi-Layer Perceptrons (MLP) in a sequential manner for learning view-specific representations. A cross-view discovery tensor was calculated using the latent representations from all types of views. A VCDN was then trained with the cross-view discovery tensor to produce the final predictions. M2AE is an end-to-end model, where both view-specific attentive encoders and VCDN are trained jointly.
