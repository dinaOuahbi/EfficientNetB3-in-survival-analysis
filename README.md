## Summary  

This project aims to predict the survival of patients with pancreatic ductal adenocarcinoma (PDAC) using deep learning models trained on digitized histological slides. Whole-slide images stained with hematoxylin and eosin (H&E) or hematoxylin, eosin, and saffron (HES) were used alongside RNA-seq and exome data from a private cohort (Besançon, n=206) and the TCGA cohort (n=166).  

### Methodology  
We applied **EfficientNet** to classify tissue regions and predict patient prognosis. The workflow includes:  
- Tile extraction using QuPath  
- Reinhard normalization  
- Training **EfficientNet** models for tissue classification:  
  - Stroma vs. others  
  - Normal vs. tumor  
- Combining model results to generate annotated slides  
- Independent tile prediction with Grad-CAM visualization  
- Survival model integrating deep learning features
  
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/annotation_example.png)
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/annotation_example1.png)

![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/GRAD_CAM_normal_class_activation.png)
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/GRAD_CAM_normal_class_activation_ex2.png)
  

### Link to the DenseNet Version  
This methodology follows the same approach as our previous study using **DenseNet**. For more details on the original implementation, refer to: [https://github.com/dinaOuahbi/PDAC-DeepLearning-Survival-Prediction]  

Our findings suggest that deep learning can assist in predicting PDAC prognosis and improving treatment decisions.

![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/km_train_b3.png)
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/km_test_b3.png)
