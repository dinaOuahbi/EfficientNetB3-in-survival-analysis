# EfficientNetB3-in-survival-analysis


Entrainement a été effectuer sur une base de données composée uniquement de tuile TCGA issues de 40 lames TCGA annoté par l’anapath FG. Nous avons construit deux modèles CNN basés sur EfficientNetb3, le premier sépare les tuiles selon 3 classes [Duodénum, N_T, Stroma] tandis que le deuxième sépare les tuiles en deux classes. [Normal, Tumor]. Ci-dessous l’histoire d’entrainement de ce modèle avec à gauche le CNN1, et à droite le CNN2.

## Annotations
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/annotation_example.png)
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/annotation_example1.png)

## Grad_CAM normal class activation 
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/GRAD_CAM_normal_class_activation.png)
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/GRAD_CAM_normal_class_activation_ex2.png)


## survival analysis result : COX PH / train maxstat 
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/km_train_b3.png)
![Image of aciduino on protoboard](https://github.com/dinaOuahbi/EfficientNetB3-in-survival-analysis/blob/main/km_test_b3.png)
