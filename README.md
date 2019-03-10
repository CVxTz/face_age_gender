# face_age_gender

Data : https://www.openu.ac.il/home/hassner/Adience/data.html

Description of the approach : https://medium.com/@CVxTz/predicting-apparent-age-and-gender-from-face-picture-keras-tensorflow-a99413d8fd5e

requirements : Keras, tensorflow, numpy, PIL, cv2


# Predicting apparent Age and Gender from face picture : Keras + Tensorflow

<span class="figcaption_hack">Source :
[https://www.openu.ac.il/home/hassner/Adience/data.html](https://www.openu.ac.il/home/hassner/Adience/data.html)</span>

Predicting the apparent age and gender from a picture is a very interesting
problem from a technical point of view but can also be very useful when applied
to better understand consumer segments or a user base for example. It can be
used to infer the age or gender of a user and use this information to make
personalized products and experiences for each user.

In this post we will train a model to predict those attributes given a face
picture.

### Data :

We use data from
[https://www.openu.ac.il/home/hassner/Adience/data.html](https://www.openu.ac.il/home/hassner/Adience/data.html)
which is a dataset of face photos in the wild that are labeled into 8 age groups
(0–2, 4–6, 8–13, 15–20, 25–32, 38–43, 48–53, 60-) and into 2 gender classes.<br>
There are around 26,580 images (with missing labels in some cases) that are
pre-split into 5 folds.

### Existing results :

As this dataset is usually used as a benchmark for this type of tasks in many
research papers, I was able to find many prior **accuracy** results for apparent
age and gender prediction =>

[[1]
https://www.openu.ac.il/home/hassner/Adience/EidingerEnbarHassner_tifs.pdf](https://www.openu.ac.il/home/hassner/Adience/EidingerEnbarHassner_tifs.pdf)
<br> Gender : 76.1±0.9<br> Age : 45.1±2.6

[2]
[https://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf](https://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf)<br>
Gender : 86.8±1.4<br> Age : 50.7±5.1

[3]
[https://arxiv.org/pdf/1702.04280.pdf](https://arxiv.org/pdf/1702.04280.pdf)<br>
Gender : 91<br> Age : 61.3±3.7

### Preprocessing :

Faces are cropped and aligned using this tool :
[https://www.openu.ac.il/home/hassner/Adience/code.html#inplanealign](https://www.openu.ac.il/home/hassner/Adience/code.html#inplanealign)

### Data augmentation :

We use Random shift, Zoom, Horizontal Flip as a form of data augmentation to
create synthetic examples used during training to improve the generalization of
the model.

### Model :

We use a [Resnet](https://arxiv.org/pdf/1512.03385.pdf) architecture pre-trained
on ImageNet :

Resnet ( Deep Residual Networks ) are an architecture that reduces the
under-fitting and optimization issues that occur in deep neural networks.

<span class="figcaption_hack">Residual Block :
[https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)</span>

### Results :

We train the model 3 times for each fold and average the predictions and get the
following results : <br> Gender : 89.4±1.4<br> Age : 57.1±5.3<br> Those results
are better than [1] and [2] probably because of the bagging and the pretrained
weights but worse than [3] probably because Sighthound Inc used a bigger and
internal Faces dataset for pretraining.

Code to reproduce the results can be found at :
[https://github.com/CVxTz/face_age_gender](https://github.com/CVxTz/face_age_gender)
