# **Image Generation**

## **코드 설명** 
<br/>

이번 과제의 목표는generative adversarial network (GAN) 완성하여 이미지 생성하는 것이다. Pytorch를 이용하여 GAN 네트워크를 구현한다.

<br/>

### **1. Dataset 설정**
제공된 CelebA 데이터셋을 training data로 사용한다. CelebA 데이터셋은 대표적인 얼굴 데이터 셋이며 약 20만개 정도의 이미지로 구성된다. 각 이미지는 178x218 해상도로 존재하는데 이 과제에는 64x64로 crop된 이미지를 input으로 사용하여 GAN 모델 이용해 가짜 이미지를 만들어낸다. Gradient descent algorithm이 더 빠르게 최적화 지점을 찾을 수 있게 input 데이터 정규화(input normalization)를 시행한다. 

<br/>

### **2. 모델 설계**
Generative adversarial network를 구축한다. GAN 네트워크는 generator와 discriminator, 두 개의 모델이 동시에 적대적인 과정으로 학습한다. Generator는 제공되는 코드 구조를 그대로 사용한다. Discriminator는 3 * 64 * 64 이미지를 input으로 받아, 여러 개의 Conv2d, LeakyReLU와 BatchNorm layer를 걸쳐서 최종적으로 sigmoid function 거친 확률 값을 output으로 출력한다. 고성능의 모델을 구축하기 위해 weight initialization 기법을 사용하여 generator와 discriminator 모델의 가중치를 평균 0, 표준편차 0.02의 정규 분포에서 random하게 초기화한다.

<br/>

### **3. Loss 함수와 Optimizer 정의**
Binary cross entropy (BCE) 함수를 손실함수로 사용하며 ADAM optimizer를 optimizer로 사용한다. ADAM optimizer의 learning rate은 0.0002, beta1은 0.5, beta2는 0.999로 설정한다.

<br/>

### **4. 모델 학습**
GAN 네트워크 학습시키는 과장은 두 부분으로 이뤄진다: Discriminator 업데이트와 generator 업데이트하는 과정이다. Batch 사이즈, epoch횟수 등 같은 hyperparameter들이 설정한 뒤에 batch으로 나눈 데이터셋이 반복적으로 훈련시킨다.

<br/>

> #### **Discriminator 학습**
Discriminator는 주어진 input이 진짜인지 가짜인지 판단하는 역할을 가진다. 먼저, real 데이터로만 이루어진 batch를 만들어 discriminator에 통과시킨다. 그 output으로 log(D(x))의 loss를 계산하고 backpropagation 과정에서의 gradient들을 계산한다. 그 다음에, fake 데이터로만 이루어진 batch를 만들어 discriminator에 통과시키고, 그 output으로 (log(1-D(G(z))))의 loss를 계산하여 gradient 값을 구하며 discriminator를 업데이트한다. 이를 반복하여 시행하며 discriminator 모델을 학습시킨다.

<br/>

> #### **Generator 학습**
그럴듯한 fake 이미지를 만들기 위한 노력으로 log(1-D(G(z)))를 최소화함으로써 generator를 학습시킨다. Discriminator를 이용해 generator의 output을 판별해주며, 진짜 label 값을 이용해 generator의 loss를 구한다. 그 다음에, 구해진 loss값으로 gradient를 구하고, 최종적으로는 optimizer를 이용해 generator를 업데이트한다. 이를 반복하여 시행하며 generator 모델을 훈련시킨다.


<br/>

## **실험 결과**
<br/>

GAN 모델 학습 끝난 뒤에 3000장의 가짜 이미지를 생성하여 local directory에 저장한다. 학습된 모델의 성능을 판단할 수 있게 Fréchet inception distance (FID) score를 계산한다. 70이하의 FID-score를 달성하는 것이 목표로 한다. 가짜 이미지 생성 모델의 성능 향상할 수 있도록 epoch수 조정과 일반화 테크닉을 적용한다.

<br/>

> Input normalization, weight normalization 테크닉을 적용하고 optimal epoch는 5로 설정할 때 FID-score가 **68.76**로 측정되었다. 

<br/>

## **Code Explanation**
<br/>

The goal of this assignment is to build a generative adversarial network (GAN) which generates new, artificial images. A GAN network is created using Pytorch.

<br/>

### **1. Loading the dataset**
The provided CelebA dataset is used as training data. CelebA dataset is a large-scale face attributes dataset with more than 200,000 celebrity images, each with 40 attribute annotations. Each image is of 178x218 resolution, but in this assignment, 64x64 cropped images are used as input of the GAN model. For faster convergence, the input normalization is performed on the input dataset.

<br/>

### **2. Defining the model**
A generative adversarial network (GAN) is created. The GAN network consists of a generator and a discriminator, where both parts train in parallel. The generator is defined as per the provided codes. As for the discriminator, it accepts 3x64x64 image as input, and passes through multiple Conv2d, LeakyReLU and BatchNorm layers before calculating the probability value as output from a sigmoid function. Weight initialization technique is also applied so that all weights are randomly initialized to mean = 0, standard deviation = 0.2.

<br/>

### **3. Defining loss function & optimizer**
Binary cross entropy (BCE) function is used as loss function, whereas ADAM optimizer is used as optimizer. The learning rate, beta1, and beta2 values of the ADAM optimizer is set to 0.0002, 0.5, and 0.999 respectively.

<br/>

### **4. Training the model**

The training of GAN network is split up into two parts: updating the discriminator and updating the generator. The hyperparameters of the model, such as batch size, epoch number, etc. are arbitrarily set. The dataset, which is divided into minibatches, are fed to the model for training.

<br/>

> #### **Training the discriminator**
The role of the discriminator is to classify whether the given input as real or fake. First, a batch of real samples from the training set is constructed and passed forward through the discriminator. The loss log(D(x)) is calculated before performing backpropagatin to calculate the gradients. Secondly, a batch of fake samples is constructed using the current generator and passed forward through the discriminator. The loss log(1-D(G(z))) is calculated and the gradients are accumulated with a backward pass. With the gradients accumulated from both the all-real and all-fake batches, the optimizer of the discriminator is called. This process is repeated to train the discriminator.

<br/>

> #### **Training the generator**
In order to generate better fakes, the generator is trained by minimizing log(1-D(G(z))). Using the discriminator to classify the generator output, the loss of the generator is computed with real labels. Then, the gradients of the generator are computed in a backward pass, before finally updating the parameters of the generator with an optimizer step. This process is repeated to train the generator.

<br/>

## **Experiment results**
<br/>

After completing the training of the GAN model, 3000 fake images are generated by the model before storing them at a local directory. The Fréchet inception distance (FID) score of the model is calculated to evaluate the performance of the trained model. The epoch number is adjusted and regularization techniques are applied in order to improve the performance of the image generation model.

<br/>

> After applying input normalization, weight normalization techniques and the optimal epoch of 5, the FID-score is calculated as **68.76**.