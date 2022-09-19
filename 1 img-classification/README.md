# **Image Classification**


## **코드 설명** 
<br/>
CIFAR-10 이미지 데이터를 분류할 수 있는 artificial neural network 모델 세우는 것이 이 과제의 목표이다. Pytorch를 이용하여 Multi-layer Perceptron을 구현한다.

<br/>

### **1. Dataset 설정**
Torchvision에서 불러온 CIFAR-10 데이터를 training과 test data로 사용한다. CIFAR-10에 포함된 이미지의 크기는 3x32x32로, 이는 32x32 pixel 크기의 이미지가 RGB channel으로 이뤄져 있다. 학습과정을 안전하게 할 수 있도록 normalization 기법을 적용하여 데이터셋의 output은 [0.0, 1.0]의 범위로 정규화 된 Tensor로 변환한다.
- CIFAR-10 mean: 0.4914, 0.4822, 0.4465
- CIFAR-10 standard deviation: 0.247, 0.243, 0.261

<br/>

### **2. 모델 설계**
Multilayer perceptron을 구축한다. 모델 학습은 모델은 3개의 Linear layer로 구성되어 있고 구조의 dimension는 아래와 같이 설정한다:
- 첫번째 Layer
    - Input size: 32 * 32 * 3
    - Output size: 256
- 두번째 Layer
    - Input size: 256
    - Output size: 128
- 세번째 Layer
    - Input size: 128
    - Output size: 10

<br/>

### **3. Loss 함수와 Optimizer 정의**
Cross entropy loss 함수를 손실함수로 사용하며 momentum 개념을 적용되는 Stochastic Gradient Descent를 optimizer로 사용한다.

<br/>

### **4. 모델 학습**
변화된 이미지 데이터를 반복해서 Multi-layer Perceptron모델에 입력으로 제공하고, 최적화를 한다. 반복 횟수(epochs)는 임의의 값으로 설정하고 여러 번의 시도를 통하여 optimal한 training 횟수를 찾아낸다. 입력 값을 forward propagation시켜주고 에러 값(loss)을 계산한다. Overfitting을 방지하기 위하여 L2 regularization을 한다. L2 regularization은 모델의 weight의 제곱합을 penalty term으로 주어 loss를 최소화하는 것이다. 모델 최적화는 역전파 (back propagation)을 시켜주며 parameter 값을 업데이트를 한다. 이를 반복하여 시행하며 신경망 모델을 학습시킨다.


<br/>

## **실험 결과**
<br/>

학습용 데이터셋을 반복하며 신경망을 학습시킨 후에 모델을 검증한다. Test 데이터를 넣어 예측 정확도를 확인한다. 이미지 분류 모델의 성능 향상을 하는 목표로 hyperparameter 조정과 다양한 일반화 테크닉을 적용할 때 나온 결과를 비교한다.

<br/>

### **1. Regularization 테크닉 적용**
L2 regularization, drop-out과 normalization 기법을 적용하여 test 데이터셋에 대한 오차를 감소하도록 실험한다. Optimizer 함수는 기본 Stochastic Gradient Descent (SGD) 함수로 설정한다.

<br/>

### **2. ADAM optimizer 함수 적용**
Pytorch library 모듈에서 제공된 Adaptive Moment Estimation (ADAM) optimizer를 적용한다. 모델 학습 위해 돌리는 횟수 (epoch), learning rate등 모델 hyperparameter를 조정하여 모델 성능을 관측한다.

<br/>

### **3. SGD Optimizer에 Momentum 적용**
빠른 학습 속도와 local minima를 문제를 개선하고자 SGD에 관성 개념을 적용한다. 모델 학습 위해 돌리는 반복횟수를 조정하여 모델 예측 정확도를 기록한다. Epoch을 늘려도 어느 시점부터는 오히려 정확도가 떨어지며 optimal epoch은 90으로 설정할 때 정확도가 **52.51**%로 측정되었다.


<br/>

## **Code Explanation**
<br/>

The goal of this assignment is to train an artificial neural network to classify the CIFAR-10 image dataset. A multi-layer perceptron is created using Pytorch.

<br/>

### **1. Loading the dataset**
Using `torchvision`, the CIFAR-10 training and test datasets are loaded. The CIFAR-10 dataset consists of 60,000 32x32 color images (RGB) in 10 classes, with 6,000 images per class. Normalization method is applied to the dataset, thus the images are transformed to Tensors of normalized range [0.0, 1.0].
- CIFAR-10 mean: 0.4914, 0.4822, 0.4465
- CIFAR-10 standard deviation: 0.247, 0.243, 0.261

<br/>

### **2. Defining the model**
A multi-layer perceptron is created. The model consists of 3 linear layers. The dimension of the model is as follows.
- 1st layer
    - Input size: 32 * 32 * 3
    - Output size: 256
- 2nd layer
    - Input size: 256
    - Output size: 128
- 3rd layer
    - Input size: 128
    - Output size: 10

<br/>

### **3. Defining loss function & optimizer**
Cross entropy loss function is set as the loss function whereas Stochastic Gradient Descent (SGD) with momentum is set as the optimizer.

<br/>

### **4. Training the model**
The image dataset is looped over repeatedly, fed as input to the MLP model and optimized. The epochs (number of iterations) is set to an arbitrary value. Through multiple tries, the optimal number of iterations for training is found. Forward propagation is performed on the input values, and loss (error) is calculated. To prevent the model from overfitting, L2 regularization is performed, by adding a term to the error function used by the training algorithm to penalize large weight values. Back propagation is then performed, where the weights of the model is fine-tuned based on the error rate obtained in the previous epoch. This process is repeated over multiple iterations to train the model. 

<br/>

## **Experiment results**
<br/>

After training the model with the training dataset, the performance of the model is evaluated by feeding it with the test dataset where it predicts the class label of those inputs. To improve the accuracy of the model's predictions, the model's hyperparameters are tuned and various optimization techniques are applied.

<br/>

### **1. Applying regularization technique**
L2 regularization, drop-out and normalization techniques are applied onto the model, and its prediction accuracy on the test set is recorded. Stochastic Gradient Descent (SGD) function is used as the optimizer function.

<br/>

### **2. Applying ADAM optimizer**
Adaptive Moment Estimation (ADAM) optimizer provided by the Pytorch library is applied. The hyperparameters of the model, i.e., number of iterations (epoch), learning rate, etc., are adjusted and the performance of the model is evaluated.

<br/>

### **3. Applying SGD optimizer with momentum**
To allow fast convergence and avoid the local optima problem, SGD is applied with the principle of momentum. The number of iterations (epoch) are adjusted whilst testing the model. At some point, the accuracy of the model falls even when increasing the epoch number. Through multiple evaluations, the optimal epoch is found to be 90, where the accuracy of the model is recorded as **52.51%**.


