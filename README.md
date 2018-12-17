# Neural-Network-Implementation-Digits-Classification

Python code of Neural Network Implementation on digits classification

1. Description.pdf: description of the questions
2. Review.pdf: review of the homework
3. howtoru.txt, instruction.txt: instruction of running the code
4. num1a.py: python code of the wide hidden layer (2 layers)
5. num1b.py: python code of the deep hidde layer (3 layers)
6. mnist.py: python code to read the data

#### Observations:
1. The result of a three-layer-neural-network is slightly better than that of a two-layer-neural-network, probably because of the sufficient dataset
2. The result of train accuracy is usually better than that of test accuracy, as the model is trained with the dataset of train set. However, sometimes the test accuracy is higher than the train accuracy. This mostly occurs in the early stage of training process. As for the reason of this phenomena, I believe it’s purely coincidence.
3. The values of loss decrease as the model trains better, which makes sense.
4. The parametres mentioned above are tried and considered to be leading to better results; however, as many learners of AI concern, I don’t really understand why ends in this conclusion and this set of optimal parametres. I tried to print out some processes of this training, but didn’t yet find out any useful information.
