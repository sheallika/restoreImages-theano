-----
Classification on CIFAR-10 dataset

Model:


The following Architecture of the CNN has obtained accuracy 83.43% on CIFAR-10 data set:
1) Convolution-pool layer (with max pooling) with 32 kernels of size 3*3
2) convolution-pool layer with pool size = (1,1), nkerns=64 and kernels of size 3*3
3) batch normalisation layer with filter shape = (64,64,3,3) 
4) Flatten the out put of batch normalisation layer 
5) Drop out layer with drop out rate as 30%
6) convolution-pool layer (with max pool) with 32 1*1 kernels 
7) drop out layer with drop out rate as 20%
8) hidden layer ( without drop out)
9) softmax layer with number od outputs as 10

see the file main.ipynb for details
