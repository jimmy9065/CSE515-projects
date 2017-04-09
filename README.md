# CSE515-projects

#### LoadData.py  
Just for data viewing  
#### Preprocessing.py  
Load all the data and rescale them into uniform size  
#### Pipeline.py 
Read all the picture into tensorflow.
Generate the train set and test set.
Split the train set into batches. at the begining of each epoch, the whole train_set will be shuffled.
#### CAE.py
the graph map of convolution autoencoder.
It train an CAE, and map some train data and test data into new feature space
You can directly get them by return them for the next part or save them on driver for the next time uses.
