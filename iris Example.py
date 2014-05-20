##http://www.researchgate.net/publication/221667696_mlpy_Machine_Learning_Python

iris.shape                               # 2d numpy array, 150 observations and 4 attributes (150, 4) 
import mlpy                              # import the mlpy module 
pca = mlpy.PCA()                         # build a new PCA instance 
pca.learn(iris)                          # perform the PCA on the Iris dataset 
iris_pc = pca.transform(iris, k=2)       # project Iris on the first 2 PCs 
svm = mlpy.LibSvm(kernel_type=’linear’)  # build a new LibSVM instance 
svm.learn(iris_pc, labels)               # train the model 
labels_pred = svm.pred(iris_pc)          # test the model 
mlpy.error(labels, labels_pred)          # compute the prediction error 
