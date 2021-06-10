from tensorflow.examples.tutorials import mnist

def get_mnist_loader(path):
    train_data = mnist.input_data.read_data_sets(path, one_hot=True).train # binarized (0-1) mnist data
    return train_data
    
if __name__ == '__main__':
    batchsize = 5
    loader = get_mnist_loader("mnist")
    print(loader)
    for i in range(5):
        data = loader.next_batch(batchsize)
        # print(data)