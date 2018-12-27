import chainer.functions as F
import chainer.links as L
from chainer import datasets, iterators, optimizers, serializers,ChainList,Chain,training
from chainer.training import extensions
import cupy as cp
import random
from chainer.dataset import concat_examples
import chainer
import chainercv
import numpy as np
import os
from chainer_resnet import ResNet

#model=L.ResNet101Layers()
model= chainercv.links.model.resnet.ResNet101(n_class=10)

def data_augment(data):
        image, label = data
        xp = chainer.cuda.get_array_module(image)

        # after 0 padding, image.shape = (3, 40, 40)
        image = xp.pad(image, ((0, 0), (4, 4), (4, 4)), 'constant', constant_values=(0, 0))
        _, h, w = image.shape
        crop_size = 32

        top = random.randint(0, h - crop_size - 1)
        left = random.randint(0, w - crop_size - 1)
        if random.randint(0, 1):
            image = image[:, :, ::-1]
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        return image, label
n_layers =20
class_labels=10
model = L.Classifier(model)
model.to_gpu(0)
data_dir = '/'
x_train = np.load(os.path.join(data_dir,'x_train.npy'))
y_train = np.load(os.path.join(data_dir,'y_train.npy'))
x_test = np.load(os.path.join(data_dir,'x_test.npy'))
y_test = np.load(os.path.join(data_dir,'y_test.npy'))   
x_train =x_train.astype(np.float32)
x_test  =x_test.astype(np.float32)
x_train =x_train.transpose([0,3,1,2])/255 - 0.5
x_test  =x_test.transpose([0,3,1,2])/255 - 0.5
y_train = y_train.astype(np.int8)
y_test  = y_test.astype(np.int8)
y_train = y_train.squeeze()
y_test  = y_test.squeeze()
train_data = [(data,label) for data,label in zip(x_train,y_train)]
test_data  = [(data,label) for data,label in zip(x_test,y_test)]
train_data = chainer.datasets.TransformDataset(train_data, data_augment)
batch_size = 128
train_iter = iterators.SerialIterator(train_data, batch_size,repeat=True,shuffle=True)
test_iter  = iterators.SerialIterator(test_data, batch_size,repeat=False,shuffle=False)
optimizer = optimizers.MomentumSGD(0.01)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(10e-5))
gpu_id=0
max_epoch =50

updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='cifar_result')
trainer.extend(extensions.ExponentialShift('lr', 0.99), trigger=(1000, 'iteration'))
interval = 200
trainer.extend(training.extensions.Evaluator(
        test_iter, model, device=gpu_id), trigger=(interval, 'iteration'))
trainer.extend(training.extensions.LogReport(log_name='resnet101_cifar',trigger=(interval, 'iteration')))
trainer.extend(training.extensions.observe_lr(), trigger=(interval, 'iteration'))
trainer.extend(training.extensions.PrintReport(['epoch', 'iteration','lr' , 'main/loss', 'main/accuracy','validation/main/loss', 'validation/main/accuracy']))
trainer.run()
print(model.count_params())
chainer.serializers.save_npz(os.path.join('model.npz'), model,)

