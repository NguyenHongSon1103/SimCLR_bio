import tensorflow as tf
import numpy as np
import os
from losses import contrastive_loss
from model import Model
from config import config
from dataset import get_constrastive_data

os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

root = os.getcwd()
save_model_dir = os.path.join(root, 'trained_models')
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

## LOAD DATA ##
train_gen, val_gen = get_constrastive_data()
print('DATASET:\nTRAININGSET: %d\nVALIDATIONSET: %d'%(len(train_gen), len(val_gen)))

## BUILD MODEL ##
model_train = Model(size_1=20217, size_2=17073).make_model(True)
# training_model.summary()
with open(os.path.join(save_model_dir, 'model_train.json'), 'w') as f:
    f.write(model_train.to_json())
if config['resume']:
    print('USING PRETRAINED WEIGHTS: ', config['pretrained_weights'])
    model_train.load_weights(config['pretrained_weights'])

## DEFINE LOSS, OPTIMIZATER, CALLBACKS ##
if config['use_lr_schedule']:
    if config['steps_per_epoch'] is None:
        max_iters = config['epochs']*len(train_gen)
    else:
        max_iters = config['epochs']*config['steps_per_epoch']

    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        config['base_lr'], max_iters, config['end_lr'], power=0.9
    )
else:
    lr = 1e-3

optims = {'adam':tf.keras.optimizers.Adam, 'sgd':tf.keras.optimizers.SGD}
opt = optims['optimizer_type'](learning_rate = lr)
model_dir = os.path.join(save_model_dir, 'weights_{epoch:02d}_{val_loss:.4f}.h5')
ckpt = tf.keras.callbacks.ModelCheckpoint(model_dir, mode='min', monitor='val_loss',
                                         save_best_only=True, save_weights_only=True)
tfboard = tf.keras.callbacks.TensorBoard(os.path.join(save_model_dir, 'logs'))

## Training: 
model_train.compile(optimizer=opt, loss=contrastive_loss)

model_train.fit( train_gen, epochs=config['epochs'],
                steps_per_epoch=config['steps_per_epoch'],
                callbacks=[ckpt, tfboard], validation_data=val_gen)