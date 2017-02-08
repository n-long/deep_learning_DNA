# deep_learning_DNA

A work-in-progress for training an LSTM classifier to predict chromosomal regions prone to the inversion type of rearrangement. 

```
import numpy as np
import h5py
np.set_printoptions(threshold=np.inf)
np.random.seed(1337)

import random
from keras.utils.np_utils import *
from keras.models import *
from keras.layers import *

from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

dendro_string, tconf_string, tmad_string, tfree_string = ([] for i in range(4))
with open("dendro.I.1-1000.fa") as file:
	for line in file:
		if not line.startswith(">"):
			dendro_string.append(line.strip())
with open("tconf.I.1-1000.fa") as file:
        for line in file:
                if not line.startswith(">"):
                        tconf_string.append(line.strip())
with open("tmad.I.1-1000.fa") as file:
        for line in file:
                if not line.startswith(">"):
                        tmad_string.append(line.strip())
with open("tfree.I.1-1000.fa") as file:
        for line in file:
                if not line.startswith(">"):
                        tfree_string.append(line.strip())

ENCODE_MAPPER = {
    'A': (1, 0, 0, 0),
    'C': (0, 1, 0, 0),
    'G': (0, 0, 1, 0),
    'T': (0, 0, 0, 1),
    'a': (1, 0, 0, 0),
    'c': (0, 1, 0, 0),
    'g': (0, 0, 1, 0),
    't': (0, 0, 0, 1),
    'n': (1, 1, 1, 1),
}
NUM_CLASSES = 2
PRINT_WIDTH = 200
MATCH_SEQ = 'ac'

ONE_HOT_DIMENSION = len(list(ENCODE_MAPPER.values())[0])
ALPHABET = list(ENCODE_MAPPER.keys())
ALPHABET_SIZE = len(ALPHABET)
DECODE_MAPPER = {
    one_hot: nucleotide
    for nucleotide, one_hot in ENCODE_MAPPER.items()
}

def encode_sequence(sequence):
    return np.array([[
        ENCODE_MAPPER[nucleotide]
        for nucleotide in sequence
        ]])
dendro_sat = []
for line in dendro_string:
    dendro_sat.append(encode_sequence(line))
print len(dendro_sat)
print dendro_sat[0:1]

sequence = Input(shape=(None, ONE_HOT_DIMENSION), dtype='float32')
dropout = Dropout(0.2)(sequence)

        # bidirectional LSTM
forward_lstm = LSTM(
    output_dim=50, init='uniform', inner_init='uniform', forget_bias_init='one', return_sequences=True,
    activation='tanh', inner_activation='sigmoid',
)(dropout)
backward_lstm = LSTM(
    output_dim=50, init='uniform', inner_init='uniform', forget_bias_init='one', return_sequences=True,
    activation='tanh', inner_activation='sigmoid', go_backwards=True,
)(dropout)
blstm = merge([forward_lstm, backward_lstm], mode='concat', concat_axis=-1)

dense = TimeDistributed(Dense(NUM_CLASSES))(blstm)

self.model = Model(input=sequence, output=dense)
print 'Compiling model...'
self.model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print 'Training...'
sequence_list, y_list = self.generate_sequences((100, 150), 10 )
X_list = [
    self.encode_sequence(seq)
    for seq in sequence_list
]
for sample_i, (X, y) in enumerate(zip(X_list, y_list)):
    print '\t X.shape=%s  y.shape=%s' % (X.shape, y.shape)
    self.model.fit(
        X, y,
        verbose=0,
        batch_size=1,
        nb_epoch=NUM_EPOCHS,
                #callbacks=[ EarlyStopping(monitor='val_loss', patience=3, verbose=1) ]
   ) 
```
