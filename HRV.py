import warnings
import numpy as np
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, AveragePooling1D, Dense, Conv2D
from tensorflow.keras.layers import Dropout, Concatenate, Flatten, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape,Bidirectional
from biosppy.signals import ecg
from pyentrp import entropy as ent
from torch.nn.utils.spectral_norm import SpectralNorm

import CPSC_utils as utils


warnings.filterwarnings("ignore")


class Net(object):


    def __init__(self):
            pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    @staticmethod
    def __backbone(inp, C=0.001, initial='he_normal'):

        net = Conv1D(4, 31, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 11, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 7, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(16, 5, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(int(net.shape[1]), int(net.shape[1]))(net)

        return net

    @staticmethod
    def nnet(inputs, keep_prob, num_classes):

        branches = []
        for i in range(int(inputs.shape[-1])):
            ld = Lambda(Net.__slice, output_shape=(int(inputs.shape[1]), 1), arguments={'index': i})(inputs)
            ld = Reshape((int(inputs.shape[1]), 1))(ld)
            bch = Net.__backbone(ld)
            branches.append(bch)
        features = Concatenate(axis=1)(branches)
        features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        # features = Bidirectional(CuDNNLSTM(1, return_sequences=True), merge_mode='concat')(features)
        features = Flatten()(features)
        net = Dense(units=num_classes, activation='softmax')(features)
        return net, features



class ManFeat_HRV(object):

    FEAT_DIMENSION = 9

    def __init__(self, sig, fs=250.0):
        assert len(sig.shape) == 1, 'The signal must be 1-dimension.'
        assert sig.shape[0] >= fs * 6, 'The signal must >= 6 seconds.'
        self.sig = utils.WTfilt_1d(sig)
        self.fs = fs
        self.rpeaks, = ecg.hamilton_segmenter(signal=self.sig, sampling_rate=self.fs)
        self.rpeaks, = ecg.correct_rpeaks(signal=self.sig, rpeaks=self.rpeaks,
                                         sampling_rate=self.fs)
        self.RR_intervals = np.diff(self.rpeaks)
        self.dRR = np.diff(self.RR_intervals)

    def __get_sdnn(self): 
        return np.array([np.std(self.RR_intervals)])

    def __get_maxRR(self): 
        return np.array([np.max(self.RR_intervals)])

    def __get_minRR(self):  
        return np.array([np.min(self.RR_intervals)])

    def __get_meanRR(self): 
        return np.array([np.mean(self.RR_intervals)])

    def __get_Rdensity(self):  
        return np.array([(self.RR_intervals.shape[0] + 1) 
                         / self.sig.shape[0] * self.fs])

    def __get_pNN50(self):  
        return np.array([self.dRR[self.dRR >= self.fs*0.05].shape[0] 
                         / self.RR_intervals.shape[0]])

    def __get_RMSSD(self):  
        return np.array([np.sqrt(np.mean(self.dRR*self.dRR))])
    
    def __get_SampEn(self):  
        sampEn = ent.sample_entropy(self.RR_intervals, 
                                  2, 0.2 * np.std(self.RR_intervals))
        for i in range(len(sampEn)):
            if np.isnan(sampEn[i]):
                sampEn[i] = -2
            if np.isinf(sampEn[i]):
                sampEn[i] = -1
        return sampEn

    def extract_features(self):  
        features = np.concatenate((self.__get_sdnn(),
                self.__get_maxRR(),
                self.__get_minRR(),
                self.__get_meanRR(),
                self.__get_Rdensity(),
                self.__get_pNN50(),
                self.__get_RMSSD(),
                self.__get_SampEn(),
                ))
        assert features.shape[0] == ManFeat_HRV.FEAT_DIMENSION
        return features

