import numpy as np
import re

class Dataset:
    def __init__(self, X, y):
        self._x = X 
        self._y = y 
        self.train = None 
        self.val = None 
        self.test = None 
        self.label2num = {} 
        self.num2label = {} 
        self._transform()
        
    def __len__(self):
        return len(self._x)
    
    def _transform(self):
        '''
        Функция очистки сообщения и преобразования меток в числа.
        '''
        # Очистка текста и приведение к нижнему регистру
        self._x = np.array([
            re.sub(r'[^\w\s]', '', text.lower()) 
            for text in self._x
        ])
        
        # Маппинг меток классов
        unique_labels = np.unique(self._y)
        for idx, label in enumerate(unique_labels):
            self.label2num[label] = idx
            self.num2label[idx] = label
            
        self._y = np.array([self.label2num[label] for label in self._y])

    def split_dataset(self, val=0.1, test=0.1):
        '''
        Функция, которая разбивает набор данных на наборы train-validation-test.
        '''
        n_samples = len(self._x)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        val_size = int(n_samples * val)
        test_size = int(n_samples * test)
        train_size = n_samples - val_size - test_size
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size : train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        self.train = (self._x[train_idx], self._y[train_idx])
        self.val = (self._x[val_idx], self._y[val_idx])
        self.test = (self._x[test_idx], self._y[test_idx])