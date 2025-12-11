import numpy as np
import re

class Model:
    def __init__(self, alpha=1):
        self.vocab = set() 
        self.spam = {} 
        self.ham = {} 
        self.alpha = alpha 
        self.label2num = None 
        self.num2label = None 
        self.Nvoc = 0 
        self.Nspam = 0 
        self.Nham = 0 
        self.prior_spam = 0
        self.prior_ham = 0

    def fit(self, dataset):
        '''
        Обучение модели на тренировочной выборке.
        '''
        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        
        X_train, y_train = dataset.train
        
        # Определение индекса спама (предполагаем наличие метки 'spam')
        spam_idx = self.label2num.get('spam', 1)
        
        # Разделение выборки
        spam_msgs = X_train[y_train == spam_idx]
        ham_msgs = X_train[y_train != spam_idx]
        
        # Расчет априорных вероятностей (log space)
        total_count = len(X_train)
        self.prior_spam = np.log(len(spam_msgs) / total_count)
        self.prior_ham = np.log(len(ham_msgs) / total_count)
        
        # Заполнение частотных словарей
        for msg in spam_msgs:
            for word in msg.split():
                self.vocab.add(word)
                self.spam[word] = self.spam.get(word, 0) + 1
                self.Nspam += 1
                
        for msg in ham_msgs:
            for word in msg.split():
                self.vocab.add(word)
                self.ham[word] = self.ham.get(word, 0) + 1
                self.Nham += 1
                
        self.Nvoc = len(self.vocab)
    
    def inference(self, message):
        '''
        Предсказание класса для одного сообщения.
        '''
        message = re.sub(r'[^\w\s]', '', message.lower())
        words = message.split()
        
        # Инициализация с априорными вероятностями
        p_spam = self.prior_spam
        p_ham = self.prior_ham
        
        for word in words:
            # Сглаживание Лапласа
            p_word_spam = (self.spam.get(word, 0) + self.alpha) / (self.Nspam + self.alpha * self.Nvoc)
            p_word_ham = (self.ham.get(word, 0) + self.alpha) / (self.Nham + self.alpha * self.Nvoc)
            
            p_spam += np.log(p_word_spam)
            p_ham += np.log(p_word_ham)

        if p_spam > p_ham:
            return "spam"
        return "ham"
    
    def validation(self):
        '''
        Оценка точности на валидационной выборке.
        '''
        X_val, y_val = self._val_X, self._val_y = dataset.val # small fix for scope if needed externally, but usually passed via object
        # Исправление доступа к данным: данные хранятся в объекте dataset, который передавался в fit, 
        # но для чистоты архитектуры лучше передать их или сохранить в fit.
        # В текущей структуре задания предполагается использование атрибутов dataset.
        # Поэтому здесь предполагаем, что метод вызывается после fit, где можно было бы сохранить ссылки, 
        # но так как fit не сохраняет dataset в self, используем переданные данные.
        # *В рамках задания файлы разделены, поэтому допустим следующий подход:*
        # Ошибка архитектуры задания: validation() не принимает аргументов.
        # Придется сохранить ссылки на валидацию внутри fit (добавил это ниже в fit).
        pass 

    # ПЕРЕПИСАННЫЙ МЕТОД FIT (обнови в файле model.py чтобы работали validation и test без аргументов)
    def fit(self, dataset):
        self.label2num = dataset.label2num
        self.num2label = dataset.num2label
        
        # Сохраняем ссылки на данные для методов validation и test
        self._val_data = dataset.val
        self._test_data = dataset.test
        
        X_train, y_train = dataset.train
        
        spam_idx = self.label2num.get('spam', 1)
        spam_msgs = X_train[y_train == spam_idx]
        ham_msgs = X_train[y_train != spam_idx]

        self.prior_spam = np.log(len(spam_msgs) / len(X_train))
        self.prior_ham = np.log(len(ham_msgs) / len(X_train))

        for msg in spam_msgs:
            for word in msg.split():
                self.vocab.add(word)
                self.spam[word] = self.spam.get(word, 0) + 1
                self.Nspam += 1
                
        for msg in ham_msgs:
            for word in msg.split():
                self.vocab.add(word)
                self.ham[word] = self.ham.get(word, 0) + 1
                self.Nham += 1
        self.Nvoc = len(self.vocab)

    def validation(self):
        correct = 0
        X_val, y_val = self._val_data
        for i in range(len(X_val)):
            pred = self.inference(X_val[i])
            true_label = self.num2label[y_val[i]]
            if pred == true_label:
                correct += 1
        return correct / len(X_val)

    def test(self):
        correct = 0
        X_test, y_test = self._test_data
        for i in range(len(X_test)):
            pred = self.inference(X_test[i])
            true_label = self.num2label[y_test[i]]
            if pred == true_label:
                correct += 1
        return correct / len(X_test)