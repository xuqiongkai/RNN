__author__ = 'xuqiongkai'

class Reader():
    def read_imdb(self, positive_path, negative_path):
        pos_data = []
        neg_data = []
        with open(positive_path) as pf:
            for line in pf:
                pos_data.append(line.split(' '))
        with open(negative_path) as nf:
            for line in nf:
                neg_data.append(line.split(' '))
        return pos_data, neg_data
