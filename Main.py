__author__ = 'xuqiongkai'
import Reader
import Model
import tensorflow as tf
import Classifier


def chain_test(word_model, pos_data, neg_data):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # for d in ['/cpu:0', '/cpu:1']:
    #     with tf.device(d):
    #         print d
    classifier = Classifier.ChainClassifier(word_model, class_num=2, unit_num=100)

    sess.run(tf.initialize_all_variables())

    c = 0
    for t in range(20):

        for i in range(200):
            print t, i
            classifier.train_sample(pos_data[i], 0, sess)
            classifier.train_sample(neg_data[i], 1, sess)
            c += 1
            if c % 50 == 0:
                print 'round:', t, 'instance:', i

                bIdx = 4000
                eIdx = 4020
                tp = fp = tn = fn = 0.0
                for i in range(bIdx, eIdx):
                    res = classifier.test_sample(pos_data[i], sess)
                    print 'pos: ', res
                    if res[0, 0] > res[0, 1]:
                        tp += 1
                    else:
                        fp += 1
                for i in range(bIdx,eIdx):
                    res = classifier.test_sample(neg_data[i], sess)
                    print 'neg: ', res
                    if res[0, 0] < res[0, 1]:
                        tn += 1
                    else:
                        fn += 1
                cor = tp + tn
                all = cor + fp + fn
                print("tp: {}\t tn: {}".format(tp, tn))
                print("pre: {}".format(cor / all))

    sess.close()

def plain_test(word_model, pos_data, neg_data):
    classifier = Classifier.PlainClassifier(word_model, 2)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for t in range(50):
        if t % 5 == 0:
            print 'round:', t

            bIdx = 4000
            eIdx = 5000
            tp = fp = tn = fn = 0.0
            for i in range(bIdx,eIdx):
                res = classifier.test_sample(pos_data[i], sess)
                if res[0, 0] > res[0, 1]:
                    tp += 1
                else:
                    fp += 1
            for i in range(bIdx,eIdx):
                res = classifier.test_sample(neg_data[i], sess)
                if res[0, 0] < res[0, 1]:
                    tn += 1
                else:
                    fn += 1
            cor = tp + tn
            all = cor + fp + fn
            print("tp: {}\t tn: {}".format(tp, tn))
            print("pre: {}".format(cor / all))

        for i in range(4000):
            #print pos_data[i]
            #print neg_data[i]

            classifier.train_sample(pos_data[i], 0, sess)
            classifier.train_sample(neg_data[i], 1, sess)

    sess.close()



#word_model = Model.Glove('Models/GloveVec/vectors.6B.100d.txt')
word_model = Model.Glove('Models/GloveVec/vectors.6B.100d.small.txt')

pos_data_path = './Dataset/rotten_imdb/plot.tok.gt9.5000'
neg_data_path = './Dataset/rotten_imdb/quote.tok.gt9.5000'
pos_data, neg_data = Reader.Reader().read_imdb(pos_data_path, neg_data_path)

#plain_test(word_model, pos_data, neg_data)
chain_test(word_model, pos_data, neg_data)