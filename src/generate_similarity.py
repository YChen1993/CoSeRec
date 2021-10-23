import  math
import os
import pickle

class ItemCFBasedSimilarity:
    def __init__(self,data_file=None, similarity_path=None, model_type='ItemCF'):
        self.similarity_path = similarity_path
        self.train_data_dict = self._load_train_data(data_file)
        self.model_type = model_type
        self.similarity_dict = self.load_similarity_dict(self.similarity_path)

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user,item,record in data:
            train_data_dict.setdefault(user,{})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path = './similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self,data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-3]
            for itemid in items:
                train_data.append((userid,itemid,int(1)))
        return self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self,train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()
        for idx, (u, items) in enumerate(train.items()):
            if idx%2000 == 12:
                print("proceeded: ", idx)
                break
            if self.model_type == 'ItemCF':
                for i in items.keys():
                    N.setdefault(i,0)
                    N[i] += 1
                    for j in items.keys():
                        if i == j:
                            continue
                        C.setdefault(i,{})
                        C[i].setdefault(j,0)
                        C[i][j] += 1
            elif self.model_type == 'ItemCF_IUF':
                for i in items.keys():
                    N.setdefault(i,0)
                    N[i] += 1
                    for j in items.keys():
                        if i == j:
                            continue
                        C.setdefault(i,{})
                        C[i].setdefault(j,0)
                        C[i][j] += 1 / math.log(1 + len(items) * 1.0)
        self.itemSimBest = dict()
        for idx, (cur_item, related_items) in enumerate(C.items()):
            if idx%2000 == 0:
                print("proceeded: ", idx)
            self.itemSimBest.setdefault(cur_item,{})
            for related_item, score in related_items.items():
                self.itemSimBest[cur_item].setdefault(related_item,0);
                self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
        self._save_dict(self.itemSimBest, save_path=save_path)

    def load_similarity_dict(self, similarity_dict_path):
        if not similarity_dict_path:
            raise ValueErroe('invalid path')
        elif not os.path.exists(similarity_dict_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        with open(similarity_dict_path, 'rb') as read_file:
            similarity_dict = pickle.load(read_file)
        return similarity_dict

    def most_similar(self, item, top_k=8):
        top_k_items_with_score = sorted(self.similarity_dict[item].items(),key=lambda x : x[1], \
                                    reverse=True)[0:top_k]
        return list(map(lambda x: x[0], top_k_items_with_score))


def test_item_based_cf():
    cf_based_similarity  =  ItemCFBasedSimilarity(data_file='../data/Sports_and_Outdoors_sample.txt',
                                similarity_path='../data/item_cf_iuf_similarity.pkl',
                                model_type='ItemCF_IUF')
    print(cf_based_similarity.most_similar('2', top_k=4))

if __name__ == "__main__":
    test_item_based_cf()