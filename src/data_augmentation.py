import random
import copy
import itertools

class CombinatorialEnumerate(object):
    """Given M type of augmentations, and a original sequence, successively call \
    the augmentation 2*C(M, 2) times can generate total C(M, 2) augmentaion pairs. 
    In another word, the augmentation method pattern will repeat after every 2*C(M, 2) calls.
    
    For example, M = 3, the argumentation methods to be called are in following order: 
    a1, a2, a1, a3, a2, a3. Which formed three pair-wise augmentations:
    (a1, a2), (a1, a3), (a2, a3) for multi-view contrastive learning.
    """
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2, \
                item_similarity_model=None, insert_rate=0.3, \
                max_insert_num_per_pos=3, substitute_rate=0.3, n_views=5):
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma), 
                            Insert(item_similarity_model, insert_rate=insert_rate, 
                                max_insert_num_per_pos=max_insert_num_per_pos),
                            Substitute(item_similarity_model, substitute_rate=substitute_rate)]
        self.n_views = n_views
        # length of the list == C(M, 2)
        self.augmentation_idx_list = self.__get_augmentation_idx_order()
        self.total_augmentation_samples = len(self.augmentation_idx_list)
        self.cur_augmentation_idx_of_idx = 0     

    def __get_augmentation_idx_order(self):
        augmentation_idx_list = []
        for (view_1, view_2) in itertools.combinations([i for i in range(self.n_views)], 2):
            augmentation_idx_list.append(view_1)
            augmentation_idx_list.append(view_2)
        return augmentation_idx_list

    def __call__(self, sequence):
        augmentation_idx = self.augmentation_idx_list[self.cur_augmentation_idx_of_idx]
        augment_method = self.data_augmentation_methods[augmentation_idx]
        # keep the index of index in range(0, C(M,2))
        self.cur_augmentation_idx_of_idx += 1
        self.cur_augmentation_idx_of_idx = self.cur_augmentation_idx_of_idx % self.total_augmentation_samples
        # print(augment_method.__class__.__name__)
        return augment_method(sequence)

class Random(object):
    """Randomly pick one data augmentation type every time call"""
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2, \
                item_similarity_model=None, insert_rate=0.3, \
                max_insert_num_per_pos=3, substitute_rate=0.3,\
                augment_threshold=-1,
                augment_type_for_short='SIM'):
        self.augment_threshold = augment_threshold
        self.augment_type_for_short = augment_type_for_short
        if self.augment_threshold == -1:
            self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta), 
                                Insert(item_similarity_model, insert_rate=insert_rate, 
                                    max_insert_num_per_pos=max_insert_num_per_pos),
                                Substitute(item_similarity_model, substitute_rate=substitute_rate)]
            print("Total augmentation numbers: ", len(self.data_augmentation_methods))
        elif self.augment_threshold > 0:
            print("short sequence augment type:", self.augment_type_for_short)
            if self.augment_type_for_short == 'SI':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate)]
            elif self.augment_type_for_short == 'SIM':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Mask(gamma=gamma)]

            elif self.augment_type_for_short == 'SIR':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Reorder(beta=gamma)]
            elif self.augment_type_for_short == 'SIC':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Crop(tao=tao)]
            elif self.augment_type_for_short == 'SIMR':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Mask(gamma=gamma), Reorder(beta=gamma)]
            elif self.augment_type_for_short == 'SIMC':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Mask(gamma=gamma), Crop(tao=tao)]
            elif self.augment_type_for_short == 'SIRC':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                    Reorder(beta=gamma), Crop(tao=tao)]
            else:
                print("all aug set for short sequences")
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    Substitute(item_similarity_model, substitute_rate=substitute_rate),
                                   Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma)]                
            self.long_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                    max_insert_num_per_pos=max_insert_num_per_pos, 
                                    augment_threshold=self.augment_threshold),
                                Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma),
                                Substitute(item_similarity_model, substitute_rate=substitute_rate)]
            print("Augmentation methods for Long sequences:", len(self.long_seq_data_aug_methods))
            print("Augmentation methods for short sequences:", len(self.short_seq_data_aug_methods))
        else:
            raise ValueError("Invalid data type.")

    def __call__(self, sequence):
        if self.augment_threshold == -1:
            #randint generate int x in range: a <= x <= b
            augment_method_idx = random.randint(0, len(self.data_augmentation_methods)-1)
            augment_method = self.data_augmentation_methods[augment_method_idx]
            # print(augment_method.__class__.__name__) # debug usage
            return augment_method(sequence)
        elif self.augment_threshold > 0:
            seq_len = len(sequence)
            if seq_len > self.augment_threshold:
                #randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.long_seq_data_aug_methods)-1)
                augment_method = self.long_seq_data_aug_methods[augment_method_idx]
                # print(augment_method.__class__.__name__) # debug usage
                return augment_method(sequence)
            elif seq_len <= self.augment_threshold:
                #randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.short_seq_data_aug_methods)-1)
                augment_method = self.short_seq_data_aug_methods[augment_method_idx]
                # print(augment_method.__class__.__name__) # debug usage
                return augment_method(sequence)                

def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
#     print("offline: ",top_k_one, "online: ", top_k_two)
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]
    
class Insert(object):
    """Insert similar items every time call"""
    def __init__(self, item_similarity_model, insert_rate=0.4, max_insert_num_per_pos=1,
            augment_threshold=14):
        self.augment_threshold = augment_threshold
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.insert_rate = insert_rate
        self.max_insert_num_per_pos = max_insert_num_per_pos
    

        
    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        insert_nums = max(int(self.insert_rate*len(copied_sequence)), 1)
        insert_idx = random.sample([i for i in range(len(copied_sequence))], k = insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos/insert_nums)))
                if self.ensemble:
                    top_k_one = self.item_sim_model_1.most_similar(item,
                                            top_k=top_k, with_score=True)
                    top_k_two = self.item_sim_model_2.most_similar(item,
                                            top_k=top_k, with_score=True)
                    inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two)
                else:
                    inserted_sequence += self.item_similarity_model.most_similar(item,
                                            top_k=top_k)
            inserted_sequence += [item]

        return inserted_sequence

                    
class Substitute(object):
    """Substitute with similar items"""
    def __init__(self, item_similarity_model, substitute_rate=0.1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        substitute_nums = max(int(self.substitute_rate*len(copied_sequence)), 1)
        substitute_idx = random.sample([i for i in range(len(copied_sequence))], k = substitute_nums)
        inserted_sequence = []
        for index in substitute_idx:
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index],
                                        with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index],
                                        with_score=True)
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
                copied_sequence[index] = substitute_items[0]
            else:
            
                copied_sequence[index] = copied_sequence[index] = self.item_similarity_model.most_similar(copied_sequence[index])[0]
        return copied_sequence

class Crop(object):
    """Randomly crop a subseq from the original sequence"""
    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao*len(copied_sequence))
        #randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1)
        if sub_seq_length<1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index:start_index+sub_seq_length]
            return cropped_seq

class Mask(object):
    """Randomly mask k items given a sequence"""
    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma*len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k = mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence

class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""
    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta*len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1)
        sub_seq = copied_sequence[start_index:start_index+sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index+sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq

if __name__ == '__main__':
    reorder = Reorder(beta=0.2)
    sequence=[14052, 10908,  2776, 16243,  2726,  2961, 11962,  4672,  2224,
    5727,  4985,  9310,  2181,  3428,  4156, 16536,   180, 12044, 13700]
    rs = reorder(sequence)
    crop = Crop(tao=0.2)
    rs = crop(sequence)
    # rt = RandomType()
    # rs = rt(sequence)
    n_views = 5
    enum_type = CombinatorialEnumerateType(n_views=n_views)
    for i in range(40):
        if i == 20:
            print('-------')
        es = enum_type(sequence)