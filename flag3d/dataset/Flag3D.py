import random
import pickle
from mmengine.registry import DATASETS
from torch.utils.data import Dataset


@DATASETS.register_module()
class Flag3D(Dataset):
    def __init__(self, phase, path, voter_number):
        self.phase = phase
        self.voter_number = voter_number
        self.path = path
        with open(path, 'rb') as f:
            p = pickle.load(f)
            self.data = p

    def __getitem__(self, index):
        if self.phase == 'test':
            # test phase
            data_item = {'video': self.data['test'][index]['feat'],
                         'final_score': self.data['test'][index]['score1'] + self.data['test'][index]['score2']}
            target_obj = self.data['train'].copy()
            random.shuffle(target_obj)
            target_list = []
            for item in target_obj:
                if item['frame_dir'].split('A')[1].split('R')[0] == \
                        self.data['test'][index]['frame_dir'].split('A')[1].split('R')[0] \
                        and item['frame_dir'].split('P')[1].split('A')[0] != \
                        self.data['test'][index]['frame_dir'].split('P')[1].split('A')[0]:
                    target_item = {'video': item['feat'],
                                   'final_score': item['score1'] + item['score2'],
                                   }
                    target_list.append(target_item)
                if len(target_list) == self.voter_number:
                    return data_item, target_list
        elif self.phase == 'train':
            # train phase
            data_item = {'video': self.data['train'][index]['feat'],
                         'final_score': self.data['train'][index]['score1'] + self.data['train'][index]['score2']}
            target_obj = self.data['train'].copy()
            random.shuffle(target_obj)
            target = {}
            for item in target_obj:
                if item['frame_dir'].split('A')[1].split('R')[0] == \
                        self.data['train'][index]['frame_dir'].split('A')[1].split('R')[0]:
                    target = {'video': item['feat'],
                              'final_score': item['score1'] + item['score2'],
                              }
            return data_item, target

    def __len__(self):
        if self.phase == 'test':
            return len(self.data['test'])
        else:
            return len(self.data['train'])

    # def delta(self):
    #     '''
    #         RT: builder group
    #     '''
    #     delta = []
    #     dataset = self.data[self.subset]
    #     for i in range(len(dataset)):
    #         for j in range(i + 1, len(dataset)):
    #             delta.append(
    #                 abs(
    #                     self.data[self.subset][i]['score1'] + self.data[self.subset][i]['score2'] -
    #                     self.data[self.subset][j]['score1'] - self.data[self.subset][j]['score2']))
    #     return delta
