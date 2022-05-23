import torch

path = '/Users/qizhao/Documents/project/steinerTree/REST-main/save/DAC21/rsmt50b.pt'

dict = torch.load(path)
dict['nnsteiner_dict'] = dict['actor_state_dict']
torch.save(dict, '/Users/qizhao/Documents/project/steinerTree/REST-main/save/nnsteiner.pt')



path = '/data/qiz/steiner/REST-main/save/DAC21/rsmt50b.pt'

dict = torch.load(path)
dict['nnsteiner_dict'] = dict['actor_state_dict']
torch.save(dict, '/data/qiz/steiner/nnsteiner.pt')