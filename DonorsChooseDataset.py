import torch
from torchtext import data
from torchtext.vocab import GloVe

class DCDataset(data.TabularDataset):
    
    #sort by the longest field, which is project essay
    @staticmethod
    def sort_key(ex):
        return len(ex.project_essay_1)

    #default is to use CPU and batch of 32. Change device to 0 for GPU
    @classmethod
    def iters(cls, batch_size = 32, device = -1):
        TEXT = data.Field(include_lengths=True)
        LABEL = data.Field(sequential=False, use_vocab = False)
        ID = data.Field(sequential=False)
        
        train, val, test = cls.splits(
            path='.', train='train.csv', skip_header=True,
            validation='val.csv', test='dev.csv', format='csv', 
            fields=[('id', ID), ('project_title', TEXT),
            ('project_resource_summary', TEXT), ('project_essay_1', TEXT), ('project_essay_2', TEXT), ('project_is_approved', LABEL)])

        #vocab is shared across all the text fields
        #CAUTION: GloVe will download all embeddings locally (862 MB).  If not interested, remove "vectors"
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
        ID.build_vocab(train)
        
  
        return data.BucketIterator.splits(
            (train, val, test), 
            batch_size=batch_size, device=device)