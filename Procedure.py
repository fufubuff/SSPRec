
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    

    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):

    u_batch_size = world.config['test_u_batch_size']

    dataset: utils.BasicDataset

    testDict: dict = dataset.testDict

    Recmodel: model.LightGCN

    # eval mode with no dropout

    Recmodel = Recmodel.eval()

    max_K = max(world.topks)

    if multicore == 1:

        pool = multiprocessing.Pool(CORES)

    # 初始化指标

    results = {'precision': np.zeros(len(world.topks)),

               'recall': np.zeros(len(world.topks)),

               'ndcg': np.zeros(len(world.topks)),

               'coverage': np.zeros(len(world.topks)),

               'gini': np.zeros(len(world.topks)),

               'entropy': np.zeros(len(world.topks))

              }

    with torch.no_grad():

        users = list(testDict.keys())

        try:

            assert u_batch_size <= len(users) / 10

        except AssertionError:

            print(f"test_u_batch_size is too big for this dataset, try a smaller one {len(users) // 10}")

        users_list = []

        rating_list = []

        groundTrue_list = []

        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):

            allPos = dataset.getUserPosItems(batch_users)

            groundTrue = [testDict[u] for u in batch_users]

            batch_users_gpu = torch.Tensor(batch_users).long()

            batch_users_gpu = batch_users_gpu.to(world.device)



            rating = Recmodel.getUsersRating(batch_users_gpu)

            exclude_index = []

            exclude_items = []

            for range_i, items in enumerate(allPos):

                exclude_index.extend([range_i] * len(items))

                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1<<10)

            _, rating_K = torch.topk(rating, k=max_K)

            users_list.append(batch_users)  # 确保每次循环中都添加 batch_users

            rating_list.append(rating_K.cpu())

            groundTrue_list.append(groundTrue)



        assert total_batch == len(users_list)

        X = zip(rating_list, groundTrue_list)

        if multicore == 1:

            pre_results = pool.map(test_one_batch, X)

        else:

            pre_results = []

            for x in X:

                pre_results.append(test_one_batch(x))

        

        scale = float(u_batch_size/len(users))

        for result in pre_results:

            results['recall'] += result['recall']

            results['precision'] += result['precision']

            results['ndcg'] += result['ndcg']

        results['recall'] /= float(len(users))

        results['precision'] /= float(len(users))

        results['ndcg'] /= float(len(users))



        # 计算 Coverage, Gini, 和 Entropy 基于不同的 top-k

        for k_index, K in enumerate(world.topks):

            all_recommended = torch.cat([ranking[:, :K] for ranking in rating_list]).view(-1).to(world.device)

            unique_items = torch.unique(all_recommended)

            item_counts = torch.bincount(all_recommended, minlength=dataset.m_items)



            # Coverage 

            results['coverage'][k_index] = len(unique_items) / dataset.m_items



            # Gini Index

            probs = item_counts.float() / len(all_recommended)

            sorted_probs = torch.sort(probs).values

            n = len(sorted_probs)

            results['gini'][k_index] = (torch.sum((2 * torch.arange(1, n+1).float().to(world.device) - n - 1) * sorted_probs) / (n - 1)).item()



            # Entropy

            epsilon = 1e-10

            results['entropy'][k_index] = (-torch.sum(probs * torch.log(probs + epsilon))).item()



        print('After Testing:')

        if world.tensorboard:

            for k_index, K in enumerate(world.topks):

                w.add_scalar(f'Test/Coverage@{K}', results['coverage'][k_index], epoch)

                w.add_scalar(f'Test/Gini@{K}', results['gini'][k_index], epoch)

                w.add_scalar(f'Test/Entropy@{K}', results['entropy'][k_index], epoch)

            

            w.add_scalars(f'Test/Recall@{world.topks}',

                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)

            w.add_scalars(f'Test/Precision@{world.topks}',

                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)

            w.add_scalars(f'Test/NDCG@{world.topks}',

                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)

        

        if multicore == 1:

            pool.close()

        print(results)

        return results