
import time
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim*2)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb_id, pos_emb_id, neg_emb_id, users_emb_ego, pos_emb_ego, neg_emb_ego = self.getEmbedding(users, pos, neg)
        _, _, users_lm_emb, items_lm_emb = self.computer()

        # 基于ID的评分
        pos_scores_id = self.compute_scores(users_emb_id, pos_emb_id)
        neg_scores_id = self.compute_scores(users_emb_id, neg_emb_id)

        # 基于语言模型的评分
        pos_scores_lm = self.compute_scores(users_lm_emb[users.long()], items_lm_emb[pos.long()])
        neg_scores_lm = self.compute_scores(users_lm_emb[users.long()], items_lm_emb[neg.long()])

        # 计算损失
        loss_id = torch.mean(torch.nn.functional.softplus(neg_scores_id - pos_scores_id))
        loss_lm = torch.mean(torch.nn.functional.softplus(neg_scores_lm - pos_scores_lm))
        total_loss =loss_id +  0.5 * loss_lm

        # 正则化损失
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2))/float(len(users))
    
        return total_loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config: dict, 
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__init_weight()
        self.contrastive_loss = InfoNCELoss(temperature=config.get('temperature', 0.5))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])
    def __init_weight(self):
        self.margin = 0.1
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.user_lm_T = nn.Parameter(F.normalize(self.dataset.user_lm_T.to(self.device)), requires_grad=True)
        self.item_lm = self.dataset.item_lm.to(self.device)
        self.item_lm = nn.Parameter(F.normalize(self.item_lm, dim=1), requires_grad=True)
        self.phi3 = nn.Parameter(torch.tensor(self.dataset.phi3, dtype=torch.float32).to(self.device), requires_grad=False)
        self.phi4 = nn.Parameter(torch.tensor(self.dataset.phi4, dtype=torch.float32).to(self.device), requires_grad=False)
        self.phi2 = nn.Parameter(torch.tensor(self.dataset.phi2, dtype=torch.float32).to(self.device), requires_grad=False)
        self.selected_k = self.config.get('selected_k',1000)
        print(self.phi3.shape)
        print(self.phi4.shape)
        print(self.phi2.shape)
        print("First row of phi2:", self.phi2[0])
        print("First row of phi3:", self.phi3[0])
        print("First row of phi4:", self.phi4[0])
        self.phi1 = torch.matmul(self.phi3, self.phi4)
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initializer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is ready to go(dropout:{self.config['dropout']})")
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        users_emb = self.embedding_user.weight  # U x D
        items_emb = self.embedding_item.weight  # I x D
        all_emb = torch.cat([users_emb, items_emb])  # (U + I) x D
        embs = [all_emb]
    
    # Dropout
        if self.config['dropout']:
            if self.training:
                print("dropping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
    
    # Propagate through layers
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
    
        embs = torch.stack(embs, dim=1)  # (U + I) x L x D
        light_out = torch.mean(embs, dim=1)  # (U + I) x D
        users_id_emb, items_id_emb = torch.split(light_out, [self.num_users, self.num_items])
        k = self.selected_k
    # 保留前k个关键词
        start_time = time.time()

        # 计算 self.user_lm
        k = self.selected_k
        phi1_subset = self.phi1[:, :k]  # U x k
        user_lm_T_subset = self.user_lm_T[:k, :]  # k x D
        phi2_subset = self.phi2[:, :k] 
        self.user_lm = 0.1 * torch.matmul(phi1_subset, user_lm_T_subset) + \
                       0.9 * torch.matmul(phi2_subset, user_lm_T_subset)

        # 结束计时
        end_time = time.time()
        time_cost = end_time - start_time
        #print(f"计算 user_lm 的时间开销：{time_cost:.4f} 秒")
        users_lm_emb = self.user_lm
        items_lm_emb = self.item_lm
        return users_id_emb, items_id_emb, users_lm_emb, items_lm_emb
    
    def getEmbedding(self, users, pos_items, neg_items):
        users_id_emb, items_id_emb, users_lm_emb, items_lm_emb = self.computer()
        users_emb = users_id_emb[users]
        pos_emb = items_id_emb[pos_items]
        neg_emb = items_id_emb[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    
    def bpr_loss(self, users, pos, neg):
        users_emb_id, pos_emb_id, neg_emb_id, users_emb_ego, pos_emb_ego, neg_emb_ego = self.getEmbedding(users, pos, neg)
        _, _, users_lm_emb, items_lm_emb = self.computer()

        # 基于ID的评分
        pos_scores_id = self.compute_scores(users_emb_id, pos_emb_id)
        neg_scores_id = self.compute_scores(users_emb_id, neg_emb_id)

        # 基于语言模型的评分
        pos_scores_lm = self.compute_scores(users_lm_emb[users.long()], items_lm_emb[pos.long()])
        neg_scores_lm = self.compute_scores(users_lm_emb[users.long()], items_lm_emb[neg.long()])

        # 计算损失
        loss_id = torch.mean(torch.nn.functional.softplus(neg_scores_id - pos_scores_id))
        loss_lm = torch.mean(torch.nn.functional.softplus(neg_scores_lm - pos_scores_lm))
        total_loss =  0.5 *loss_id + loss_lm
         # 用户对比学习损失
        contrastive_loss_user = self.calcSSL(users_emb_id, users_lm_emb[users.long()])

        # 物品对比学习损失
        contrastive_loss_item = self.calcSSL(pos_emb_id, items_lm_emb[pos.long()])

        # 总对比学习损失
        contrastive_loss = contrastive_loss_user + contrastive_loss_item
        # 总损失
        
        total_loss+=0.0001*contrastive_loss
        

        # 正则化损失
        
        reg_loss = (1/2)*(users_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2))/float(len(users))
    
        return total_loss, reg_loss
    
    def contrastive_loss_fn(self, users, pos, neg):
        _, _, users_lm_emb, items_lm_emb = self.computer()

        # 基于语言模型的对比损失
        pos_users_emb = users_lm_emb[users.long()]
        pos_items_emb = items_lm_emb[pos.long()]
        neg_items_emb = items_lm_emb[neg.long()]

        lm_loss_pos = self.contrastive_loss(pos_users_emb, pos_items_emb)
        lm_loss_neg = self.contrastive_loss(pos_users_emb, neg_items_emb)

        lm_loss = lm_loss_pos + lm_loss_neg
        return lm_loss


    def compute_distances(self, emb1, emb2):
        return torch.sum((emb1 - emb2) ** 2, dim=1)

    def getUsersRating(self, users):
        users_id_emb, items_id_emb, users_lm_emb, items_lm_emb = self.computer()

        users_emb_id = users_id_emb[users.long()]
        items_emb_id = items_id_emb

        users_emb_lm = users_lm_emb[users.long()]
        items_emb_lm = items_lm_emb

        # 计算基于ID嵌入的评分
        rating_id = torch.matmul(users_emb_id, items_emb_id.t())

        # 计算基于语言模型嵌入的评分
        rating_lm = torch.matmul(users_emb_lm, items_emb_lm.t())

        # 加权和
        rating = 0.5 *self.f(rating_id) + self.f(rating_lm)
        return rating

    def forward(self, users, items):
        users_id_emb, items_id_emb, users_lm_emb, items_lm_emb = self.computer()

        users_emb_id = users_id_emb[users]
        items_emb_id = items_id_emb[items]

        users_emb_lm = users_lm_emb[users]
        items_emb_lm = items_lm_emb[items]

        # 计算基于ID嵌入的评分
        inner_pro_id = torch.mul(users_emb_id, items_emb_id)
        gamma_id = torch.sum(inner_pro_id, dim=1)

        # 计算基于语言模型嵌入的评分
        inner_pro_lm = torch.mul(users_emb_lm, items_emb_lm)
        gamma_lm = torch.sum(inner_pro_lm, dim=1)

        # 加权和
        gamma =  0.5 *gamma_id + gamma_lm
        return gamma

    def load_embeddings(file_path):
        data = np.load(file_path)
        return torch.tensor(data['embeddings'], dtype=torch.float32)

    def compute_scores(self, user_emb, item_emb):
        return torch.sum(torch.mul(user_emb, item_emb), dim=1)
    
    def calcSSL(self, lmLat, gnnLat):
        # 对输入进行归一化，防止数值溢出
        lmLat = F.normalize(lmLat, p=2, dim=1)
        gnnLat = F.normalize(gnnLat, p=2, dim=1)
        
        # 计算正样本得分
        posScore = torch.exp(torch.sum(lmLat * gnnLat, dim=1)).to(self.device)
        
        # 检查正样本得分是否包含无穷值或NaN
        if torch.isnan(posScore).any() or torch.isinf(posScore).any():
            print("Warning: posScore contains NaN or Inf")

        # 计算负样本得分
        negScore = torch.sum(torch.exp(gnnLat @ torch.transpose(lmLat, 0, 1)), dim=1).to(self.device)
        
        # 检查负样本得分是否包含无穷值或NaN
        if torch.isnan(negScore).any() or torch.isinf(negScore).any():
            print("Warning: negScore contains NaN or Inf")

        # 计算对比学习损失
        uLoss = torch.sum(-torch.log(posScore / (negScore + 1e-8) + 1e-8)).to(self.device)
        
        # 检查对比学习损失是否包含无穷值或NaN
        if torch.isnan(uLoss).any() or torch.isinf(uLoss).any():
            print("Warning: uLoss contains NaN or Inf")
        
        return uLoss
    def loss_fn(self, p, z):  # cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    def poly_feature(self, x):
        user_e = self.projector(x) 
        xx = self.bn(user_e).T @ self.bn(user_e)
        poly = (self.a * xx + self.polyc) ** self.degree 
        return poly.mean().log()
    def calc_diversity_metrics(self, recommended_lists, total_items):
        """
        基于topK推荐列表计算多样性指标
        Args:
            recommended_lists: List[List[item_id]] 每个用户的topK推荐列表
            total_items: 物品总数
        Returns:
            coverage, gini, entropy
        """
        # 展平所有推荐物品
        all_recommended = torch.cat([torch.tensor(lst) for lst in recommended_lists]).to(self.device)
        
        # 计算覆盖率
        unique_items = torch.unique(all_recommended)
        coverage = len(unique_items) / total_items
        
        # 计算物品出现概率
        item_counts = torch.bincount(all_recommended, minlength=total_items)
        total_recs = len(all_recommended)
        probs = item_counts.float() / total_recs if total_recs > 0 else item_counts.float()
        
        # 计算基尼指数
        sorted_probs = torch.sort(probs)[0]
        n = total_items
        gini = torch.sum((2 * torch.arange(1, n+1).float().to(self.device) - n - 1) * sorted_probs) / (n - 1)
        
        # 计算熵
        epsilon = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + epsilon))
        
        return coverage.item(), gini.item(), entropy.item()
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(cl_loss)
