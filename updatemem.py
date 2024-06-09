import math
import torch
from torch.nn import functional as F
import torch.nn as nn
from basic_modules import *
import copy


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)     #大于0，硬收缩
    return output


class Memory(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, choices=None, hard_shrink=True):
        super(Memory, self).__init__()
        self.mem_dim = mem_dim   #插槽
        self.fea_dim = fea_dim
        self.choice = choices
        self.hard_shrink = hard_shrink
        self.memMatrix = nn.Parameter(torch.empty(mem_dim, fea_dim))
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)      # 随机化参数

    def get_update_query(self, mem, max_indices, score, query, train):

        m, d = mem.size()
        if train:
            query_update = torch.zeros((m, d)).cuda()
            random_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                # ex = update_indices[0][i]
                if a != 0:
                    # random_idx = torch.randperm(a)[0]
                    # idx = idx[idx != ex]
                    #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                    # random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
                else:
                    query_update[i] = 0
                    # random_update[i] = 0

            return query_update

        else:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                # ex = update_indices[0][i]
                if a != 0:
                    # idx = idx[idx != ex]
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                else:
                    query_update[i] = 0

            return query_update


    def update(self, query, keys, train=True):

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)

        if train:
            # top-1 queries (of each memory) update (weighted sum) & random pick
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_query,
                                                 query, train)

        else:
            # only weighted sum update when test
            query_update = self.get_update_query(keys, gathering_indices, softmax_score_query,
                                                 query, train)

        # top-1 update
        # query_update = query_reshape[updating_indices][0]
        # updated_memory = F.normalize(query_update + keys, dim=1)
        return query_update

    def get_score(self, mem, query):
        # bs, deep, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))  # b X deep X h X w X m
        score = score.view(-1, m)  # (b X deep X h X w) X m

        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)
        return score_query, score_memory

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        att_weight = F.linear(input=x, weight=self.memMatrix)  # [N,C] by [M,C]^T --> [N,M] 线性变换
        att_weight = F.softmax(att_weight, dim=1)  # NxM

        # if use hard shrinkage
        if self.shrink_thres > 0 and self.hard_shrink:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)  # [N,M]
            # normalize
            att_weight = F.normalize(att_weight, p=1, dim=1)  # [N,M]

        # out slot
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C] N是查询项的数目，C与记忆槽的维数相同

        return dict(out=out, att_weight=att_weight, mem=self.memMatrix)

        # return dict(output=out, att=att_weight)


class ML_MemAE_SC(nn.Module):
    def __init__(self, num_in_ch, features_root, mem_dim, ano_mem_dim, shrink_thres, mem_usage, skip_ops, hard_shrink_opt):
        super(ML_MemAE_SC, self).__init__()
        self.num_in_ch = num_in_ch
        self.mem_dim = mem_dim
        self.ano_mem_dim = ano_mem_dim
        self.shrink_thres = shrink_thres
        self.hard_shrink_opt = hard_shrink_opt
        self.mem_usage = mem_usage
        self.num_mem = sum(mem_usage)
        self.skip_ops = skip_ops
        self.mem3 = nn.Parameter(torch.empty(mem_dim, 256))
        self.mem3_ano = nn.Parameter(torch.empty(mem_dim, 256))
        # self.mem3 = F.normalize(torch.rand((mem_dim, 256), dtype=torch.float), dim=1).cuda()
        self.reset_parameters()






        self.in_conv = inconv(num_in_ch, features_root)         #double_conv
        self.down_1 = down(features_root, features_root * 2)                #Conv3d  double_conv
        self.down_2 = down(features_root * 2, features_root * 4)
        self.down_3 = down(features_root * 4, features_root * 8)

        # memory modules
        # self.mem3 = Memory(mem_dim=self.mem_dim, fea_dim=features_root * 8, #* 2 * 32 * 32,
        #                    shrink_thres=self.shrink_thres, hard_shrink=self.hard_shrink_opt) if self.mem_usage[3] else None
        # self.mem3_ano = Memory(mem_dim=self.ano_mem_dim, fea_dim=features_root * 8,  # * 2 * 32 * 32,
        #                    shrink_thres=self.shrink_thres, hard_shrink=self.hard_shrink_opt) if self.mem_usage[3] else None
        # self.memory = Memory_up(memory_size=10, feature_dim=256, key_dim=512, temp_update=0.1, temp_gather=0.1)

        # self.up_4 = outconv(features_root * 16, features_root * 8)
        self.out_conv0 = outconv(features_root * 16, features_root * 8)
        self.up_3 = up(features_root * 8, features_root * 4, op=self.skip_ops[-1])
        self.up_2 = up(features_root * 4, features_root * 2, op=self.skip_ops[-2])
        self.up_1 = up(features_root * 2, features_root, op=self.skip_ops[-3])
        # self.out_conv = outconv(features_root, num_in_ch)
        self.out_conv1 = outconv(features_root, num_in_ch)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mem3.size(1))
        self.mem3.data.uniform_(-stdv, stdv)  # 随机化参数

    def get_score(self, mem, query):
        bs, deep, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))  # b X deep X h X w X m
        score = score.view(bs * deep * h * w, m)  # (b X deep X h X w) X m

        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)

        return score_query, score_memory


    def get_update_query(self, mem, max_indices, update_indices, score, query, train):

        m, d = mem.size()
        if train:
            query_update = torch.zeros((m, d)).cuda()
            random_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                # ex = update_indices[0][i]
                if a != 0:
                    # random_idx = torch.randperm(a)[0]
                    # idx = idx[idx != ex]
                    #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                    # random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
                else:
                    query_update[i] = 0
                    # random_update[i] = 0

            return query_update

        else:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                # ex = update_indices[0][i]
                if a != 0:
                    # idx = idx[idx != ex]
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                else:
                    query_update[i] = 0

            return query_update


    def update(self, query, keys, shrink_thres=0.0025, train=True):

        self.shrink_thres = shrink_thres

        batch_size, deep, h, w, dims = query.size()  # b X deep X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * deep * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)

        if train:
            # top-1 queries (of each memory) update (weighted sum) & random pick
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)



        else:
            # only weighted sum update when test
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)

        # top-1 update
        # query_update = query_reshape[updating_indices][0]
        # updated_memory = F.normalize(query_update + keys, dim=1)


        return updated_memory.detach()


    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()  # (b X deep X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')

        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return pointwise_loss

    def spread_loss(self, query, keys, train):
        batch_size, deep, h, w, dims = query.size()  # b X deep X h X w X d

        loss = torch.nn.TripletMarginLoss(margin=1.0)

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * deep * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        # 1st, 2nd closest memories
        pos = keys[gathering_indices[:, 0]]
        neg = keys[gathering_indices[:, 1]]

        spreading_loss = loss(query_reshape, pos.detach(), neg.detach())

        return spreading_loss

    def gather_loss(self, query, keys, train):

        batch_size, deep, h, w, dims = query.size()  # b X deep X h X w X d

        loss_mse = torch.nn.MSELoss()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * deep * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss

    def read(self, query, updated_memory):

        batch_size, deep, h, w, dims = query.size()  # b X deep X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)

        query_reshape = query.contiguous().view(batch_size * deep * h * w, dims)

        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory)  # (b X deep X h X w) X dim
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # (b X deep X h X w) X 2d
        updated_query = updated_query.view(batch_size, deep, h, w, 2 * dims)
        updated_query = updated_query.permute(0, 4, 1, 2, 3)

        return updated_query, softmax_score_query, softmax_score_memory



    def forward(self, x, mem=True, mem_ano=False, train=True, up=True):
        """
        :param x: size [bs,C*seq_len,H,W]
        :return:
        """
        x0 = self.in_conv(x)     # x0: 2, 32, 16, 256, 256
        x1 = self.down_1(x0)     # x1: 2, 64, 8,  128, 128
        x2 = self.down_2(x1)     # x2: 2, 128,4,  64,  64
        x3 = self.down_3(x2)     # x3: 2, 256,2,  32,  32
        feas = x3.clone()



        # train
        if train and mem:
            query = F.normalize(feas, dim=1)
            query = query.permute(0, 2, 3, 4, 1)  # b X deep X h X w X d
            # gathering loss
            gathering_loss = self.gather_loss(query, self.mem3, train)
            # spreading_loss
            spreading_loss = self.spread_loss(query, self.mem3, train)
            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, self.mem3)
            # update
            updated_memory = self.update(query, self.mem3, train)

        if train and mem_ano:
            query = F.normalize(feas, dim=1)
            query = query.permute(0, 2, 3, 4, 1)  # b X deep X h X w X d
            # gathering loss
            gathering_loss_ano = self.gather_loss(query, self.mem3_ano, train)
            # spreading_loss
            spreading_loss_ano = self.spread_loss(query, self.mem3_ano, train)
            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, self.mem3_ano)
            # update
            updated_memory_ano = self.update(query, self.mem3_ano, train)

            # return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss

        # test
        else:
            query = F.normalize(feas, dim=1)
            query = query.permute(0, 2, 3, 4, 1)  # b X deep X h X w X dims
            # gathering loss
            gathering_loss = self.gather_loss(query, self.mem3, train)

            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, self.mem3)

            # update
            updated_memory = self.mem3

            # return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss

        recon = self.out_conv0(updated_query)
        # recon = self.up_3(x3, x2 if self.skip_ops[-1] != "none" else None)
        recon = self.up_3(recon, x2 if self.skip_ops[-1] != "none" else None)
        recon = self.up_2(recon, x1 if self.skip_ops[-2] != "none" else None)
        recon = self.up_1(recon, x0 if self.skip_ops[-3] != "none" else None)
        recon = self.out_conv1(recon)  # Conv3d

        if up and mem:
            outs = dict(recon=recon, mem=updated_memory, mem_ano=torch.zeros(1, 1))  # dummy attention weights

        if up and mem_ano:
            outs = dict(recon=recon, mem=torch.zeros(1, 1), mem_ano=updated_memory_ano)  # dummy attention weights

        if not mem and not mem_ano:
            outs = dict(recon=recon, att_weight3=torch.zeros(1, 1), att_weight3_ano=torch.zeros(1, 1),
                        mem=torch.zeros(1, 1), mem_ano=torch.zeros(1, 1))  # dummy attention weights


        if train and mem:
            return outs, feas, gathering_loss, spreading_loss

        if train and mem_ano:
            return outs, feas, gathering_loss_ano, spreading_loss_ano

        else:
            return outs, feas, gathering_loss


