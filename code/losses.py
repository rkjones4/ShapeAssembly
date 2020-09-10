import torch
import torch.nn as nn
import faiss
import numpy as np
import math
from copy import deepcopy

def robust_norm(var, dim=2):
    return ((var ** 2).sum(dim=dim) + 1e-8).sqrt()

class FScore():
    def __init__(self, device):
        self.dimension = 3
        self.k = 1
        #
        self.device = device
        self.gpu_id = torch.cuda.current_device()
        self.faiss_gpu = hasattr(faiss, 'StandardGpuResources')

        if self.faiss_gpu:
            # we need only a StandardGpuResources per GPU
            self.res = faiss.StandardGpuResources()
            # self.res.setTempMemoryFraction(0.1)
            #self.res.setTempMemory(
            #    4 * (1024 * 1024 * 1024)
            #)  # Bytes, the single digit is basically GB)
            self.flat_config = faiss.GpuIndexFlatConfig()
            self.flat_config.device = self.gpu_id

        # place holder
        self.forward_loss = torch.FloatTensor([0])
        self.backward_loss = torch.FloatTensor([0])

    def build_nn_index(self, database):
        """
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        """
        # index = faiss.GpuIndexFlatL2(self.res, self.dimension, self.flat_config)  # dimension is 3
        index_cpu = faiss.IndexFlatL2(self.dimension)

        if self.faiss_gpu:
            index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, index_cpu)
        else:
            index = index_cpu

        index.add(database)
        return index

    def search_nn(self, index, query, k):
        """
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: Variable of Nxk, type FloatTensor, in GPU
                 I: Variable of Nxk, type LongTensor, in GPU
        """
        D, I = index.search(query, k)

        D_var = torch.from_numpy(np.ascontiguousarray(D))
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))
        if self.gpu_id >= 0:
            D_var = D_var.to(self.device)
            I_var = I_var.to(self.device)

        return D_var, I_var

    def getAvgDist(self, index, query):
        D, I = index.search(query, 2)

        m_d = math.sqrt(np.percentile(D[:,1],90))
        return m_d

    def getOpMatch(self, points):
        return (points.max(axis = 0) - points.min(axis = 0)).max() / 100

    def score(self, predict_pc_6, gt_pc_6, use_normals=True):
        """
        :param predict_pc: Bx3xM Variable in GPU
        :param gt_pc: Bx3xN Variable in GPU
        :return:
        """
        if self.gpu_id >= 0:
            predict_pc_6 = predict_pc_6.to(self.device)
            gt_pc_6 = gt_pc_6.to(self.device)

        predict_pc = predict_pc_6[:, :3, :]
        gt_pc = gt_pc_6[:, :3, :]
        #
        predict_pcn = predict_pc_6[:, 3:, :]
        gt_pcn = gt_pc_6[:, 3:, :]

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(
            torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxMx3
        gt_pc_np = np.ascontiguousarray(
            torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxNx3

        # selected_gt: Bxkx3xM
        selected_gt_by_predict = torch.FloatTensor(
            predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2]
        )
        # selected_predict: Bxkx3xN
        selected_predict_by_gt = torch.FloatTensor(
            gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2]
        )
        if use_normals:
            # normals
            selected_gt_by_predictn = torch.FloatTensor(
                predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2]
            )
            selected_predict_by_gtn = torch.FloatTensor(
                gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2]
            )

        if self.gpu_id >= 0:
            selected_gt_by_predict = selected_gt_by_predict.to(self.device)
            selected_predict_by_gt = selected_predict_by_gt.to(self.device)
            if use_normals:
                selected_gt_by_predictn = selected_gt_by_predictn.to(self.device)
                selected_predict_by_gtn = selected_predict_by_gtn.to(self.device)

        # process each batch independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_var = self.search_nn(index_gt, predict_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_gt_by_predict[i, k, ...] = gt_pc[i].index_select(
                    1, I_var[:, k]
                )
                if use_normals:
                    selected_gt_by_predictn[i, k, ...] = gt_pcn[i].index_select(
                        1, I_var[:, k]
                    )

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_var = self.search_nn(index_predict, gt_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_predict_by_gt[i, k, ...] = predict_pc[i].index_select(
                    1, I_var[:, k]
                )
                if use_normals:
                    selected_predict_by_gtn[i, k, ...] = predict_pcn[i].index_select(
                        1, I_var[:, k]
                    )

        index_gt = self.build_nn_index(gt_pc_np[i])

        dist = self.getAvgDist(index_gt, gt_pc_np[0])


        # compute loss ===================================================
        # selected_gt(Bxkx3xM) vs predict_pc(Bx3xM)
        r_to_gt = predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict) - selected_gt_by_predict #Forward as in reconstructed to ground truth?
        gt_to_r = selected_predict_by_gt - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt) #Backward as in ground truth to reconstructed?

        r_to_gt = robust_norm(
            selected_gt_by_predict
            - predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict),
            dim=2
        ).cpu().detach().numpy()

        gt_to_r = robust_norm(
            selected_predict_by_gt
            - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt),
            dim=2
        ).cpu().detach().numpy()



        r_to_gt = r_to_gt.flatten()
        gt_to_r = gt_to_r.flatten()

        ones = np.ones(r_to_gt.shape)

        dt = dist

        precision = (100 / ones.shape[0]) * np.sum(ones[r_to_gt < dt])
        recall = (100 / ones.shape[0]) * np.sum(ones[gt_to_r < dt])

        return (2*precision*recall) / (precision + recall + 1e-8)
    
    
class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()
        self.dimension = 3
        self.k = 1
        #
        self.device = device
        self.gpu_id = torch.cuda.current_device()

        self.faiss_gpu = hasattr(faiss, 'StandardGpuResources')
        if self.faiss_gpu:
            # we need only a StandardGpuResources per GPU
            self.res = faiss.StandardGpuResources()
            # self.res.setTempMemoryFraction(0.1)
            #self.res.setTempMemory(
            #    4 * (1024 * 1024 * 1024)
            #)  # Bytes, the single digit is basically GB)
            self.flat_config = faiss.GpuIndexFlatConfig()
            self.flat_config.device = self.gpu_id


    def build_nn_index(self, database):
        """
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        """
        # index = faiss.GpuIndexFlatL2(self.res, self.dimension, self.flat_config)  # dimension is 3
        index_cpu = faiss.IndexFlatL2(self.dimension)

        if self.faiss_gpu:
            index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, index_cpu)
        else:
            index = index_cpu

        index.add(database)
        return index

    def search_nn(self, index, query, k):
        """
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: Variable of Nxk, type FloatTensor, in GPU
                 I: Variable of Nxk, type LongTensor, in GPU
        """
        D, I = index.search(query, k)

        D_var = torch.from_numpy(np.ascontiguousarray(D))
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))
        if self.gpu_id >= 0:
            D_var = D_var.to(self.device)
            I_var = I_var.to(self.device)

        return D_var, I_var

    def forward(self, predict_pc_6, gt_pc_6, thresh, keep_dim=False, use_normals=True):
        """
        :param predict_pc: Bx3xM Variable in GPU
        :param gt_pc: Bx3xN Variable in GPU
        :return:
        """

        predict_pc = predict_pc_6[:, :3, :]
        gt_pc = gt_pc_6[:, :3, :]
        #
        if use_normals:
            predict_pcn = predict_pc_6[:, 3:, :]
            gt_pcn = gt_pc_6[:, 3:, :]

        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(
            torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxMx3
        gt_pc_np = np.ascontiguousarray(
            torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxNx3

        # selected_gt: Bxkx3xM
        selected_gt_by_predict = torch.FloatTensor(
            predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2]
        )
        # selected_predict: Bxkx3xN
        selected_predict_by_gt = torch.FloatTensor(
            gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2]
        )

        if use_normals:
            # normals
            selected_gt_by_predictn = torch.FloatTensor(
                predict_pc_size[0], self.k, predict_pc_size[1], predict_pc_size[2]
            )
            selected_predict_by_gtn = torch.FloatTensor(
                gt_pc_size[0], self.k, gt_pc_size[1], gt_pc_size[2]
            )

        if self.gpu_id >= 0:
            selected_gt_by_predict = selected_gt_by_predict.to(self.device)
            selected_predict_by_gt = selected_predict_by_gt.to(self.device)
            if use_normals:
                selected_gt_by_predictn = selected_gt_by_predictn.to(self.device)
                selected_predict_by_gtn = selected_predict_by_gtn.to(self.device)

        # process each batch independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_var = self.search_nn(index_gt, predict_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_gt_by_predict[i, k, ...] = gt_pc[i].index_select(
                    1, I_var[:, k]
                )
                if use_normals:
                    selected_gt_by_predictn[i, k, ...] = gt_pcn[i].index_select(
                        1, I_var[:, k]
                    )

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_var = self.search_nn(index_predict, gt_pc_np[i], self.k)

            # process nearest k neighbors
            for k in range(self.k):
                selected_predict_by_gt[i, k, ...] = predict_pc[i].index_select(
                    1, I_var[:, k]
                )
                if use_normals:
                    selected_predict_by_gtn[i, k, ...] = predict_pcn[i].index_select(
                        1, I_var[:, k]
                    )

        # compute loss ===================================================

        # selected_gt(Bxkx3xM) vs predict_pc(Bx3xM)
        forward_loss_element = robust_norm(
            selected_gt_by_predict
            - predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict)
        )
        self.forward_loss = forward_loss_element.mean()
        self.forward_loss_array = forward_loss_element.mean(dim=1).mean(dim=1)

        # selected_predict(Bxkx3xN) vs gt_pc(Bx3xN)
        backward_loss_element = robust_norm(
            selected_predict_by_gt
            - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt)
        )  # BxkxN
        self.backward_loss = backward_loss_element.mean()
        self.backward_loss_array = backward_loss_element.mean(dim=1).mean(dim=1)
        #
        
        if keep_dim:
            self.forward_loss = torch.relu(forward_loss_element - thresh).squeeze(1).mean(dim=1)
            self.backward_loss = torch.relu(backward_loss_element - thresh).squeeze(1).mean(dim=1)
            return self.forward_loss + self.backward_loss
        else:
            self.forward_loss = torch.relu(forward_loss_element.flatten() - thresh).mean()
            self.backward_loss = torch.relu(backward_loss_element.flatten() - thresh).mean()
            return self.forward_loss + self.backward_loss


    def __call__(self, predict_pc, gt_pc, ang_wt, keep_dim=False):
        # start_time = time.time()
        loss = self.forward(predict_pc, gt_pc, ang_wt, keep_dim)
        # print(time.time()w-start_time)
        return loss


def weighted_mae_loss(input, target, weight):
    return torch.sum(weight * (input - target).abs())

closs = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight = torch.tensor(0.2))
celoss = nn.CrossEntropyLoss(reduction='sum')

# 0 -> start
# 1 -> Cuboid
# 2 -> attach
# 3 -> reflect
# 4 -> translate
# 5 -> squeeze
# 6 -> end

# 0 - 7 -> commandIndex
# 7 -> 18 -> cuboid1
# 18 -> 29 -> cuboid2
# 29 -> 40 -> cuboid3
# 40 -> 43 -> l,h,w
# 43 -> 46 -> x1,y1,z1
# 46 -> 49 -> x2,y2,z2
# 49 - 52 -> symaxis
# 52 -> num
# 53 -> scale
# 54 - 60 face
# 60 - 62 - UV
# 62 -> aligned

# BBOX DIMS 40 42

class ProgLoss(nn.Module):
    def forward(self, pred, target, weight):       
        
        if target.shape[1] > pred.shape[1]:
            target = target[:,:pred.shape[1],:]
            weight = weight[:,:pred.shape[1],:]
                        
        commands = torch.argmax(target[:,:,:7], dim = 2).flatten()
        pcommands = torch.argmax(pred[:,:,:7], dim = 2).flatten()
        
        cmdc = (commands == pcommands).sum().item() * 1.0
        
        cmd_loss = celoss(
            pred[:,:,:7].view(-1,7),
            commands
        )
        
        cub_inds = (commands==1).nonzero().flatten()
        ap_inds = (commands==2).nonzero().flatten()
        ref_inds = (commands==3).nonzero().flatten()
        trans_inds = (commands==4).nonzero().flatten()
        sq_inds = (commands==5).nonzero().flatten()

        cub_prm_loss = 0.
        align_loss = 0.
        align_num_pos = 0.
        align_num_neg = 0.
        align_posc = 0.
        align_negc = 0.
        
        if cub_inds.sum() > 0:        
            cub_prm_loss = weighted_mae_loss(
                pred[:,cub_inds,:], target[:,cub_inds,:], weight[:,cub_inds,:]
            )

        if cub_inds.sum() > 1:
            align_pos_inds = cub_inds[1:][(target[:, cub_inds[1:], 62].flatten() == 1.).nonzero().flatten()]
            align_neg_inds = cub_inds[1:][(target[:, cub_inds[1:], 62].flatten() == 0.).nonzero().flatten()]

            align_num_pos += align_pos_inds.shape[0]
            align_num_neg += align_neg_inds.shape[0]

            align_loss = closs(
                pred[:, cub_inds[1:], 62].flatten(),
                target[:, cub_inds[1:], 62].flatten()
            )
            
            align_posc = (pred[:, align_pos_inds, 62] > 0.).sum().float()
            align_negc = (pred[:, align_neg_inds, 62] <= 0.).sum().float()
            
        xyz_prm_loss = 0
        cubc = 0
        ap_cub_loss = 0
        
        if ap_inds.sum() > 0:
            
            xyz_prm_loss = weighted_mae_loss(
                pred[:,ap_inds,:], target[:,ap_inds,:], weight[:,ap_inds,:]
            )
                
            ap_cube_1s = torch.argmax(target[:,ap_inds,7:18], dim=2).flatten()
            ap_cube_2s = torch.argmax(target[:,ap_inds,18:29], dim=2).flatten()

            ap_pcube_1s = torch.argmax(pred[:,ap_inds,7:18], dim=2).flatten()
            ap_pcube_2s = torch.argmax(pred[:,ap_inds,18:29], dim=2).flatten()
        
            ap_cubc = ((ap_pcube_1s == ap_cube_1s) * (ap_pcube_2s == ap_cube_2s)).sum().item() * 1.0

            cubc = ap_cubc
        
            ap_cub_loss = celoss(
                pred[:,ap_inds,7:18].view(-1, 11),
                ap_cube_1s
            )
        
            ap_cub_loss += celoss(
                pred[:,ap_inds,18:29].view(-1, 11),
                ap_cube_2s
            )
        
        cub_loss = ap_cub_loss

        axisc = 0.
        axis_loss = 0.
        sym_cubc = 0.
        sym_cub_loss = 0.
        sym_prm_loss = 0.
        
        if ref_inds.sum() > 0:
            ref_cube_1s = torch.argmax(target[:,ref_inds,7:18], dim=2).flatten()
            ref_pcube_1s = torch.argmax(pred[:,ref_inds,7:18], dim=2).flatten()

            ref_cubc = ((ref_pcube_1s == ref_cube_1s)).sum().item() * 1.0

            ref_axis = torch.argmax(target[:, ref_inds, 49:52], dim=2).flatten()
            ref_paxis = torch.argmax(pred[:, ref_inds, 49:52], dim=2).flatten()
            ref_axisc = ((ref_paxis == ref_axis)).sum().item() * 1.0
            
            ref_axis_loss = celoss(
                pred[:,ref_inds,49:52].view(-1, 3),
                ref_axis
            )
            
            ref_cub_loss = celoss(
                pred[:,ref_inds,7:18].view(-1, 11),
                ref_cube_1s
            )

            axisc += ref_axisc
            axis_loss += ref_axis_loss
            sym_cubc += ref_cubc
            sym_cub_loss += ref_cub_loss
            
        if trans_inds.sum() > 0:

            trans_cube_1s = torch.argmax(target[:,trans_inds,7:18], dim=2).flatten()  
            trans_pcube_1s = torch.argmax(pred[:,trans_inds,7:18], dim=2).flatten()

            trans_cubc = ((trans_pcube_1s == trans_cube_1s)).sum().item() * 1.0
            
            trans_axis = torch.argmax(target[:, trans_inds, 49:52], dim=2).flatten()
            trans_paxis = torch.argmax(pred[:, trans_inds, 49:52], dim=2).flatten()
            trans_axisc = ((trans_paxis == trans_axis)).sum().item() * 1.0
            trans_axis_loss = celoss(
                pred[:,trans_inds,49:52].view(-1, 3),
                trans_axis
            )

            trans_cub_loss = celoss(
                pred[:,trans_inds,7:18].view(-1, 11),
                trans_cube_1s
            )

            trans_prm_loss = weighted_mae_loss(
                pred[:,trans_inds,:], target[:,trans_inds,:], weight[:,trans_inds,:]
            )

            axisc += trans_axisc
            axis_loss += trans_axis_loss
            sym_cubc += trans_cubc
            sym_cub_loss += trans_cub_loss
            sym_prm_loss += trans_prm_loss

        uv_prm_loss = 0
        sq_cubc = 0
        sq_cub_loss = 0
        face_loss = 0
        facec = 0
        
        if sq_inds.sum() > 0:
            
            uv_prm_loss = weighted_mae_loss(
                pred[:,sq_inds,:],
                target[:,sq_inds,:],
                weight[:,sq_inds,:]
            )
                
            sq_cube_1s = torch.argmax(target[:,sq_inds,7:18], dim=2).flatten()
            sq_cube_2s = torch.argmax(target[:,sq_inds,18:29], dim=2).flatten()
            sq_cube_3s = torch.argmax(target[:,sq_inds,29:40], dim=2).flatten()
            faces = torch.argmax(target[:, sq_inds, 54:60], dim=2).flatten()

            sq_pcube_1s = torch.argmax(pred[:,sq_inds,7:18], dim=2).flatten()
            sq_pcube_2s = torch.argmax(pred[:,sq_inds,18:29], dim=2).flatten()
            sq_pcube_3s = torch.argmax(pred[:,sq_inds,29:40], dim=2).flatten()
            pfaces = torch.argmax(pred[:,sq_inds,54:60], dim=2).flatten()
            
            sq_cubc = (
                (sq_pcube_1s == sq_cube_1s) * \
                (sq_pcube_2s == sq_cube_2s) * \
                (sq_pcube_3s == sq_cube_3s)
            ).sum().item() * 1.0

            facec = (faces == pfaces).sum().item() * 1.0
        
            sq_cub_loss = celoss(
                pred[:,sq_inds,7:18].view(-1, 11),
                sq_cube_1s
            )
        
            sq_cub_loss += celoss(
                pred[:,sq_inds,18:29].view(-1, 11),
                sq_cube_2s
            )

            sq_cub_loss += celoss(
                pred[:,sq_inds,29:40].view(-1, 11),
                sq_cube_3s
            )

            face_loss = celoss(
                pred[:,sq_inds,54:60].view(-1, 6),
                faces
            )
            
        losses = {
            'cmd': cmd_loss,
            'cub_prm': cub_prm_loss,
            'xyz_prm': xyz_prm_loss,
            'uv_prm': uv_prm_loss,
            'sym_prm': sym_prm_loss,#
            'cub': cub_loss,
            'sq_cub': sq_cub_loss,
            'sym_cub': sym_cub_loss,#
            'axis': axis_loss,#
            'face': face_loss,
            'cmdc': cmdc,
            'cubc': cubc,
            'sq_cubc': sq_cubc,
            'sym_cubc': sym_cubc,#
            'axisc': axisc,#
            'facec': facec,
            'align': align_loss,
            'palignc': align_posc,            
            'nalignc': align_negc,
            'nan': align_num_neg,
            'nap': align_num_pos,
            'na': ap_inds.shape[0] * 1.0,
            'nc': cub_inds.shape[0] * 1.0,
            'ns': (ref_inds.shape[0] * 1.0) + (trans_inds.shape[0] * 1.0),
            'nsq': sq_inds.shape[0] * 1.0
        }
        return losses

