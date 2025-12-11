import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from model_util import PointConv, PointConvD, UpsampleFlow, PointWarping, FlowEmbeddingLayer as Embedding, \
    BidirectionalLayerFeatCosine as PCFeatCosine
from model_util import CrossLayerLightFeatCosine as CrossLayer, SceneFlowEstimatorResidual as RCEstimator
from model_util import index_points_group, Conv1d, square_distance, knn_point_cosine, knn_point
from model_util import default, extract, cosine_beta_schedule, SinusoidalPosEmb

import time
from tqdm.auto import tqdm

scale = 1.0


def transform(PCs, R, T):
    newPCs = []
    for i in range(len(PCs)):
        newPCs.append((torch.bmm(R, PCs[i]) + T.unsqueeze(-1)))
    return newPCs


def quat2mat(q):
    '''
    :param q: Bx4
    :return: R: BX3X3
    '''
    batch_size = q.shape[0]
    w, x, y, z = q[:, 0].unsqueeze(1), q[:, 1].unsqueeze(1), q[:, 2].unsqueeze(1), q[:, 3].unsqueeze(1)
    Nq = torch.sum(q ** 2, dim=1, keepdim=True)
    s = 2.0 / Nq
    wX = w * x * s;
    wY = w * y * s;
    wZ = w * z * s
    xX = x * x * s;
    xY = x * y * s;
    xZ = x * z * s
    yY = y * y * s;
    yZ = y * z * s;
    zZ = z * z * s
    a1 = 1.0 - (yY + zZ);
    a2 = xY - wZ;
    a3 = xZ + wY
    a4 = xY + wZ;
    a5 = 1.0 - (xX + zZ);
    a6 = yZ - wX
    a7 = xZ - wY;
    a8 = yZ + wX;
    a9 = 1.0 - (xX + yY)
    R = torch.cat([a1, a2, a3, a4, a5, a6, a7, a8, a9], dim=1).view(batch_size, 3, 3)
    return R


class PointConvEncoder(nn.Module):
    def __init__(self, weightnet=8):
        super(PointConvEncoder, self).__init__()
        feat_nei = 32
        feat_list = [32, 64, 128, 256, 512]
        self.level0_lift = Conv1d(3, feat_list[0])
        self.level0 = PointConv(feat_nei, feat_list[0] + 3, feat_list[0], weightnet=weightnet)
        self.level0_1 = Conv1d(feat_list[0], feat_list[1])

        self.level1 = PointConvD(2048, feat_nei, feat_list[1] + 3, feat_list[1], weightnet=weightnet)
        self.level1_0 = Conv1d(feat_list[1], feat_list[1])
        self.level1_1 = Conv1d(feat_list[1], feat_list[2])

        self.level2 = PointConvD(512, feat_nei, feat_list[2] + 3, feat_list[2], weightnet=weightnet)
        self.level2_0 = Conv1d(feat_list[2], feat_list[2])
        self.level2_1 = Conv1d(feat_list[2], feat_list[3])

        self.level3 = PointConvD(256, feat_nei, feat_list[3] + 3, feat_list[3], weightnet=weightnet)
        self.level3_0 = Conv1d(feat_list[3], feat_list[3])
        self.level3_1 = Conv1d(feat_list[3], feat_list[4])

        self.level4 = PointConvD(64, feat_nei, feat_list[4] + 3, feat_list[4], weightnet=weightnet)

    def forward(self, xyz, color):
        feat_l0 = self.level0_lift(color)
        feat_l0 = self.level0(xyz, feat_l0)
        feat_l0_1 = self.level0_1(feat_l0)

        # l1
        pc_l1, feat_l1, fps_l1 = self.level1(xyz, feat_l0_1)
        feat_l1 = self.level1_0(feat_l1)
        feat_l1_2 = self.level1_1(feat_l1)

        # l2
        pc_l2, feat_l2, fps_l2 = self.level2(pc_l1, feat_l1_2)
        feat_l2 = self.level2_0(feat_l2)
        feat_l2_3 = self.level2_1(feat_l2)

        # l3
        pc_l3, feat_l3, fps_l3 = self.level3(pc_l2, feat_l2_3)
        feat_l3 = self.level3_0(feat_l3)
        feat_l3_4 = self.level3_1(feat_l3)

        # l4
        pc_l4, feat_l4, fps_l4 = self.level4(pc_l3, feat_l3_4)

        return [xyz, pc_l1, pc_l2, pc_l3, pc_l4], [feat_l0, feat_l1, feat_l2, feat_l3, feat_l4], [fps_l1, fps_l2,
                                                                                                  fps_l3, fps_l4]


class RecurrentUnit(nn.Module):
    def __init__(self, feat_ch, feat_new_ch, latent_ch, cross_mlp1, cross_mlp2, weightnet=8, flow_channels=[64, 64],
                 flow_mlp=[64, 64]):
        super(RecurrentUnit, self).__init__()
        nei = 32
        neighbors = 9

        self.bid = PCFeatCosine(nei, feat_new_ch + feat_ch, cross_mlp1)
        self.fe = Embedding(nei, cross_mlp1[-1], cross_mlp2)
        self.flow = DiffusionRelativePC(neighbors, in_channel=cross_mlp2[-1] + feat_ch, latent_channel=latent_ch,
                                        mlp=flow_channels, channels=flow_channels)

    def forward(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat, gt_flow=None):

        c_feat1 = torch.cat([feat1, feat1_new], dim=1)
        c_feat2 = torch.cat([feat2, feat2_new], dim=1)

        feat1_new, feat2_new = self.bid(pc1, pc2, c_feat1, c_feat2, feat1, feat2)
        fe = self.fe(pc1, pc2, feat1_new, feat2_new, feat1, feat2)
        new_feat1 = torch.cat([feat1, fe], dim=1)

        if self.training:
            feat_flow, flow, loss = self.flow(pc1, pc1, up_feat, new_feat1, up_flow, gt_flow)
            return flow, feat1_new, feat2_new, feat_flow, loss
        else:
            feat_flow, flow = self.flow(pc1, pc1, up_feat, new_feat1, up_flow, gt_flow)
            return flow, feat1_new, feat2_new, feat_flow


class DiffusionRelativePC(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn=False, use_leaky=True, \
                 return_inter=False, radius=None, use_relu=False, channels=[64, 64], clamp=[-200, 200], scale_dif=1.0):
        super(DiffusionRelativePC, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.use_relu = use_relu
        self.fc = nn.Conv1d(channels[-1], 3, 1)
        self.clamp = clamp

        last_channel = in_channel + 3 + 64 + 3

        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

        # build diffusion
        timesteps = 300
        sampling_timesteps = 1
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps=timesteps).float()
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.01
        self.scale = scale_dif

        time_dim = 64
        dim = 16
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        # q(x_t | x_{t-1})
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
        # q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.iters = 1

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.scale * torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def forward(self, xyz1, xyz2, points1, points2, flow, flow_gt):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        batch_size = flow.shape[0]
        n = flow.shape[2]

        if self.training:
            flow_gt = flow_gt

            def z_score_normalize(data):
                mean = np.mean(data, axis=1)
                std_dev = np.std(data, axis=1)
                normalized_data = (data - mean) / std_dev
                return normalized_data

            gt_delta_flow = flow_gt
            gt_delta_flow = torch.where(torch.isinf(gt_delta_flow), torch.zeros_like(gt_delta_flow), gt_delta_flow)
            gt_delta_flow = gt_delta_flow.detach()

            t = torch.randint(0, self.timesteps, (batch_size,), device=flow.device).long()
            noise = (self.scale * torch.randn_like(gt_delta_flow)).float()

            delta_flow = self.q_sample(x_start=gt_delta_flow, t=t, noise=noise)

            for i in range(self.iters):
                delta_flow = delta_flow.detach()
                time = self.time_mlp(t)

                time = time.unsqueeze(1).repeat(1, n, 1)

                if self.radius is None:
                    sqrdists = square_distance(xyz1, xyz2)
                    dists, knn_idx = torch.topk(sqrdists, self.nsample, dim=-1, largest=False, sorted=False)
                    neighbor_xyz = index_points_group(xyz2, knn_idx)
                    direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

                    grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx)
                    time = time.unsqueeze(-2).repeat(1, 1, self.nsample, 1)
                    delta_flow = delta_flow.permute(0, 2, 1)
                    delta_flow = delta_flow.unsqueeze(-2).repeat(1, 1, self.nsample, 1)

                    new_points = torch.cat([grouped_points2, direction_xyz, delta_flow, time], dim=-1)

                    new_points = new_points.permute(0, 3, 2, 1)

                else:
                    new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
                    new_points = new_points.permute(0, 1, 3, 2)

                point1_graph = points1

                # r
                r = new_points
                for i, conv in enumerate(self.mlp_r_convs):
                    r = conv(r)
                    if i == 0:
                        grouped_points1 = self.fuse_r(point1_graph)
                        r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                    if self.bn:
                        r = self.mlp_r_bns[i](r)
                    if i == len(self.mlp_r_convs) - 1:
                        r = self.sigmoid(r)
                    else:
                        r = self.relu(r)

                # z
                z = new_points
                for i, conv in enumerate(self.mlp_z_convs):
                    z = conv(z)
                    if i == 0:
                        grouped_points1 = self.fuse_z(point1_graph)
                        z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                    if self.bn:
                        z = self.mlp_z_bns[i](z)
                    if i == len(self.mlp_z_convs) - 1:
                        z = self.sigmoid(z)
                    else:
                        z = self.relu(z)

                    if i == len(self.mlp_z_convs) - 2:
                        z = torch.max(z, -2)[0].unsqueeze(-2)

                z = z.squeeze(-2)

                point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
                point1_expand = r * point1_graph_expand
                point1_expand = self.fuse_r_o(point1_expand)

                h = new_points
                for i, conv in enumerate(self.mlp_h_convs):
                    h = conv(h)
                    if i == 0:
                        h = h + point1_expand
                    if self.bn:
                        h = self.mlp_h_bns[i](h)
                    if i == len(self.mlp_h_convs) - 1:

                        if self.use_relu:
                            h = self.relu(h)
                        else:
                            h = self.tanh(h)
                    else:
                        h = self.relu(h)
                    if i == len(self.mlp_h_convs) - 2:
                        h = torch.max(h, -2)[0].unsqueeze(-2)

                h = h.squeeze(-2)

                new_points = (1 - z) * points1 + z * h

                if self.mlp2:
                    for _, conv in enumerate(self.mlp2):
                        new_points = conv(new_points)

                update = self.fc(new_points)
                delta_flow = update[:, :3, :].clamp(self.clamp[0], self.clamp[1])

                loss_df = F.mse_loss(delta_flow, gt_delta_flow)

            return new_points, delta_flow, loss_df

        else:
            batch, device, total_timesteps, sampling_timesteps, eta = batch_size, flow.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta

            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
            img = (self.scale * torch.randn_like(flow)).float()

            for time, time_next in time_pairs:
                t = torch.full((batch,), time, device=device, dtype=torch.long)
                delta_flow = img

                for i in range(self.iters):
                    delta_flow = delta_flow.detach()

                    time = self.time_mlp(t)
                    time = time.unsqueeze(1).repeat(1, n, 1)

                    if self.radius is None:
                        sqrdists = square_distance(xyz1, xyz2)
                        dists, knn_idx = torch.topk(sqrdists, self.nsample, dim=-1, largest=False, sorted=False)
                        neighbor_xyz = index_points_group(xyz2, knn_idx)
                        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

                        grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx)
                        time = time.unsqueeze(-2).repeat(1, 1, self.nsample, 1)
                        delta_flow = delta_flow.permute(0, 2, 1)
                        delta_flow = delta_flow.unsqueeze(-2).repeat(1, 1, self.nsample, 1)

                        new_points = torch.cat([grouped_points2, direction_xyz, delta_flow, time], dim=-1)
                        new_points = new_points.permute(0, 3, 2, 1)

                    else:
                        new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
                        new_points = new_points.permute(0, 1, 3, 2)

                    point1_graph = points1

                    # r
                    r = new_points
                    for i, conv in enumerate(self.mlp_r_convs):
                        r = conv(r)
                        if i == 0:
                            grouped_points1 = self.fuse_r(point1_graph)
                            r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample,
                                                                                                   1)
                        if self.bn:
                            r = self.mlp_r_bns[i](r)
                        if i == len(self.mlp_r_convs) - 1:
                            r = self.sigmoid(r)
                        else:
                            r = self.relu(r)

                    # z
                    z = new_points
                    for i, conv in enumerate(self.mlp_z_convs):
                        z = conv(z)
                        if i == 0:
                            grouped_points1 = self.fuse_z(point1_graph)
                            z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample,
                                                                                                   1)
                        if self.bn:
                            z = self.mlp_z_bns[i](z)
                        if i == len(self.mlp_z_convs) - 1:
                            z = self.sigmoid(z)
                        else:
                            z = self.relu(z)

                        if i == len(self.mlp_z_convs) - 2:
                            z = torch.max(z, -2)[0].unsqueeze(-2)

                    z = z.squeeze(-2)

                    point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample,
                                                                                                   1)
                    point1_expand = r * point1_graph_expand
                    point1_expand = self.fuse_r_o(point1_expand)

                    h = new_points
                    for i, conv in enumerate(self.mlp_h_convs):
                        h = conv(h)
                        if i == 0:
                            h = h + point1_expand
                        if self.bn:
                            h = self.mlp_h_bns[i](h)
                        if i == len(self.mlp_h_convs) - 1:
                            if self.use_relu:
                                h = self.relu(h)
                            else:
                                h = self.tanh(h)
                        else:
                            h = self.relu(h)
                        if i == len(self.mlp_h_convs) - 2:
                            h = torch.max(h, -2)[0].unsqueeze(-2)

                    h = h.squeeze(-2)

                    new_points = (1 - z) * points1 + z * h

                    if self.mlp2:
                        for _, conv in enumerate(self.mlp2):
                            new_points = conv(new_points)

                    update = self.fc(new_points)
                    delta_flow = update[:, :3, :].clamp(self.clamp[0], self.clamp[1])

                pred_noise = self.predict_noise_from_start(img, t, delta_flow)

                if time_next < 0:
                    delta_flow = delta_flow
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = (self.scale * torch.randn_like(flow)).float()

                img = delta_flow * alpha_next.sqrt() + c * pred_noise + sigma * noise
            return new_points, delta_flow


class GWeightedSVD(nn.Module):

    def __init__(self):
        super(GWeightedSVD, self).__init__()

    def forward(self, src, rcs, weights):
        eps = 1e-4
        B, _, N = src.shape
        src_corres = src + rcs

        sum_weights = torch.sum(weights, dim=1, keepdim=True) + eps
        norm_weights = weights / sum_weights

        weights_expanded = norm_weights.unsqueeze(1)
        src_mean = torch.sum(src * weights_expanded, dim=2, keepdim=True) / (
                torch.sum(weights_expanded, dim=2, keepdim=True) + eps)
        src_corres_mean = torch.sum(src_corres * weights_expanded, dim=2, keepdim=True) / (
                torch.sum(weights_expanded, dim=2, keepdim=True) + eps)

        src_centered = src - src_mean
        src_corres_centered = src_corres - src_corres_mean

        weight_matrix = torch.diag_embed(norm_weights)
        cov_mat = torch.matmul(src_centered, torch.matmul(weight_matrix, src_corres_centered.transpose(1, 2)))

        try:
            u, s, vh = torch.linalg.svd(cov_mat, full_matrices=False)
        except Exception as e:
            r = torch.eye(3, device=src.device).repeat(B, 1, 1)
            t = torch.zeros(B, 3, device=src.device)
            return r, t

        v = vh.transpose(1, 2)  # [B, 3, 3] (Vh is V^T in torch.linalg.svd, transpose to get V)
        tm_determinant = torch.det(torch.matmul(v, u.transpose(1, 2)))
        det_matrix = torch.diag_embed(
            torch.cat((torch.ones(B, 2, device=src.device), tm_determinant.unsqueeze(1)), dim=1))
        r = torch.matmul(v, torch.matmul(det_matrix, u.transpose(1, 2)))

        t = src_corres_mean.squeeze(-1) - torch.einsum('bij,bj->bi', r, src_mean.squeeze(-1))

        return r, t


class WeightPred(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=32):
        super(WeightPred, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(hidden_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=False), nn.BatchNorm1d(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        x = self.conv1(features)
        x = self.conv2(x)
        weights = self.sigmoid(x.squeeze(1))
        return weights


class RCP_LO(nn.Module):
    def __init__(self):
        super(RCP_LO, self).__init__()

        self.scale = scale
        nei = 32
        weightnet = 8
        feat_list = [32, 64, 128, 256, 512]

        self.encoder = PointConvEncoder(weightnet=weightnet)
        # l0:
        self.refine0 = RecurrentUnit(feat_ch=feat_list[0], feat_new_ch=feat_list[0], latent_ch=64,
                                     cross_mlp1=[feat_list[0], feat_list[0]], cross_mlp2=[feat_list[0], feat_list[0]],
                                     weightnet=weightnet, flow_channels=[64, 64], flow_mlp=[64, 64])
        # l1:
        self.refine1 = RecurrentUnit(feat_ch=feat_list[1], feat_new_ch=feat_list[1], latent_ch=64,
                                     cross_mlp1=[feat_list[1], feat_list[1]], cross_mlp2=[feat_list[1], feat_list[1]],
                                     weightnet=weightnet)
        # l2:
        self.refine2 = RecurrentUnit(feat_ch=feat_list[2], feat_new_ch=feat_list[2], latent_ch=64,
                                     cross_mlp1=[feat_list[2], feat_list[2]], cross_mlp2=[feat_list[2], feat_list[2]],
                                     weightnet=weightnet)
        # l3:
        self.cross3 = CrossLayer(nei, feat_list[4], [feat_list[4], feat_list[4]], [feat_list[3], feat_list[3]])
        self.RCE3 = RCEstimator(feat_list[3], feat_list[3], channels=[feat_list[3], 64], mlp=[], weightnet=weightnet)

        # deconv
        self.deconv4_3 = Conv1d(feat_list[4], feat_list[3])
        self.deconv3_2 = Conv1d(feat_list[3], feat_list[2])
        self.deconv2_1 = Conv1d(feat_list[2], feat_list[1])
        self.deconv1_0 = Conv1d(feat_list[1], feat_list[0])

        self.weight3 = WeightPred()
        self.weight2 = WeightPred()
        self.weight1 = WeightPred()
        self.weight0 = WeightPred()
        self.svd = GWeightedSVD()

        # upsample
        self.upsample = UpsampleFlow()
        self.loss = multiScaleLoss_svd()

        # loss
        self.w_x = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.w_q = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True)
        self.w_f = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.w_r = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)

    def forward(self, xyz1, xyz2, color1, color2, qq_gt, t_gt):
        # xyz1, xyz2: B, N, 3

        # l0
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1)  # B 3 N
        color2 = color2.permute(0, 2, 1)

        pc1s, feat1s, idx1s = self.encoder(xyz1, color1)
        pc2s, feat2s, idx2s = self.encoder(xyz2, color2)

        if self.training:
            R_gt = quat2mat(qq_gt)
            t_gt = torch.squeeze(t_gt, dim=2)

            T_gt = torch.zeros(R_gt.shape[0], 4, 4).cuda()
            T_gt[:, :3, :3] = R_gt
            T_gt[:, :3, 3] = t_gt
            T_gt[:, 3, 3] = 1.0

            pct_all = transform([pc1s[0], pc1s[1], pc1s[2], pc1s[3]], R_gt, t_gt)
            gt_rcs = [p - c for p, c in zip(pct_all, pc1s[:4])]

        # l4
        feat1_l4_3 = self.upsample(pc1s[3], pc1s[4], feat1s[4])
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)
        feat2_l4_3 = self.upsample(pc2s[3], pc2s[4], feat2s[4])
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        # l3
        c_feat1_l3 = torch.cat([feat1s[3], feat1_l4_3], dim=1)
        c_feat2_l3 = torch.cat([feat2s[3], feat2_l4_3], dim=1)

        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1s[3], pc2s[3], c_feat1_l3, c_feat2_l3, feat1s[3], feat2s[3])
        feat3, RCs3, _ = self.RCE3(pc1s[3], feat1s[3], cross3)

        w3 = self.weight3(feat3)
        R3, t3 = self.svd(pc1s[3], RCs3, w3)
        pc1_all = transform([pc1s[0], pc1s[1], pc1s[2], pc1s[3]], R3, t3)
        pc1s[0], pc1s[1], pc1s[2], pc1s[3] = pc1_all

        T3 = torch.zeros(R3.shape[0], 4, 4).cuda()
        T3[:, :3, :3] = R3
        T3[:, :3, 3] = t3
        T3[:, 3, 3] = 1.0

        feat1_l3_2 = self.upsample(pc1s[2], pc1s[3], feat1_new_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)
        feat2_l3_2 = self.upsample(pc2s[2], pc2s[3], feat2_new_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        # l2
        up_RCs2 = self.upsample(pc1s[2], pc1s[3], self.scale * RCs3)
        up_feat2 = self.upsample(pc1s[2], pc1s[3], feat3)
        if self.training:
            l2_rc_gt = pct_all[2] - pc1s[2]
            RCs2, feat1_new_l2, feat2_new_l2, feat2, loss_l2 = self.refine2(pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2,
                                                                            feat1s[2], feat2s[2], up_RCs2, up_feat2,
                                                                            l2_rc_gt)
        else:
            l2_rc_gt = None
            RCs2, feat1_new_l2, feat2_new_l2, feat2 = self.refine2(pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2, feat1s[2],
                                                                   feat2s[2], up_RCs2, up_feat2, l2_rc_gt)

        w2 = self.weight2(feat2)
        delta_R2, delta_t2 = self.svd(pc1s[2], RCs2, w2)
        pc1_all = transform([pc1s[0], pc1s[1], pc1s[2]], delta_R2, delta_t2)
        pc1s[0], pc1s[1], pc1s[2] = pc1_all

        T2_ = torch.zeros(delta_R2.shape[0], 4, 4).cuda()
        T2_[:, :3, :3] = delta_R2
        T2_[:, :3, 3] = delta_t2
        T2_[:, 3, 3] = 1.0
        T2 = torch.matmul(T2_, T3)

        feat1_l2_1 = self.upsample(pc1s[1], pc1s[2], feat1_new_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)
        feat2_l2_1 = self.upsample(pc2s[1], pc2s[2], feat2_new_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        # l1
        up_RCs1 = self.upsample(pc1s[1], pc1s[2], self.scale * RCs2)
        up_feat1 = self.upsample(pc1s[1], pc1s[2], feat2)

        if self.training:
            l1_rc_gt = pct_all[1] - pc1s[1]
            RCs1, feat1_new_l1, feat2_new_l1, feat1, loss_l1 = self.refine1(pc1s[1], pc2s[1], feat1_l2_1, feat2_l2_1,
                                                                            feat1s[1], feat2s[1], up_RCs1, up_feat1,
                                                                            l1_rc_gt)
        else:
            l1_rc_gt = None
            RCs1, feat1_new_l1, feat2_new_l1, feat1 = self.refine1(pc1s[1], pc2s[1], feat1_l2_1, feat2_l2_1, feat1s[1],
                                                                   feat2s[1], up_RCs1, up_feat1, l1_rc_gt)

        w1 = self.weight1(feat1)
        delta_R1, delta_t1 = self.svd(pc1s[1], RCs1, w1)
        pc1_all = transform([pc1s[0], pc1s[1]], delta_R1, delta_t1)
        pc1s[0], pc1s[1] = pc1_all

        T1_ = torch.zeros(delta_R1.shape[0], 4, 4).cuda()
        T1_[:, :3, :3] = delta_R1
        T1_[:, :3, 3] = delta_t1
        T1_[:, 3, 3] = 1.0
        T1 = torch.matmul(T1_, T2)

        feat1_l1_0 = self.upsample(pc1s[0], pc1s[1], feat1_new_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)
        feat2_l1_0 = self.upsample(pc2s[0], pc2s[1], feat2_new_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        # l0
        up_RCs0 = self.upsample(pc1s[0], pc1s[1], self.scale * RCs1)
        up_feat0 = self.upsample(pc1s[0], pc1s[1], feat1)

        if self.training:
            l0_rc_gt = pct_all[0] - pc1s[0]
            RCs0, feat1_new_l0, feat2_new_l0, feat0, loss_l0 = self.refine0(pc1s[0], pc2s[0], feat1_l1_0, feat2_l1_0,
                                                                            feat1s[0], feat2s[0], up_RCs0, up_feat0,
                                                                            l0_rc_gt)
        else:
            l0_rc_gt = None
            RCs0, feat1_new_l0, feat2_new_l0, feat0 = self.refine0(pc1s[0], pc2s[0], feat1_l1_0, feat2_l1_0, feat1s[0],
                                                                   feat2s[0], up_RCs0, up_feat0, l0_rc_gt)

        w0 = self.weight0(feat0)
        confi0 = Gen_confinence(pc1s[0], RCs0)
        w0 = w0 * confi0
        delta_R0, delta_t0 = self.svd(pc1s[0], RCs0, w0)
        # pc1_all = transform([pc1s[0]],  delta_R0,delta_t0 )
        # pc1s[0] = pc1_all

        T0_ = torch.zeros(delta_R0.shape[0], 4, 4).cuda()
        T0_[:, :3, :3] = delta_R0
        T0_[:, :3, 3] = delta_t0
        T0_[:, 3, 3] = 1.0
        T0 = torch.matmul(T0_, T1)

        if self.training:
            all_RCs = [RCs0, RCs1, RCs2, RCs3]
            all_gt_rcs = [l0_rc_gt, l1_rc_gt, l2_rc_gt, gt_rcs[3]]
            T_pred = [T0, T1, T2, T3]
            loss = self.loss(T_pred, T_gt, all_RCs, all_gt_rcs, loss_f2=loss_l2, loss_f1=loss_l1, loss_f0=loss_l0,
                             w_x=self.w_x, w_q=self.w_q, w_f=self.w_f, w_r=self.w_r)
            return T0, loss
        else:
            return T0


def Gen_confinence(PC, flow):
    src_keypts = PC.permute(0, 2, 1)
    tgt_keypts = PC.permute(0, 2, 1) + flow.permute(0, 2, 1)

    src_dist = torch.norm((src_keypts[:, :, None, :] - src_keypts[:, None, :, :]), dim=-1)
    target_dist = torch.norm((tgt_keypts[:, :, None, :] - tgt_keypts[:, None, :, :]), dim=-1)
    cross_dist = torch.abs(src_dist - target_dist)

    SC_dist_thre = 0.05
    SC_measure = torch.clamp(1.0 - cross_dist ** 2 / SC_dist_thre ** 2, min=0)
    SC_measure = (cross_dist < SC_dist_thre).float()

    confidence = cal_leading_eigenvector(SC_measure, method='power')  # b,n

    return confidence


def cal_leading_eigenvector(M, method='power'):
    """
    Calculate the leading eigenvector using power iteration algorithm or torch.symeig
    Input:
        - M:      [bs, num_corr, num_corr] the compatibility matrix
        - method: select different method for calculating the learding eigenvector.
    Output:
        - solution: [bs, num_corr] leading eigenvector
    """
    if method == 'power':

        leading_eig = torch.ones_like(M[:, :, 0:1])
        leading_eig_last = leading_eig
        for i in range(20):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
        return leading_eig
    elif method == 'eig':
        e, v = torch.symeig(M, eigenvectors=True)
        leading_eig = v[:, :, -1]
        return leading_eig
    else:
        exit(-1)


class multiScaleLoss_svd(nn.Module):
    def __init__(self):
        super(multiScaleLoss_svd, self).__init__()

    def forward(self, pred_T, T_gt, pred_flows, gt_flow, loss_f2=None, loss_f1=None, loss_f0=None, w_x=0, w_q=0, w_f=0,
                w_r=0, alpha=[1.6, 0.8, 0.4, 0.2]):

        device = T_gt.device
        R_gt = T_gt[:, :3, :3]
        t_gt = T_gt[:, :3, 3]
        Identity = torch.eye(3, device=device).expand(T_gt.size(0), 3, 3)

        loss1 = 0.0
        loss2 = 0.0
        for i in range(len(pred_T)):
            T_pred = pred_T[i]
            R_pred = T_pred[:, :3, :3]
            t_pred = T_pred[:, :3, 3]

            resi_R = torch.norm(R_pred.transpose(2, 1) @ R_gt - Identity, dim=(1, 2)) / torch.sqrt(
                torch.tensor(3.0, device=device))
            resi_t = torch.norm(t_pred - t_gt, dim=1)
            loss1 += alpha[i] * (
                    torch.mean(resi_t) * torch.exp(-w_x) + w_x + torch.mean(resi_R) * torch.exp(-w_q) + w_q)

        resrc_list = [loss_f0, loss_f1, loss_f2]

        for i, lf in enumerate(resrc_list):
            if lf is not None:
                loss2 += 1e-5 * alpha[i] * lf.mean()

        return loss2 + loss1


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    import torch
    from thop import profile, clever_format

    pc = torch.randn((1, 8192, 3)).float().cuda()
    q = torch.randn((1, 4, 1)).float().cuda()
    t = torch.randn((1, 3, 1)).float().cuda()
    model = RCP_LO().cuda()

    macs, params = profile(model, inputs=(pc, pc, pc, pc, q, t))
    macs, params = clever_format([macs, params], "%.3f")
    # print(macs, params)
    total = sum([param.nelement() for param in model.parameters()])  
    # print("Number of parameter: %.2fM" % (total/1e6))
