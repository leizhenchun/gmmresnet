import torch
import torchinfo
import torchsummary
from torch import nn, Tensor
from torch.nn import BatchNorm1d
from torch.nn import functional as F

from asvspoof19.as19_experiment import get_parameter
from asvspoof21.as21_msgmm_experiment import AS21MultiScaleGMMExperiment, AS21MultiScaleGMM2PathExperiment
from model.gmm import GMMLayer, GMM


def exp_parameters():
    exp_param = get_parameter()
    exp_param['asvspoof_root_path'] = '/home/lzc/lzc/ASVspoof/'

    exp_param['batch_size'] = 32
    exp_param['batch_size_test'] = 128

    exp_param['feature_size'] = 60
    exp_param['feature_num'] = 400
    exp_param['feature_num_test'] = 400
    exp_param['feature_file_extension'] = '.h5'

    exp_param['feature_keep_in_memory'] = True
    exp_param['feature_keep_in_memory_debug'] = False

    exp_param['gmm_index_trans'] = False
    exp_param['gmm_regroup_num'] = 1
    exp_param['gmm_shuffle'] = False

    exp_param['gmmlayer_index_trans'] = False
    exp_param['gmmlayer_regroup_num'] = 1
    exp_param['gmmlayer_shuffle'] = False

    # exp_param['gmm_size'] = 512
    # exp_param['groups'] = 1
    exp_param['weight_decay'] = 0.0

    exp_param['num_epochs'] = 100

    exp_param['lr'] = 0.0001
    exp_param['min_lr'] = 1e-8
    exp_param['use_regularization_loss'] = False
    exp_param['use_scheduler'] = True

    exp_param['test_train2019'] = False
    exp_param['test_dev2019'] = True
    exp_param['test_eval2019'] = True
    exp_param['evaluate_asvspoof2021'] = True
    exp_param['evaluate_asvspoof2021_df'] = True

    exp_param['test_data_basic'] = True
    exp_param['test_data_ufm'] = False
    exp_param['test_data_adaptive'] = False

    return exp_param


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.concatenate([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, first=False, use_max_pool=False) -> None:
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, first=False, use_max_pool=False):
        super(ResidualBlock, self).__init__()
        self.first = first

        # if not self.first:
        #     self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        # self.bn1 = nn.LayerNorm([400, ])

        # self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=groups,
                               bias=False)

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=groups,
                               bias=False)

        if in_channels != out_channels:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             padding=0,
                                             kernel_size=1,
                                             stride=1,
                                             groups=groups,
                                             bias=False)

        else:
            self.downsample = False

        self.use_max_pool = use_max_pool
        if self.use_max_pool:
            self.mp = nn.MaxPool1d(2)

    def forward(self, x):
        identity = x

        # if not self.first:
        #     out = self.bn1(x)
        #     out = self.relu(out)
        # else:
        #     out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity

        if self.use_max_pool:
            out = self.mp(out)

        return out


class FusionBlock(nn.Module):
    def __init__(self, ) -> None:
        super(FusionBlock, self).__init__()

        self.latlayer64 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.latlayer128 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.latlayer256 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)

        # self.resblock64 = ResidualBlock(in_channels=64, out_channels=64)
        # self.resblock128 = ResidualBlock(in_channels=128, out_channels=128)
        # self.resblock256 = ResidualBlock(in_channels=256, out_channels=256)
        # self.resblock512 = ResidualBlock(in_channels=512, out_channels=512)

    def forward(self, x: Tensor) -> Tensor:
        x64 = x[:, 0:64, :]
        x128 = x[:, 64:64 + 128, :]
        x256 = x[:, 64 + 128:64 + 128 + 256, :]
        x512 = x[:, 64 + 128 + 256:, :]

        x128 = self.latlayer64(x64) + x128
        x256 = self.latlayer128(x128) + x256
        x512 = self.latlayer256(x256) + x512

        # x64 = self.resblock64(x64)
        # x128 = self.resblock128(x128)
        # x256 = self.resblock256(x256)
        # x512 = self.resblock512(x512)

        x = torch.cat((x64, x128, x256, x512), dim=1)

        return x


class MSGMMResNet(nn.Module):
    def __init__(self, gmm_1024, gmm_512, gmm_256, gmm_128, gmm_64, groups=1, group_width=512,
                 gmmlayer_regroup_num=1, gmmlayer_index_trans=False, gmmlayer_shuffle=False) -> None:
        super(MSGMMResNet, self).__init__()

        self.groups = groups
        self.group_width = group_width

        self.relu = nn.ReLU()

        self.gmm = GMM.concatenate_gmms((gmm_1024, gmm_512, gmm_256, gmm_128, gmm_64), groups=groups)
        self.gmm_layer = GMMLayer(self.gmm, requires_grad=False, regroup_num=gmmlayer_regroup_num,
                                  index_trans=gmmlayer_index_trans, shuffle=gmmlayer_shuffle)

        # idx_trans = gmm_idx_trans
        # self.gmm_layer_64 = GMMLayer(gmm_64, requires_grad=False, index_trans=gmmlayer_index_trans)
        # self.gmm_layer_128 = GMMLayer(gmm_128, requires_grad=False, index_trans=gmmlayer_index_trans)
        # self.gmm_layer_256 = GMMLayer(gmm_256, requires_grad=False, index_trans=gmmlayer_index_trans)
        # self.gmm_layer_512 = GMMLayer(gmm_512, requires_grad=False, index_trans=gmmlayer_index_trans)
        # self.gmm_layer_1024 = GMMLayer(gmm_1024, requires_grad=False, index_trans=gmmlayer_index_trans)
        #
        # self.latlayer128 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, groups=groups)
        # self.latlayer256 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, groups=groups)
        # self.latlayer512 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, groups=groups)
        # self.latlayer1024 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1, groups=groups)

        # self.se64 = SqueezeExcitation1d(input_channels=64, squeeze_channels=64 // 4)
        # self.se128 = SqueezeExcitation1d(input_channels=128, squeeze_channels=128 // 4)
        # self.se256 = SqueezeExcitation1d(input_channels=256, squeeze_channels=256 // 4)
        # self.se512 = SqueezeExcitation1d(input_channels=512, squeeze_channels=512 // 4)
        # self.se1024 = SqueezeExcitation1d(input_channels=1024, squeeze_channels=1024 // 4)

        # self.fusionblock1 = FusionBlock()
        # self.fusionblock2 = FusionBlock()
        # self.fusionblock3 = FusionBlock()
        # self.fusionblock4 = FusionBlock()
        # self.fusionblock5 = FusionBlock()
        # self.fusionblock6 = FusionBlock()

        channels = groups * group_width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=self.gmm.size(), out_channels=channels, kernel_size=1, stride=1,
                      padding=0, dilation=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups, first=True)
        self.block2 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        self.block3 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        self.block4 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        self.block5 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        self.block6 = ResidualBlock(in_channels=channels, out_channels=channels, groups=groups)
        # self.block7 = Residual_block(in_channels=512, out_channels=512, use_max_pool=False)
        # self.block8 = Residual_block(in_channels=512, out_channels=512)
        # self.block9 = Residual_block(in_channels=in_channels, out_channels=in_channels)
        # self.block10 = Residual_block(in_channels=in_channels, out_channels=in_channels)

        self.bn1 = BatchNorm1d(channels)
        self.bn2 = BatchNorm1d(channels)
        self.bn3 = BatchNorm1d(channels)
        self.bn4 = BatchNorm1d(channels)
        self.bn5 = BatchNorm1d(channels)
        self.bn6 = BatchNorm1d(channels)
        # self.bn7 = BatchNorm1d(512)
        # self.bn8 = BatchNorm1d(512)
        # self.bn9 = BatchNorm1d(512)
        # self.bn10 = BatchNorm1d(512)

        self.pool = nn.AdaptiveMaxPool1d(1)  # AdaptiveMaxPool1d(1)
        # self.pool_avg = nn.AdaptiveAvgPool1d(1)

        self.output_size = channels * 6  # + 1024 * 3  # + 768 + 1024 + 1280

        # self.sub_classifiers = nn.ModuleList()
        # for i in range(self.groups):
        #     self.sub_classifiers.append(nn.Linear(self.output_size // self.groups, 2))

        self.sub_classifiers = nn.Conv1d(self.output_size, self.groups * 2, kernel_size=1, groups=self.groups)
        # self.classifier = nn.Linear(self.groups * 2, 2, bias=False)

        # self.sub_classifiers = nn.Sequential(
        #     nn.Conv1d(self.output_size, self.groups * 256, kernel_size=1, groups=self.groups),
        #     nn.BatchNorm1d(self.groups * 256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(self.groups * 256, self.groups * 2, kernel_size=1, groups=self.groups),
        # )

        # self.attention = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        # x64 = self.gmm_layer_64(x)
        # x128 = self.gmm_layer_128(x)
        # x256 = self.gmm_layer_256(x)
        # x512 = self.gmm_layer_512(x)
        # x1024 = self.gmm_layer_1024(x)

        x = self.gmm_layer(x)

        # x64 = self.se64(x64)
        # x128 = self.se128(x128)
        # x256 = self.se256(x256)
        # x512 = self.se512(x512)

        # x = self.latlayer128(x64) + x128
        # x = self.latlayer256(x) + x256
        # x = self.latlayer512(x) + x512
        # x = self.latlayer1024(x) + x1024

        # x = torch.cat((x64, x128, x256, x512, x1024), dim=1)

        # x = self.fusionblock1(x)
        # x = self.fusionblock2(x)
        # x = self.fusionblock3(x)
        # x = self.fusionblock4(x)
        # x = self.fusionblock5(x)
        # x = self.fusionblock6(x)

        x = self.stem(x)

        y = []

        x = self.block1(x)
        y.append(self.pool(self.relu(self.bn1(x))))
        # y.append(self.pool(x))

        x = self.block2(x)
        y.append(self.pool(self.relu(self.bn2(x))))
        # y.append(self.pool(x))

        x = self.block3(x)
        y.append(self.pool(self.relu(self.bn3(x))))
        # y.append(self.pool(x))

        x = self.block4(x)
        y.append(self.pool(self.relu(self.bn4(x))))
        # y.append(self.pool(x))

        x = self.block5(x)
        y.append(self.pool(self.relu(self.bn5(x))))
        # y.append(self.pool(x))

        x = self.block6(x)
        y.append(self.pool(self.relu(self.bn6(x))))
        # y.append(self.pool(x))

        # x = torch.cat(y, dim=1)

        z = []
        for g in range(self.groups):
            for y_i in y:
                z.append(y_i[:, g * self.group_width:(g + 1) * self.group_width])
        z = torch.cat(z, dim=1)

        z = self.sub_classifiers(z)
        z = z.squeeze(2)
        z = torch.split(z, 2, dim=1)
        result = list(z)

        z_assembly = sum(z) / len(z)
        # z_assembly = self.classifier(z)
        result.append(z_assembly)

        # y = torch.split(x, 2, dim=1)
        #
        # y = torch.concatenate(y, dim=2)
        # y = y * self.attention(y)
        # y = self.pool(y)
        # y = y.squeeze(2)
        # result.append(y)

        # y = torch.concatenate(y, dim=1)
        # y = y.squeeze(2)
        # y = self.classifier(y)
        # result.append(y)

        if self.training:
            return result
        else:
            return z_assembly

        # return result

        # result = []
        # result.append(self.classifier(x))

        # for g in range(self.groups):
        #     xg = torch.cat(
        #         [x[:, (self.groups * i + g) * self.group_width:(self.groups * i + g + 1) * self.group_width] for i
        #          in range(6)], dim=1)
        #     result.append(self.sub_classifiers[g](xg))

        # result = sum(result) / len(result)
        # return result


class MSGMMResNet2P(nn.Module):
    def __init__(self,
                 gmm_b_1024, gmm_b_512, gmm_b_256, gmm_b_128, gmm_b_64,
                 gmm_s_1024, gmm_s_512, gmm_s_256, gmm_s_128, gmm_s_64,
                 groups=1, group_width=512,
                 gmmlayer_regroup_num=1, gmmlayer_index_trans=False, gmmlayer_shuffle=False,
                 ) -> None:
        super(MSGMMResNet2P, self).__init__()

        self.path1 = MSGMMResNet(gmm_b_1024, gmm_b_512, gmm_b_256, gmm_b_128, gmm_b_64,
                                 groups=groups,
                                 group_width=group_width,
                                 gmmlayer_regroup_num=gmmlayer_regroup_num,
                                 gmmlayer_index_trans=gmmlayer_index_trans,
                                 gmmlayer_shuffle=gmmlayer_shuffle)

        self.path2 = MSGMMResNet(gmm_s_1024, gmm_s_512, gmm_s_256, gmm_s_128, gmm_s_64,
                                 groups=groups,
                                 group_width=group_width,
                                 gmmlayer_regroup_num=gmmlayer_regroup_num,
                                 gmmlayer_index_trans=gmmlayer_index_trans,
                                 gmmlayer_shuffle=gmmlayer_shuffle)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.path1(x)
        x2 = self.path2(x)
        result = x1
        result.extend(x2)

        # result = x1 + x2 / 2
        return result


class AS21MultiScaleGMMResNetExperiment(AS21MultiScaleGMMExperiment):
    def __init__(self, model_type, feature_type, access_type, parm):
        super(AS21MultiScaleGMMResNetExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def get_net(self, num_classes=2):
        if self.model_type == 'GMM_ResNet2':
            model = MSGMMResNet(self.gmm_1024, self.gmm_512, self.gmm_256, self.gmm_128, self.gmm_64,
                                groups=self.parm['groups'],
                                group_width=self.parm['group_width'],
                                gmmlayer_regroup_num=self.parm['gmmlayer_regroup_num'],
                                gmmlayer_index_trans=self.parm['gmmlayer_index_trans'],
                                gmmlayer_shuffle=self.parm['gmmlayer_shuffle'],
                                )

        model = model.cuda()
        torchinfo.summary(model, (2, self.parm['feature_size'], self.parm['feature_num']), depth=5)
        torchsummary.summary(model, (self.parm['feature_size'], self.parm['feature_num']))

        return model

    def compute_loss(self, output, target, criterion):
        temperature = 3.0

        if isinstance(output, list):
            output_path = output[:-1]
            output_assembly = output[-1]

            # return criterion(output_assembly, target)

            # if self.epoch >= self.num_epochs - 10:
            #     return criterion(output_assembly, target)

            loss_list = [criterion(o, target) for o in output]
            loss = sum(loss_list) / len(loss_list)

            # if self.epoch >= self.num_epochs - 10:
            #     return loss
            #
            # output_ref = F.softmax(output_assembly / temperature, dim=1)
            # kl_loss_list = [self.kl_loss((o / temperature).log_softmax(dim=1), output_ref) for o in output_path]
            # kl_loss = sum(kl_loss_list) / len(kl_loss_list)
            # kl_loss = temperature * temperature * kl_loss
            #
            # loss = loss + 1.0 * kl_loss

        else:
            loss = criterion(output, target)

        return loss


class AS21MultiScaleGMMResNet2PExperiment(AS21MultiScaleGMM2PathExperiment):
    def __init__(self, model_type, feature_type, access_type, parm):
        super(AS21MultiScaleGMMResNet2PExperiment, self).__init__(model_type, feature_type, access_type, parm=parm)

    def get_net(self, num_classes=2):
        if self.model_type == 'MSGMM_ResNet_2P':
            model = MSGMMResNet2P(self.gmm_bonafide_1024, self.gmm_bonafide_512, self.gmm_bonafide_256,
                                  self.gmm_bonafide_128, self.gmm_bonafide_64,
                                  self.gmm_spoof_1024, self.gmm_spoof_512, self.gmm_spoof_256, self.gmm_spoof_128,
                                  self.gmm_spoof_64,
                                  groups=self.parm['groups'],
                                  group_width=self.parm['group_width'],
                                  gmmlayer_regroup_num=self.parm['gmmlayer_regroup_num'],
                                  gmmlayer_index_trans=self.parm['gmmlayer_index_trans'],
                                  gmmlayer_shuffle=self.parm['gmmlayer_shuffle'],
                                  )

        model = model.cuda()
        torchsummary.summary(model, (self.parm['feature_size'], self.parm['feature_num']))

        return model


if __name__ == '__main__':
    exp_param = exp_parameters()

    access_type = 'LA'
    feature_type = 'LFCC21NN'

    exp_param['lr'] = 0.0001
    exp_param['min_lr'] = 0.0
    exp_param['weight_decay'] = 0.0

    model_type = 'GMM_ResNet2'  # 'MSGMM_ResNet_2P'  # GMM_ResNet_2Path    GMM_SENet_2Path

    # exp_param['num_epochs'] = 80

    exp_param['groups'] = 8
    exp_param['group_width'] = 256

    exp_param['gmm_index_trans'] = False
    exp_param['gmm_regroup_num'] = 8
    exp_param['gmm_shuffle'] = False

    exp_param['gmmlayer_index_trans'] = False
    exp_param['gmmlayer_regroup_num'] = 1
    exp_param['gmmlayer_shuffle'] = False

    # exp_param['gmm_type'] = 'gmm_aug2_rb4'
    exp_param['data_augmentation'] = ['Original', 'RB4']  # 'RB4'
    exp_param[
        'gmm_file_dir'] = r'/home/lzc/lzc/ASVspoof/ASVspoof2021exp/GMM_aug2_rb4_{}'.format(feature_type)

    for _ in range(5):
        AS21MultiScaleGMMResNetExperiment(model_type, feature_type, access_type, parm=exp_param).run()

    # exp_param['lr'] = 0.0005
    # AS21MultiScaleGMMResNetExperiment(model_type, feature_type, access_type, parm=exp_param).run()
