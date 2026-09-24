import torch
import torch.nn as nn


# SE Block
class SEBlock(nn.Module):
    """Computes importance score for each channel"""
    def __init__(self, c, r=16):
        super(SEBlock, self).__init__()
        # Reduce feature maps to (1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
			nn.Linear(c, max(1, c//r)),
			nn.ReLU(inplace=True),
			nn.Linear(max(1, c//r), c),
			# Scale to att weights to 0-1
			nn.Sigmoid()
		)
    
    def forward(self, x):
        b, c, w, h = x.size()
        a = self.avg_pool(x).view(b, c)
        a = self.fc(a).view(b, c, 1, 1)
        return x * a

# CBAM
class CAttention(nn.Module):
    def __init__(self, c, r=16):
        super(CAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
			nn.Linear(c, c//r),
			nn.ReLU(),
			nn.Linear(c//r, c)
		)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.avg_pool(x).view(x.size()[0], -1)
        x2 = self.max_pool(x).view(x.size()[0], -1)
        x1 = self.mlp(x1)
        x2 = self.mlp(x2)
        x = x1 + x2
        return self.sigmoid(x).unsqueeze(-1).unsqueeze(-1)

class SAttention(nn.Module):
    def __init__(self, ks):
        super(SAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=ks, padding=ks//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_pool, max_pool], dim=1)
        a = self.sigmoid(self.conv(x))
        return a

class CBAM(nn.Module):
    def __init__(self, c, r, ks):
        super(CBAM, self).__init__()
        self.channel_att = CAttention(c, r)
        self.spatial_att = SAttention(ks)
        
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class NumericalBranch(nn.Module):
    def __init__(self, n_features=15, hidden_size=128, num_layers=2,
                 lstm_dropout=0.3, feature_dropout=0.2):
        super(NumericalBranch, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.seqBlocks = nn.Sequential(
			nn.ConvTranspose2d(2 * hidden_size, 512, kernel_size=4, stride=4),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1),
   			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Dropout2d(feature_dropout),
			nn.ConvTranspose2d(512, 512, kernel_size=4, stride=4),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
   			nn.Conv2d(512, 512, 3, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
		)
    
    def forward(self, x):
        _, (h_n, _) = self.bilstm(x)
        
        forward_h = h_n[-2]   # [B, 128]
        backward_h = h_n[-1]  # [B, 128]
        
        x = torch.cat([forward_h, backward_h], dim=1) # [B, 256]
        x = x.unsqueeze(-1).unsqueeze(-1)	# [B, 256, 1, 1]
        x = self.seqBlocks(x) # [B, 512, 16, 16]
        return x

class Multimodal(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers,
                 lstm_dropout=0.3, feature_dropout=0.2):
        super(Multimodal, self).__init__()
        self.numerical_branch = NumericalBranch(
            n_features, hidden_size, num_layers,
            lstm_dropout=lstm_dropout,
            feature_dropout=feature_dropout,
        )
        self.se_visual = SEBlock(512)
        self.se_numerical = SEBlock(512)
        self.cbam = CBAM(1024, r=16, ks=7)
        self.branch_att = nn.Sequential(
			nn.Linear(512+512, 64),
			nn.ReLU(),
			nn.Linear(64, 2)
		)
        
        self.encoder_1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
		)
        self.encoder_2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
		)
        self.encoder_3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
		)
        self.encoder_4 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
		)
        self.encoder_5 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
		)
        self.pool = nn.MaxPool2d(2, 2)
    
        self.upconv1 = nn.ConvTranspose2d(1024, 256, 2, stride=2)
        self.bn_up1 = nn.BatchNorm2d(256)
        self.decoder_1 = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
		)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.bn_up2 = nn.BatchNorm2d(128)
        self.decoder_2 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
		)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.bn_up3 = nn.BatchNorm2d(64)
        self.decoder_3 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
		)
        
        self.upconv4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.bn_up4 = nn.BatchNorm2d(32)
        self.decoder_4 = nn.Sequential(
			nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
		)
        
        self.out_layer = nn.Conv2d(32, 1, 1)
    
    def forward(self, x, s, return_att=False, branch_gates=None, return_stages=False):
        stages = {} if return_stages else None
        x_input = x
        if return_stages:
            stages['input'] = x_input
        
        x = x.unsqueeze(1)
        x1 = self.encoder_1(x)
        if return_stages:
            stages['encoder1'] = x1
        
        x2_in = self.pool(x1)
        x2 = self.encoder_2(x2_in)
        if return_stages:
            stages['encoder2'] = x2
        
        x3_in = self.pool(x2)
        x3 = self.encoder_3(x3_in)
        if return_stages:
            stages['encoder3'] = x3
        
        x4_in = self.pool(x3)
        x4 = self.encoder_4(x4_in)
        if return_stages:
            stages['encoder4'] = x4
        
        x5_in = self.pool(x4)
        x5 = self.encoder_5(x5_in)
        if return_stages:
            stages['encoder5'] = x5
        
        s = self.numerical_branch(s)
        if return_stages:
            stages['seq_branch'] = s
        
        x5 = self.se_visual(x5)
        s = self.se_numerical(s)
        
        gap = nn.functional.adaptive_avg_pool2d
        g_img = torch.flatten(gap(x5, 1), 1)
        g_seq = torch.flatten(gap(s, 1), 1)
        
        att_logits = self.branch_att(torch.cat([g_img, g_seq], dim=1))
        att_weights = torch.softmax(att_logits, dim=1)
        
        if branch_gates is not None:
            if not torch.is_tensor(branch_gates):
                branch_gates = torch.tensor(branch_gates, dtype= att_weights.dtype, device=att_weights.device)
            branch_gates = branch_gates.view(1, 2).expand_as(att_weights)
            att_weights = att_weights * branch_gates
            att_weights = att_weights / (att_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        w_img = att_weights[:, 0].view(-1, 1, 1, 1)
        w_seq = att_weights[:, 1].view(-1, 1, 1, 1)
        
        x5 = x5 * w_img
        s = s * w_seq
        x_s = torch.cat([x5, s], dim=1)
        if return_stages:
            stages['before_cbam'] = x_s
        
        x_s = self.cbam(x_s)
        if return_stages:
            stages['after_cbam'] = x_s
        
        x6 = self.upconv1(x_s)
        x6 = self.bn_up1(x6)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.decoder_1(x6)
        if return_stages:
            stages['decoder_1'] = x6
        
        x7 = self.upconv2(x6)
        x7 = self.bn_up2(x7)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.decoder_2(x7)
        if return_stages:
            stages['decoder_2'] = x7
        
        x8 = self.upconv3(x7)
        x8 = self.bn_up3(x8)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.decoder_3(x8)
        if return_stages:
            stages['decoder_3'] = x8
        
        x9 = self.upconv4(x8)
        x9 = self.bn_up4(x9)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.decoder_4(x9)
        if return_stages:
            stages['decoder_4'] = x9
        
        out = self.out_layer(x9)
        out = out.squeeze(1)
        if return_stages:
            stages['output'] = out.unsqueeze(-1)
        
        if not return_att and not return_stages:
            return out
        
        if return_att:
            with torch.no_grad():
                avg_out = torch.mean(x_s, dim=1, keepdim=True)
                max_out, _ = torch.max(x_s, dim=1, keepdim=True)
                spatial_input = torch.cat([avg_out, max_out], dim=1)
                sa = self.cbam.spatial_att(spatial_input)
                sa_up = nn.functional.interpolate(
                    sa, size=x_input.shape[-2:], mode='bilinear', align_corners=False
                )
                sa_up = sa_up.squeeze(1).detach().cpu()

            att_dict = {
                "branch_weights": att_weights.detach().cpu(),
                "spatial_map": sa_up
            }

        if return_att and not return_stages:
            return out, att_dict
        elif not return_att and return_stages:
            return out, stages
        else:
            return out, att_dict, stages


def load_pretrained_multimodal(
    model,
    checkpoint,
    n_pretrained_features=15,
    freeze_visual=True,
):
    """Load LAB weights into a real-data model with extra input features."""
    source_state = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )
    target_state = model.state_dict()

    expanded_lstm_keys = {
        "numerical_branch.bilstm.weight_ih_l0",
        "numerical_branch.bilstm.weight_ih_l0_reverse",
    }

    for key, source_value in source_state.items():
        if key not in target_state:
            continue

        target_value = target_state[key]

        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            continue

        if (
            key in expanded_lstm_keys
            and source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] == n_pretrained_features
            and target_value.shape[1] >= n_pretrained_features
        ):
            expanded_value = target_value.clone()
            expanded_value[:, :n_pretrained_features] = source_value
            target_state[key] = expanded_value

    model.load_state_dict(target_state)

    if freeze_visual:
        frozen_module_names = [
            "encoder_1",
            "encoder_2",
            "encoder_3",
            "encoder_4",
            "encoder_5",
            "se_visual",
        ]

        for module_name in frozen_module_names:
            module = getattr(model, module_name)
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        model.frozen_module_names = frozen_module_names

    return model



