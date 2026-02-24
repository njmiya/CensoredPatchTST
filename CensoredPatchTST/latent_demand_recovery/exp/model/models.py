import os
import torch
from pypots.imputation import DLinear, TimesNet, PatchTST, SAITS, BRITS, ETSformer, Autoformer, Informer, Transformer, \
    iTransformer, GPVAE, CSDI, ImputeFormer
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse, calc_mre
from torch import nn

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")


class AsymmetricMAE(nn.Module):
    """
    非对称MAE损失函数 (Pinball Loss / Quantile Loss 的变体)
    用于解决生鲜缺货数据的系统性低估问题

    Logic:
    - 如果模型预测值 < 真实值: 给予更大的惩罚 (quantile q)
    - 如果模型预测值 > 真实值: 给予较小的惩罚 (1-q)

    对于 Demand Recovery 任务，宁愿模型稍微激进一点(高估)，也不希望它保守(低估)。
    因此设置 q > 0.5。
    """

    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, inputs, results):
        # indicating_mask: 训练时人为 mask 掉的部分
        pos_mask = inputs['indicating_mask']
        targets = inputs['X_ori']
        preds = results['imputed_data']

        # 计算残差 (Pred - Target)
        # Pinball Loss 标准公式: max( (y-y_hat)*q, (y_hat-y)*(1-q) )
        # 对应残差 r = targets - preds

        residuals = targets - preds

        # Loss calculation
        loss = torch.max(self.q * residuals, (self.q - 1) * residuals)

        # 只计算 mask 部分的 loss
        loss = loss * pos_mask

        # 避免除以0
        return loss.sum() / (pos_mask.sum() + 1e-9)


def load_model(CONFIG):
    model_name = CONFIG.get('model', 'TimesNet')
    saving_path = os.path.join(CONFIG['saving_path'], model_name)

    if model_name == 'CensoredPatchTST':
        print(">>> Initializing CensoredPatchTST with Asymmetric Loss (Innovation Mode) <<<")

        # 24 小时为一周期，步长 12 小时进行平滑
        patch_size = 24
        stride = 12

        model = PatchTST(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            patch_size=patch_size,
            patch_stride=stride,
            n_layers=3,
            d_model=128,
            n_heads=8,
            d_k=16,
            d_v=16,
            d_ffn=256,
            dropout=0.1,  # 稍微增加 dropout 以应对稀疏噪声
            attn_dropout=0.1,

            # 倾向于输出 P70 分位数的预测，
            training_loss=AsymmetricMAE(q=0.7),

            ORT_weight=1,
            MIT_weight=1,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['EPOCHS'],
            patience=CONFIG['patience'],
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=1e-5),
            device=DEVICE,
            saving_path=saving_path,
        )
    elif model_name == 'SAITS':
        model = SAITS(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            n_layers=CONFIG['n_layers'],
            d_model=CONFIG['d_model'],
            d_ffn=CONFIG['d_ffn'],
            n_heads=CONFIG['n_heads'],
            d_k=CONFIG['d_k'],
            d_v=CONFIG['d_v'],
            dropout=CONFIG['dropout'],
            attn_dropout=CONFIG['attn_dropout'],
            epochs=CONFIG['EPOCHS'],
            batch_size=CONFIG['batch_size'],
            ORT_weight=1,
            MIT_weight=1,
            saving_path=saving_path,
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device=DEVICE,
            patience=CONFIG['patience'],
        )
    elif model_name == 'DLinear':
        model = DLinear(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            moving_avg_window_size=CONFIG['patch_len'] // 2 * 2 + 1,
            individual=False,
            d_model=CONFIG['d_model'],
            ORT_weight=1,
            MIT_weight=1,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['EPOCHS'],
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device=DEVICE,
            saving_path=saving_path,
            patience=CONFIG['patience'],
        )
    elif model_name == 'TimesNet':
        model = TimesNet(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            n_layers=CONFIG['n_layers'],
            top_k=7,
            d_model=CONFIG['d_model'],
            d_ffn=CONFIG['d_ffn'],
            n_kernels=5,
            dropout=CONFIG['dropout'],
            apply_nonstationary_norm=True,
            epochs=CONFIG['EPOCHS'],
            batch_size=CONFIG['batch_size'],
            saving_path=saving_path,
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device=DEVICE,
            patience=CONFIG['patience'],
        )
    elif model_name == 'iTransformer':
        model = iTransformer(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            n_layers=CONFIG['n_layers'],
            d_model=CONFIG['d_model'],
            n_heads=CONFIG['n_heads'],
            d_k=CONFIG['d_k'],
            d_v=CONFIG['d_v'],
            d_ffn=CONFIG['d_ffn'],
            dropout=CONFIG['dropout'],
            attn_dropout=CONFIG['attn_dropout'],
            ORT_weight=1,
            MIT_weight=1,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['EPOCHS'],
            patience=CONFIG['patience'],
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device=DEVICE,
            saving_path=saving_path,
        )
    elif model_name == 'GPVAE':
        model = GPVAE(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            latent_size=4,
            encoder_sizes=(64, 64),
            decoder_sizes=(64, 64),
            kernel="cauchy",
            beta=0.01,
            M=1,
            K=1,
            sigma=1.0,
            length_scale=48.0,
            kernel_scales=3,
            window_size=3,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['EPOCHS'],
            patience=CONFIG['patience'],
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device=DEVICE,
            saving_path=saving_path,
        )
    elif model_name == 'CSDI':
        model = CSDI(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            n_layers=CONFIG['n_layers'],
            n_heads=CONFIG['n_heads'],
            n_channels=CONFIG['d_ffn'],
            d_time_embedding=CONFIG['d_model'],
            d_feature_embedding=CONFIG['d_model'],
            d_diffusion_embedding=CONFIG['d_model'],
            n_diffusion_steps=30,
            target_strategy="random",
            schedule='linear',
            beta_start=0.0001,
            beta_end=0.04,
            is_unconditional=False,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['EPOCHS'],
            patience=CONFIG['patience'],
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device=DEVICE,
            saving_path=saving_path
        )
    elif model_name == 'ImputeFormer':
        model = ImputeFormer(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            n_layers=CONFIG['n_layers'],
            d_input_embed=CONFIG['d_model'],
            d_learnable_embed=CONFIG['d_model'],
            d_proj=CONFIG['d_model'],
            d_ffn=CONFIG['d_ffn'],
            n_temporal_heads=CONFIG['n_heads'],
            dropout=CONFIG['dropout'],
            input_dim=1,
            output_dim=1,
            ORT_weight=1,
            MIT_weight=1,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['EPOCHS'],
            patience=CONFIG['patience'],
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device=DEVICE,
            saving_path=saving_path,
        )
    elif model_name == 'PatchTST':
        model = PatchTST(
            n_steps=CONFIG['n_steps'],
            n_features=CONFIG['n_features'],
            patch_size=CONFIG['patch_len'],
            patch_stride=CONFIG['patch_len'],
            n_layers=CONFIG['n_layers'],
            d_model=CONFIG['d_model'],
            n_heads=CONFIG['n_heads'],
            d_k=CONFIG['d_k'],
            d_v=CONFIG['d_v'],
            d_ffn=CONFIG['d_ffn'],
            dropout=CONFIG['dropout'],
            attn_dropout=CONFIG['attn_dropout'],
            ORT_weight=1,
            MIT_weight=1,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['EPOCHS'],
            patience=CONFIG['patience'],
            optimizer=Adam(lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']),
            device=DEVICE,
            saving_path=saving_path,
        )
    else:
        raise NotImplementedError(f'{model_name} is not implemented')

    return model
