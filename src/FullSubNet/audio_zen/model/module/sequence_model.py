import torch
import torch.nn as nn

from src.DCCRN.DCCRN import EPS


class ComplexSequenceModel(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size,
            num_layers,
            bidirectional,
            sequence_model="GRU",
            output_activate_function="Tanh"
    ):
        """
        序列模型，可选 LSTM 或 CRN，支持子带输入

        Args:
            input_size: 每帧输入特征大小
            output_size: 每帧输出特征大小
            hidden_size: 序列模型隐层单元数量
            num_layers:  层数
            bidirectional: 是否为双向
            sequence_model: LSTM | GRU
            output_activate_function: Tanh | ReLU
        """
        super().__init__()
        # Sequence layer
        if sequence_model == "LSTM":
            self.sequence_model_real = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.sequence_model_img = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif sequence_model == "GRU":
            self.sequence_model_real = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.sequence_model_img = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if bidirectional:
            self.complex_fc_act_output_layer = ComplexLinearAct(hidden_size * 2, output_size, output_activate_function)
        else:
            self.complex_fc_act_output_layer = ComplexLinearAct(hidden_size, output_size, output_activate_function)


    def forward(self, x):
        """
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        assert x.dim() == 4
        self.sequence_model_real.flatten_parameters()
        self.sequence_model_img.flatten_parameters()

        # contiguous 使元素在内存中连续，有利于模型优化，但分配了新的空间
        # 建议在网络开始大量计算前使用一下
        x = x.permute(0, 1, 3, 2).contiguous()  # [B, C, F, T] => [B, C, T, F]
        sequence_out_real = self.sequence_model_real(x[:,0,:,:])[0] - self.sequence_model_img(x[:,1,:,:])[0]
        sequence_out_img = self.sequence_model_real(x[:,1,:,:])[0] + self.sequence_model_img(x[:,0,:,:])[0]
        output_real, output_img = self.complex_fc_act_output_layer(sequence_out_real, sequence_out_img)
        output_real = output_real.unsqueeze(1)
        output_img = output_img.unsqueeze(1)
        output_real = output_real.permute(0, 1, 3, 2).contiguous()  # [B, C, T, F] => [B, C, F, T]
        output_img = output_img.permute(0, 1, 3, 2).contiguous()  # [B, C, T, F] => [B, C, F, T]
        output = torch.stack([output_real,output_img], dim=1)
        return output


class ComplexLinearAct(nn.Module):
    """ One Complex Convolution as explained in paper, followed by activation and normalization"""
    def __init__(self, input_size, output_size, output_activate_function):
        super(ComplexLinearAct, self).__init__()
        self.linear_real = nn.Linear(input_size, output_size)
        self.linear_imag = nn.Linear(input_size, output_size)
        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            else:
                raise NotImplementedError(f"Not implemented activation function {self.activate_function}")

            self.act_real = self.activate_function
            self.act_imag = self.activate_function

        self.output_activate_function = output_activate_function

    def forward(self, input_real, input_imag):
        """
        :param input_real: shape [Batch, input_size, F, T]
        :param input_imag: shape [Batch, output_size, F, T]
        """
        output_real = self.linear_real(input_real) - self.linear_imag(input_imag)
        output_imag = self.linear_real(input_imag) + self.linear_imag(input_real)
        if self.output_activate_function:
            output_real = self.act_real(output_real)
            output_imag = self.act_imag(output_imag)
        return output_real, output_imag


def choose_norm(norm_type, channel_size):
    """ Currently only BN gives good results"""
    if norm_type == "CLN":
        return ChannelwiseLayerNorm(channel_size)
    else:
        assert norm_type == 'BN'  # either CLN or BN. both are
    return RealBatchNorm(channel_size)


class RealBatchNorm(nn.Module):
    """ Regular Batch norm on each of the inputs
    Batch norm in train mode isn't causal"""

    def __init__(self, channel_size):
        super(RealBatchNorm, self).__init__()
        self.batch_norm_real = nn.BatchNorm2d(channel_size)
        self.batch_norm_imag = nn.BatchNorm2d(channel_size)

    def forward(self, real_input, imag_input):
        return self.batch_norm_real(real_input), self.batch_norm_imag(imag_input)


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN). Currently doesn't give good results"""

    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma_real = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))
        self.beta_real = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))
        self.gamma_imag = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))
        self.beta_imag = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma_real.data.fill_(1)
        self.gamma_imag.data.fill_(1)
        self.beta_real.data.zero_()
        self.beta_imag.data.zero_()

    def forward(self, y_real, y_imag):
        mean_real = torch.mean(y_real, dim=1, keepdim=True)  # [M, 1, Freq, T]
        var_real = torch.var(y_real, dim=1, keepdim=True, unbiased=False)  # [M, 1, Freq, T]
        cLN_y_real = self.gamma_real * (y_real - mean_real) / torch.pow(var_real + EPS, 0.5) + self.beta_real

        mean_imag = torch.mean(y_imag, dim=1, keepdim=True)  # [M, 1, Freq, T]
        var_imag = torch.var(y_imag, dim=1, keepdim=True, unbiased=False)  # [M, 1, Freq, T]
        cLN_y_imag = self.gamma_imag * (y_imag - mean_imag) / torch.pow(var_imag + EPS, 0.5) + self.beta_imag

        return cLN_y_real, cLN_y_imag


def _print_networks(nets: list):
    print(f"This project contains {len(nets)} networks, the number of the parameters: ")
    params_of_all_networks = 0
    for i, net in enumerate(nets, start=1):
        params_of_network = 0
        for param in net.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network

    print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")


if __name__ == '__main__':
    import datetime

    with torch.no_grad():
        ipt = torch.rand(1, 2, 257, 1000)
        model = ComplexSequenceModel(
            input_size=257,
            output_size=2,
            hidden_size=512,
            bidirectional=False,
            num_layers=3,
            sequence_model="LSTM"
        )

        start = datetime.datetime.now()
        opt = model(ipt)
        end = datetime.datetime.now()
        print(f"{end - start}")
        _print_networks([model, ])
