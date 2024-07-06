import torch.nn as nn
import torch.nn.functional as F
import torch
import functools


# __all__ = ["CycleGen", "CycleDis", "UNetDown", "UNetUp"]
OPTIMIZER = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class UNetDown(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """

    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """

    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class UnetFinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.
    """

    def __init__(self, in_size, out_size):
        super(UnetFinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class CycleGen(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(CycleGen, self).__init__()

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 1024)

        self.up1 = UNetUp(1024, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = UnetFinalLayer(128, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        return self.final(u4, d1)

    # extract feature from this layer
    def extract_feature(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        return d5


class CycleDis(nn.Module):
    def __init__(self, input_shape):
        super(CycleDis, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2**4, width // 2**4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=get_norm_layer("instance"),
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
    ):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc=1,
        ndf=64,
        n_layers=3,
        norm_layer=get_norm_layer("instance"),
        use_sigmoid=False,
        gpu_ids=[],
    ):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        num_downs=8,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class linear_mlp(nn.Module):
    def __init__(self, indim, out_dim):
        super(linear_mlp, self).__init__()
        self.linear = nn.Linear(indim, out_dim)
        self.norm = nn.InstanceNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class mlp_res_block(nn.Module):
    def __init__(self, indim, out_dim):
        super(mlp_res_block, self).__init__()
        self.linear1 = linear_mlp(indim, out_dim)
        self.linear2 = linear_mlp(out_dim, out_dim)

    def forward(self, x):
        y = self.linear1(x)
        y = self.linear2(y)
        x = y + x
        return x


class downsample(nn.Module):
    def __init__(self, indim, out_dim):
        super(downsample, self).__init__()
        self.linear = linear_mlp(indim, out_dim)
        self.resblock = mlp_res_block(out_dim, out_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.resblock(x)
        return x


class upsample(nn.Module):
    def __init__(self, indim, out_dim):
        super(upsample, self).__init__()
        self.resblock = mlp_res_block(indim, out_dim)

        self.linear = linear_mlp(out_dim, out_dim)

    def forward(self, x):
        x = self.resblock(x)
        x = self.linear(x)
        return x


class MLPGenerator(nn.Module):
    def __init__(self, input_dim=90, output_dim=90):
        super(MLPGenerator, self).__init__()
        res_dim = 8
        self.network = nn.Sequential(
            linear_mlp(input_dim, 64),
            linear_mlp(64, 32),
            linear_mlp(32, 16),
            linear_mlp(16, 8),
            mlp_res_block(res_dim, res_dim),
            mlp_res_block(res_dim, res_dim),
            mlp_res_block(res_dim, res_dim),
            mlp_res_block(res_dim, res_dim),
            # mlp_res_block(res_dim, res_dim),
            # mlp_res_block(res_dim, res_dim),
            linear_mlp(8, 16),
            linear_mlp(16, 32),
            linear_mlp(32, 64),
            nn.Linear(64, output_dim),
            # nn.Sigmoid(),
            nn.Tanh(),
        )
        # res_dim = 1024
        # self.network = nn.Sequential(
        #     linear_mlp(input_dim, 128),
        #     linear_mlp(128, 256),
        #     linear_mlp(256, 512),
        #     linear_mlp(512, 1024),
        #     mlp_res_block(res_dim, res_dim),
        #     mlp_res_block(res_dim, res_dim),
        #     mlp_res_block(res_dim, res_dim),
        #     mlp_res_block(res_dim, res_dim),
        #     # mlp_res_block(res_dim, res_dim),
        #     # mlp_res_block(res_dim, res_dim),
        #     linear_mlp(1024, 512),
        #     linear_mlp(512, 256),
        #     linear_mlp(256, 128),
        #     nn.Linear(128, output_dim),
        #     # nn.Sigmoid(),
        #     nn.Tanh(),
        # )

    def forward(self, x, label=None):
        return self.network(x)


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim=90):
        super(MLPDiscriminator, self).__init__()
        # self.network = nn.Sequential(
        #     # First hidden layer
        #     nn.Linear(input_dim, 128),
        #     nn.Tanh(),
        #     # Second hidden layer
        #     nn.Linear(128, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 32),
        #     nn.Tanh(),
        #     # # Output layer
        #     nn.Linear(32, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, 4),
        #     nn.Sigmoid(),  # Sigmoid to output a probability
        # )
        # self.network = nn.Sequential(
        #     # First hidden layer
        #     linear_mlp(input_dim, 64),
        #     linear_mlp(64, 32),
        #     linear_mlp(32, 16),
        #     mlp_res_block(16, 16),
        #     mlp_res_block(16, 16),
        #     mlp_res_block(16, 16),
        #     nn.Linear(16, 4),
        #     # nn.Tanh(),
        #     nn.Sigmoid(),
        # )
        self.network = nn.Sequential(
            # First hidden layer
            linear_mlp(input_dim, 128),
            linear_mlp(128, 256),
            mlp_res_block(256, 256),
            mlp_res_block(256, 256),
            mlp_res_block(256, 256),
            linear_mlp(256, 128),
            linear_mlp(128, 64),
            linear_mlp(64, 32),
            linear_mlp(32, 16),
            nn.Linear(16, 4),
            # nn.Tanh(),
            nn.Sigmoid(),
        )
        # self.network = nn.Sequential(
        #     # First hidden layer
        #     nn.Linear(input_dim, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 256),
        #     nn.Tanh(),
        #     linear_mlp(256, 512),
        #     linear_mlp(512, 1024),
        #     linear_mlp(1024, 512),
        #     linear_mlp(512, 256),
        #     # Second hidden layer
        #     nn.Linear(256, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 32),
        #     nn.Tanh(),
        #     # nn.Linear(128, 64),
        #     # nn.Tanh(),
        #     # nn.Linear(64, 32),
        #     # nn.Tanh(),
        #     # # Output layer
        #     nn.Linear(32, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, 4),
        #     # nn.Linear(16, 1),
        #     # nn.Sigmoid(),  # Sigmoid to output a probability
        # )

    def forward(self, x):
        return self.network(x)


class Pixel2PixelDiscriminator(nn.Module):
    def __init__(self, input_dim=90 * 2):
        super(Pixel2PixelDiscriminator, self).__init__()
        # self.network = nn.Sequential(
        #     # First hidden layer
        #     linear_mlp(input_dim, 124),
        #     linear_mlp(124, 256),
        #     linear_mlp(256, 124),
        #     linear_mlp(124, 64),
        #     linear_mlp(64, 32),
        #     linear_mlp(32, 16),
        #     mlp_res_block(16, 16),
        #     mlp_res_block(16, 16),
        #     mlp_res_block(16, 16),
        #     nn.Linear(16, 4),
        #     # nn.Tanh(),
        #     nn.Sigmoid(),
        # )
        self.network = nn.Sequential(
            # First hidden layer
            linear_mlp(input_dim, 256),
            linear_mlp(256, 512),
            mlp_res_block(512, 512),
            mlp_res_block(512, 512),
            mlp_res_block(512, 512),
            linear_mlp(512, 256),
            linear_mlp(256, 124),
            linear_mlp(124, 64),
            linear_mlp(64, 32),
            linear_mlp(32, 16),
            nn.Linear(16, 4),
            # nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, source_sample, target_sample):
        return self.network(torch.cat((source_sample, target_sample), 1))


if __name__ == "__main__":

    # gen_model = UnetGenerator(num_downs=8)
    # Create an instance of the generator
    generator = MLPGenerator()

    # Generate a random input vector of size 90
    input_vector = torch.randn(1, 90)  # Batch size of 1, dimension of 90

    # Generate a sample using the MLP generator
    generated_sample = generator(input_vector)
    print(generated_sample)
    print(generated_sample.shape)
