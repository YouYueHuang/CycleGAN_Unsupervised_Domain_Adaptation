from .base_options import BaseOptions


class CycleGanTrainOptions(BaseOptions):
    def __init__(self):
        self.mode = "train"
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Network ---------- #
        parser.add_argument('--model', type=str, default="cycle_gan", choices = ['mcd','cycle_mcd'],
                            help="Method Name")
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization, default CycleGAN did not use dropout')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # ---------- Define Dataset ---------- #
        parser.add_argument('--datasetA', type=str, 
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = "gta_train")
        parser.add_argument('--datasetB', type=str, 
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = "city_train")
        # ---------- Optimizers ---------- #
        parser.add_argument('--opt', type=str, default="adam", choices=['sgd', 'adam'],
                            help="cycle gan network optimizer")
        parser.add_argument("--adjustLr", action="store_false",
                            help='whether you change lr')
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        # ---------- Hyperparameters ---------- #
        parser.add_argument('--lsgan', action='store_false', help='do not use least square GAN, if specified, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lambdaA', type=float, default=10.0,
                            help='weight for cycle loss (A -> A -> A)')
        parser.add_argument('--lambdaB', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambdaIdentity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        # ---------- Train Details ---------- #
        parser.add_argument('--epochStart', type=int, default=1, 
                            help='the starting epoch count.')
        parser.add_argument('--nEpochStart', type=int, default=100, 
                            help='# of epoch at starting learning rate')
        parser.add_argument('--nEpochDecay', type=int, default=100, 
                            help='# of epoch to linearly decay learning rate to zero')
        # ---------- Experiment Setting ---------- #
        parser.add_argument('--name', type=str, default='cycle_gan', 
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--displayWidth', type=int, default=4,
                            help='frequency of showing training results on screen')

        return parser
