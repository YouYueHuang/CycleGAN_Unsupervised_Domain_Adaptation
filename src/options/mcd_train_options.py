from .base_options import BaseOptions


class McdTrainOptions(BaseOptions):
    def __init__(self):
        self.mode = "train"
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # ---------- Define Network ---------- #
        parser.add_argument('--model', type=str, default="mcd", choices = ['mcd','cycle_mcd'],
                            help="Method Name")
        parser.add_argument('--net', type=str, default="drn_d_38", help="network structure",
                            choices=['fcn', 'psp', 'segnet', 'fcnvgg',
                                     "drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22",
                                     "drn_d_38", "drn_d_54", "drn_d_105"])
        parser.add_argument('--res', type=str, default='50', metavar="ResnetLayerNum",
                            choices=["18", "34", "50", "101", "152"], help='which resnet 18,50,101,152')
        # ---------- Define Dataset ---------- #
        parser.add_argument('--sourceDataset', type=str, nargs = '+', choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = ["gta_train"])
        parser.add_argument('--targetDataset', type=str, nargs = '+',
                choices=["gta_train", "gta_val", "city_train", "city_val"],
                default = ["city_train"])
        # ---------- Optimizers ---------- #
        parser.add_argument('--opt', type=str, default="sgd", choices=['sgd', 'adam'],
                            help="network optimizer")
        parser.add_argument("--adjustLr", action="store_true",
                            help='whether you change lr')
        parser.add_argument('--lr_policy', type=str, default='lambda',
                            help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        # ---------- Train Details ---------- #
        parser.add_argument('--k', type=int, default=4,
                            help='how many steps to repeat the generator update')
        parser.add_argument("--nTimesDLoss", type=int, default=1)
        parser.add_argument("--bgLoss", action= "store_true",
                            help='whether you add background loss')
        parser.add_argument('--dLoss', type=str, default="diff",
                            choices=['mysymkl', 'symkl', 'diff'],
                            help="choose from ['mysymkl', 'symkl', 'diff']")
        parser.add_argument('--epochStart', type=int, default=1, 
                            help='the starting epoch count.')
        parser.add_argument('--nEpochStart', type=int, default=10, 
                            help='# of epoch at starting learning rate')
        parser.add_argument('--nEpochDecay', type=int, default=10, 
                            help='# of epoch to linearly decay learning rate to zero')
        # ---------- Experiment Setting ---------- #
        parser.add_argument('--name', type=str, default='mcd_da', 
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--displayWidth', type=int, default=3,
                            help='frequency of showing training results on screen')

        return parser
