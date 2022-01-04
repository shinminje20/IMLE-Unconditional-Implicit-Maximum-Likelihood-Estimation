import torch.nn as nn
from GeneratorArchitectures import *

def get_arch_args(args):
    """Returns the kwargs for building the architecture specified by [args]."""
    pass

def reduce_dim(x):
    """Returns [x] with its dimensionality reduced.
    """

def generate_camnet_code(bs, map_nc, code_nc, h, w, level):
    """Returns the code for IMLE sampling at level [level].

    Args:
    bs      -- batch size of current input
    map_nc  -- number of mapping channels of the model
    code_nc -- number of code channels of the model
    h       -- height of the current input's lowest-resolution image
    w       -- width of the current input's lowest-resolution image
    level   -- level of the CAMNet module have a code sampled
    """
    return torch.randn(bs, map_nc + code_nc * w * h * (2 ** (2 * l))).to(device)




def sample_imle_code(generator,
    targets,
    dci_num_comp_indices=10,
    dci_num_simp_indices=10,
    sample_perturbation_magnitude=1,
    block_size=100,
    thread_size=10
    num_samples_per_img=32,
    project_dim=1000,
    mini_batch_size=20,
    num_outer_iterations=5000):
    """
    Args:
    generator               -- a CAMNet generator
    targets                 -- list of all targets, one for each level at which
                                sampling takes place

    dci_num_comp_indices    --
    dci_num_simp_indices    --
    perturbation_magnitude  --
    block_size              --
    thread_size             --
    num_samples_per_img     --
    project_dim             --
    mini_batch_size         --
    num_outer_iterations    -- number of outer iterations for DCI query
    """



    for level in range(generator.num_levels):
        dci_db = DCI(project_dim, dci_num_comp_indices, dci_num_simp_indices,
            block_size, thread_size)
        cur_sampled_code = generate_camnet_code(bs, model.map_nc, model.code_nc,
            h, w, level)
        dci_db.add(x)



    best_sample_idx, _ = dci_db.query(target_feat, num_outer_iterations)










class Generator(nn.Module):

    def __init__(self, method):
        if method == "camnet":
            self.model = CAMNet(**get_arch_args(args))
            self.imle = True
        else:
            raise ValueError(f"Unknown architecture '{arch}'")

    def forward(self, input):
        if self.imle:
            pass
        else:
            return self.model(input)
