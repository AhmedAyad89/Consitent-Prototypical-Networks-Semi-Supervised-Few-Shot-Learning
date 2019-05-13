# Copyright (c) 2018 Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell,
# Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richars S. Zemel.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
from fewshot.configs.config_factory import RegisterConfig


@RegisterConfig("mini-imagenet", "basic")
class BasicConfig(object):
  """Standard CNN with prototypical layer."""

  def __init__(self):
    self.name = "mini-imagenet_basic"
    self.model_class = "basic"
    self.height = 84
    self.width = 84
    self.num_channel = 3
    self.steps_per_valid = 2000
    self.steps_per_log = 100
    self.steps_per_save = 2000
    self.filter_size = [[3, 3, 3, 64]] + [[3, 3, 64, 64]] * 3
    self.strides = [[1, 1, 1, 1]] * 4
    self.pool_fn = ["max_pool"] * 4
    self.pool_size = [[1, 2, 2, 1]] * 4
    self.pool_strides = [[1, 2, 2, 1]] * 4
    self.conv_act_fn = ["relu"] * 4
    self.conv_init_method = None
    self.conv_init_std = [1.0e-2] * 4
    self.wd = 5e-5
    self.learn_rate = 1e-4
    self.normalization = "batch_norm"
    self.lr_scheduler = "fixed"
    self.max_train_steps = 100000
    self.lr_decay_steps = list(range(0, self.max_train_steps, 20000)[1:])
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x, range(
            len(self.lr_decay_steps))))
    self.similarity = "euclidean"


@RegisterConfig("mini-imagenet", "basic-pretrain")
class BasicPretrainConfig(BasicConfig):

  def __init__(self):
    super(BasicPretrainConfig, self).__init__()
    self.max_train_steps = 4000
    self.lr_decay_steps = [2000, 2500, 3000, 3500]
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(1,
                  len(self.lr_decay_steps) + 1)))
    self.similarity = "euclidean"


@RegisterConfig("mini-imagenet", "kmeans-refine")
class KMeansRefineConfig(BasicConfig):

  def __init__(self):
    super(KMeansRefineConfig, self).__init__()
    self.name = "mini-imagenet_kmeans-refine"
    self.model_class = "kmeans-refine"
    self.num_cluster_steps = 1




@RegisterConfig("mini-imagenet", "kmeans-refine-mask")
class KMeansRefineDistractorMSV3Config(BasicConfig):

  def __init__(self):
    super(KMeansRefineDistractorMSV3Config, self).__init__()
    self.name = "mini-imagenet_kmeans-refine-mask"
    self.model_class = "kmeans-refine-mask"
    self.num_cluster_steps = 1


@RegisterConfig("mini-imagenet", "basic-LP")
class BasicLP(BasicConfig):

  def __init__(self):
    super(BasicLP, self).__init__()
    self.name = "mini-imagenet_basic-LP"
    self.model_class = "basic-LP"


@RegisterConfig("mini-imagenet", "basic-VAT")
class BasicVAT(BasicConfig):

  def __init__(self):
    super(BasicVAT, self).__init__()
    self.name = "mini-imagenet_basic-VAT"
    self.model_class = "basic-VAT"
    self.VAT_step_size = 2.5
    self.labeled_weight = 0.2
    self.VAT_weight = 1.0

@RegisterConfig("mini-imagenet", "basic-VAT-ENT")
class BasicVAT_ENTConfig(BasicVAT):
  def __init__(self):
    super(BasicVAT_ENTConfig, self).__init__()
    self.name = "mini-imagenet_basic-VAT-ENT-multi"
    self.model_class = "basic-VAT-ENT"
    self.ENT_weight = 1.0
    self.VAT_ENT_step_size = 2.5
    self.max_train_steps = 120000

@RegisterConfig("mini-imagenet", "basic-ENT")
class BasicENTConfig(BasicConfig):
  def __init__(self):
    super(BasicENTConfig, self).__init__()
    self.name = "mini-imagenet_basic-ENT"
    self.model_class = "basic-ENT"
    self.ENT_weight = 0.75
    self.ENT_step_size = 2.5
    self.max_train_steps = 120000


@RegisterConfig("mini-imagenet", "basic-matching-ENT")
class BasicMatchingENTConfig(BasicConfig):
  def __init__(self):
    super().__init__()
    self.name = "mini-imagenet_basic-matching-ENT"
    self.model_class = "basic-matching-ENT"
    self.ENT_weight = 0.75
    self.ENT_step_size = 2.5
    self.max_train_steps = 120000
    self.stop_grad_unlbl = False
    self.stop_grad_lbl  = True
    self.stop_grad_lbl_logits = True
    self.match_to_labeled = False

@RegisterConfig("mini-imagenet", "basic-ENT-graphVAT")
class BasicENTGraphVATConfig(BasicVAT):
  def __init__(self):
    super().__init__()
    self.name = "mini-imagenet_basic-ENT-graphVAT"
    self.model_class = "basic-ENT-graphVAT"
    self.ENT_weight = 1.0
    self.ENT_weight = 1.5
    self.ENT_step_size = 1.0


@RegisterConfig("mini-imagenet-all", "basic-VAT-ENT")
class BasicVAT_ENTConfigAll(BasicVAT):
  def __init__(self):
    super().__init__()
    self.name = "mini-imagenet_basic-VAT-ENT-multi-all"
    self.model_class = "basic-VAT-ENT"
    self.ENT_weight = 1.0
    self.VAT_ENT_step_size = 3.0
    self.max_train_steps = 120000
    self.labeled_weight = 0.0



@RegisterConfig("mini-imagenet", "kmeans-refine-radius")
class KMeansRefineDistractorConfig(BasicVAT_ENTConfig):

  def __init__(self):
    super(KMeansRefineDistractorConfig, self).__init__()
    self.name = "mini-imagenet_kmeans-refine-radius"
    self.model_class = "kmeans-refine-radius"
    self.num_cluster_steps = 1


@RegisterConfig("mini-imagenet", "kmeans-radius")
class KMeansRadiusConfig(BasicVAT_ENTConfig):
  def __init__(self):
    super().__init__()
    self.name = "mini-imagenet_kmeans-radius"
    self.model_class = "kmeans-radius"
    self.num_cluster_steps = 1


@RegisterConfig("mini-imagenet", "VAT-refine-prototypes")
class RefineVATPrototypes(BasicVAT):
  def __init__(self):
    super(BasicVAT, self).__init__()
    self.name = "mini-imagenet_VAT-refine-prototypes"
    self.model_class = "VAT-refine-prototypes"
    self.VAT_weight = 1.0
    self.ENT_weight = 1.0
    self.inference_step_size = 0.005
    self.num_steps = 10
    self.VAT_eps = 4.0
    
    
@RegisterConfig("mini-imagenet", "pairwise-VAT-ENT")
class PairwiseVAT_ENTConfig(BasicVAT_ENTConfig):
  def __init__(self):
    super(PairwiseVAT_ENTConfig, self).__init__()
    self.name = "mini-imagenet_pairwise-VAT-ENT"
    self.model_class = "pairwise-VAT-ENT"


@RegisterConfig("mini-imagenet", "kmeans-refine-VAT-ENT")
class KMeansRefineVAT_ENTConfig(BasicVAT_ENTConfig):

  def __init__(self):
    super(KMeansRefineVAT_ENTConfig, self).__init__()
    self.name = "mini-imagenet_kmeans-refine-VAT-ENT"
    self.model_class = "kmeans-refine-VAT-ENT"
    self.num_cluster_steps = 1
    self.VAT_ENT_step_size = 2.0
