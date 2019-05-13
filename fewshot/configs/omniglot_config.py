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


@RegisterConfig("omniglot", "basic")
class BasicConfig(object):
  """Standard CNN on Omniglot with prototypical layer."""

  def __init__(self):
    self.name = "omniglot_basic"
    self.model_class = "basic"
    self.height = 28
    self.width = 28
    self.num_channel = 1
    self.steps_per_valid = 1000
    self.steps_per_log = 100
    self.steps_per_save = 1000
    self.filter_size = [[3, 3, 1, 64]] + [[3, 3, 64, 64]] * 3
    self.strides = [[1, 1, 1, 1]] * 4
    self.pool_fn = ["max_pool"] * 4
    self.pool_size = [[1, 2, 2, 1]] * 4
    self.pool_strides = [[1, 2, 2, 1]] * 4
    self.conv_act_fn = ["relu"] * 4
    self.conv_init_method = None
    self.conv_init_std = [1.0e-2] * 4
    self.wd = 5e-5
    self.learn_rate = 1e-3
    self.normalization = "batch_norm"
    self.lr_scheduler = "fixed"
    self.lr_decay_steps = [4000, 6000, 8000, 12000, 14000, 16000, 18000, 20000,
                           22000, 24000, 26000, 28000]
    self.max_train_steps = 20000
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(len(self.lr_decay_steps))))
    self.similarity = "euclidean"

@RegisterConfig("omniglot", "refine")
class RefineConfig(BasicConfig):
  def __init__(self):
    super(RefineConfig, self).__init__()
    self.name = "omniglot_refine"
    self.model_class = "refine"



@RegisterConfig("omniglot", "basic-pretrain")
class BasicTestConfig(BasicConfig):

  def __init__(self):
    super(BasicTestConfig, self).__init__()
    self.lr_decay_steps = [2000, 2500, 3000, 3500]
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(1, len(self.lr_decay_steps) + 1)))
    self.max_train_steps = 4000


@RegisterConfig("omniglot", "basic-test")
class BasicPretrainConfig(BasicConfig):

  def __init__(self):
    super(BasicPretrainConfig, self).__init__()
    self.lr_decay_steps = [30, 60, 90]
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(1, len(self.lr_decay_steps) + 1)))
    self.max_train_steps = 100
    self.steps_per_valid = 10
    self.steps_per_log = 10
    self.steps_per_save = 10

@RegisterConfig("omniglot", "basic-VAT")
class BasicVATConfig(BasicConfig):

  def __init__(self):
    super(BasicVATConfig, self).__init__()
    self.name = "omniglot_basic-VAT"
    self.model_class = "basic-VAT"
    self.VAT_step_size = 1.0
    self.labeled_weight = 0.2
    self.VAT_weight = 1.0


@RegisterConfig("omniglot", "basic-VAT-ENT")
class BasicVAT_ENTConfig(BasicVATConfig):
  def __init__(self):
    super(BasicVAT_ENTConfig, self).__init__()
    self.name = "omniglot_basic-VAT-ENT"
    self.model_class = "basic-VAT-ENT"
    self.ENT_weight = 1.0
    self.VAT_ENT_step_size = 1.0
    self.max_train_steps = 20000

@RegisterConfig("omniglot", "basic-ENT")
class BasicENTConfig(BasicConfig):
  def __init__(self):
    super().__init__()
    self.name = "omniglot_basic-ENT"
    self.model_class = "basic-ENT"
    self.ENT_weight = 1.5
    self.ENT_step_size = 1.0
    self.max_train_steps = 20000


@RegisterConfig("omniglot", "basic-matching-ENT")
class BasicMatchingENTConfig(BasicConfig):
  def __init__(self):
    super().__init__()
    self.name = "omniglot_basic-matching-ENT-CE"
    self.model_class = "basic-matching-ENT"
    self.ENT_weight = 1.5
    self.ENT_step_size = 1.0
    self.max_train_steps = 20000
    self.stop_grad_unlbl = False
    self.stop_grad_lbl  = True
    self.stop_grad_lbl_logits = True
    self.match_to_labeled = False
    self.non_matching = False


@RegisterConfig("omniglot", "basic-ENT-graphVAT")
class BasicENTGraphVATConfig(BasicVATConfig):
  def __init__(self):
    super().__init__()
    self.name = "omniglot_basic-ENT-graphVAT"
    self.model_class = "basic-ENT-graphVAT"
    self.ENT_weight = 1.0
    self.ENT_weight = 1.5
    self.ENT_step_size = 1.0

@RegisterConfig("omniglot", "basic-VAT-prototypes")
class BasicVAT_PrototypesConfig(BasicVATConfig):

  def __init__(self):
    super(BasicVAT_PrototypesConfig, self).__init__()
    self.name = "omniglot_basic-VAT-prototypes"
    self.model_class = "basic-VAT-prototypes"
    self.VAT_weight = 1

@RegisterConfig("omniglot", "VAT-refine")
class RefineVAT(BasicVATConfig):
  def __init__(self):
    super(BasicVATConfig, self).__init__()
    self.name = "omniglot_VAT-refine"
    self.model_class = "VAT-refine"
    self.VAT_weight = 1.0


@RegisterConfig("omniglot", "VAT-refine-prototypes")
class RefineVATPrototypes(BasicVATConfig):
  def __init__(self):
    super(BasicVATConfig, self).__init__()
    self.name = "omniglot_VAT-refine-prototypes"
    self.model_class = "VAT-refine-prototypes"
    self.VAT_weight = 1.0
    self.inference_step_size = 0.09
    self.num_steps = 10
    self.VAT_eps = 4.0

@RegisterConfig("omniglot", "kmeans-refine")
class KMeansRefineConfig(BasicConfig):

  def __init__(self):
    super(KMeansRefineConfig, self).__init__()
    self.name = "omniglot_kmeans-refine"
    self.model_class = "kmeans-refine"
    self.num_cluster_steps = 1


@RegisterConfig("omniglot", "kmeans-refine-VAT-ENT")
class KMeansRefineVAT_ENTConfig(BasicVAT_ENTConfig):

  def __init__(self):
    super(KMeansRefineVAT_ENTConfig, self).__init__()
    self.name = "omniglot_kmeans-refine-VAT-ENT"
    self.model_class = "kmeans-refine-VAT-ENT"
    self.num_cluster_steps = 1


@RegisterConfig("omniglot", "kmeans-refine-test")
class KMeansRefineTestConfig(KMeansRefineConfig):

  def __init__(self):
    super(KMeansRefineTestConfig, self).__init__()
    self.lr_decay_steps = [30, 60, 90]
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(1, len(self.lr_decay_steps) + 1)))
    self.max_train_steps = 100
    self.steps_per_valid = 10
    self.steps_per_log = 10
    self.steps_per_save = 10


@RegisterConfig("omniglot", "kmeans-refine-radius")
class KMeansRefineRadiusConfig(BasicVAT_ENTConfig):

  def __init__(self):
    super(KMeansRefineRadiusConfig, self).__init__()
    self.name = "omniglot_kmeans-refine-radius"
    self.model_class = "kmeans-refine-radius"
    self.num_cluster_steps = 1

@RegisterConfig("omniglot", "kmeans-radius")
class KMeansRadiusConfig(KMeansRefineRadiusConfig):
  def __init__(self):
    super().__init__()
    self.name = "omniglot_kmeans-radius"
    self.model_class = "kmeans-radius"


@RegisterConfig("omniglot", "kmeans-refine-radius-test")
class KMeansRefineRadiusTestConfig(KMeansRefineRadiusConfig):

  def __init__(self):
    super(KMeansRefineRadiusTestConfig, self).__init__()
    self.lr_decay_steps = [30, 60, 90]
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(1, len(self.lr_decay_steps) + 1)))
    self.max_train_steps = 100
    self.steps_per_valid = 10
    self.steps_per_log = 10
    self.steps_per_save = 10


@RegisterConfig("omniglot", "kmeans-refine-mask")
class KMeansRefineMaskConfig(BasicConfig):

  def __init__(self):
    super(KMeansRefineMaskConfig, self).__init__()
    self.name = "omniglot_kmeans-refine-mask"
    self.model_class = "kmeans-refine-mask"
    self.num_cluster_steps = 1


@RegisterConfig("omniglot", "kmeans-refine-mask-test")
class KMeansRefineMaskTestConfig(KMeansRefineMaskConfig):

  def __init__(self):
    super(KMeansRefineMaskTestConfig, self).__init__()
    self.lr_decay_steps = [30, 60, 90]
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(1, len(self.lr_decay_steps) + 1)))
    self.max_train_steps = 100
    self.steps_per_valid = 10
    self.steps_per_log = 10
    self.steps_per_save = 10


@RegisterConfig("omniglot", "persistent")
class PersistentConfig(BasicConfig):

  def __init__(self):
    super().__init__()
    self.name = "omniglot_persistent"
    self.model_class = "persistent"
    self.persistent_reg = None
    self.trainable = True
    self.n_train_classes = 4112
    self.proto_dim = 64
    self.classification_weight = 0.005


@RegisterConfig("omniglot", "persistent-SSL")
class PersistentSSLConfig(BasicConfig):

  def __init__(self):
    super().__init__()
    self.name = "omniglot_persistent-SSL"
    self.model_class = "persistent-SSL"
    self.persistent_reg = None
    self.trainable = True
    self.n_train_classes = 4112
    self.proto_dim = 64
    self.classification_weight = 0.005
    self.VAT_weight = 1.0
    self.ENT_weight = 1.0


@RegisterConfig("omniglot", "basic-pairwise")
class PairwiseConfig(BasicConfig):

  def __init__(self):
    super().__init__()
    self.name = "omniglot_basic_pairwise"
    self.model_class = "basic-pairwise"


@RegisterConfig("omniglot", "pairwise-VAT-ENT")
class PairwiseVAT_ENTConfig(BasicVATConfig):
  def __init__(self):
    super(PairwiseVAT_ENTConfig, self).__init__()
    self.name = "omniglot_pairwise-VAT-ENT"
    self.model_class = "pairwise-VAT-ENT"
    self.ENT_weight = 0.5
    self.max_train_steps = 30000