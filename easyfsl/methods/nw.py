"""
See original implementation at
https://github.com/facebookresearch/low-shot-shrink-hallucinate
"""
from typing import Optional
import copy

import torch
from torch import Tensor, nn

from .few_shot_classifier import FewShotClassifier
from nwhead.nw import NWNet


class NW(FewShotClassifier):
    """
    Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra.
    "Matching networks for one shot learning." (2016)
    https://arxiv.org/pdf/1606.04080.pdf

    Matching networks extract feature vectors for both support and query images. Then they refine
    these feature by using the context of the whole support set, using LSTMs. Finally they compute
    query labels using their cosine similarity to support images.

    Be careful: while some methods use Cross Entropy Loss for episodic training, Matching Networks
    output log-probabilities, so you'll want to use Negative Log Likelihood Loss.
    """

    def __init__(
        self,
        *args,
        num_classes: int,
        support_set: Optional[Tensor] = None,
        nis_scalar: float = -10,
        class_dropout: float = 0,
        fine_tuning_steps: int = 0,
        fine_tuning_lr: float = 1e-4,
        kernel_type: str = 'euclidean',
        debug_mode: bool = False,
        **kwargs,
    ):
        """
        Build Matching Networks by calling the constructor of FewShotClassifier.
        Args:
            feature_dimension: dimension of the feature vectors extracted by the backbone.
            support_encoder: module encoding support features. If none is specific, we use
                the default encoder from the original paper.
            query_encoder: module encoding query features. If none is specific, we use
                the default encoder from the original paper.
        """
        super().__init__(*args, **kwargs)

        # self.feature_dimension = feature_dimension

        self.nwnet = NWNet(self.backbone, 
                        num_classes,
                        # feat_dim=self.feature_dimension,
                        support_dataset=support_set,
                        kernel_type=kernel_type,
                        # subsample_classes=args.subsample_classes,
                        # proj_dim=args.proj_dim,
                        debug_mode=debug_mode,
                        # do_held_out_training=args.do_held_out_training,
                        # held_out_class=held_out_class,
                        # use_nis_embedding=args.use_nis_embedding,
                        nis_scalar=nis_scalar,
                        class_dropout=class_dropout,
                        # cl2n=args.cl2n,
                        device='cuda')

        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract features from the support set with full context embedding.
        Store contextualized feature vectors, as well as support labels in the one hot format.

        Args:
            support_images: images of the support set of shape (n_support, **image_shape)
            support_labels: labels of support set images of shape (n_support, )
        """
        self.support_images = support_images
        self.support_labels = support_labels

    def forward(self, query_images: Tensor, query_labels: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        Predict query labels based on their cosine similarity to support set features.
        Classification scores are log-probabilities.

        Args:
            query_images: images of the query set of shape (n_query, **image_shape)
        Returns:
            a prediction of classification scores for query images of shape (n_query, n_classes)
        """
        if self.training:
            return self.nwnet(query_images, query_labels)
        else:
            task_support_data=(self.support_images, self.support_labels, None)
            batch_size = self.support_images.shape[0]
            
            # Fine-tune the backbone on the support set
            # Make a copy of the model to avoid modifying the original one
            temp_model = copy.deepcopy(self.nwnet)
            temp_model.train()
            if self.fine_tuning_steps > 0:
                with torch.enable_grad():
                    optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.fine_tuning_lr)
                    for _ in range(self.fine_tuning_steps):
                        perm = torch.randperm(batch_size)
                        support_images, support_labels = self.support_images[perm], self.support_labels[perm]
                        qinputs, sinputs = support_images[:batch_size//2], support_images[batch_size//2:]
                        qlabels, slabels = support_labels[:batch_size//2], support_labels[batch_size//2:]
                        sdata = (sinputs, slabels, None)
                        log_probs = temp_model(qinputs, qlabels, support_data=sdata)
                        loss = nn.functional.nll_loss(
                            log_probs, qlabels 
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            temp_model.eval()
            return temp_model.predict(query_images, mode='full', 
                                      support_data=task_support_data)


    @staticmethod
    def is_transductive() -> bool:
        return False
