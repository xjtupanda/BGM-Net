from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Hungarian Algorithm for bipartite-graph matching
# modified from DETR.
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the query and the video moments.
    
    """

    def __init__(self, cost_sim: float = 1):
        """Creates the matcher

        Params:
            cost_sim: This is the relative weight of similarity score in the matching cost
        """
        super().__init__()
        self.cost_sim = cost_sim
        assert cost_sim != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, query_labels):
        """ Performs the matching

        Params:
            outputs: This is a tensor that contains the similarity score tensor:
                shape: [num_captions, batch_size, num_moments] with similarity scores between queries and video moments.

            query_labels: This is a list of query's index to video (len(query_labels) = #captions)
                For example, the first video in a batch has 5 GT moments and 5 corresponding captions, the first 5 elements
                    of the list is [0, 0, 0, 0, 0, ...]
                    
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected queries (sorted in order)
                - index_j is the indices of the corresponding selected video moment (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_captions, num_moments) (so usually = num_captions)
        """
        num_captions, bs, num_moments = outputs.shape
        
        # We flatten to compute the cost matrices in a batch
        # TODO: consider Softmax on dim 0 for queries in the same video


        cost_sim = -outputs

        # Final cost matrix
        C = self.cost_sim * cost_sim.cpu()        # [num_captions, bs, num_moments]
        labels = np.array(query_labels)
        
        indices = [linear_sum_assignment(C[np.where(labels==i)[0], i]) for i in range(bs)]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

if __name__ == '__main__':
    Matcher = HungarianMatcher()
    outputs = torch.sigmoid(torch.randn((64, 4, 10)))   # bs=4 videos, each 10 clips, [20, 20, 20, 4] captions
    labels = [0 for _ in range(20)] + [1 for _ in range(20)] + [2 for _ in range(20)] + [3 for _ in range(4)]
    indices = Matcher(outputs, labels)  # list of len(bs), each is a tuple of 2 elements,
                                        # (cap_indices, moment_indices), both of len(num_caps) of this video
    print('debugging')