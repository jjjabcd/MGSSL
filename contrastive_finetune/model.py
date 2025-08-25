import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from api.emb.gnn.finetune.model import GNN

class GNNForContrastive(nn.Module):
    """
    GNN model for contrastive learning, using the pre-trained GNN as a backbone.
    """
    def __init__(self, cfg):
        super().__init__()
        
        self.gnn = GNN(
            num_layer=cfg.num_layer, 
            emb_dim=cfg.emb_dim, 
            JK=cfg.JK, 
            drop_ratio=0, # No dropout during fine-tuning/inference
            gnn_type=cfg.gnn_type
        )
        self.pool = global_mean_pool

    def forward(self, data):
        """ Processes a batch of graph data. """
        node_representation = self.gnn(data.x, data.edge_index, data.edge_attr)
        graph_embedding = self.pool(node_representation, data.batch)
        return graph_embedding

    def load_pretrained_backbone(self, ckpt_path: str):
        """ Loads weights from a pre-trained checkpoint into the GNN backbone. """
        try:
            # The checkpoint for the GNN backbone is just the state_dict of the GNN class
            self.gnn.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        except FileNotFoundError:
            pass  # Train from scratch if checkpoint not found
        except Exception as e:
            pass  # Train from scratch if loading fails