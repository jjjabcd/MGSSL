import torch
from api.emb.base.trainer import BaseTrainer
from api.emb.gnn.contrastive_finetune.dataset import GNNDataset, gnn_collate_fn
from api.emb.gnn.contrastive_finetune.model import GNNForContrastive

class GNNTrainer(BaseTrainer):
    """ Trainer for the GNN model, inheriting from BaseTrainer. """
    
    def __init__(self, cfg):
        super().__init__(cfg, GNNDataset, GNNForContrastive, gnn_collate_fn)

    def _get_fixed_vector(self):
        """Creates the fixed vector for '[unused1]' for the GNN."""
        emb_dim = self.cfg.emb_dim
        fixed_vec_val = 1.0 / emb_dim
        return torch.full((emb_dim,), fill_value=fixed_vec_val, device=self.device)

    def _forward(self, batch):
        """ Forward pass for a GNN batch. """
        anc_graphs = batch.get('anchor_graph')
        pos_graphs = batch.get('positive_graph')
        neg_graphs = batch.get('negative_graph')
        
        anc_texts = batch['anchor_text']
        pos_texts = batch['positive_text']
        neg_texts = batch['negative_text']
        
        batch_size = len(anc_texts)
        anc = torch.zeros(batch_size, self.cfg.emb_dim, device=self.device)
        pos = torch.zeros(batch_size, self.cfg.emb_dim, device=self.device)
        neg = torch.zeros(batch_size, self.cfg.emb_dim, device=self.device)

        if anc_graphs: anc[ [i for i, smi in enumerate(anc_texts) if smi != '[unused1]'] ] = self.model(anc_graphs.to(self.device))
        if pos_graphs: pos[ [i for i, smi in enumerate(pos_texts) if smi != '[unused1]'] ] = self.model(pos_graphs.to(self.device))
        if neg_graphs: neg[ [i for i, smi in enumerate(neg_texts) if smi != '[unused1]'] ] = self.model(neg_graphs.to(self.device))

        fixed_vector = self._get_fixed_vector()
        for i, smi in enumerate(anc_texts):
            if smi == '[unused1]': anc[i, :] = fixed_vector
        for i, smi in enumerate(pos_texts):
            if smi == '[unused1]': pos[i, :] = fixed_vector
        for i, smi in enumerate(neg_texts):
            if smi == '[unused1]': neg[i, :] = fixed_vector
        
        return anc, pos, neg