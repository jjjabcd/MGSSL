import torch
import numpy as np
from typing import List
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from rdkit import Chem
from tqdm import tqdm

from api.emb.base.embedder import BaseEmbedder
from api.emb.gnn.contrastive_finetune.config import GNNConfig
from api.emb.gnn.contrastive_finetune.model import GNNForContrastive
from api.emb.gnn.finetune.loader import mol_to_graph_data_obj_simple

class GNNEmbedder(BaseEmbedder):
    """ Embedder for GNN models. """
    def __init__(self, cfg: GNNConfig, ckpt_path: str, device: str = None, load_backbone_only: bool = False):
        super().__init__(cfg, ckpt_path, device)
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = GNNForContrastive(self.cfg).to(self.device)
        
        if load_backbone_only:
            self.model.load_pretrained_backbone(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=True)['model']
            self.model.load_state_dict(state_dict)
            print(f"Loaded fine-tuned GNN weights from {ckpt_path}")
            
        self.model.eval()

    @torch.no_grad()
    def embed_one(self, smiles: str) -> np.ndarray:
        if smiles == '[unused1]':
            fixed_vec_val = 1.0 / self.cfg.emb_dim
            return np.full(self.cfg.emb_dim, fill_value=fixed_vec_val, dtype=np.float32)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.cfg.emb_dim, dtype=np.float32)

        graph = mol_to_graph_data_obj_simple(mol)
        graph_batch = Batch.from_data_list([graph]).to(self.device)
        embedding = self.model(graph_batch)
        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed(self, smiles_list: List[str], batch_size: int = 256) -> np.ndarray:
        """
        Embeds a list of SMILES strings in batches, with special handling for '[unused1]'.
        """
        print(f"Processing {len(smiles_list)} SMILES with batch size {batch_size}...")
        final_embeddings = np.zeros((len(smiles_list), self.cfg.emb_dim), dtype=np.float32)
        
        fixed_vec_val = 1.0 / self.cfg.emb_dim
        fixed_vector = np.full(self.cfg.emb_dim, fill_value=fixed_vec_val, dtype=np.float32)
        
        regular_graphs = []
        regular_indices = []

        for i, smi in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Converting SMILES to molecular graphs"):
            if smi == '[unused1]':
                final_embeddings[i] = fixed_vector
            else:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    regular_graphs.append(mol_to_graph_data_obj_simple(mol))
                    regular_indices.append(i)
                # If MolFromSmiles fails, it will be left as a zero vector, which is acceptable.

        if regular_graphs:
            print(f"Processing {len(regular_graphs)} valid molecules in batches...")
            # Use PyG's DataLoader for efficient batching of graph objects
            loader = DataLoader(regular_graphs, batch_size=batch_size, shuffle=False)
            
            # Keep track of the current position in the regular_embeddings
            current_pos = 0
            for batch in tqdm(loader, desc="GNN Embedding", total=len(loader), leave=True):
                batch = batch.to(self.device)
                batch_embeddings = self.model(batch).cpu().numpy()
                
                num_in_batch = batch.num_graphs
                # Get the original indices for this batch
                batch_original_indices = regular_indices[current_pos : current_pos + num_in_batch]
                # Place embeddings into the final array
                final_embeddings[batch_original_indices] = batch_embeddings
                
                current_pos += num_in_batch
            
        print(f"Embedding completed. Shape: {final_embeddings.shape}")
        return final_embeddings