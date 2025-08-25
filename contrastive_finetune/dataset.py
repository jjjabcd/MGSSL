import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from torch_geometric.data import Batch

from api.emb.base.dataset import BaseContrastiveDataset
from api.emb.gnn.finetune.loader import mol_to_graph_data_obj_simple

class GNNDataset(BaseContrastiveDataset):
    """ GNN-specific contrastive dataset. Handles SMILES to graph conversion. """

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        anchor_smi, pos_smi, neg_smi = row["anchor"], row["positive"], row["negative"]

        # Special handling for '[unused1]' token before Mol conversion
        anchor_mol = None if anchor_smi == '[unused1]' else Chem.MolFromSmiles(anchor_smi)
        positive_mol = None if pos_smi == '[unused1]' else Chem.MolFromSmiles(pos_smi)
        negative_mol = None if neg_smi == '[unused1]' else Chem.MolFromSmiles(neg_smi)

        # Filter out triplets with other invalid SMILES
        if (anchor_smi != '[unused1]' and not anchor_mol) or \
           (pos_smi != '[unused1]' and not positive_mol) or \
           (neg_smi != '[unused1]' and not negative_mol):
            return None

        anchor_graph = mol_to_graph_data_obj_simple(anchor_mol) if anchor_mol else None
        positive_graph = mol_to_graph_data_obj_simple(positive_mol) if positive_mol else None
        negative_graph = mol_to_graph_data_obj_simple(negative_mol) if negative_mol else None
        
        return {
            "anchor_graph": anchor_graph, "positive_graph": positive_graph, "negative_graph": negative_graph,
            "anchor_text": anchor_smi, "positive_text": pos_smi, "negative_text": neg_smi,
        }

def gnn_collate_fn(batch):
    """ Custom collate function for the GNNDataset. """
    batch = [item for item in batch if item is not None]
    if not batch: return None
    collated = {}
    
    # Batch graph objects, filtering out None values
    for key in ["anchor", "positive", "negative"]:
        graph_key = f"{key}_graph"
        graphs = [d[graph_key] for d in batch if d[graph_key] is not None]
        if graphs:
            collated[graph_key] = Batch.from_data_list(graphs)
    
    # Collect text
    for key in ["anchor", "positive", "negative"]:
        text_key = f"{key}_text"
        collated[text_key] = [d[text_key] for d in batch]

    return collated