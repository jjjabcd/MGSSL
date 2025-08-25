from api.emb.base.config import BaseConfig

class GNNConfig(BaseConfig):
    """
    Configuration for GNN, inheriting from BaseConfig.
    Adds GNN-specific architectural parameters and checkpoint path.
    """
    def __init__(self,
                 # Path to the pre-trained GNN backbone model
                 pretrained_ckpt_path: str,
                 
                 # GNN Architecture (from your model.py)
                 num_layer: int = 5,
                 emb_dim: int = 300,
                 JK: str = "last",
                 gnn_type: str = "gin",
                 
                 **kwargs):
        """
        Args:
            pretrained_ckpt_path (str): Path to the pre-trained GNN backbone checkpoint.
            num_layer (int): Number of GNN message passing layers.
            emb_dim (int): Embedding dimension of the GNN backbone. This is the final output dim.
            JK (str): Jumping knowledge connection type ("last", "sum", "max", "concat").
            gnn_type (str): The type of GNN layer to use (e.g., "gin", "gcn").
            **kwargs: Arguments for BaseConfig (e.g., train_csv_path, lr, epochs).
        """
        super().__init__(**kwargs)
        
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.JK = JK
        self.gnn_type = gnn_type

class GNNEmbeddingConfig:
    """
    Configuration for GNN embedding (without training parameters).
    """
    def __init__(self,
                 num_layer: int = 5,
                 emb_dim: int = 300,
                 JK: str = "last",
                 gnn_type: str = "gin",
                 device: str = "cuda:0",
                 embedding_batch_size: int = 256,
                 vectordb_batch_size: int = 1000):
        """
        Args:
            num_layer (int): Number of GNN message passing layers.
            emb_dim (int): Embedding dimension of the GNN backbone.
            JK (str): Jumping knowledge connection type.
            gnn_type (str): The type of GNN layer to use.
            device (str): Device to use for inference.
            embedding_batch_size (int): Batch size for embedding generation.
            vectordb_batch_size (int): Batch size for adding to vector database.
        """
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.JK = JK
        self.gnn_type = gnn_type
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self.vectordb_batch_size = vectordb_batch_size