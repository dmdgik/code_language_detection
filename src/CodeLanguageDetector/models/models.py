from torch import nn, Tensor
from typing import List, Dict
from transformers import AutoModel


class CodeLanguageModelLSTM(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        embedding_dim: int,
        hid_size: int,
        n_layers: int,
        mlp_layers: List[int],
    ) -> None:
        """
        TGLanguageModelLSTM constructor.

        Args:
            n_tokens (int): Number of tokens in the vocabulary.
            embedding_dim (int): Dimension of the token embeddings.
            hid_size (int): Hidden size of the LSTM.
            mlp_layers (list): List of dimensions for the MLP layers.
            n_layers (int, optional): Number of LSTM layers. Default is 4.
            
        """
        super(CodeLanguageModelLSTM, self).__init__()

        assert len(mlp_layers) >= 1, "As minimum 1 linear layer needed (out layer)"

        self.content_emb = nn.Embedding(n_tokens, embedding_dim)
        self.content_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hid_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(hid_size, mlp_layers[0]))
        for i in range(len(mlp_layers) - 1):
            self.linears.append(nn.ReLU())
            self.linears.append(nn.BatchNorm1d(mlp_layers[i]))
            self.linears.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))

        nn.init.xavier_uniform_(self.content_lstm.weight_ih_l0)
        nn.init.orthogonal_(self.content_lstm.weight_hh_l0)
        nn.init.constant_(self.content_lstm.bias_ih_l0, 0)
        nn.init.constant_(self.content_lstm.bias_hh_l0, 0)

    def forward(self, inp: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass of the TGLanguageModelLSTM.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.

        """
        x = inp["input_ids"]
        x = self.content_emb(x)
        _, (x, _) = self.content_lstm(x)
        x = x[-1]

        for module in self.linears:
            x = module(x)

        return x


class CodeLanguageModelBERT(nn.Module):
    def __init__(
        self, 
        bert_model_name: str, 
        mlp_layers:  List[int] = [768, 512, 256, 84]
    ) -> None:
        """
        TGLanguageModelBERT constructor.

        Args:
            bert_model_name (str): Name of the pre-trained BERT model.
            mlp_layers (List[int]): List of dimensions for the MLP layers. Default is [768, 512, 256, 84].

        """
        super(CodeLanguageModelBERT, self).__init__()

        assert len(mlp_layers) >= 2, "As minimum 2 linear layers needed"

        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(mlp_layers[0], mlp_layers[1]))
        for i in range(len(mlp_layers) - 2):
            self.linears.append(nn.ReLU())
            self.linears.append(nn.BatchNorm1d(mlp_layers[i + 1]))
            self.linears.append(nn.Linear(mlp_layers[i + 1], mlp_layers[i + 2]))

    def forward(self, inp: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass of the TGLanguageModelBERT.

        Args:
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Attention mask.

        Returns:
            Tensor: Output tensor.

        """
        input_ids = inp["input_ids"]
        attention_mask = inp["attention_mask"]
        bert_out = self.bert_model(input_ids, attention_mask)
        x = bert_out.pooler_output

        for module in self.linears:
            x = module(x)

        return x
