# Graph Neural Network Specifications

## Architectures
The study implements two primary GNN architectures using PyTorch Geometric:

### 1. Graph Convolutional Network (GCN)
A standard 2-layer GCN operating on the spatial domain. It aggregates node features from neighborhood structures defined by the trade matrices.
- Activation: LeakyReLU
- Regularization: Dropout (p=0.2)

### 2. Temporal Graph Network (`temporal_graph`)
An advanced architecture combining spatial graph convolutions with temporal sequence modeling. It utilizes an LSTM layer to encode temporal dynamics, followed by graph attention layers to model spatial contagion, and a final temporal attention mechanism over the sequence.

## Graph Construction
Nodes represent the 20 European countries. Edges represent bilateral trade dependencies. Adjacency matrices ($A_t$) are constructed dynamically for each time step $t$.

## Graph Variants Evaluated
To isolate the structural contribution of trade data, we systematically evaluate 8 graph variants:

1. `directed_trade`: Raw asymmetric import/export flows.
2. `log_trade`: Log-transformed trade flows to dampen magnitude disparities.
3. `import_dependence`: Trade flows normalized by the importing country's GDP.
4. `top_k_incoming`: Sparsified graph retaining only the top-K import partners per node.
5. `reversed`: Transposed adjacency matrix (testing directionality of contagion).
6. `undirected`: Symmetrized matrix (Imports + Exports).
7. `degree_preserving_random`: A random graph maintaining the exact degree distribution of the true trade network (null model).
8. `identity_no_trade`: An identity matrix $I$, assuming no spatial spillovers (spatial ablation).

## Rationale
The inclusion of `identity_no_trade` and `degree_preserving_random` serves as critical ablation studies. If trade-based graphs fail to statistically outperform these null structures, the hypothesized utility of explicit topological modeling for inflation contagion is rejected.
