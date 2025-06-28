# Molecular Bioavailability Prediction GNN+MLP
This repository provides a complete pipeline for predicting molecular activity (binary classification) using graph neural networks. 
Specifically, it targets the NR-AR (Androgen Receptor) endpoint from the Tox21 dataset to track bioactivity. Molecules are represented 
as graphs and learned using AttentiveFP and a custom GNN+MLP model implemented in PyTorch Geometric and RDKit. The reason I chose to 
do it this way was because I can use the GNN to learn atom-level representations, where each layer progressively encodes an atoms local 
chemical enviornment creating an embedding. The embeddings are then pooled to create a graph level representation for the whole 
molecule.The graph is then passed into the MLP which consists of several non-linear layers which reads the molecule embedding and the 
MLP predicts binary bioactivity.

## How to Run (Visual Studio Code)
To get this project to work follow these steps:
1. Open in Visual Studio Code.
2. Click the search bar at the top of the project.
3. Navigate to "Show and run Commands" and select.
4. Navigate to "Python: Create Enviorment".
5. Create a venv enviroment.
6. pip install any libraries as needed.
7. Run the project with the python command through terminal.

