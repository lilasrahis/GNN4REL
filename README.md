# GNN4REL
Graph Neural Networks for Predicting Circuit Reliability Degradation. TCAD 2022, [paper](https://ieeexplore.ieee.org/document/9852805)

**Contact**
Lilas Alrahis (lma387@nyu.edu)

### Overview 
We employ graph neural networks (GNNs) to accurately estimate the impact of process variations and device aging on the delay of any path within a circuit. GNN4REL empowers designers to perform rapid and accurate reliability estimations without accessing transistor models, standard-cell libraries, or even STA; these components are all incorporated into the GNN model via training by the foundry.

### About This Repo
This repo contains the source code of the netlist-to-subgraphs transformation from our paper (TCAD '22, see the [Citation](#citation-and-acknowledgement) Section). The scripts released here parse the designs in technology-mapped Netlist format (FinFET 14nm). Please contact Lilas Alrahis (lma387@nyu.edu) if you wish to expand Verilog netlists parsing to handle different tech libs. We release the benchmarks and STA measurements to replicate the results presented in Fig.14 in our paper, i.e., predicting the average process-variation-induced degradation.

## Conversion to Graphs
**Benchmarks**

The `./perl_codes` directory contains the `adder`, `multiplier`, `max`, `b15`, and `b17` synthesized netlists used in (TCAD'22).

**Scripts**

The following scripts are required for the conversion:  
`./perl_codes/TheCircuit.pm`: a Perl module we create to ease circuit's parsing. This module is required by our parser `./perl_codes/netlist_to_subgraph_directed.pl`

`./perl_codes/netlist_to_subgraph_directed.pl`: a Perl script that reads one synthesized gate-level netlist at a time (or a number of netlists) in a given dataset and converts the dataset into a single graph. It assigns unique numerical IDs (0 to N-1) to the nodes (gates). N represents the total number of nodes (gates) in the dataset. It will create a directory under `../Path_PNA/data/` which includes:

- The extracted features will be dumped in `feat.txt`. The ith line in feat.txt represents the feature vector of the node ID = the ith line in `count.txt`
- The existence of an edge i between two vertices u and v is represented by the entry of ith line in `link.txt`
- The `cell.txt` file includes the mapping between node IDs and gate instances

**Running the Conversion**   
1) Modify line 6 in `./perl_codes/netlist_to_subgraph_directed.pl` and place the full path to `theCircuit.pm`.
2) Perform the conversion:  
    ```sh
    $ cd ./perl_codes
    $ perl netlist_to_subgraph_directed.pl -i test_adder -f test_adder -m 0 > log_adder.txt
    $ cd ../
    ```
## Degradation Estimation (Subgraph Regression)
**STA Results and Labels**
- We build degradation-aware libraries and perform STA. Here, we release the degradation information obtained from STA.
- The `./Path_PNA/data/degradation_info` direcotry contains the extracted 1000 timing-paths per design and the corresponding delay degradation.
- For example `./Path_PNA/data/degradation_info/adder` contains the following 4 files:
    - `paths.txt` the extracted timing paths and corresponding node IDs.
    - `adder_degradation_std.txt` the STD of the process-variation-induced degradation per path.
    - `adder_degradation_max.txt` the max process-variation-induced degradation per path.
    - `adder_degradation_avg.txt` the average process-variation-induced degradation per path.

**Example: Predict the Average Process-Variation-Induced Delay Degradation per Path**

To predict the average process-variation-induced delay degradation per adder path:
1) Copy the `./Path_PNA/data/degradation_info/adder/paths.txt` to the generated `./Path_PNA/data/test_adder`:
    ```sh
    $ cp ./Path_PNA/data/degradation_info/adder/paths.txt ./Path_PNA/data/test_adder/paths.txt
    ```
2) Copy the `./Path_PNA/data/degradation_info/adder/adder_degradation_avg.txt` to the generated `./Path_PNA/data/test_adder` as the label file:
    ```sh
    $ cp ./Path_PNA/data/degradation_info/adder/adder_degradation_avg.txt ./Path_PNA/data/test_adder/label.txt
    ```
3) Train the model and predict. The `./Path_PNA/Main.py` first extracts subgraphs around the timing-paths and then trains a PNA model.
    ```sh
    $ cd ./Path_PNA
    $ python Main.py  --no-parallel  --file-name test_adder --links-name link.txt  --hop 1  --filename Results_adder_average.txt  >  log_adder_average.txt
    ```

## Citation and Acknowledgement

If you find the code useful, please cite our paper:
* TCAD 2022:
```
@ARTICLE{gnn4rel,
author={Alrahis, Lilas and Knechtel, Johann and Klemme, Florian and Amrouch, Hussam and Sinanoglu, Ozgur},
journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
title={{GNN4REL}: Graph Neural Networks for Predicting Circuit Reliability Degradation},
year={2022},  volume={41},  number={11},  pages={3826-3837},
doi={10.1109/TCAD.2022.3197521}}
```

We owe many thanks to the authors of "Principal Neighbourhood Aggregation for Graph Nets", NeurIPS 2020, for making their PNA code available.
