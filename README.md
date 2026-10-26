<h1 align="center"> Domain-Discriminative Graph Foundation Model </a></h2>


### Run Script



The `__DDGFM.py` script implements the complete workflow including pretraining, downstream graph classification, and downstream node classification. The `utils.py` file contains definitions of all utility functions and classes.



Pre-training and downstream fine-tuning:
```bash
python  __DDGFM.py   --pre_dataset  PubMed_CiteSeer_Photo_amazonratings_romanempire_squirrel_cornell_Facebook_CoauthorCS_WikiCS    --adapt_dataset  Cora  
```





