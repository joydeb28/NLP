# Example of embeddings using ELMo on TensorFlow Hub
This repository is an example of word embeddings using ELMo (Embeddings from Language Models) on TensorFlow Hub.

The comparison_test module can calculate cosine similarity between given two sentences.


### Usage
- In case of using exmaple.sh
```bash
sh example.sh
```

- In case of executing on your terminal
```bash
python comparison_test.py [sentence1] [sentence2]
```

### Example
```bash
bash-3.2$ python comparison_test.py "people read the book" "the book people read"

[Cosine Similarity]
"people read the book" vs "the book people read"
ELMo: 0.83875865
NNLM: 1.0
```
