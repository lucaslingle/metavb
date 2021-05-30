# metavb
Official implementation of experiments from 'Meta-Learning with Variational Bayes'. 
The paper describes a scalable method for designing fast-adapting latent variable models, 
in a deep generative setting. 

## Dependencies
```
python3.6
tensorflow==1.13.1
tensorflow_probability==0.6.0
tensorflow_datasets==1.0.2
numpy==1.16.4
scipy==1.3.0
matplotlib==3.1.0
moviepy==1.0.0
```

## Getting started
#### Step 1.

Install the dependencies. I suggest creating separate python environment for this project.
You can do this easily using virtualenv or Anaconda 3.

#### Step 2. 

Clone this repo, and navigate to it. 

#### Step 3. 

Activate the virtualenv or conda environment. 
To run the code for a given experiment, navigate to the corresponding subfolder and follow the instructions in the README for that subfolder.

```
The first subfolder corresponds to the quantitative experiments for benchmarking the inference algorithms. 
The second subfolder corresponds to quantitative experiments for benchmarking various deep generative models.
The third subfolder contains code for our more sophisticated models, which serve as the basis of our qualitative experiments, 
    including 'generating from memory', 'iterative reading' and 'resizing memory'.
```


## Misc.

Please note that this is a research codebase and as most of our efforts were focused on algorithmic improvements,
the resulting code is somewhat messy. For simplicity, we provide the code as-is, rather than refactoring.

If you find the code or paper helpful in your research, please cite the following bibtex:
```
@article{Lingle2021,
    title = {Meta-Learning with Variational Bayes},
    author = {Lucas D. Lingle},
    journal = {arXiv preprint arxiv:2103.02265},
    year = {2021}
}
```

