# LinkedNN

Neural network for extracting LD features from SNPs

- [Installation](#installation)
- [Usage](#usage)
- [Vignette](#vignette)

---




<h2 id="installation">Installation</h2>

#### Quick start
```
pip install linkedNN
```

To test the installation you can apply the pretrained model from the paper to predict from a simulated dataset:

```bash
$ git clone https://github.com/the-smith-lab/LinkedNN.git
$ linkedNN --wd LinkedNN/Example_data/ --seed 1 --predict
using saved model from epoch 438
	test indices 0 to 0 out of 1
target 0 MRAE (no-logged): 0.136
target 1 MRAE (no-logged): 0.259
target 2 MRAE (no-logged): 0.289
```

The LD layer by itself can be accessed using:

```
from linkedNN.models import ld_layer
```

GPU compatibility: The code should work out of the box on a CPU, but to train on GPUs you need to sync torch with the particular CUDA version on your computer:

```
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### GSL installation
There may be additional requirements depending on the specific platform.
In particular, installation of GNU Scientific Library (GSL) is sometimes needed for running simulations with `msprime`.
See the `msprime` documentation---https://tskit.dev/msprime/docs/stable/installation.html---for up-to-date instructions.








---
<h2 id="usage">Usage</h2>

The following are explanations of command-line flags for `linkedNN`.

#### Preprocessing
The program trains on datasets simulated usins `msprime` or `SLiM`. Before training, simulated tree sequences are preprocessed to (i) add mutations, (ii) sample SNPs, and (iii) write binary files.
This can be applied to individual simulations, a range of simulation ID's, or all simulations in the specified directory; toggle this using the `--simid` flag.
The working directory for `linkedNN` must itself contain a folder with tree sequences called `TreeSeqs/` and a separate folder with the corresponding targets called `Targets/`, the latter saved as ".npy" format.

Example preprocessing command:

```
linkedNN --preprocess \
         --wd <path> \
         --seed <int> \
         --num_snps <int> \
         --n <int> \
         --l <int> \
         --hold_out <int> \
         --simid <int>
```
- `preprocess`:	runs the preprocessing pipeline.
- `wd`: path to output directory.
- `seed`: random number seed ($>0$). The random number seed determines the names of outputs,so it's important to use different seeds for different analyses.
- `num_snps`: fixed number of SNPs to extract; it is recommended to use the number in your empirical dataset.
- `n`: number of diploid individuals; it is recommended to use the `n` from your empirical dataset.
- `l`: chromosome length; it is recommented to use `l` from your empirical dataset.
- `hold_out`: number of simulations from the full set to hold out for testing.
- `simid`: (optional) either (i) an individual simulation ID, (ii) a comma-separated range of IDs; if excluded, all ID's are preprocessed.

#### Training
After preprocessing all simulations, `linkedNN` can train a model using:

```
linkedNN --train \
         --wd <path> \
         --seed <int> \
         --batch_size <int>
```

- `train`: runs the training pipeline.
- `batch_size`: the size of mini-batches

#### Testing
To predict on held-out test data, run:
```
linkedNN --predict \
         --wd <path> \
         --seed <int> \
         --batch_size <int>
```


#### Empirical applications
To predict from an empirical VCF: leave in rare alleles, subset for a particular chromosome, and run the below command.
```
linkedNN --predict \
         --wd <path> \
         --seed <int> \
         --batch_size <int> \
         --empirical <path>
```
- `empirical`: is the path and prefix for the vcf file (without ".vcf").









---
<h2 id="vignette">Vignette</h2>

Below is a complete, example workflow with `LinkedNN` to provide a sense what inputs and outputs to expect at each stage in the pipeline.


#### Simulating training data
`LinkedNN` expects tree sequences, so you can use whatever program produces this output, i.e., msprime, SLiM, tsinfer.
For this vignette, we will run one hundred small simulations using a script provided in the GitHub repo.
However, note that 50,000 simulations and hundreds of training epochs may be required to train successfully.

```
git clone https://github.com/the-smith-lab/LinkedNN.git
for i in {1..100}
do
    echo "simulation ID $i"
    python LinkedNN/Misc/sim_demog.py $i 500,1e3 1e2,1e3 1e2,1e3 tempdir/
done
```


#### Preprocess
```
linkedNN --preprocess --wd tempdir/ --seed 2 --num_snps 5000 --n 10 --l 1e8 --hold_out 25
```


#### Train
```
linkedNN --train --wd tempdir/ --seed 2 --batch_size 10 --max_epochs 10
```
The new `max_epochs` flag is used here to limit the number of training epochs (default=1000).


#### Test
```
linkedNN --predict --wd tempdir/ --seed 2 --batch_size 10
```
---







**How to cite:**


**Source code:** github.com/the-smith-lab/LinkedNN
