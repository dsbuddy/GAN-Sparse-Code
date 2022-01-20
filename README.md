# Sparse Code Approximation

## Code

* `GenerateSparseCodesDictionary.ipynb` - Downloads CIFAR10 data and generates sparse codes given dictionary
* `cnn.ipynb` - Pipeline to learn sparse codes given patches and previously learned LCA sparse codes from `GenerateSparseCodesDictionary.ipynb` and saves results
* `cnn-full-image.ipynb` - Pipeline to learn sparse codes given full image and previously learned LCA sparse codes from patches from `GenerateSparseCodesDictionary.ipynb` and saves results

## Data

* `newfilters50k_nomom.mat` - Dictionary file for 128 elements 
* `dic256.mat` - Dictionary file for 256 elements
* `dic512.mat` - Dictionary file for 512 elements

# Steps
* Run `GenerateSparseCodesDictionary.ipynb` and then run `cnn.ipynb`
