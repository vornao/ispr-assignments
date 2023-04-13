# Assignment 2 - Track 3

Implement from scratch an RBM and apply it to DSET3. 
The RBM should be implemented fully by you (both CD-1 training and inference steps) 
but you are free to use library functions for the rest (e.g. image loading and management, etc.).

1. Train an RBM with a number of hidden neurons selected by you 
   (single layer) on the MNIST data (use the training set split provided by the website).
2. Use the trained RBM to encode a selection of test images (e.g. using one per digit type) 
   using the corresponding activation of the hidden neurons.
3. Reconstruct the original test images from their hidden encoding 
   and confront the reconstructions with the original image 
   (use a suitable quantitative metric to assess the reconstraction quality.
   Also choose few examples to confront visually).