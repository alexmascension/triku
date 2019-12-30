# Triku, the Highly Variable Gene selector

Triku, hedgehog in euskera (basque), is an algorithm for Highly Variable Gene selection. 
This algorithm was inspired in the *Droplet scRNA-seq is not zero-inflated* paper from Valentine Svensson. 
The paper describes that, when using technical controls in droplet based single cell RNA-seq 
experiments, the number of zeros in the data based on the mean expression for each gene follows a negative 
binomial distribution. The paper then states that additional zeros on count matrices from biological 
samples are due to biological variation.

EXPLAIN THE REST OF THE README