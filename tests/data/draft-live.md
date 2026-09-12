# What a claim costs to check

## Introduction

Attention-only sequence models changed what a translation system costs to train.
On the WMT 2014 English-to-French translation task the Transformer reached its
state-of-the-art BLEU score after training for 21 days on eight GPUs [1]. The
same architecture now underpins the models that read the very papers this draft
cites.

Structure prediction followed the same path. AlphaFold predicts protein
structures with atomic accuracy even where no similar structure is known [2].

## Tooling

The numerical work behind both results rests on a small stack of open libraries.
NumPy is the foundational array programming library of the Python scientific
ecosystem [3]. SciPy builds on it with the fundamental algorithms for scientific
computing in Python [4].

## Data practice

Reuse is a property of the data, not of the goodwill of its authors: research
data should be findable, accessible, interoperable and reusable [5]. Association
alone cannot answer a causal question, however large the dataset [6].

## The part that does not hold up

A recent scientometric study reports that cross-domain retrieval quality
collapses by 31% under distribution shift in citation graphs [7]. That figure is
quoted here exactly as this draft's author found it.

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. & Polosukhin, I. Attention is all you need. arXiv:1706.03762 (2017).
[2] Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583-589 (2021). doi:10.1038/s41586-021-03819-2
[3] Harris, C. R., Millman, K. J., van der Walt, S. J. et al. Array programming with NumPy. Nature 585, 357-362 (2020). doi:10.1038/s41586-020-2649-2
[4] Virtanen, P., Gommers, R., Oliphant, T. E. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17, 261-272 (2020). doi:10.1038/s41592-019-0686-2
[5] Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J. et al. The FAIR Guiding Principles for scientific data management and stewardship. Scientific Data 3, 160018 (2016). doi:10.1038/sdata.2016.18
[6] Pearl, J. & Mackenzie, D. The Book of Why: The New Science of Cause and Effect (Basic Books, 2018).
[7] Marchetti, L. R., Osei, K. & Lindqvist, P. Cross-domain retrieval collapse under distribution shift in citation graphs. Journal of Applied Scientometrics 14, 220-238 (2021).
