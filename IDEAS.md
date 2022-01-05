# Ideas

## On-Manifold Augmentations and the Corruptions that Lead to Them
Ke noted that it makes a lot of sense to generate data as follows: (1) sample an image, (2) corrupt it, (3) use a generative model to decorrupt it twice. This gives two different images that look real and look different where the corruption was replaced. The advantage of this is as follows: the corruption operator is surjective (ie. there are multiple images that differ only in the information deleted by the corruption), so inverting it should lead to many different modes of outputs!

_With this in mind, what corruptions make sense?_
1) **Cropping** out regions of the image easily deletes information, but it's unclear what the right amount/way to crop is. This actually strikes me as an operator that'd need to be used sparingly outside of the background of an image.
2) **Downsampling** should allow the generator to fill in fine details. _I'm interested to see if this is helpful since it shouldn't really distributionally change an image at all and this operator should lead to the greatest sameness between images as per human vision. Therefore we can evaluate how sensitivity between images (what contrastive loss focuses on) is important for the classification task.
2) **Decolorization** could lead the generator to generate wildly different dominant colors for positive images, which would would be good according to the augmentations found useful in SimCLR

_Is corruption everything?_ I don't think so, since positives can differ by **rotation**, **flipping**, and small **edge-aligned crops**, and it's difficult to see a conditional generative model doing this! In other words, we should think of the generative model/corruptions as providing special spicy augmentations.

_How do we actually implement this?_
 - Rotation, flipping, and edge-aligned crops are easy to implement with the right TorchVision transforms and a simple `DataSet` we already have
 - CAMNet should be able to do both super-resolution and colorization.
 - Cropping doesn't seem like something CAMNet can easily do?

## Sampling Corruptions
On a basic level, we want a function that can take in an image and output a corrupted version of it. Ideally, it should be trainable without gradients flowing from the contrastive learner; we can bring in RL or a gradient-estimation technique here.
