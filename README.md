# ISICLE
**I**mage **S**ynthes**i**s for **C**ontrastive **Le**arning

### Environment
```
python==3.9.6
pytorch==1.9
torchvision==0.10.0
tqdm==4.62.2
```

### Running Code


### TODOs


### Questions

**Wednesday 9/15/2021**:
**Q**: How come loss is divided by two?
**A:**

**Q**: Replace with CAMNet?
**A:**

**Q**: What kind of task to actually do? Super-resolution? Coloring? Generating specific random crops?


**A:**

**Q**: Datasets? CIFAR-10 seems really low-res for this?

**Q**: Standardizing data... When we do contrastive learning with real data, we can precompute its mean and STD. Here, we have to generate it. Options:
 - train the generator to output pre-standardized samples
 - assume its output will approximate the mean/std of the input dataset
