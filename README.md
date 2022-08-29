# IMLE

* Conditional IMLE refactoring CAM-NET

* Unconditional IMLE with Gaussian Mixture noise


# Conditional IMLE refactoring CAM-NET

Current CAM-NET has complexity in triple loops usage when training model. The goal here is to improve usability and complexity of CAM-NET model training procedure by reducing triple loops to double loops while keeping the performance of CAM-NET.


## Objectives

* subsample_size
* num_iteration
* outer_loop
* KorKMinusOne
* CIMLEDataLoader

## Subsample Size

`subsample_size` argument determines the size of subsample for each `outer_loop` iteration. `subsample_size=len(dataset)` by default.

## Num iteration (Number of Iteration)

`num_iteration` argument determines the number of iterations for each subsampled data.

## Outer Loop
`outer_loop` argument is used instead of `epoch`. Each `outer_loop` iteration is one subsampling. Thus, one `epoch` concept, use of entire datasets, is depends on `subsample_size` value. (For example, if `subsample_size=400, len(dataset)=1200`, then 3 `outer_loops` considered as an `epoch`.

## KorKMinusOne

KorKMinusOne (KKM), is to track of how many times each data has been used. 

### How KorKMinusOne works

```
class KorKMinusOne
INPUT: idxs(list of index), shuffle

    idxs -> idxs, counter -> 0, shuffle -> shuffle
    
    function pop
        
        if counter reaches to end of idxs 
           then counter -> 0
           
           if shuffle = True then
              randomize the elements position in idxs
        
        result -> idxs[counter]
        counter -> counter + 1

        return result
```

`idxs` input is a list that maps each data's positional index. Example:
```
kkm = KorKMinusOne(range(len(data_tr)), shuffle=True)
```
`shuffle` is to dertmine whether to randomize `idxs` at each `epoch`. `shuffle = False` by default.

## CIMLEDataLoader

CIMLEDataLoader is a iterator object that subsamples data and returns chained dataloaders lazily, and the number of chained dataloaders are determined by `subsample_size` and `num_iteration` arguments.

Another apporach to implement CIMLEDataLoader was to modify DataLoader source code to sample dataset when `iter` method gets called. However, `len` of dataloader is initialized in `__init__` and updating length of dataloader dynamically as chained dataloader generate was making conflict with other inner methods that are utilizing `__len__()` method. Thus, instead, iterator object has been created. 

### How CIMLEDataLoader works

#### Initialization
```
def __init__(self, dataset, kkm, model, corruptor, z_gen, loss_fn, num_samples, sample_parallelism, code_bs, subsample_size=None, num_iteration=1, pin_memory: bool = False, shuffle: Optional[bool] = None, batch_size: Optional[int] = 1, num_workers: int = 0, drop_last: bool = False)
```

CIMLE DataLoader takes additional arguments upon normal Dataloader. kkm, model, corruptor, z_gen, loss_fn, num_samples, sample_parallelism, code_bs, subsample_size, num_iteration are additionals. Key arguments here are `kkm`, `num_iteration`, and `subsample_size`.

During initialization, `num_chained_loaders` determines how many chained dataloader to be generated. This is calculated as follows:

```
if num_iteration is greater than (subsample_size // batch_size)          # subsample_size // batch_size is how many batch data can be fitted             
    then num_chained_loaders -> num_iteration // (subsample_size // batch_size)
else
    num_chained_loaders -> 1
```

By looping through range of `num_chained_loaders`, DataLoaders with `subsample_size // batch_size` iterations are generated and appended into a list. When the loop reached to the last one, it checkes `num_iteration % (subsample_size // batch_size) != 0`. 

This checks whether the `num_iteration` evenly divisible by number of iterations for each subsample `(subsample_size // batch_size)`. If it is divisible, then another DataLoader is generated as previous, otherwise, a DataLoader with `num_iteration % (subsample_size // batch_size)` amounts of iterations is generated which is less iterations than previously generated DataLoaders and appended to the list.

At last, list of DataLoaders will be chained by `itertools.chain` method. 

## Outcome and Usage

By using `CIMLEDataLoader`, user will see less loop complexity, have better readability, and usability.

- previous training structure
```
loader = DataLoader(.....)
for epoch in epochs

    for (x, ys) in loader
        
        #### Sampling ####
        batch_loader = DataLoader(.....)
        
        for (x, latent, ys) in batch_loader
        
            #### Training ####
```

- updated strcuture
```
k_or_k_minus_one = KorKMinusOne(range(len(data_tr)), shuffle=True)
loader = CIMLEDataLoader(dataset, k_or_k_minus_one,  model, corruptor, z_gen, loss_fn, ...)

for loop in range(args.outer_loop)

    for (x, latent, ys) in loader        #### Sampling lazily ####
        #### Training ####
```

# Unconditional IMLE with Gaussian Mixture noise

## Objectives
* Style GAN
