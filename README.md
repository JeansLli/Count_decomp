# Count_decomp
This repository is based on [Persistable](https://github.com/LuisScoccola/persistable).

First, complile the Cython code
```
python setup.py build_ext --inplace
```

Then 
```
cd persistable
```

## The first version
Create random points and build bifiltrations on them. Run
```
python experiment_count_decomp_v1.py

```
The results are stored in `path-to-repository/experiment_result_v1`

## The second version 
Fix a simplicial complex (complete graph with n vertices) and build different bifiltrations on it. Run
```
python experiment_count_decomp_v2.py
```
The results are stored in `path-to-repository/experiment_result_v2`
