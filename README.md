# Count_decomp

First, complile the Cython code
```
python setup.py build_ext --inplace
```

Then 
```
cd persistable
```

## The first vision is to create random points and build bifiltrations on them. Run
```
python experiment_count_decomp_v1.py

```
The results are stored in `path-to-repository/experiment_result_v1`

## The second vision is to fix a simplicial complex (complete graph with n vertices) and build different bifiltrations on it. Run
```
python experiment_count_decomp_v2.py
```
The results are stored in `path-to-repository/experiment_result_v2`
