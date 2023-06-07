# Count_decomp

First, complile the Cython code
```
python setup.py build_ext --inplace
```

Then 
```
cd persistable
```

The first vision is creating random points and building bifiltration on them
```
python experiment_count_decomp_v1.py

```
The results are stored in `path-to-repository/experiment_result_v1`

The second vision is fixing a simplicial complex (complete graph with n vertices) and building different bifiltrations on it
```
python experiment_count_decomp_v2.py
```
The results are stored in `path-to-repository/experiment_result_v2`
