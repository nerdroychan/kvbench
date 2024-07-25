This example shows a benchmark that reads popular records in the store, running with different
number of threads.

The workload file, `readpopular.toml` is as follows:

```toml
[global]
threads = 1
repeat = 1
klen = 8
vlen = 16
kmin = 0
kmax = 1000000

[[benchmark]]
set_perc = 100
get_perc = 0
del_perc = 0
repeat = 1
dist = "incrementp"
report = "hidden"

[[benchmark]]
timeout = 1
set_perc = 0
get_perc = 100
del_perc = 0
dist = "zipfian"
zipf_theta = 1.0
report = "finish"
```

In the first phase, all worker threads fill the key space of the store, and the metrics are hidden.
In the second phase, worker threads execute the read-only workload that accesses Zipfian keys for
1 second and report once when finished.

The script file `run.sh` runs this benchmark against multiple stores with different number of
threads. The number of threads are dynamically adjusted via `global.threads` environment variable.

Results:

[readpopular](readpopular.pdf)
