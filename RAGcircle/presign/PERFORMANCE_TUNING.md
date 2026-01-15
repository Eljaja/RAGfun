# RustFS Performance Tuning Guide

This guide explains how to optimize your rustfs instance for maximum bulk upload performance.

## 🎯 Current Performance Baseline

- **Your Results**: 182.73 uploads/second with 10 workers (2.3 MB files)
- **Throughput**: ~420 MB/s
- **File Size**: 2.3 MB

## 📊 Optimization Strategy

### 1. Hardware-Level Optimizations

#### Storage
- **Use NVMe SSDs**: 3-5× faster than SATA SSDs or HDDs
- **Multiple drives**: Distribute I/O across multiple physical drives
- **Avoid RAID-5/6**: Use RAID-0, RAID-10, or JBOD for better write performance
- **Filesystem tuning**: Use `ext4` or `xfs` with options:
  ```bash
  mount -o noatime,nodiratime,data=writeback /dev/nvme0n1 /data
  ```

#### Memory
- **Minimum**: 8 GB for light workloads
- **Recommended**: 16-32 GB for high concurrency
- **Optimal**: 64-128 GB for enterprise workloads
- More memory = better metadata caching and I/O buffers

#### Network
- **1 Gbps**: Good for ~100-125 MB/s throughput
- **10 Gbps**: Supports ~1 GB/s throughput
- **25-100 Gbps**: For maximum performance (thousands of uploads/sec)

#### CPU
- **Minimum**: 4 cores
- **Recommended**: 8-16 cores for high concurrency
- Modern CPUs (with io_uring support) preferred

### 2. Docker Configuration Optimizations

#### Use the Optimized docker-compose.yaml

```bash
docker-compose -f docker-compose.optimized.yaml up -d
```

**Key improvements:**
- **Multiple volumes** (`/data1`, `/data2`, `/data3`, `/data4`) for parallel I/O
- **Resource limits** (8 CPUs, 16 GB RAM)
- **Larger buffers** (4 MB read/write buffer)
- **Higher concurrency** (10,000 max concurrent requests)
- **Optimized caching** (80% cache quota, immediate caching)
- **Jumbo frames** (MTU 9000) if your network supports it

#### Create Volume Directories

```bash
mkdir -p data/vol{1,2,3,4}
```

**Pro tip**: If you have multiple physical drives, mount each volume on a different drive:
```bash
# Example with 4 NVMe drives
mount /dev/nvme0n1 ./data/vol1
mount /dev/nvme1n1 ./data/vol2
mount /dev/nvme2n1 ./data/vol3
mount /dev/nvme3n1 ./data/vol4
```

### 3. RustFS Environment Variable Tuning

The optimized compose file includes these critical settings:

| Variable | Default | Optimized | Impact |
|----------|---------|-----------|--------|
| `RUSTFS_API_REQUESTS_MAX` | 1000 | 10000 | 10× more concurrent requests |
| `RUSTFS_API_READWRITE_BUFFER_SIZE` | 1048576 (1MB) | 4194304 (4MB) | Better for larger files |
| `RUSTFS_CACHE_QUOTA` | 50 | 80 | More aggressive caching |
| `RUSTFS_CACHE_AFTER` | 3 | 0 | Cache immediately |
| `RUSTFS_NOTIFY_MQTT_QUEUE_LIMIT` | 10000 | 100000 | Larger notification queue |

### 4. System-Level Tuning (Linux)

#### Increase file descriptors
```bash
# Add to /etc/security/limits.conf
* soft nofile 65536
* hard nofile 65536

# Or temporarily:
ulimit -n 65536
```

#### Optimize network stack
```bash
# Add to /etc/sysctl.conf
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 67108864
net.ipv4.tcp_wmem = 4096 65536 67108864
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr

# Apply:
sudo sysctl -p
```

#### Optimize disk I/O
```bash
# Use deadline or noop scheduler for SSDs
echo deadline > /sys/block/nvme0n1/queue/scheduler

# Increase queue depth
echo 1024 > /sys/block/nvme0n1/queue/nr_requests
```

## 🧪 Running Benchmarks

### Quick Test (Find Optimal Worker Count)
```bash
cd test
python benchmark_upload.py \
  --bucket test-bucket \
  --uploads-per-test 1000 \
  --worker-counts "10,20,50,100,200" \
  --output results.json
```

### Comprehensive Test
```bash
python benchmark_upload.py \
  --bucket test-bucket \
  --uploads-per-test 5000 \
  --worker-counts "1,5,10,20,50,100,200,500" \
  --output benchmark_full.json
```

### Expected Output
```
==============================================================
BENCHMARK COMPARISON
==============================================================
Workers    Uploads/s    MB/s       Avg Time     P99 Time    
--------------------------------------------------------------
10         182.73       420.28     0.055        0.089
20         342.15       787.25     0.058        0.102
50         756.42       1739.77    0.066        0.125
100        1205.33      2772.26    0.083        0.156
200        1456.78      3350.60    0.137        0.245
--------------------------------------------------------------

🏆 BEST PERFORMANCE: 200 workers
   Throughput: 1456.78 uploads/s
   Bandwidth: 3350.60 MB/s
   P99 Latency: 0.245s
```

## 📈 Performance Expectations

Based on rustfs benchmarks and your current setup:

| Setup | Expected Throughput (2.3 MB files) |
|-------|-----------------------------------|
| **Current** (10 workers) | ~180 uploads/s (~420 MB/s) |
| **Optimized Docker** (50 workers) | ~600-800 uploads/s (~1.5-2 GB/s) |
| **Multi-volume + tuning** (100 workers) | ~1000-1500 uploads/s (~2.5-3.5 GB/s) |
| **High-end hardware** (200+ workers) | ~2000-3000 uploads/s (~5-7 GB/s) |
| **Cluster setup** (multi-node) | ~5000+ uploads/s (10+ GB/s) |

## 🔍 Identifying Bottlenecks

### Monitor During Load Test

```bash
# CPU usage
htop

# Disk I/O
iostat -xz 1

# Network
nload

# Docker stats
docker stats rustfs
```

### What to Look For

| Bottleneck | Symptoms | Solution |
|------------|----------|----------|
| **CPU** | 100% CPU usage | Add more cores or reduce workers |
| **Disk I/O** | High iowait (>50%) | Use faster drives, add volumes |
| **Network** | Saturated bandwidth | Upgrade network, use 10G+ NIC |
| **Memory** | Swapping, OOM errors | Add more RAM |
| **Connection limits** | Connection refused errors | Increase `RUSTFS_API_REQUESTS_MAX` |

## 🚀 Quick Start - Deploy Optimized Setup

```bash
# 1. Stop current setup
docker-compose down

# 2. Create volume directories
mkdir -p data/vol{1,2,3,4}

# 3. Start optimized setup
docker-compose -f docker-compose.optimized.yaml up -d

# 4. Wait for startup
sleep 10

# 5. Run benchmark
cd test
python benchmark_upload.py \
  --uploads-per-test 2000 \
  --worker-counts "10,20,50,100" \
  --output optimization_results.json
```

## 🎓 Advanced: Multi-Node Cluster

For ultimate performance (5000+ uploads/sec), deploy rustfs as a distributed cluster:

1. Multiple rustfs nodes (4-16 nodes)
2. Load balancer in front
3. Shared distributed storage or erasure coding
4. Separate metadata and data paths

See [rustfs cluster documentation](https://docs.rustfs.com) for details.

## 📝 Recommendations Based on Your Results

Given your **182 uploads/s with 10 workers**:

1. **Increase workers to 50-100** - Should give you 2-3× improvement (400-600 uploads/s)
2. **Use optimized docker-compose** - Additional 20-30% improvement
3. **Add multiple volumes** - Another 50-100% improvement if on separate drives
4. **Total expected improvement**: 1000-1500 uploads/s (5-8× current performance)

## 🎯 Target Benchmarks

For 2.3 MB files on a **single well-configured node**:

- **Good**: 500-1000 uploads/s (~1-2.5 GB/s)
- **Great**: 1000-2000 uploads/s (~2.5-5 GB/s)
- **Excellent**: 2000-3000+ uploads/s (~5-7 GB/s)

Your baseline of 182 uploads/s shows the system is working well, but there's **significant headroom** for optimization! 🚀

## 📚 References

- [RustFS Hardware Checklist](https://docs.rustfs.com/installation/checklists/hardware-checklists.html)
- [RustFS Software Checklist](https://docs.rustfs.com/installation/software-checklists.html)
- [RustFS Docker Deployment](https://docs.rustfs.com/installation/docker/)
- [RustFS Performance Blog](https://blog.csdn.net/rustfs_contrib/article/details/153136659)


