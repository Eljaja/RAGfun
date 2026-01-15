#!/bin/bash
# Setup script for optimized rustfs deployment

set -e

echo "============================================================"
echo "RustFS Performance Optimization Setup"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create data volumes
echo "Creating volume directories..."
mkdir -p data/vol{1,2,3,4}
echo -e "${GREEN}✓${NC} Created data/vol1, data/vol2, data/vol3, data/vol4"

# Check if running on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo ""
    echo "Checking system configuration..."
    
    # Check file descriptor limits
    CURRENT_ULIMIT=$(ulimit -n)
    if [ "$CURRENT_ULIMIT" -lt 65536 ]; then
        echo -e "${YELLOW}⚠${NC}  File descriptor limit is low: $CURRENT_ULIMIT"
        echo "   Recommended: Run 'ulimit -n 65536' or add to /etc/security/limits.conf"
    else
        echo -e "${GREEN}✓${NC} File descriptor limit: $CURRENT_ULIMIT"
    fi
    
    # Check if io_uring is available
    if [ -d "/sys/kernel/debug/io_uring" ]; then
        echo -e "${GREEN}✓${NC} io_uring is available"
    else
        echo -e "${YELLOW}⚠${NC}  io_uring not detected (kernel 5.10+ recommended)"
    fi
    
    # Check available memory
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM_GB" -lt 8 ]; then
        echo -e "${YELLOW}⚠${NC}  Low memory: ${TOTAL_MEM_GB}GB (recommend 16GB+ for high performance)"
    else
        echo -e "${GREEN}✓${NC} Available memory: ${TOTAL_MEM_GB}GB"
    fi
    
    # Check if running in Docker
    if [ -f /.dockerenv ]; then
        echo -e "${YELLOW}⚠${NC}  Running inside Docker container"
    fi
fi

echo ""
echo "============================================================"
echo "Deployment Options"
echo "============================================================"
echo ""
echo "1. Start optimized rustfs:"
echo "   docker-compose -f docker-compose.optimized.yaml up -d"
echo ""
echo "2. Run benchmark:"
echo "   cd test"
echo "   python benchmark_upload.py --worker-counts '10,20,50,100'"
echo ""
echo "3. Monitor performance:"
echo "   docker stats rustfs"
echo ""
echo "============================================================"
echo "Performance Tuning Tips"
echo "============================================================"
echo ""
echo "For maximum performance:"
echo ""
echo "📁 Storage:"
echo "   - Use NVMe SSDs (not SATA or HDD)"
echo "   - Mount each volume on a separate physical drive"
echo "   - Use ext4 or xfs with noatime,nodiratime options"
echo ""
echo "🧠 Memory:"
echo "   - Allocate 16-32GB for high concurrency"
echo "   - Disable swap or use minimal swap"
echo ""
echo "🌐 Network:"
echo "   - Use 10Gbps+ networking for best throughput"
echo "   - Enable jumbo frames (MTU 9000) if supported"
echo ""
echo "⚙️  System:"
echo "   - Increase file descriptors: ulimit -n 65536"
echo "   - Use Linux kernel 5.10+ (for io_uring)"
echo "   - Set I/O scheduler to deadline or noop for SSDs"
echo ""
echo "See PERFORMANCE_TUNING.md for detailed guide"
echo ""
echo "============================================================"

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo ""
    read -p "Start optimized rustfs now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting rustfs with optimized configuration..."
        docker-compose -f docker-compose.optimized.yaml up -d
        echo ""
        echo -e "${GREEN}✓${NC} RustFS started!"
        echo ""
        echo "Monitor logs: docker-compose -f docker-compose.optimized.yaml logs -f"
        echo "Check status: docker-compose -f docker-compose.optimized.yaml ps"
    fi
else
    echo -e "${RED}✗${NC} docker-compose not found"
    echo "   Install Docker Compose to proceed"
fi

echo ""
echo "Setup complete! 🚀"


