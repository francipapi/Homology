#!/bin/bash

echo "🚀 Launching Optuna Dashboard..."
echo ""
echo "This will start a web interface to monitor optimization progress."
echo "Open your browser to: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the dashboard."
echo ""

# Activate conda environment if needed
source activate myenv 2>/dev/null || conda activate myenv || true

# Look for any database files
DB_FILE=""
if [ -f "torch_mlp_optimization.db" ]; then
    DB_FILE="torch_mlp_optimization.db"
elif [ -f "torch_mlp_bayesian_optimization.db" ]; then
    DB_FILE="torch_mlp_bayesian_optimization.db"
else
    # Look for any .db file
    DB_FILE=$(ls *.db 2>/dev/null | head -n 1)
fi

if [ ! -z "$DB_FILE" ]; then
    echo "✅ Found optimization database: $DB_FILE"
    echo "🌐 Starting dashboard on http://localhost:8080"
    echo ""
    optuna-dashboard "sqlite:///$DB_FILE"
else
    echo "⚠️  No optimization database found."
    echo ""
    echo "To create one:"
    echo "   1. Run: ./run_optimization.sh"
    echo "   2. Wait for at least one trial to complete"
    echo "   3. Then run this dashboard script again"
    echo ""
    echo "Available files:"
    ls -la *.db 2>/dev/null || echo "   No .db files found"
fi