mpirun --report-bindings --use-hwthread-cpus --mca pml ob1 --mca btl self,vader --map-by ppr:2:core --bind-to core -np $5 ../bin/xhpcg $1 $2 $3 $4
#mpirun --use-hwthread-cpus --report-bindings --mca pml ob1 --mca btl self,vader --map-by numa --bind-to core -np $5 ../bin/xhpcg $1 $2 $3 $4
