
# Set the output to a png file
set terminal svg size 800,800

# set grid
# The file we'll write to
set output 'approximations.svg'
set datafile separator ','
set key noautotitle
set size 1,1
set origin 0,0
set xrange[0.5:1.5] 
set yrange[-0.5:1] 
set multiplot layout 3,3 rowsfirst scale 1.1,0.9 title "Chebyshev expansions of length N approximating f(x)=sin(4x^3)exp(-4x^4)"
set title 'N=1'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:3 with lines lw 3 linecolor rgb "#1DB100"
set title 'N=2'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:4 with lines lw 3 linecolor rgb "#1DB100"
set title 'N=3'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:5 with lines lw 3 linecolor rgb "#1DB100"
set title 'N=4'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:6 with lines lw 3 linecolor rgb "#1DB100"
set title 'N=5'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:7 with lines lw 3 linecolor rgb "#1DB100"
set title 'N=6'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:8 with lines lw 3 linecolor rgb "#1DB100"
set title 'N=7'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:9 with lines lw 3 linecolor rgb "#1DB100"
set title 'N=8'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:10 with lines lw 3 linecolor rgb "#1DB100"
set title 'N=9'
plot 'plot_data.csv' using 1:2 with lines linecolor rgb "black", 'plot_data.csv' using 1:11 with lines lw 3 linecolor rgb "#1DB100"

unset multiplot
# The graphic title





unset multiplot