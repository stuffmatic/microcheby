
# Set the output to a png file
set terminal svg size 800,800

# set grid
# The file we'll write to
set output 'sinx.png'
set datafile separator ','
set key noautotitle
set size 1,1
set origin 0,0
set xrange[-0.1:7.04] 
set yrange[-1.2:1.2] 
set multiplot layout 3,3 rowsfirst scale 1.1,0.9
set title 'N=1'
plot 'plot_data.csv' using 1:2 with lines, 'plot_data.csv' using 1:3 with lines
set title 'N=2'
plot 'plot_data.csv' using 1:2 with lines, 'plot_data.csv' using 1:4 with lines
set title 'N=3'
plot 'plot_data.csv' using 1:2 with lines, 'plot_data.csv' using 1:5 with lines
set title 'N=4'
plot 'plot_data.csv' using 1:2 with lines lw 3, 'plot_data.csv' using 1:6 with lines lw 3
plot 'plot_data.csv' using 1:2 with lines, 'plot_data.csv' using 1:7 with lines
plot 'plot_data.csv' using 1:2 with lines, 'plot_data.csv' using 1:8 with lines
plot 'plot_data.csv' using 1:2 with lines, 'plot_data.csv' using 1:9 with lines
plot 'plot_data.csv' using 1:2 with lines, 'plot_data.csv' using 1:10 with lines
plot 'plot_data.csv' using 1:2 with lines, 'plot_data.csv' using 1:11 with lines

unset multiplot
# The graphic title





unset multiplot