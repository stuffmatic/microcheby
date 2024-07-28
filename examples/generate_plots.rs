use std::fs;

use microcheby::ChebyshevApprox;

fn x_values(x_min: f32, x_max: f32, n_samples: usize) -> Vec<f32> {
    let mut result = vec![];
    for i in 0..n_samples {
        result.push(x_min + (x_max - x_min) / (n_samples as f32) * (i as f32));
    }
    result
}

fn approx_values<const N: usize, F>(x_values: &Vec<f32>, f: F) -> Vec<f32>
where
    F: Fn(f32) -> f32,
{
    let approx = ChebyshevApprox::<N>::fit(
        *x_values.first().unwrap(), 
        *x_values.last().unwrap(), 
    f);

    x_values.iter().map(|x| approx.eval(*x)).collect()
}

fn write_csv_columns(columns: &Vec<Vec<f32>>, csv_path: &str) {
    let n_samples = columns[0].len();

    let mut csv_string = String::new();
    
    for row_idx in 0..n_samples {
        let mut row: Vec<String> = vec![];
        for col_idx in 0..columns.len() {
            row.push(format!("\"{}\"", columns[col_idx][row_idx].to_string()))
        }
        
        csv_string.push_str(row.join(",").as_str());
        csv_string.push_str("\n");
    }
    fs::write(csv_path, csv_string).expect("Failed to write plot data");
}

fn main() {
    // Generate csv files for plotting with gnuplot
    let output_path = "plots/plot_data.csv";
    let n_samples = 120;
    let f = |x: f32| (2.0 * x).sin();  
    let x_min = -0.1;
    let x_max = 7.1;

    let mut columns: Vec<Vec<f32>> = vec![];
    let x_values = x_values(x_min, x_max, n_samples);
    columns.push(x_values.clone());
    let f_values: Vec<f32> = x_values.iter().map(|x| f(*x)).collect();
    columns.push(f_values);
    columns.push(approx_values::<1, _>(&x_values, f));
    columns.push(approx_values::<2, _>(&x_values, f));
    columns.push(approx_values::<3, _>(&x_values, f));
    columns.push(approx_values::<4, _>(&x_values, f));
    columns.push(approx_values::<5, _>(&x_values, f));
    columns.push(approx_values::<6, _>(&x_values, f));
    columns.push(approx_values::<7, _>(&x_values, f));
    columns.push(approx_values::<8, _>(&x_values, f));
    columns.push(approx_values::<9, _>(&x_values, f));

    write_csv_columns(&columns, output_path);
    

}