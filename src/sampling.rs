use crate::sample::Sample;

pub (crate) fn window(mut x : f32) -> f32
{
    x = x*2.0 - 1.0;
    x *= x;
    x = 1.0 - x;
    let x2 = x*x;
    let x3 = x2*x;
    x2*0.56 + x3*0.44
}
pub (crate) fn generate_sinc_table(filter_size : usize, adjusted_freq : f32) -> Vec<f32>
{
    let mut sinc_table = vec!(0.0f32; filter_size);
    let mut energy = 0.0;
    for i in 0..filter_size
    {
        let t = (i as f32 + if filter_size % 2 == 0 { 0.0 } else { 0.5 }) / (filter_size as f32);
        let w = window(t);
        let x = (t * 2.0 - 1.0) * filter_size as f32 / 2.0 * adjusted_freq * std::f32::consts::PI;
        let e = if x == 0.0 { 1.0 } else { x.sin() / x } * w;
        energy += e;
        sinc_table[i] = e;
    }
    for i in 0..filter_size
    {
        sinc_table[i] /= energy;
    }
    println!("filter energy: {}. normalizing.", energy);
    sinc_table
}
pub (crate) fn interpolate(table : &[f32], i : usize, t : f32) -> f32
{
    let a = table[i%table.len()];
    let b = table[(i+1)%table.len()];
    a * (1.0 - t) + b * t
}
pub (crate) fn interpolate_sample(table : &[Sample], i : usize, t : f32) -> Sample
{
    let a = table[i%table.len()];
    let b = table[(i+1)%table.len()];
    a * (1.0 - t) + b * t
}
pub (crate) fn resample(in_data : &[Sample], factor : f64) -> Vec<Sample>
{
    let mut new_data = vec!(Sample { l : 0.0, r : 0.0 }; (in_data.len() as f64 * factor) as usize);
    for i in 0..new_data.len()
    {
        let _j = i as f64 / factor;
        let j = _j as usize;
        let t = _j - j as f64;
        new_data[i] = interpolate_sample(in_data, j, t as f32);
    }
    new_data
}

pub (crate) fn do_freq_split(in_data : &[Sample], samplerate : f64, filter_size_s : f64, freq : f64) -> (Vec<Sample>, Vec<Sample>)
{
    let bandwidth_length = 1.0 / freq;
    let filter_size = ((samplerate * filter_size_s) as usize).max(1);
    if filter_size % 2 == 0
    {
        println!("filter size ({}, around {}ms) is even. building sinc table", filter_size, filter_size_s*1000.0);
    }
    else
    {
        println!("filter size ({}, around {}ms) is NOT even (is odd). building sinc table", filter_size, filter_size_s*1000.0);
    }
    
    let adjusted_freq = (freq * 2.0 / samplerate) as f32;
    let sinc_table = generate_sinc_table(filter_size, adjusted_freq);
    
    let mut out_lo = vec!(Sample { l : 0.0, r : 0.0 }; in_data.len());
    let mut out_hi = vec!(Sample { l : 0.0, r : 0.0 }; in_data.len());
    for i in 0..in_data.len()
    {
        let o = filter_size/2;
        let mut sum = Sample { l : 0.0, r : 0.0 };
        for j in 0..filter_size
        {
            let s = (i + j) as isize - o as isize;
            if s < 0 || s as usize >= in_data.len()
            {
                continue;
            }
            sum += in_data[s as usize] * sinc_table[j];
        }
        out_lo[i] = sum;
        out_hi[i] = in_data[i] - out_lo[i];
    }
    (out_lo, out_hi)
}
