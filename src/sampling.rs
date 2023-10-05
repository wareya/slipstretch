use crate::sample::Sample;

pub (crate) fn window(mut x : f32) -> f32
{
    // This is a Welch window, squared and cubed, then interpolated between the squared and cubed copies such that the x=0.25 and x=0.75 points are slightly above 0.5.
    x = x*2.0 - 1.0;
    x *= x;
    x = 1.0 - x;
    let x2 = x*x;
    let x3 = x2*x;
    x2*0.56 + x3*0.44
}
pub (crate) fn sinc_f64(mut x : f64) -> f64
{
    x *= std::f64::consts::PI;
    if x == 0.0 { 1.0 } else { x.sin() / x }
}
pub (crate) fn sinc(mut x : f32) -> f32
{
    x *= std::f32::consts::PI;
    if x == 0.0 { 1.0 } else { x.sin() / x }
}
pub (crate) fn generate_sinc_table(filter_size : usize, adjusted_freq : f32) -> (Vec<f32>, Vec<f32>)
{
    let mut sinc_table = vec!(0.0f32; filter_size);
    let mut tangent_table = vec!(0.0f32; filter_size);
    let mut energy = 0.0;
    for i in 0..filter_size
    {
        let t = (i as f64 + if filter_size % 2 == 0 { 0.0 } else { 0.5 }) / (filter_size as f64);
        let w = window(t as f32) as f64;
        let x = (t * 2.0 - 1.0) * filter_size as f64 / 2.0 * adjusted_freq as f64;
        let e = sinc_f64(x) * w;
        let a = sinc_f64(x-0.0001) * w;
        let b = sinc_f64(x+0.0001) * w;
        let d = (b-a)*5000.0;
        energy += e;
        sinc_table[i] = e as f32;
        tangent_table[i] = d as f32;
    }
    println!("filter energy: {}. normalizing.", energy);
    for i in 0..filter_size
    {
        sinc_table[i] /= energy as f32;
        tangent_table[i] /= energy as f32;
    }
    (sinc_table, tangent_table)
}
/*
pub (crate) fn interpolate_kernel(table : &[f32], tangents : &[f32], i : usize, t : f32) -> f32
{
    let a = table[i%table.len()];
    let b = table[(i+1)%table.len()];
    //let at = tangents[i%tangents.len()];
    //let bt = tangents[(i+1)%tangents.len()];
    a * (1.0 - t) + b * t
}
pub (crate) fn interpolate_sample(table : &[Sample], i : usize, t : f32) -> Sample
{
    let a = table[i%table.len()];
    let b = table[(i+1)%table.len()];
    a * (1.0 - t) + b * t
}
*/
// sinc resampler
pub (crate) fn resample(in_data : &[Sample], factor : f64) -> Vec<Sample>
{
    if factor == 1.0
    {
        return in_data.to_vec();
    }
    // TODO: use interpolated sinc table instead of using sinc directly
    /*
    let filter_size = 128;
    let (kernel, tangents) = generate_sinc_table(filter_size, 1.0/8.0);
    println!("{}", (kernel[45]-kernel[43])/2.0);
    println!("{}", tangents[44]);
    let fast_sinc = |mut x : f32| -> f32
    {
        x *= 8.0;
        0.0
    };
    for k in kernel.iter()
    {
        print!("{:.4}, ", k);
    }
    println!("");
    println!("");
    for i in 1..kernel.len()-1
    {
        print!("{:.4}, ", (kernel[i+1] - kernel[i-1])/2.0);
    }
    let o = filter_size/2;
    */
    println!("");
    let mut new_data = vec!(Sample { l : 0.0, r : 0.0 }; (in_data.len() as f64 * factor) as usize);
    let halflobe_range = 16;
    for i in 0..new_data.len()
    {
        let mut sum = Sample::default();
        let mut e = 0.0;
        
        let source_center = i as f64 / factor;
        let floored_pos = source_center.floor() as isize;
        let mut range = halflobe_range;
        let mut t_factor = 1.0;
        if factor < 1.0 // downsampling
        {
            range = (16.0 / factor) as isize;
            t_factor = factor as f32;
        }
        for j in -range..=range
        {
            let source = floored_pos + j;
            let t = (source as f64 - source_center) as f32;
            let mut w = window((t / range as f32)/2.0+0.5);
            w *= w;
            let sample = in_data.get(source as usize).map(|x| *x).unwrap_or_default();
            let energy = sinc(t * t_factor)*w;
            sum += sample * energy;
            e += energy;
        }
        sum /= e;
        new_data[i] = sum;
    }
    new_data
}

pub (crate) fn do_freq_split(in_data : &[Sample], samplerate : f64, filter_size_s : f64, freq : f64) -> (Vec<Sample>, Vec<Sample>)
{
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
    let (sinc_table, _) = generate_sinc_table(filter_size, adjusted_freq);
    
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

pub (crate) fn pitch_analysis(data : &[Sample], samplerate : f64, pos : isize, len : isize) -> (f32, f32)
{
    let mut len = len as usize * 2;
    len = len.clamp(441, 4410); // FIXME ????
    
    let pos = (pos - (len/2) as isize).max(0);
    while pos as usize + len > data.len()
    {
        len /= 2; // FIXME ???? why not just set it to data.len() - pos
    }
    
    let signal = &data[pos as usize..pos as usize + len];
    let mean = signal.iter().map(|x| x.l + x.r).sum::<f32>() / len as f32;
    
    let base_square_sum = signal.iter().map(|x| (x.l + x.r - mean) * (x.l + x.r - mean)).sum::<f32>();
    
    let mut correlations : Vec<f32> = Vec::new();
    for i in 0..len/2
    {
        let mut square_sum = 0.0;
        let mut j = 0;
        while j < len && i + j < signal.len()
        {
            let a = signal[j];
            let b = signal[i + j];
            square_sum += (a.l + a.r - mean) * (b.l + b.r - mean);
            j += 8; // FIXME: performance hack
        }
        correlations.push(square_sum / base_square_sum * (len as f32 / (j/8) as f32));
    }
    let mut found_negative = false;
    let mut max = 0.0;
    let mut max_i = 0;
    for i in 0..correlations.len()
    {
        if found_negative
        {
            if correlations[i] > max
            {
                max = correlations[i];
                max_i = i;
            }
        }
        if correlations[i] < 0.0
        {
            found_negative = true;
        }
    }
    if max_i > 0
    {
        let info = ((samplerate / max_i as f64) as f32, max);
        //println!("{}\t{}\t{}", info.0, pos, info.1);
        return info;
    }
    (440.0, 0.0)
}
/*
fn lerp(a : f64, b : f64, t : f64) -> f64
{
    a * (1.0 - t) + b * t
}
fn lerp_f32(a : f32, b : f32, t : f32) -> f32
{
    a * (1.0 - t) + b * t
}
*/