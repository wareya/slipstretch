use std::str::FromStr;

#[derive(Clone, Copy, Debug)]
struct Sample
{
    l: f32,
    r: f32,
}

impl Sample
{
    fn energy_sq(&self) -> f32
    {
        self.l*self.l + self.r*self.r
    }
}
impl core::ops::Add<Sample> for Sample
{
    type Output = Sample;
    fn add(self, other: Sample) -> Sample
    {
        Sample { l: self.l + other.l, r: self.r + other.r }
    }
}
impl core::ops::Sub<Sample> for Sample
{
    type Output = Sample;
    fn sub(self, other: Sample) -> Sample
    {
        Sample { l: self.l - other.l, r: self.r - other.r }
    }
}
impl core::ops::Mul<f32> for Sample
{
    type Output = Sample;
    fn mul(self, scalar: f32) -> Sample
    {
        Sample { l: self.l * scalar, r: self.r * scalar }
    }
}
impl core::ops::Div<f32> for Sample
{
    type Output = Sample;
    fn div(self, divisor: f32) -> Sample
    {
        Sample { l: self.l / divisor, r: self.r / divisor }
    }
}
impl core::ops::AddAssign<Sample> for Sample
{
    fn add_assign(&mut self, other: Sample)
    {
        self.l += other.l;
        self.r += other.r;
    }
}
impl core::ops::SubAssign<Sample> for Sample
{
    fn sub_assign(&mut self, other: Sample)
    {
        self.l -= other.l;
        self.r -= other.r;
    }
}

impl core::ops::MulAssign<f32> for Sample
{
    fn mul_assign(&mut self, scalar: f32)
    {
        self.l *= scalar;
        self.r *= scalar;
    }
}

impl core::ops::DivAssign<f32> for Sample
{
    fn div_assign(&mut self, divisor: f32)
    {
        self.l /= divisor;
        self.r /= divisor;
    }
}

fn lerp(a : f32, b : f32, t : f32) -> f32
{
    a * (1.0 - t) + b * t
}
fn window(mut x : f32) -> f32
{
    x = x*2.0 - 1.0;
    x = x*x;
    x = 1.0 - x * (2.0 - x);
    x = 2.0 * (x * x) * (1.5 - x);
    x
}

fn calc_overlap_energy(pos_a_source : &[Sample], pos_a : isize, pos_b_source : &[Sample], pos_b : isize, len : isize) -> f32
{
    let mut energy = 0.0;
    for j in 0..len
    {
        let ap = pos_a + j;
        let bp = pos_b + j;
        if ap < 0 || ap as usize >= pos_a_source.len() || bp as usize >= pos_b_source.len()
        {
            continue;
        }
        
        let t = (j as f32 + 0.5) / (len as f32);
        let w = window(t);
        
        let a = pos_a_source[ap as usize];
        let b = pos_b_source[bp as usize] * w;
        energy += (a + b).energy_sq();
    }
    energy
}

fn find_best_overlap(pos_a_source : &[Sample], pos_a : isize, pos_b_source : &[Sample], pos_b : isize, len : isize, range : isize, min_offs : isize) -> isize
{
    let subdiv = 5;
    let mut best_energy = 0.0;
    let mut best_offset = 0;
    
    let pass_count = 4;
    for pass in 1..=pass_count
    {
        let base_offset = best_offset;
        for i in -subdiv..=subdiv
        {
            let offset = range*i/subdiv.pow(pass) + base_offset;
            if offset < min_offs
            {
                continue;
            }
            let central_bias = window((i+subdiv) as f32 / (subdiv*2) as f32) + 2.0;
            let energy = calc_overlap_energy(pos_a_source, pos_a + offset, pos_b_source, pos_b, len) * central_bias;
            if energy > best_energy
            {
                best_energy = energy;
                best_offset = offset;
            }
        }
    }
    best_offset
}

fn do_timestretch(in_data : &[Sample], samplerate : f64, scale_length : f64, window_secs : f64, known_offsets : Option<&Vec<isize>>) -> (Vec<Sample>, Vec<isize>)
{
    let window_size = ((samplerate * window_secs      ) as isize).max(4);
    let search_dist = ((samplerate * window_secs / 4.0) as isize).min(window_size/2-1).max(0);
    
    let mut out_data = vec![Sample { l: 0.0, r: 0.0 }; (in_data.len() as f64 * scale_length as f64) as usize];
    let mut envelope = vec![0.0; out_data.len()];
    
    let mut lapping = 2;
    if scale_length < 1.0
    {
        lapping = lapping.max((2.0 / scale_length.min(1.0)).ceil() as isize);
    }
    println!("lapping: {}", lapping);
    
    let mut offsets = Vec::new();
    for i in 0..out_data.len() as isize/window_size*lapping
    {
        let chunk_pos_out = i*window_size/lapping;
        let pure_offset = ((1.0 - 2.0_f64.powf(1.0 - scale_length)) * (window_size/2) as f64) as isize; // this is a guess
        let chunk_pos_in = chunk_pos_out * in_data.len() as isize / out_data.len() as isize - pure_offset;
        
        let min_offs = known_offsets.map(|x| x[i as usize]).unwrap_or(-1000000);
        let mut offs = 0;
        if i > 0
        {
            offs = find_best_overlap(&out_data[..], chunk_pos_out, &in_data[..], chunk_pos_in, window_size, search_dist, min_offs);
        }
        offsets.push(offs);
        
        for j in 0..window_size
        {
            if (chunk_pos_out + j + offs) as usize >= out_data.len() || (chunk_pos_in + j) as usize >= in_data.len()
            {
                break;
            }
            let t = (j as f32 + 0.5) / (window_size as f32);
            let w = window(t);
            let d = in_data[(chunk_pos_in + j) as usize];
            out_data[(chunk_pos_out + j + offs) as usize] += d*w;
            envelope[(chunk_pos_out + j + offs) as usize] += w;
        }
    }
    for i in 0..out_data.len()
    {
        if envelope[i] != 0.0
        {
            out_data[i] /= envelope[i];
        }
    }
    (out_data, offsets)
}

fn do_freq_split(in_data : &[Sample], samplerate : f64, freq : f64) -> (Vec<Sample>, Vec<Sample>)
{
    let bandwidth_length = 1.0 / freq;
    let filter_size_ms = bandwidth_length * 8.0; // 8.0 - filter bandwidth constant
    let filter_size = ((samplerate * filter_size_ms) as usize).max(1);
    if filter_size % 2 == 0
    {
        println!("filter size ({}) is even. building sinc table", filter_size);
    }
    else
    {
        println!("filter size ({}) is NOT even (is odd). building sinc table", filter_size);
    }
    
    let mut sinc_table = vec!(0.0f32; filter_size);
    let adjusted_freq = (freq * 2.0 / samplerate) as f32;
    let mut energy = 0.0;
    for i in 0..filter_size
    {
        let t = (i as f32 + if filter_size % 2 == 0 { 0.0 } else { 0.5 }) / (filter_size as f32);
        let w = window(t);
        let x = (t * 2.0 - 1.0) * filter_size as f32 / 2.0 * adjusted_freq * std::f32::consts::PI;
        let e = if x == 0.0
        {
            1.0
        }
        else
        {
            x.sin() / x
        } * w;
        energy += e;
        sinc_table[i] = e;
    }
    for i in 0..filter_size
    {
        sinc_table[i] /= energy;
    }
    println!("filter energy: {}. normalizing.", energy);
    
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

fn main() -> Result<(), hound::Error>
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4
    {
        println!("usage: slipstretch input.wav output.wav time_scale [full_spectrum_mode_window_size]");
        return Ok(());
    }
    
    let mut reader = hound::WavReader::open(&args[1])?;
    let sample_format = reader.spec().sample_format;

    let in_data: Vec<f32> = match sample_format
    {
        hound::SampleFormat::Int => reader.samples::<i16>().map(|sample| sample.unwrap() as f32 / 32768.0).collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|sample| sample.unwrap()).collect()
    };

    let in_data : Vec<Sample> = in_data
        .chunks(reader.spec().channels.into())
        .map(|chunk| match chunk.len()
        {
            2 => Sample { l: chunk[0], r: chunk[1] },
            1 => Sample { l: chunk[0], r: chunk[0] },
            count => panic!("unsupported audio channel count {} (only 1- and 2-channel audio supported)", count)
        }).collect();
    
    let samplerate = reader.spec().sample_rate;
    
    let time_factor = f64::from_str(&args[3]).unwrap_or_else(|_| panic!("third argument must be a number"));
    let multiband_window_size = args.get(4).map(|x| f64::from_str(x).unwrap_or_else(|_| panic!("fourth argument must be a number"))).unwrap_or(0.0);
    let do_multiband = multiband_window_size == 0.0;
    
    let output = if do_multiband
    {
        let window_secs_bass      = 0.2;
        let window_secs_mid       = 0.16;
        let window_secs_treble    = 0.08;
        let window_secs_presence  = 0.005;
        
        let (bass, _temp)      = do_freq_split(&in_data[..], samplerate as f64, 400.0);
        let (mid, _temp)       = do_freq_split(&_temp[..]  , samplerate as f64, 1600.0);
        let (treble, presence) = do_freq_split(&_temp[..]  , samplerate as f64, 4800.0);
        
        println!("timestretching presence frequency band...");
        let (out_presence , presence_offs) = do_timestretch(&presence [..], samplerate as f64, time_factor, window_secs_presence, None);
        println!("timestretching treble frequency band...");
        let (out_treble   , treble_offs  ) = do_timestretch(&treble   [..], samplerate as f64, time_factor, window_secs_treble, Some(&presence_offs));
        println!("timestretching mid frequency band...");
        let (out_mid      , mid_offs     ) = do_timestretch(&mid      [..], samplerate as f64, time_factor, window_secs_mid, Some(&treble_offs));
        println!("timestretching bass frequency band...");
        let (out_bass     , _            ) = do_timestretch(&bass     [..], samplerate as f64, time_factor, window_secs_bass, Some(&mid_offs));
        
        let mut out_data = out_bass.clone();
        for (i, val) in out_data.iter_mut().enumerate()
        {
            *val += out_mid[i];
            *val += out_treble[i];
            *val += out_presence[i];
        }
        
        out_data
    }
    else
    {
        println!("timestretching full-spectrum audio...");
        let (out_raw, _) = do_timestretch(&in_data[..], samplerate as f64, time_factor, multiband_window_size, None);
        out_raw
    };
    
    let out: Vec<_> = output.into_iter().flat_map(|sample| vec![sample.l, sample.r]).collect();

    let spec = hound::WavSpec
    {
        channels: 2,
        sample_rate: samplerate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&args[2], spec)?;

    for sample in out
    {
        writer.write_sample(sample)?;
    }

    println!("Audio data saved to {}", args[2]);

    Ok(())
}