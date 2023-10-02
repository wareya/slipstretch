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

fn find_best_overlap(search_subdiv : isize, search_pass_count : u32, pos_a_source : &[Sample], pos_a : isize, pos_b_source : &[Sample], pos_b : isize, len : isize, range : isize, min_offs : isize) -> isize
{
    let mut best_energy = 0.0;
    let mut best_offset = 0;
    
    for pass in 1..=search_pass_count
    {
        let base_offset = best_offset;
        for i in -search_subdiv..=search_subdiv
        {
            let p = search_subdiv.pow(pass);
            let offset = range*i/p as isize + base_offset;
            if offset < min_offs
            {
                continue;
            }
            let mut d = (i+search_subdiv) as f32 / (search_subdiv*2) as f32;
            d = d*2.0-1.0;
            d *= len as f32 / 5000.0;
            d *= i as f32 / p as f32;
            
            let central_bias = window(d*0.5+0.5) + 2.0;
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

fn do_timestretch(in_data : &[Sample], samplerate : f64, args : &Args, window_secs : f64, known_offsets : Option<&Vec<isize>>) -> (Vec<Sample>, Vec<isize>)
{
    let slip_range = args.slip_range.min(0.5);
    let length_scale = args.length_scale;
    let window_size = ((samplerate * window_secs                  ) as isize).max(4);
    let search_dist = ((samplerate * window_secs * args.slip_range) as isize).min(window_size/2-1).max(0);
    
    let mut out_data = vec![Sample { l: 0.0, r: 0.0 }; (in_data.len() as f64 * length_scale as f64) as usize];
    let mut envelope = vec![0.0; out_data.len()];
    
    let mut lapping = 2;
    if length_scale < 1.0
    {
        lapping = lapping.max((2.0 / length_scale.min(1.0)).ceil() as isize);
    }
    println!("lapping: {}", lapping);
    
    let mut offsets = Vec::new();
    for i in 0..out_data.len() as isize/window_size*lapping
    {
        let chunk_pos_out = i*window_size/lapping;
        let pure_offset = ((1.0 - 2.0_f64.powf(1.0 - length_scale.powf((lapping - 1) as f64))) * (window_size/2) as f64) as isize; // this is a guess
        let chunk_pos_in = chunk_pos_out * in_data.len() as isize / out_data.len() as isize - pure_offset;
        
        let min_offs = known_offsets.map(|x| x[i as usize]).unwrap_or(-1000000);
        let mut offs = 0;
        if i > 0
        {
            offs = find_best_overlap(args.search_subdiv, args.search_pass_count, &out_data[..], chunk_pos_out, &in_data[..], chunk_pos_in, window_size, search_dist, min_offs);
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

fn generate_sinc_table(filter_size : usize, adjusted_freq : f32) -> Vec<f32>
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
fn interpolate(table : &[f32], i : usize, t : f32) -> f32
{
    let a = table[i%table.len()];
    let b = table[(i+1)%table.len()];
    a * (1.0 - t) + b * t
}
fn interpolate_sample(table : &[Sample], i : usize, t : f32) -> Sample
{
    let a = table[i%table.len()];
    let b = table[(i+1)%table.len()];
    a * (1.0 - t) + b * t
}
fn resample(in_data : &[Sample], factor : f64) -> Vec<Sample>
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

fn do_freq_split(in_data : &[Sample], samplerate : f64, cutoff_steepness : f64, freq : f64) -> (Vec<Sample>, Vec<Sample>)
{
    let bandwidth_length = 1.0 / freq;
    let filter_size_ms = bandwidth_length * cutoff_steepness;
    let filter_size = ((samplerate * filter_size_ms) as usize).max(1);
    if filter_size % 2 == 0
    {
        println!("filter size ({}) is even. building sinc table", filter_size);
    }
    else
    {
        println!("filter size ({}) is NOT even (is odd). building sinc table", filter_size);
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

use clap::Parser;
use std::path::PathBuf;

/// An audio stretcher.
#[derive(Debug, Parser)]
struct Args {
    /// Input wav filename.
    in_file_name: String,

    /// Output wav filename
    out_file_name: String,

    /// Length scale. A value of 0.5 makes the audio play twice as fast.
    /// To convert a 120bpm song to 140bpm, you would run 120/140 through a calculator and use that value.
    #[arg(short = 'l', long, default_value_t=1.0)]
    length_scale: f64,

    /// Pitch scale. A value of 0.5 makes the audio be pitched down by an octave.
    /// To pitch shift by semitones, run 2^(<semitones>/12) through a calculator and use that value.
    #[arg(short = 'p', long, default_value_t=1.0)]
    pitch_scale: f64,

    /// The range in which the shifting stage of the algorithm is allowed to try to shift each window to reduce phasing artifacts.
    /// Recommendation: Leave at default, or set to lower than default if you're having bad flam or tempo shifting artifacts.
    /// Higher values will have more flam/tempo shifting artifacts. Lower values will have more phasing artifacts.
    /// Default: 0.25. Maximum: 0.5.
    #[arg(short = 's', long, default_value_t=0.25)]
    slip_range: f64,

    /// Window size for full-band pitch shifting, in seconds.
    /// This causes the algorithm to run without splitting the audio up into multiple frequency bands first.
    /// If unset, the algorithm will run on frequency bands instead, which is better for most types of audio.
    /// Useful values range from 0.2 to 0.01, depending on the audio. 0.08 works decently well for speech.
    /// Recommendation: Do not use. Leave unset.
    #[arg(short = 'z', long, default_value_t=0.0)]
    fullband_window_secs: f64,

    /// Number of times to subdivide the slip range in each direction when searching for good chunk alignment.
    #[arg(long, default_value_t=5)]
    search_subdiv: isize,
    
    /// Number of increasingly subdivided search passes to do when searching for good chunk alignment.
    #[arg(long, default_value_t=4)]
    search_pass_count: u32,
    
    /// In multiband mode (the default mode), the chunk window size (in seconds) used when stretching the bass frequency band.
    /// Smaller chunk window sizes give better time-domain but worse frequency-domain results.
    #[arg(long, default_value_t=0.2)]
    window_secs_bass : f64,
    /// In multiband mode, the chunk window size used when stretching the mid frequency band.
    #[arg(long, default_value_t=0.16)]
    window_secs_mid : f64,
    /// In multiband mode, the chunk window size used when stretching the treble frequency band.
    #[arg(long, default_value_t=0.08)]
    window_secs_treble : f64,
    /// In multiband mode, the chunk window size used when stretching the presence frequency band.
    #[arg(long, default_value_t=0.008)]
    window_secs_presence : f64,
    
    /// For multiband mode, the cutoff frequency between the bass and mid frequency bands.
    #[arg(long, default_value_t=400.0)]
    cutoff_bass_mid : f64,
    /// For multiband mode, the cutoff frequency between the mid and treble frequency bands.
    #[arg(long, default_value_t=1600.0)]
    cutoff_mid_treble : f64,
    /// For multiband mode, the cutoff frequency between the treble and presence frequency bands.
    #[arg(long, default_value_t=4800.0)]
    cutoff_treble_presence : f64,
    
    /// The steepness of the filter that separates each frequency bands.
    /// The filter is a windowed sinc filter, not an IIR filter.
    /// This value is proportional to (but not equal to) the number of lobes present in the windowed sinc kernel.
    /// Higher values are slower, because a larger filter must be used. Lower values have more phasing artifacts where the frequency bands cross.
    /// Extremely high values will produce pre-ringing artifacts on sharp transients.
    #[arg(long, default_value_t=4.0)]
    cutoff_steepness : f64,
}

fn main() -> Result<(), hound::Error>
{
    let mut args = Args::parse();
    
    let mut reader = hound::WavReader::open(&args.in_file_name)?;
    let sample_format = reader.spec().sample_format;

    let in_data: Vec<f32> = match sample_format
    {
        hound::SampleFormat::Int => reader.samples::<i16>().map(|sample| sample.unwrap() as f32 / 32768.0).collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|sample| sample.unwrap()).collect()
    };

    let mut in_data : Vec<Sample> = in_data
        .chunks(reader.spec().channels.into())
        .map(|chunk| match chunk.len()
        {
            2 => Sample { l: chunk[0], r: chunk[1] },
            1 => Sample { l: chunk[0], r: chunk[0] },
            count => panic!("unsupported audio channel count {} (only 1- and 2-channel audio supported)", count)
        }).collect();
    
    let samplerate = reader.spec().sample_rate;
    let do_multiband = args.fullband_window_secs == 0.0;
    
    if args.pitch_scale < 1.0
    {
        println!("pre-resampling audio for pitch stretching...");
        in_data = resample(&in_data[..], 1.0/args.pitch_scale);
    }
    args.length_scale *= args.pitch_scale;
    
    let mut output = if do_multiband
    {
        let (bass, _temp)      = do_freq_split(&in_data[..], samplerate as f64, args.cutoff_steepness, args.cutoff_bass_mid);
        let (mid, _temp)       = do_freq_split(&_temp[..]  , samplerate as f64, args.cutoff_steepness, args.cutoff_mid_treble);
        let (treble, presence) = do_freq_split(&_temp[..]  , samplerate as f64, args.cutoff_steepness, args.cutoff_treble_presence);
        
        println!("timestretching presence frequency band...");
        let (out_presence , presence_offs) = do_timestretch(&presence [..], samplerate as f64, &args, args.window_secs_presence, None);
        println!("timestretching treble frequency band...");
        let (out_treble   , treble_offs  ) = do_timestretch(&treble   [..], samplerate as f64, &args, args.window_secs_treble, Some(&presence_offs));
        println!("timestretching mid frequency band...");
        let (out_mid      , mid_offs     ) = do_timestretch(&mid      [..], samplerate as f64, &args, args.window_secs_mid, Some(&treble_offs));
        println!("timestretching bass frequency band...");
        let (out_bass     , _            ) = do_timestretch(&bass     [..], samplerate as f64, &args, args.window_secs_bass, Some(&mid_offs));
        
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
        let (out_raw, _) = do_timestretch(&in_data[..], samplerate as f64, &args, args.fullband_window_secs, None);
        out_raw
    };
    if args.pitch_scale > 1.0
    {
        println!("post-resampling audio for pitch stretching...");
        output = resample(&output[..], 1.0/args.pitch_scale);
    }
    
    let out: Vec<_> = output.into_iter().flat_map(|sample| vec![sample.l, sample.r]).collect();

    let spec = hound::WavSpec
    {
        channels: 2,
        sample_rate: samplerate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&args.out_file_name, spec)?;

    for sample in out
    {
        writer.write_sample(sample)?;
    }

    println!("Audio data saved to {}", args.out_file_name);

    Ok(())
}