extern crate hound;

use hound::{WavReader, WavWriter, Error};

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
    fn energy(&self) -> f32
    {
        self.energy_sq().sqrt()
    }
    fn energy_abs(&self) -> f32
    {
        self.l.abs() + self.r.abs()
    }
}

impl core::ops::Add<Sample> for Sample
{
    type Output = Sample;

    fn add(self, other: Sample) -> Sample
    {
        Sample
        {
            l: self.l + other.l,
            r: self.r + other.r,
        }
    }
}

impl core::ops::Sub<Sample> for Sample
{
    type Output = Sample;

    fn sub(self, other: Sample) -> Sample
    {
        Sample
        {
            l: self.l - other.l,
            r: self.r - other.r,
        }
    }
}

impl core::ops::Mul<f32> for Sample
{
    type Output = Sample;

    fn mul(self, scalar: f32) -> Sample
    {
        Sample
        {
            l: self.l * scalar,
            r: self.r * scalar,
        }
    }
}

impl core::ops::Div<f32> for Sample
{
    type Output = Sample;

    fn div(self, divisor: f32) -> Sample
    {
        Sample
        {
            l: self.l / divisor,
            r: self.r / divisor,
        }
    }
}


impl core::ops::AddAssign<Sample> for Sample {
    fn add_assign(&mut self, other: Sample) {
        self.l += other.l;
        self.r += other.r;
    }
}

impl core::ops::SubAssign<Sample> for Sample {
    fn sub_assign(&mut self, other: Sample) {
        self.l -= other.l;
        self.r -= other.r;
    }
}

impl core::ops::MulAssign<f32> for Sample {
    fn mul_assign(&mut self, scalar: f32) {
        self.l *= scalar;
        self.r *= scalar;
    }
}

impl core::ops::DivAssign<f32> for Sample {
    fn div_assign(&mut self, divisor: f32) {
        self.l /= divisor;
        self.r /= divisor;
    }
}

fn window(mut x : f32) -> f32
{
    x = x*2.0 - 1.0;
    x = x*x;
    x = 1.0 - x * (2.0 - x);
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
        
        let mut t = (j as f32 + 0.5) / (len as f32);
        let w = window(t);
        
        let a = pos_a_source[ap as usize];
        let b = pos_b_source[bp as usize] * w;
        energy += (a + b).energy_abs();
    }
    energy
}

fn find_best_overlap(pos_a_source : &[Sample], pos_a : isize, pos_b_source : &[Sample], pos_b : isize, len : isize, range : isize) -> isize
{
    let subdiv = 5;
    let mut best_energy = 0.0;
    let mut best_offset = 0;
    
    let pass_count = 5;
    for pass in 1..=pass_count
    {
        let base_offset = best_offset;
        for i in -subdiv..=subdiv
        {
            let offset = range*i/subdiv.pow(pass) + base_offset;
            let energy = calc_overlap_energy(pos_a_source, pos_a + offset, pos_b_source, pos_b, len);
            if energy > best_energy
            {
                best_energy = energy;
                best_offset = offset;
            }
        }
    }
    
    best_offset
}

fn do_timestretch(in_data : &[Sample], freq : f64, scale_length : f64, window_secs : f64) -> Vec<Sample>
{
    let window_size = ((freq * window_secs    ) as isize).max(4);
    let search_dist = ((freq * window_secs/4.0) as isize).min(window_size/2-1).max(0);
    
    let mut out_data = vec![Sample { l: 0.0, r: 0.0 }; (in_data.len() as f64 * scale_length as f64) as usize];
    let mut envelope = vec![0.0; out_data.len()];
    
    let lapping = 2;
    
    for i in 0..out_data.len() as isize/window_size*lapping
    {
        let chunk_pos_out = i*window_size/lapping;
        let mut chunk_pos_in = chunk_pos_out * in_data.len() as isize / out_data.len() as isize;
        
        let offs = if i > 0
        {
            find_best_overlap(&out_data[..], chunk_pos_out, &in_data[..], chunk_pos_in, window_size, search_dist)
        }
        else
        {
            0
        };
        
        for j in 0..window_size
        {
            if (chunk_pos_out + j + offs) as usize >= out_data.len() || (chunk_pos_in + j) as usize >= in_data.len()
            {
                break;
            }
            let mut t = (j as f32 + 0.5) / (window_size as f32);
            let w = window(t);
            let d = in_data[(chunk_pos_in + j) as usize];
            out_data[(chunk_pos_out + j + offs) as usize] += d*w;
            envelope[(chunk_pos_out + j + offs) as usize] += w;
        }
    }
    for i in 0..out_data.len()
    {
        out_data[i] /= envelope[i];
    }
    out_data
}

fn main() -> Result<(), Error>
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3
    {
        println!("usage: slipstretch input.wav output.wav");
        return Ok(());
    }
    
    // Load audio from a WAV file into a Vec<i16>
    let mut reader = WavReader::open(&args[1])?;
    let in_data: Vec<i16> = reader.samples::<i16>().map(|sample| sample.unwrap()).collect();
    
    let freq = reader.spec().sample_rate;
    
    let window_secs = 0.05;
    
    let in_data : Vec<Sample> = in_data
        .chunks(2)
        .map(|chunk| match chunk.len()
        {
            2 => Sample
            {
                l: chunk[0] as f32 / 32768.0,
                r: chunk[1] as f32 / 32768.0,
            },
            1 => Sample
            {
                l: chunk[0] as f32 / 32768.0,
                r: 0.0,
            },
            count => panic!(
                "unsupported audio channel count {} (only 1- and 2-channel audio supported)",
                count
            ),
        })
        .collect();
    
    let out_data = do_timestretch(&in_data[..], freq as f64, 1.1, 0.08);
    
    // Convert stereo audio data back to a Vec<i16> for writing
    let out_i16: Vec<i16> = out_data
        .into_iter()
        .flat_map(|sample| vec![(sample.l * 32768.0) as i16, (sample.r * 32768.0) as i16])
        .collect();

    // Save modified audio data to a WAV file
    let spec = hound::WavSpec
    {
        channels: 2, // Stereo
        sample_rate: 44100, // Sample rate (adjust as needed)
        bits_per_sample: 16, // 16-bit integer
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::create(&args[2], spec)?;

    for sample in out_i16
    {
        writer.write_sample(sample)?;
    }

    println!("Audio data saved to {}", args[2]);

    Ok(())
}