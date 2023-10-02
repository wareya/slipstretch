mod sample;
use sample::Sample;
mod sampling;
use sampling::*;
mod frontend;
use frontend::*;

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
            
            // bias towards center
            let mut d = (i+search_subdiv) as f32 / (search_subdiv*2) as f32;
            d = d*2.0-1.0;
            d *= len as f32 / 5000.0;
            d *= i as f32 / p as f32;
            let mut central_bias = window(d*0.5+0.5) + 2.0;
            
            if offset == 0 // bias slightly towards the exact zero offset
            {
                central_bias += 1.0;
            }
            
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

fn do_timestretch(in_data : &[Sample], out_data : &mut Vec<Sample>, samplerate : f64, args : &Args, window_secs : f64, known_offsets : Option<&Vec<isize>>) -> Vec<isize>
{
    let slip_range = args.slip_range.min(0.5);
    let length_scale = args.length_scale;
    let window_size = ((samplerate * window_secs                  ) as isize).max(4);
    let search_dist = ((samplerate * window_secs * args.slip_range) as isize).min(window_size/2-1).max(0);
    
    let out_len = (in_data.len() as f64 * length_scale as f64) as usize;
    assert!(out_data.len() == (in_data.len() as f64 * length_scale as f64) as usize);
    // if this is multi-pass, out_data contains data from previous passes, otherwise all zeroes
    // we use the previous pass to help reduce phase cancellation artifacts
    // however, it's hard (maybe impossible) to do envelope adjustment 100% correctly with just a single buffer
    // so we also write to our own buffer and copy the original previous pass data so we can mix them together properly at the end
    let mut own_data = vec![Sample { l: 0.0, r: 0.0 }; out_len];
    let mut orig_data = out_data.clone();
    
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
            own_data[(chunk_pos_out + j + offs) as usize] += d*w;
            envelope[(chunk_pos_out + j + offs) as usize] += w;
        }
    }
    for i in 0..out_data.len()
    {
        if envelope[i] != 0.0
        {
            own_data[i] /= envelope[i];
        }
        out_data[i] = orig_data[i] + own_data[i];
    }
    offsets
}


fn main() -> Result<(), hound::Error>
{
    use clap::Parser;
    let mut args = Args::parse();
    
    let (mut in_data, samplerate) = frontend_acquire_audio(&args);
    
    let do_multiband = args.fullband_window_secs == 0.0;
    
    if args.pitch_scale < 1.0
    {
        println!("pre-resampling audio for pitch stretching...");
        in_data = resample(&in_data[..], 1.0/args.pitch_scale);
    }
    args.length_scale *= args.pitch_scale;
    
    let out_len = (in_data.len() as f64 * args.length_scale as f64) as usize;
    let mut out_data = vec![Sample { l: 0.0, r: 0.0 }; out_len];
    
    if do_multiband
    {
        let (bass, _temp)      = do_freq_split(&in_data[..], samplerate as f64, args.cutoff_steepness, args.cutoff_bass_mid);
        let (mid, _temp)       = do_freq_split(&_temp[..]  , samplerate as f64, args.cutoff_steepness, args.cutoff_mid_treble);
        let (treble, presence) = do_freq_split(&_temp[..]  , samplerate as f64, args.cutoff_steepness, args.cutoff_treble_presence);
        
        println!("timestretching presence frequency band...");
        let presence_offs = do_timestretch(&presence [..], &mut out_data, samplerate as f64, &args, args.window_secs_presence, None);
        println!("timestretching treble frequency band...");
        let treble_offs   = do_timestretch(&treble   [..], &mut out_data, samplerate as f64, &args, args.window_secs_treble, Some(&presence_offs));
        println!("timestretching mid frequency band...");
        let mid_offs      = do_timestretch(&mid      [..], &mut out_data, samplerate as f64, &args, args.window_secs_mid, Some(&treble_offs));
        println!("timestretching bass frequency band...");
        let _             = do_timestretch(&bass     [..], &mut out_data, samplerate as f64, &args, args.window_secs_bass, Some(&mid_offs));
    }
    else
    {
        println!("timestretching full-spectrum audio...");
        let _ = do_timestretch(&in_data[..], &mut out_data, samplerate as f64, &args, args.fullband_window_secs, None);
    }
    if args.pitch_scale > 1.0
    {
        println!("post-resampling audio for pitch stretching...");
        out_data = resample(&out_data[..], 1.0/args.pitch_scale);
    }
    
    frontend_save_audio(&args, &out_data[..], samplerate);

    Ok(())
}