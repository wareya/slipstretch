use crate::sample::Sample;
use crate::sampling::*;

#[derive(Clone, Debug)]
pub struct SlipStretchArgs {
    pub length_scale: f64,
    pub pitch_scale: f64,

    pub slip_range: f64,

    pub fullband_window_secs: f64,

    pub search_subdiv: isize,
    pub search_pass_count: u32,
    
    pub window_secs_bass : f64,
    pub window_secs_mid : f64,
    pub window_secs_treble : f64,
    pub window_secs_presence : f64,
    
    pub window_minimum_lapping : isize,
    
    pub cutoff_bass_mid : f64,
    pub cutoff_mid_treble : f64,
    pub cutoff_treble_presence : f64,
    
    pub cutoff_steepness : f64,
    pub filter_length : f64,
    
    pub combined_energy_estimation : bool,
    
    pub smart_band_offset : bool,
    
    pub amplitude_bass : f64,
    pub amplitude_mid : f64,
    pub amplitude_treble : f64,
    pub amplitude_presence : f64,
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
            
            // bias towards center
            let mut d = (i+search_subdiv) as f32 / (search_subdiv*2) as f32;
            d = d*2.0-1.0;
            d *= len as f32 / 5000.0;
            d *= i as f32 / p as f32;
            let mut central_bias = window(d*0.5+0.5) + 1.5;
            
            if offset == 0 && range < 450 // bias slightly towards the exact zero offset if this is probably the presence frequency bin
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

fn do_timestretch(in_data : &[Sample], out_data : &mut Vec<Sample>, samplerate : f64, args : &SlipStretchArgs, which_pass : i32, window_secs : f64, known_offsets : Option<&Vec<(isize, isize)>>) -> Vec<(isize, isize)>
{
    let slip_range = args.slip_range.min(0.5);
    let length_scale = args.length_scale;
    let window_size = ((samplerate * window_secs             ) as isize).max(4);
    let search_dist = ((samplerate * window_secs * slip_range) as isize).min(window_size/2-1).max(0);
    
    let out_len = (in_data.len() as f64 * length_scale as f64) as usize;
    assert!(out_data.len() == (in_data.len() as f64 * length_scale as f64) as usize);
    // if this is multi-pass and also not the first pass, out_data contains data from previous passes, otherwise all zeroes
    // we use the previous pass to help reduce phase cancellation artifacts
    // however, it's hard (maybe impossible) to do envelope adjustment 100% correctly with just a single buffer
    // so we also write to our own buffer and copy the original previous pass data so we can mix them together properly at the end
    let mut own_data = vec![Sample { l: 0.0, r: 0.0 }; out_len];
    let orig_data = out_data.clone();
    
    let mut envelope = vec![0.0; out_data.len()];
    
    let mut lapping = 2;
    if length_scale < 1.0
    {
        lapping = lapping.max((2.0 / length_scale.min(1.0)).ceil() as isize);
    }
    let raw_lapping = lapping;
    lapping = lapping.max(args.window_minimum_lapping);
    println!("lapping: {}", lapping);
    
    let amp = match which_pass
    {
        0 => args.amplitude_bass,
        1 => args.amplitude_mid,
        2 => args.amplitude_treble,
        3 => args.amplitude_presence,
        _ => 1.0,
    } as f32;
    
    let mut offsets = Vec::new();
    for i in 0..out_data.len() as isize/window_size*lapping
    {
        let chunk_pos_out = i*window_size/lapping;
        
        // this is a guess
        let mut smart_offset = if args.smart_band_offset { ((1.0 - 2.0_f64.powf(1.0 - length_scale.powf((raw_lapping - 1) as f64))) * (window_size*2/3) as f64) as isize } else { 0 };
        
        let min_fadein = (args.length_scale.max(1.0) * 2.0).floor() as isize;
        smart_offset = smart_offset * i.min(min_fadein) / min_fadein; // the smart offset has to fade in or else the first split second of audio can sound wet
        let chunk_pos_in = chunk_pos_out * in_data.len() as isize / out_data.len() as isize - smart_offset;
        
        let mut min_offs = -1000000;
        if let Some(known) = known_offsets
        {
            let _i = known.binary_search_by(|pair| pair.0.cmp(&chunk_pos_out));
            if let Ok(i) = _i
            {
                min_offs = known[i].1;
            }
            else if let Err(i) = _i
            {
                if i > 0
                {
                    min_offs = known[i-1].1;
                    if i < known.len()
                    {
                        min_offs = min_offs.max(known[i].1);
                    }
                }
            }
        }
        let mut offs = 0;
        if i > 0
        {
            offs = find_best_overlap(args.search_subdiv, args.search_pass_count, &out_data[..], chunk_pos_out, &in_data[..], chunk_pos_in, window_size, search_dist, min_offs);
        }
        offsets.push((chunk_pos_out, offs));
        
        for j in 0..window_size
        {
            if (chunk_pos_out + j + offs) as usize >= out_data.len() || (chunk_pos_in + j) as usize >= in_data.len()
            {
                break;
            }
            let t = (j as f32 + 0.5) / (window_size as f32);
            let w = window(t) * amp;
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

pub fn run_slipstretch(samples : &[Sample], samplerate : f64, mut args : SlipStretchArgs) -> Vec<Sample>
{
    let do_multiband = args.fullband_window_secs == 0.0;
    
    let mut in_data = samples;
    let mut _dummy = Vec::new();
    if args.pitch_scale < 1.0
    {
        println!("pre-resampling audio for pitch stretching...");
        _dummy = resample(&in_data[..], 1.0/args.pitch_scale);
        in_data = &_dummy[..];
    }
    args.length_scale *= args.pitch_scale;
    
    let out_len = (in_data.len() as f64 * args.length_scale as f64) as usize;
    
    let mut out_data = if (args.length_scale - 1.0).abs() > 0.00001
    {
        if do_multiband
        {
            let get_filter_length = |cutoff|
            {
                if args.filter_length != 0.0
                {
                    args.filter_length
                }
                else
                {
                    1.0 / cutoff * args.cutoff_steepness
                }
            }.max(0.001);
            let (bass, _temp)      = do_freq_split(&in_data[..], samplerate, get_filter_length(args.cutoff_bass_mid       ), args.cutoff_bass_mid);
            let (mid, _temp)       = do_freq_split(&_temp[..]  , samplerate, get_filter_length(args.cutoff_mid_treble     ), args.cutoff_mid_treble);
            let (treble, presence) = do_freq_split(&_temp[..]  , samplerate, get_filter_length(args.cutoff_treble_presence), args.cutoff_treble_presence);
            
            if args.combined_energy_estimation
            {
                println!("using combined energy estimation");
                
                let mut out_data = vec![Sample { l: 0.0, r: 0.0 }; out_len];
                println!("timestretching presence frequency band...");
                let presence_offs = do_timestretch(&presence [..], &mut out_data, samplerate, &args, 0, args.window_secs_presence, None);
                println!("timestretching treble frequency band...");
                let treble_offs   = do_timestretch(&treble   [..], &mut out_data, samplerate, &args, 1, args.window_secs_treble, Some(&presence_offs));
                println!("timestretching mid frequency band...");
                let mid_offs      = do_timestretch(&mid      [..], &mut out_data, samplerate, &args, 2, args.window_secs_mid, Some(&treble_offs));
                println!("timestretching bass frequency band...");
                let _             = do_timestretch(&bass     [..], &mut out_data, samplerate, &args, 3, args.window_secs_bass, Some(&mid_offs));
                out_data
            }
            else
            {
                println!("using independent energy estimation");
                
                println!("timestretching presence frequency band...");
                let mut out_presence = vec![Sample { l: 0.0, r: 0.0 }; out_len];
                let presence_offs = do_timestretch(&presence [..], &mut out_presence, samplerate, &args, 0, args.window_secs_presence, None);
                println!("timestretching treble frequency band...");
                let mut out_treble = vec![Sample { l: 0.0, r: 0.0 }; out_len];
                let treble_offs   = do_timestretch(&treble   [..], &mut out_treble  , samplerate, &args, 1, args.window_secs_treble, Some(&presence_offs));
                println!("timestretching mid frequency band...");
                let mut out_mid = vec![Sample { l: 0.0, r: 0.0 }; out_len];
                let mid_offs      = do_timestretch(&mid      [..], &mut out_mid     , samplerate, &args, 2, args.window_secs_mid, Some(&treble_offs));
                println!("timestretching bass frequency band...");
                let mut out_bass = vec![Sample { l: 0.0, r: 0.0 }; out_len];
                let _             = do_timestretch(&bass     [..], &mut out_bass    , samplerate, &args, 3, args.window_secs_bass, Some(&mid_offs));
                
                for (i, val) in out_bass.iter_mut().enumerate()
                {
                    *val += out_mid[i];
                    *val += out_treble[i];
                    *val += out_presence[i];
                }
        
                out_bass
            }
        }
        else
        {
            println!("timestretching full-spectrum audio...");
            let mut out_data = vec![Sample { l: 0.0, r: 0.0 }; out_len];
            do_timestretch(&in_data[..], &mut out_data, samplerate, &args, -1, args.fullband_window_secs, None);
            out_data
        }
    }
    else
    {
        in_data.iter().map(|x| *x).collect::<_>()
    };
    if args.pitch_scale > 1.0
    {
        println!("post-resampling audio for pitch stretching...");
        out_data = resample(&out_data[..], 1.0/args.pitch_scale);
    }
    out_data
}