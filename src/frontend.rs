use clap::Parser;
use std::path::PathBuf;

use crate::sample::Sample;

/// An audio stretcher.
#[derive(Debug, Parser)]
pub (crate) struct Args {
    /// Input wav filename.
    pub (crate) in_file_name: String,

    /// Output wav filename
    pub (crate) out_file_name: String,

    /// Length scale. A value of 0.5 makes the audio play twice as fast.
    /// To convert a 120bpm song to 140bpm, you would run 120/140 through a calculator and use that value.
    #[arg(short = 'l', long, verbatim_doc_comment, default_value_t=1.0)]
    pub (crate) length_scale: f64,

    /// Pitch scale. A value of 0.5 makes the audio be pitched down by an octave.
    /// To pitch shift by semitones, run 2^(<semitones>/12) through a calculator and use that value.
    #[arg(short = 'p', long, verbatim_doc_comment, default_value_t=1.0)]
    pub (crate) pitch_scale: f64,

    /// The range in which the shifting stage of the algorithm is allowed to try to shift each window to reduce phasing artifacts.
    /// Recommendation: Leave at default, or set to lower than default if you're having bad flam or tempo shifting artifacts.
    /// Higher values will have more flam/tempo shifting artifacts. Lower values will have more phasing artifacts.
    /// Maximum: 0.5.
    #[arg(short = 'r', long, verbatim_doc_comment, default_value_t=0.25)]
    pub (crate) slip_range: f64,

    /// Window size for full-band pitch shifting, in seconds.
    /// This causes the algorithm to run without splitting the audio up into multiple frequency bands first.
    /// If unset, the algorithm will run on frequency bands instead, which is better for most types of audio.
    /// Useful values range from 0.2 to 0.01, depending on the audio.
    /// 0.08 works decently for speech.
    /// 0.2 works for music but you should leave this unset for music and let it do multiband mode for music.
    /// Recommendation: Do not use. Leave unset.
    #[arg(short = 'z', long, verbatim_doc_comment, default_value_t=0.0)]
    pub (crate) fullband_window_secs: f64,

    /// Number of times to subdivide the slip range in each direction when searching for good chunk alignment.
    #[arg(long, default_value_t=5)]
    pub (crate) search_subdiv: isize,
    
    /// Number of increasingly subdivided search passes to do when searching for good chunk alignment.
    #[arg(long, default_value_t=5)]
    pub (crate) search_pass_count: u32,
    
    /// In multiband mode (the default mode), the chunk window size (in seconds) used when stretching the bass frequency band.
    /// Smaller chunk window sizes give better time-domain but worse frequency-domain results.
    #[arg(long, verbatim_doc_comment, default_value_t=0.2)]
    pub (crate) window_secs_bass : f64,
    /// In multiband mode, the chunk window size used when stretching the mid frequency band.
    #[arg(long, default_value_t=0.16)]
    pub (crate) window_secs_mid : f64,
    /// In multiband mode, the chunk window size used when stretching the treble frequency band.
    #[arg(long, default_value_t=0.08)]
    pub (crate) window_secs_treble : f64,
    /// In multiband mode, the chunk window size used when stretching the presence frequency band.
    #[arg(long, default_value_t=0.008)]
    pub (crate) window_secs_presence : f64,
    
    /// For multiband mode, the cutoff frequency between the bass and mid frequency bands.
    #[arg(long, default_value_t=400.0)]
    pub (crate) cutoff_bass_mid : f64,
    /// For multiband mode, the cutoff frequency between the mid and treble frequency bands.
    #[arg(long, default_value_t=1600.0)]
    pub (crate) cutoff_mid_treble : f64,
    /// For multiband mode, the cutoff frequency between the treble and presence frequency bands.
    #[arg(long, default_value_t=4800.0)]
    pub (crate) cutoff_treble_presence : f64,
    
    /// The steepness of the filter that separates each frequency bands.
    /// The filter is a windowed sinc filter, not an IIR filter.
    /// This value is proportional to (but not equal to) the number of lobes present in the windowed sinc kernel.
    /// Higher values are slower, because a larger filter must be used.
    /// Lower values have more phasing/notching artifacts where the frequency bands cross.
    /// Extremely high values will produce pre-ringing artifacts on sharp transients, in addition to being extremely slow.
    #[arg(long, verbatim_doc_comment, default_value_t=8.0)]
    pub (crate) cutoff_steepness : f64,
    
    /// Whether to do a combined energy estimation (i.e. including the higher-frequency bands) when doing chunk sliding or not.
    /// When false, you might have long periods of obvious phase interference between frequency bands, especially if cutoff steepness is low.
    /// When true, you might have moments of temporary energy loss in mid frequencies after transients.
    #[arg(long, verbatim_doc_comment, default_value_t=false, action = clap::ArgAction::Set)]
    pub (crate) combined_energy_estimation : bool,
    
    /// Whether to offset frequency bands in a way that attempts to prevent transients from sounding "wet".
    #[arg(long, verbatim_doc_comment, default_value_t=true, action = clap::ArgAction::Set)]
    pub (crate) smart_band_offset : bool,
}

pub (crate) fn frontend_acquire_audio(args : &Args) -> (Vec<Sample>, f64)
{
    let mut reader = hound::WavReader::open(&args.in_file_name).unwrap();
    
    let in_data: Vec<f32> = match (reader.spec().sample_format, reader.spec().bits_per_sample)
    {
        (hound::SampleFormat::Int, 16) => reader.samples::<i16>().map(|sample| sample.unwrap() as f32 / 32768.0).collect(),
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().map(|sample| sample.unwrap()).collect(),
        _ => panic!("unsupported audio format (only 16-bit int and 32-bit float wav files are supported)")
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
    
    (in_data, samplerate.into())
}

pub (crate) fn frontend_save_audio(args : &Args, output : &[Sample], samplerate : f64)
{
    let out: Vec<_> = output.into_iter().flat_map(|sample| vec![sample.l, sample.r]).collect();

    let spec = hound::WavSpec
    {
        channels: 2,
        sample_rate: samplerate as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&args.out_file_name, spec).unwrap();

    for sample in out
    {
        writer.write_sample(sample).unwrap();
    }

    println!("Audio data saved to {}", args.out_file_name);
}