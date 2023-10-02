use clap::Parser;
use std::path::PathBuf;

/// An audio stretcher.
#[derive(Debug, Parser)]
pub (crate) struct Args {
    /// Input wav filename.
    pub (crate) in_file_name: String,

    /// Output wav filename
    pub (crate) out_file_name: String,

    /// Length scale. A value of 0.5 makes the audio play twice as fast.
    /// To convert a 120bpm song to 140bpm, you would run 120/140 through a calculator and use that value.
    #[arg(short = 'l', long, default_value_t=1.0)]
    pub (crate) length_scale: f64,

    /// Pitch scale. A value of 0.5 makes the audio be pitched down by an octave.
    /// To pitch shift by semitones, run 2^(<semitones>/12) through a calculator and use that value.
    #[arg(short = 'p', long, default_value_t=1.0)]
    pub (crate) pitch_scale: f64,

    /// The range in which the shifting stage of the algorithm is allowed to try to shift each window to reduce phasing artifacts.
    /// Recommendation: Leave at default, or set to lower than default if you're having bad flam or tempo shifting artifacts.
    /// Higher values will have more flam/tempo shifting artifacts. Lower values will have more phasing artifacts.
    /// Default: 0.25. Maximum: 0.5.
    #[arg(short = 's', long, default_value_t=0.25)]
    pub (crate) slip_range: f64,

    /// Window size for full-band pitch shifting, in seconds.
    /// This causes the algorithm to run without splitting the audio up into multiple frequency bands first.
    /// If unset, the algorithm will run on frequency bands instead, which is better for most types of audio.
    /// Useful values range from 0.2 to 0.01, depending on the audio. 0.08 works decently well for speech.
    /// Recommendation: Do not use. Leave unset.
    #[arg(short = 'z', long, default_value_t=0.0)]
    pub (crate) fullband_window_secs: f64,

    /// Number of times to subdivide the slip range in each direction when searching for good chunk alignment.
    #[arg(long, default_value_t=5)]
    pub (crate) search_subdiv: isize,
    
    /// Number of increasingly subdivided search passes to do when searching for good chunk alignment.
    #[arg(long, default_value_t=4)]
    pub (crate) search_pass_count: u32,
    
    /// In multiband mode (the default mode), the chunk window size (in seconds) used when stretching the bass frequency band.
    /// Smaller chunk window sizes give better time-domain but worse frequency-domain results.
    #[arg(long, default_value_t=0.2)]
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
    /// Higher values are slower, because a larger filter must be used. Lower values have more phasing artifacts where the frequency bands cross.
    /// Extremely high values will produce pre-ringing artifacts on sharp transients.
    #[arg(long, default_value_t=4.0)]
    pub (crate) cutoff_steepness : f64,
}