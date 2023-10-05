use clap::Parser;
use crate::sample::Sample;

/// An audio stretcher.
#[derive(Clone, Debug, Parser)]
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
    
    /// Whether to enforce cross-band alignment when searching for aligned offsets.
    #[arg(long, verbatim_doc_comment, default_value_t=true, action = clap::ArgAction::Set)]
    pub (crate) cross_band_search_alignment : bool,
    
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
    
    /// The minimum number of overlapping chunks under each output sample, before doing chunk realignment.
    /// Large values give less throbbing/warbling in return for worse performance and a risk of more phasing effects.
    /// Recommended: 2 (for scales close to 1.0), 3 or 4 (for scales not close to 1.0).
    /// Minimum: 2.
    #[arg(long, verbatim_doc_comment, default_value_t=2)]
    pub (crate) window_minimum_lapping: isize,
    
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
    /// Using cutoff steepness makes higher frequency bands process with a smaller, faster filter than lower frequency bands.
    /// This can make things faster, but brings the risk of introducing more energy loss at higher cutoff frequencies.
    /// That risk of energy loss is especially true if combined energy estimation is disabled.
    /// Recommended: between 4.0 and 8.0
    #[arg(long, verbatim_doc_comment, default_value_t=8.0)]
    pub (crate) cutoff_steepness : f64,
    
    /// The length of the filter that separates each frequency bands, in seconds.
    /// The filter is a windowed sinc filter, not an IIR filter.
    /// Higher values are slower, because a larger filter must be used.
    /// Lower values have more phasing/notching artifacts where the frequency bands cross.
    /// Extremely high values will produce pre-ringing artifacts on sharp transients, in addition to being extremely slow.
    /// NOTE: If specified, overrides cutoff steepness.
    /// Recommended: between 0.02 and 0.002
    #[arg(long, verbatim_doc_comment, default_value_t=0.0)]
    pub (crate) filter_length : f64,
    
    /// Whether to do a combined energy estimation (i.e. including the higher-frequency bands) when doing chunk sliding or not.
    /// When false, you might have periods of obvious phase interference between frequency bands, especially if cutoff steepness is low.
    /// When true, you might have moments of slight, temporary energy loss in mid frequencies after transients.
    #[arg(long, verbatim_doc_comment, default_value_t=true, action = clap::ArgAction::Set)]
    pub (crate) combined_energy_estimation : bool,
    
    /// Whether to offset frequency bands in a way that attempts to prevent transients from sounding "wet".
    //#[arg(long, verbatim_doc_comment, default_value_t=true, action = clap::ArgAction::Set)]
    //pub (crate) smart_band_offset : bool,
    
    /// Amplitude of the bass frequency band.
    #[arg(long, default_value_t=1.0)]
    pub (crate) amplitude_bass : f64,
    
    /// Amplitude of the mid frequency band.
    #[arg(long, default_value_t=1.0)]
    pub (crate) amplitude_mid : f64,
    
    /// Amplitude of the treble frequency band.
    #[arg(long, default_value_t=1.0)]
    pub (crate) amplitude_treble : f64,
    
    /// Amplitude of the presence frequency band.
    #[arg(long, default_value_t=1.0)]
    pub (crate) amplitude_presence : f64,
    
    /// Disable the bass-mid-treble split without disabling the presence split. Bass, mids, and treble will end up in a single band, and presence in another band.
    /// The bass-mid-treble band will use the settings for the bass band. The presence band will use the settings for the presence band. The cutoff frequency will be the treble-presence cutoff frequency.
    /// This is useful for doing multi-band resizing with speech.
    #[arg(long, verbatim_doc_comment, default_value_t=false, action = clap::ArgAction::Set)]
    pub (crate) filter_disable_bass_mid_treble_split : bool,
    
    /// Whether to attempt to detect and follow the underlying main tone and pick a window size that matches it.
    /// Has a big negative performance impact and usually makes things worse, except for very simple monophonic inputs.
    #[arg(long, verbatim_doc_comment, default_value_t=false, action = clap::ArgAction::Set)]
    pub (crate) attempt_match_frequency : bool,
}

use crate::slipstretch::SlipStretchArgs;
impl Args {
    pub (crate) fn to_slipstretch_args(&self) -> SlipStretchArgs
    {
        SlipStretchArgs
        {
            length_scale: self.length_scale,
            pitch_scale: self.pitch_scale,

            slip_range: self.slip_range,

            fullband_window_secs: self.fullband_window_secs,

            search_subdiv: self.search_subdiv,
            search_pass_count: self.search_pass_count,
            cross_band_search_alignment: self.cross_band_search_alignment,
            
            window_minimum_lapping : self.window_minimum_lapping,
            
            window_secs_bass : self.window_secs_bass,
            window_secs_mid : self.window_secs_mid,
            window_secs_treble : self.window_secs_treble,
            window_secs_presence : self.window_secs_presence,
            
            cutoff_bass_mid : self.cutoff_bass_mid,
            cutoff_mid_treble : self.cutoff_mid_treble,
            cutoff_treble_presence : self.cutoff_treble_presence,
            
            cutoff_steepness : self.cutoff_steepness,
            filter_length : self.filter_length,
            
            combined_energy_estimation : self.combined_energy_estimation,
            
            //smart_band_offset : self.smart_band_offset,
            
            amplitude_bass : self.amplitude_bass,
            amplitude_mid : self.amplitude_mid,
            amplitude_treble : self.amplitude_treble,
            amplitude_presence : self.amplitude_presence,
            
            filter_disable_bass_mid_treble_split : self.filter_disable_bass_mid_treble_split,
            
            attempt_match_frequency : self.attempt_match_frequency,
        }
    }
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

    let in_data : Vec<Sample> = in_data
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