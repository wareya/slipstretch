mod sample;
mod sampling;


mod frontend;
use frontend::*;

mod slipstretch;
use slipstretch::*;

fn main() -> Result<(), hound::Error>
{
    use clap::Parser;
    let clap_args = Args::parse();
    let args = clap_args.to_slipstretch_args();
    
    let (in_data, samplerate) = frontend_acquire_audio(&clap_args);
    
    let out_data = run_slipstretch(&in_data, samplerate as f64, args);
    
    frontend_save_audio(&clap_args, &out_data[..], samplerate);

    Ok(())
}