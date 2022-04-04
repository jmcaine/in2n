use std::cmp;
use std::sync::{Arc, Mutex};

pub mod freqs;
pub mod setup;
pub mod song;

use setup::Setup;
use song::{Note, Score, Song};

use anyhow::{Result, bail}; // thus, return Result<T> rather than Result<T, anyhow::Error>  (that is, we're using anyhow::Result rather than std::Result)
use cpal::{SampleFormat};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{Fft, FftNum, FftDirection, num_complex::Complex, algorithm::Radix4};

const ZERO_BUF: [f32; setup::BUFFER_SIZE] = [0.0; setup::BUFFER_SIZE]; // empty buffer, for filling an outbuf with zeros
const TRIAL_BINS: usize = 3;


struct WaveMaker {
	samples_per_second: f32,
	sample_clock: f32,
	frequency: f32,
}

impl WaveMaker {
	fn new(samples_per_second: f32) -> WaveMaker {
		WaveMaker {
			samples_per_second: samples_per_second,
			sample_clock: 0.0,
			frequency: 0.0,
		}
	}
	fn set_frequency(&mut self, frequency: f32) {
		self.frequency = frequency;
	}

	fn next(&mut self) -> f32 {
		self.sample_clock = (self.sample_clock + 1.0) % self.samples_per_second;
		return (self.sample_clock * self.frequency * 2.0 * std::f32::consts::PI / self.samples_per_second).sin()
	}
}

struct Processor {
	song: Song,
	bogus: [Complex<f32>; setup::SAMPLES_PER_FFT],
	buffer: [f32; setup::BUFFER_SIZE], // shared buffer; input data is placed in it; FFT is performed in-place on data, and then new output data is written into the same space; indeces march along to keep ranges separate
	in_end: usize,  // upper (non-inclusive) index of current (most recent) input; next available input would start at this index; stays ahead of out_end (or, in the extreme case, out_end can become equal to in_end)
	out_start: usize, out_end: usize, // starting index and upper (non-inclusive) index of "available for output" data - data that can be copied to output buffer
		// note that out_end is effectively "in_start", except when processing flips back to the start of the buffer, in which case the in_start will be index 0
	samples_per_second: f32,
	fft: Radix4<f32>,
	wave_maker: WaveMaker,
}

impl Processor {
	fn new(samples_per_second: f32, buffer: [f32; setup::BUFFER_SIZE]) -> Processor { // though we don't use this, since we want this on the stack
		Processor {
			song: Song::demo_c_scale(),
			bogus: [Complex::new(0f32, 0f32); setup::SAMPLES_PER_FFT],
			buffer: buffer,
			in_end: 0,
			out_start: 0, out_end: 0,
			samples_per_second: samples_per_second,
			fft: Radix4::new(setup::SAMPLES_PER_FFT, FftDirection::Forward),
			wave_maker: WaveMaker::new(samples_per_second),
		}
	}

	pub fn add(&mut self, data: &[f32]) { // add data to our internal buffer AND (possibly) process some of it (if there's enough)
		// strategy: lay data in in one fell swoop, or "split" if at end of buffer; then loop-process SAMPLES_PER_FFT samples at a time, unless out_start and out_end are stuck at the end of the buffer; then let consume() do the FFT and move out_start and out_end  only after
		let overrun = |samples| eprintln!("Ran out of room in main buffer for all of input data - consider: is the BUFFER_SIZE large enough or is data not being consumed with reliable consistency and speed? {} samples lost", samples);
		let blocks_room = (setup::BUFFER_SIZE - self.out_end) / setup::SAMPLES_PER_FFT; // the number of SAMPLES_PER_FFT-sized "blocks" remaining in our buffer

		// xfer `data` into our buffer, circling back to the start if necessary...
		if self.in_end < self.out_end { // we've wrapped to the beginning of our (quasi-circular) buffer, even though out_start and out_end are still at the tail of the buffer as the data there awaits consumption
			let xfer_samples = cmp::min(data.len(), self.out_start - self.in_end); // lay down as much of `data` as we can, but don't encroach into out_start territory
			self.xfer_in(data, 0, xfer_samples);
			if xfer_samples < data.len() { overrun(data.len() - xfer_samples); }
		} else { // out_end <= in_end; there are data in our buffer that are not yet FFT'd; "<" when the last add() left less than one full FFT-worth (SAMPLES_PER_FFT) of data (this is the most common case), "==" whenever FFT has already been performed on all samples laid in
			if self.out_end < self.in_end { assert!(blocks_room > 0); } // we should not have started laying down samples last iteration if there wasn't room for a whole FFT
			let xfer_samples = cmp::min(data.len(), setup::SAMPLES_PER_FFT * blocks_room - (self.in_end - self.out_end));
			self.xfer_in(data, 0, xfer_samples);
			if setup::BUFFER_SIZE - self.in_end < setup::SAMPLES_PER_FFT { // there is less space remaining in the buffer than we need to do one more FFT, so we have to wrap...
				self.in_end = 0; // re-start
				let remaining_data = data.len() - xfer_samples;
				if remaining_data > 0 { // we still have some `data` to absorb; so, now that we've reset in_end, lay the data down at the beginning:
					let xfer_remainder = cmp::min(remaining_data, self.out_start); // lay down as much of `data` as we can, but don't encroach into out_start territory (technically, out_start-in_end, but in_end was just set to 0, so...)
					self.xfer_in(data, xfer_samples, xfer_remainder);
					if xfer_remainder < remaining_data { overrun(remaining_data - xfer_remainder); }
				}
			}
		}

		// now, process data that is ready...
		if self.in_end < self.out_end { // we "wrapped", above, since we came to the end of our quasi-circular buffer but still had more input data
			// we will ONLY process the tail of the buffer, and let the head get processed within consume() or next time around (next call to add()); this way, out_start and out_end can remain proper markers for the consume() call
			while self.out_end + setup::SAMPLES_PER_FFT <= setup::BUFFER_SIZE { // while out_end isn't quite to the end of the buffer...
				self.process(); // bumps out_end forward by SAMPLES_PER_FFT
			}
		} else { // our in_end and out_end are marching forward from the start of the buffer...
			while self.in_end - self.out_end >= setup::SAMPLES_PER_FFT {
				self.process(); // bumps out_end forward by SAMPLES_PER_FFT
			}
		}
	}

	pub fn consume(&mut self, out_buffer: &mut[f32]) { // consume some of the internal buffer, writing it to `out_buffer` to "play" out the data
		assert!(self.out_start <= self.out_end); // out_end should never be reset to the start of our buffer without out_start, too, which only happens when this function consumes the last of the tail of the buffer
		let samples = cmp::min(out_buffer.len(), self.out_end - self.out_start);
		self.xfer_out(out_buffer, 0, samples); // updates self.out_start

		let remainder = out_buffer.len() - samples; // NOTE that there is a case below in which remainder might == 0, and it's critical that that code run when remainder == 0, resetting out_start and out_end for the next time 'round.  The same reset has to be done if there IS a remainder of data to transfer, so it's DRY to do it this way, even if a little less legible
		if self.out_start == self.out_end {
			if self.out_start + setup::SAMPLES_PER_FFT <= setup::BUFFER_SIZE { // if we've merely run out of available data to copy into `out_buffer`...
				if remainder > 0 {
					eprintln!("Underrun - not enough prepared data to fill output buffer! {} samples will be silent, which should naturally increase latency and self-correct.", remainder);
					for i in (0 .. remainder).step_by(setup::BUFFER_SIZE) { // this loop is only to treat the (unlikely?) situation in which out_buffer (and remainder) is larger than our own BUFFER_SIZE... so we'll have to copy  zeros from our ZERO_BUF more than once...
						let real_count = cmp::min(remainder - i, setup::BUFFER_SIZE);
						out_buffer[samples + i .. samples + i + real_count].copy_from_slice(&ZERO_BUF[ .. real_count]);
					}
				}
			} else { // we've come to the end of our quasi-circular buffer, and it's time to wrap to the start, where there may be plenty of input, but it's not yet going to be process()ed...
				if self.in_end >= self.out_end {
					eprintln!("!!! in_end: {}, out_end: {}, out_start: {}, remainder: {}, samples: {}", self.in_end, self.out_end, self.out_start, remainder, samples);
				}
				assert!(self.in_end < self.out_end); // should always be true, in this case, that in_end is marching along at the front of the buffer now, and it's high time we wrapped out_end to join at the front, since out_start has caught up with out_end
				// reset to the start:
				self.out_start = 0;
				self.out_end = 0;
				// process:
				if remainder > 0 {
					let available = setup::SAMPLES_PER_FFT * (cmp::min(remainder, self.in_end - self.out_end) / setup::SAMPLES_PER_FFT); // floor the possible sample count to a multiple of SAMPLES_PER_FFT, then multiply that back up to a real sample-count
					for _ in (0 .. available).step_by(setup::SAMPLES_PER_FFT) {
						self.process(); // bumps out_end forward by SAMPLES_PER_FFT
					}
					// and put processed data into out_buffer:
					self.xfer_out(out_buffer, samples, available); // updates self.out_start
				}
			}
		}
	}

	fn xfer_out(&mut self, buffer: &mut[f32], start: usize, count: usize) {
		//println!("xfer FROM buffer[{}:{}] (buffer.len = {})", start, start + count, buffer.len());
		if count > 0 {
			buffer[start .. start + count].copy_from_slice(&self.buffer[self.out_start .. self.out_start + count]);
			self.out_start += count;
		} // else nop
	}

	fn xfer_in(&mut self, buffer: &[f32], start: usize, count: usize) {
		//println!("xfer TO buffer[{}:{}] (buffer.len = {})", start, start + count, buffer.len());
		if count > 0 {
			self.buffer[self.in_end .. self.in_end + count].copy_from_slice(&buffer[start .. start + count]);
			self.in_end += count;
		} // else nop
	}

	fn process(&mut self) { // process one chunk from self.out_end to self.out_end + SAMPLES_PER_FFT
		// do the analysis, then replace the slice of self.buffer with output data according to the analysis
		for i in 0 .. setup::SAMPLES_PER_FFT {
			self.bogus[i] = Complex::new(self.buffer[self.out_end + i], 0f32);
		}
		self.fft.process(&mut self.bogus);
		let mut candidates = [0usize; TRIAL_BINS]; // lowest spot holds index (into bogus) to highest-valued peak; higher spots hold (alleged/probable) harmonics/octaves / less powerful spikes
		for i in 0 .. setup::SAMPLES_PER_FFT {
			for j in 0 .. candidates.len() {
				if self.bogus[i].re > self.bogus[candidates[j]].re {
					for k in (j + 1 .. candidates.len()).rev() {
						candidates[k] = candidates[k - 1]
					}
					candidates[j] = i;
				}
				break;
			}
		}
		//println!("candidate[0]: {}  candidate[1]: {}  candidate[2]: {}", candidates[0], candidates[1], candidates[2]);
		let mut frequencies = [0f32; TRIAL_BINS];
		for i in 0 .. TRIAL_BINS {
			frequencies[i] = candidates[i] as f32 * self.samples_per_second / (setup::SAMPLES_PER_FFT as f32);
		}
		//println!("frequencies[0]: {}  frequencies[1]: {}  frequencies[2]: {}", frequencies[0], frequencies[1], frequencies[2]);
		let target_frequency = self.song.match_current_note(frequencies);
		if target_frequency > 0.0 {
			println!("target_frequency: {}", target_frequency);
		}

		//println!("frequency: {}", frequency);
		self.wave_maker.set_frequency(target_frequency);
		for i in 0 .. setup::SAMPLES_PER_FFT { // TODO!!!
			self.buffer[self.out_end + i] = self.wave_maker.next();
		}

		// move up self.out_end:
		self.out_end = self.out_end + setup::SAMPLES_PER_FFT;
	}

}


fn main() -> Result<()> {

	// Open audio host and default devices:
	let host = cpal::default_host();
	let out_device = host.default_output_device().expect("default output device");
	let in_device = host.default_input_device().expect("default input device");
	println!("Audio host: {}; Output device: {}, Input device: {}", host.id().name(), out_device.name()?, in_device.name()?);

	// Find appropriate stream configs:
	let out_configs = out_device.supported_output_configs()?;
	let in_configs = in_device.supported_input_configs()?;

	fn match_config<T>(configs: T, match_fn: &dyn Fn(&cpal::SupportedStreamConfig) -> bool)
		-> Result<cpal::StreamConfig>
		where T: Iterator<Item = cpal::SupportedStreamConfigRange>
	{
		for (_config_index, candidate) in configs.into_iter().enumerate() {
			let config = candidate.with_max_sample_rate();
			if match_fn(&config) {
				return Ok(config.into()); // into() converts SupportedStreamConfig to StreamConfig
			}
		}
		bail!("No appropriate I/O configs found");
	}

	// Output config: require at least 44.1k, 1 channel (tyical headset or dual-speaker will interpret 1 as a mono-duplicate-left+right), and F32 samples
	let match_out_config = |config: &cpal::SupportedStreamConfig| config.sample_rate().0 >= 44100 && config.channels() == 1 && config.sample_format() == SampleFormat::F32;
	let out_config = match_config(out_configs, &match_out_config)?;
	// Input config: also I16 samples, but only 1 channel (typical microphone), and match the output sample-rate, whatever it was
	let match_in_config = |config: &cpal::SupportedStreamConfig| config.sample_rate() == out_config.sample_rate && config.channels() == 1 && config.sample_format() == SampleFormat::F32;
	let in_config = match_config(in_configs, &match_in_config)?;
	let samples_per_second = out_config.sample_rate.0 as f32;
	// out_config.channels unused?!

	let setup = Setup::new(samples_per_second);
	
	// Demo song (c scale):
	let mut c_scale = Song::new(&setup, vec![
		/*Note::new(36, 1000),
		Note::new(38, 1000),
		Note::new(40, 1000),
		Note::new(41, 1000),
		Note::new(43, 1000),
		Note::new(45, 1000),
		Note::new(47, 1000),
		Note::new(48, 1000),
		Note::new(50, 1000),
		Note::new(52, 1000),
		Note::new(53, 1000),
		Note::new(55, 1000),
		Note::new(57, 1000),
		Note::new(59, 1000),
		Note::new(60, 1000),
		Note::new(62, 1000),*/
		Note::new(64, 1000),
		/*Note::new(65, 1000),
		Note::new(67, 1000),
		Note::new(69, 1000),
		Note::new(71, 1000),
		Note::new(72, 1000),
		Note::new(74, 1000),
		Note::new(76, 1000),
		Note::new(77, 1000),
		Note::new(79, 1000),
		Note::new(81, 1000),
		Note::new(83, 1000),
		Note::new(84, 1000),*/
	]);

	// FFT / pitch-detector:
	let fft = Radix4::new(setup::SAMPLES_PER_FFT, FftDirection::Forward);
	// Wave maker (for output):
	let wave_maker = WaveMaker::new(samples_per_second);

	// our main buffer:
	let processor = Processor::new(samples_per_second, [0.0; setup::BUFFER_SIZE]); // we create the buffer on the stack, rather than new() on the heap
	let input_buffer: Processor = Processor { song: c_scale, bogus: [Complex::new(0f32, 0f32); setup::SAMPLES_PER_FFT], buffer: [0.0; setup::BUFFER_SIZE], in_end: 0,  out_start: 0, out_end: 0, samples_per_second: samples_per_second, fft: fft, wave_maker, };	// we create on the stack, rather than using Processor::new()
	let processor_arc = Arc::new(Mutex::new(processor));

	// Set up output and error functions:
	let processor_arc_1 = Arc::clone(&processor_arc);
	let output_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
		let mut ib = processor_arc_1.lock().unwrap();
		ib.consume(data);
	};
	let out_err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
	// And input and error functions:
	let processor_arc_2 = Arc::clone(&processor_arc);
	let input_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
		let mut ib = processor_arc_2.lock().unwrap();
		ib.add(data);
	};
	let in_err_fn = |err| eprintln!("an error occurred on the input audio stream: {}", err);

	// Set up and start the actual streams:
	let out_stream = out_device.build_output_stream(&out_config, output_fn, out_err_fn)?;
	let in_stream = in_device.build_input_stream(&in_config, input_fn, in_err_fn)?;
	out_stream.play()?;
	in_stream.play()?;

	// TEST: just "play" for ... seconds:
	std::thread::sleep(std::time::Duration::from_millis(4000));
	drop(in_stream);
	drop(out_stream);
	//TODO: let rb = RingBuffer::<i32>::new(2);

	Ok(())
}

/*
FFT Analysis...

T = sampling interval = 1/sample-rate
	i.e. 44k stream -- 1/44k seconds between samples
	so 1/T = sampling frequency (1/1/44k = 44k samples per second)
N = # of samples
N/2 = # of frequency bins
	HALF of FFT bins (from fft[N/2] to fft[N]) cover frequencies from 0 to sampling-frequency/2 (you can only see spectral data up to half of the sampling frequency -- 22kHz component max for a 44k stream)
		OR IS IT fft[0] to fft[N/2] ?!!!

1/N = normalization factor: multiply it by each abs(fft-value) if care for a plot (no need if just detecting largest absolute values!)

sample-rate = 44.1k
N = 4096 samples
T = 2.3E-5 (1/44.1k)
2048 frequency bins (fft[0] to fft[2047]) cover frequencies from 0 to 22k

	so, for incremental i sample:
		value = i as f32 * 2.0 * 3.14 / 64.0;
		buffer[i].re = value.sin();
	= 1 cycle per 64 points = 1 cycles per 64/44k second = 1 cycle per .00145 seconds = 689 cycles per second = 689 Hz
		so this should show up in bin 689/22k * 2048bins = 64 (bin 64, or result[64])

	AND, for similar, but one cycle every 39 samples, rather than 64:
		value = i as f32 * 2.0 * 3.14 / 39.0;
	= 1 cycle per 39 points = 1 cycles per 39/44k second = 1 cycle per .00088 seconds = 1131 cycles per second = 1131 Hz
		so this should show up in bin 1131/22k * 2048bins = 105 (bin 105, or result[105])

so, wherever we find peak magnitude, frequency = take index (bin) * half-of-sample-rate / half-of-bin-count
	or, translated: frequency(Hz) = index * 0.5*sample_rate / 0.5*N) = index * sample_rate / N (where N = number of samples)
*/
